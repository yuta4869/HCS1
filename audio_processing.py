# audio_processing.py

import datetime
import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
import collections

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import tkinter as tk

from faster_whisper import WhisperModel
from hrf2_controller import HRF2Controller, HRF2Config, ControlMode, AdaptiveConfig, GainType

import config

if sys.version_info >= (3, 9):
    from typing import ForwardRef
    HeartRateMonitor = ForwardRef('HeartRateMonitor')
    H10Monitor = ForwardRef('H10Monitor')
else:
    HeartRateMonitor = Any  # type: ignore
    H10Monitor = Any  # type: ignore


class ProsodySettings:
    """音声合成の抑揚や速度などの値をまとめて管理するための設定クラス。"""

    def __init__(self) -> None:
        # 各種拡大率
        self.intonation_scale: float = 1.0
        self.pitch_scale: float = 0.0
        self.speed_scale: float = 1.0
        self.energy_scale: float = 1.0  # VOICEVOX の volumeScale
        self.pause_duration_scale: float = 1.0  # pre/postPhonemeLength への倍率

        # HFB の有効・無効
        self.hfb_enabled: bool = False

        # 各種範囲
        self.min_intonation, self.max_intonation = 0.0, 2.0
        self.min_pitch, self.max_pitch = -0.15, 0.15
        self.min_speed, self.max_speed = 0.5, 2.0
        self.min_energy, self.max_energy = 0.0, 2.0
        self.min_pause_duration, self.max_pause_duration = 0.0, 2.0

        # GUI から接続される変数 (tkinter)
        self.intonation_var: Optional[tk.DoubleVar] = None
        self.pitch_var: Optional[tk.DoubleVar] = None
        self.speed_var: Optional[tk.DoubleVar] = None
        self.energy_var: Optional[tk.DoubleVar] = None
        self.pause_duration_var: Optional[tk.DoubleVar] = None
        self.hfb_enabled_var: Optional[tk.BooleanVar] = None

        # --- HFB: 正弦波モード用 ---
        self.sinusoidal_hfb_enabled: bool = False
        self.sinusoidal_hfb_sequence: List[float] = self._generate_sinusoidal_sequence()
        self.sinusoidal_hfb_step_index: int = 0
        if self.sinusoidal_hfb_sequence:
            try:
                self.sinusoidal_hfb_step_index = self.sinusoidal_hfb_sequence.index(1.0)
            except ValueError:
                self.sinusoidal_hfb_step_index = 0
        self.sinusoidal_hfb_enabled_var: Optional[tk.BooleanVar] = None

        # --- HFB: どのパラメータを変えるか (抑揚 / ピッチ / 速度 / 音量 / 間) ---
        # 既定は抑揚
        self.hfb_target_param: str = "intonation"
        self.hfb_target_param_var: Optional[tk.StringVar] = None

        # --- HRF2: PID制御による心拍数追従モード ---
        self.hrf2_controller = HRF2Controller()
        self.hrf2_enabled: bool = False
        self.hrf2_enabled_var: Optional[tk.BooleanVar] = None

    # ------------------------------------------------------------------
    # 基本パラメータ操作
    # ------------------------------------------------------------------
    def set_parameter(self, param_name: str, value: float) -> None:
        """param_name で指定した項目の倍率を、範囲内に収めて設定する。"""
        min_val = getattr(self, f"min_{param_name}")
        max_val = getattr(self, f"max_{param_name}")
        clamped_value = max(min_val, min(max_val, value))
        setattr(self, f"{param_name}_scale", clamped_value)

        tk_var = getattr(self, f"{param_name}_var", None)
        if tk_var is not None:
            try:
                tk_var.set(clamped_value)
            except tk.TclError:
                # GUI がまだ無い場合など
                pass

    def get_parameter(self, param_name: str) -> float:
        return getattr(self, f"{param_name}_scale")

    def get_parameter_range(self, param_name: str) -> Tuple[float, float]:
        return getattr(self, f"min_{param_name}"), getattr(self, f"max_{param_name}")

    # ------------------------------------------------------------------
    # HFB 有効・無効
    # ------------------------------------------------------------------
    def enable_hfb(self, enable: bool) -> None:
        self.hfb_enabled = enable
        print(f"HFB (Direct Adjustment) set to {'enabled' if enable else 'disabled'}.")
        if self.hfb_enabled_var is not None:
            try:
                self.hfb_enabled_var.set(enable)
            except tk.TclError:
                pass

    def is_hfb_enabled(self) -> bool:
        return self.hfb_enabled

    # ------------------------------------------------------------------
    # 正弦波 HFB (抑揚用)
    # ------------------------------------------------------------------
    def enable_sinusoidal_hfb(self, enable: bool) -> None:
        self.sinusoidal_hfb_enabled = enable
        print(f"Sinusoidal HFB set to {'enabled' if enable else 'disabled'}.")
        if self.sinusoidal_hfb_enabled_var is not None:
            try:
                self.sinusoidal_hfb_enabled_var.set(enable)
            except tk.TclError:
                pass
        if enable:
            self.reset_sinusoidal_hfb_state_values()

    def is_sinusoidal_hfb_enabled(self) -> bool:
        return self.sinusoidal_hfb_enabled

    def _generate_sinusoidal_sequence(self) -> List[float]:
        """抑揚倍率を 1.0 → 2.0 → 0.0 付近 → 1.0 とゆっくり変化させるための値列を作る。"""
        seq: List[float] = []
        # 上昇 1.0 → 2.0 (0.1 刻み, 11 個)
        for i in range(11):
            seq.append(round(1.0 + i * 0.1, 1))
        # 下降 1.9 → -0.1 (0.1 刻み, 20 個) ただし後で範囲内に切り詰める
        for i in range(20):
            seq.append(round(1.9 - i * 0.1, 1))
        # 下から 0.1 → 0.9 へ戻る (0.1 刻み, 9 個)
        for i in range(9):
            seq.append(round(0.1 + i * 0.1, 1))
        return seq or [1.0]

    def get_next_sinusoidal_intonation(self) -> float:
        """正弦波 HFB 用に、次に使う抑揚倍率を一つ進めて返す。"""
        if not self.sinusoidal_hfb_sequence:
            self.sinusoidal_hfb_step_index = 0
            return 1.0
        current_intonation = self.sinusoidal_hfb_sequence[self.sinusoidal_hfb_step_index]
        self.sinusoidal_hfb_step_index = (self.sinusoidal_hfb_step_index + 1) % len(self.sinusoidal_hfb_sequence)
        # 範囲で切り詰める
        return max(self.min_intonation, min(self.max_intonation, current_intonation))

    def reset_sinusoidal_hfb_state_values(self) -> None:
        """正弦波 HFB の内部状態を初期位置 (倍率 1.0) に戻す。"""
        initial_value = 1.0
        try:
            self.sinusoidal_hfb_step_index = self.sinusoidal_hfb_sequence.index(initial_value)
        except ValueError:
            self.sinusoidal_hfb_step_index = 0

        if self.sinusoidal_hfb_sequence:
            current_val_display = self.sinusoidal_hfb_sequence[self.sinusoidal_hfb_step_index]
        else:
            current_val_display = 1.0

        if self.intonation_var is not None:
            try:
                self.intonation_var.set(current_val_display)
            except tk.TclError:
                pass
        print(f"Sinusoidal HFB state reset. Current intonation value: {current_val_display:.2f}")

    # ------------------------------------------------------------------
    # HFB でどのパラメータを変えるか
    # ------------------------------------------------------------------
    def set_hfb_target_param(self, param_name: str) -> None:
        """HFB で操作する対象パラメータ名を設定する。"""
        valid_params = {"intonation", "pitch", "speed", "energy", "pause_duration"}
        if param_name not in valid_params:
            print(f"[ProsodySettings] Invalid hfb_target_param '{param_name}', fallback to 'intonation'.")
            param_name = "intonation"
        self.hfb_target_param = param_name
        if self.hfb_target_param_var is not None:
            try:
                self.hfb_target_param_var.set(param_name)
            except tk.TclError:
                pass
        print(f"HFB target parameter set to: {param_name}")

    def get_hfb_target_param(self) -> str:
        return self.hfb_target_param

    def get_current_mode(self) -> str:
        """
        現在のモードを文字列で返す。
        Returns:
            "Sin" - 正弦波モード
            "HRF" - 心拍数フィードバックモード（直接調整）
            "HRF2_<制御種別>" - HRF2の制御モード（例: HRF2_PID, HRF2_Adaptive, HRF2_GS）
            "Fixed" - 抑揚固定モード（どちらも無効）
        """
        if self.sinusoidal_hfb_enabled:
            return "Sin"
        elif self.hrf2_enabled:
            control_mode = getattr(self.hrf2_controller, "control_mode", None)
            if control_mode == ControlMode.GAIN_SCHEDULED:
                control_label = "GS"
            elif control_mode:
                control_label = control_mode.value
            else:
                control_label = "PID"
            return f"HRF2_{control_label}"
        elif self.hfb_enabled:
            return "HRF"
        else:
            return "Fixed"

    def get_current_mode_display(self) -> str:
        """
        現在のモードを表示用の文字列で返す。
        Returns:
            "正弦波モード (Sin)" / "心拍フィードバック (HRF)" / "心拍追従 (HRF2)" / "抑揚固定 (Fixed)"
        """
        mode = self.get_current_mode()
        if mode == "Sin":
            return "正弦波モード (Sin)"
        elif mode.startswith("HRF2"):
            control_label = mode.split("_", 1)[1] if "_" in mode else ""
            if control_label:
                label_display = "GS (GainSchedule)" if control_label == "GS" else control_label
                return f"心拍追従 (HRF2 / {label_display})"
            return "心拍追従 (HRF2)"
        elif mode == "HRF":
            return "心拍フィードバック (HRF)"
        else:
            return "抑揚固定 (Fixed)"

    # ------------------------------------------------------------------
    # HRF2: PID制御による心拍数追従
    # ------------------------------------------------------------------
    def enable_hrf2(self, enable: bool) -> None:
        """HRF2モードの有効/無効を設定"""
        self.hrf2_enabled = enable
        self.hrf2_controller.enabled = enable
        print(f"HRF2 (PID Control) set to {'enabled' if enable else 'disabled'}.")
        if self.hrf2_enabled_var is not None:
            try:
                self.hrf2_enabled_var.set(enable)
            except tk.TclError:
                pass
        if enable:
            self.hrf2_controller.reset()

    def is_hrf2_enabled(self) -> bool:
        return self.hrf2_enabled

    def get_hrf2_output(self, current_hr: float) -> Tuple[float, dict]:
        """HRF2コントローラーから抑揚レベルを取得"""
        return self.hrf2_controller.update(current_hr)

    def set_hrf2_target_hr(self, target_hr: float) -> None:
        """HRF2の目標心拍数を設定"""
        self.hrf2_controller.target_hr = target_hr

    def get_hrf2_target_hr(self) -> float:
        """HRF2の目標心拍数を取得"""
        return self.hrf2_controller.target_hr

    def set_hrf2_pid_gains(self, kp: float, ki: float, kd: float) -> None:
        """HRF2のPIDゲインを設定"""
        self.hrf2_controller.set_pid_gains(kp, ki, kd)

    # ------------------------------------------------------------------
    # HRF2: 制御モード切り替え（PID / 適応制御）
    # ------------------------------------------------------------------
    def set_hrf2_control_mode(self, mode: ControlMode) -> None:
        """HRF2の制御モードを設定"""
        self.hrf2_controller.control_mode = mode
        print(f"HRF2 control mode set to: {mode.value}")

    def get_hrf2_control_mode(self) -> ControlMode:
        """HRF2の制御モードを取得"""
        return self.hrf2_controller.control_mode

    def set_hrf2_adaptive_params(self, gamma: float, tau: float) -> None:
        """HRF2の適応制御パラメータを設定"""
        self.hrf2_controller.set_adaptive_params(gamma, tau)

    def get_hrf2_adaptive_theta(self) -> float:
        """HRF2の適応パラメータθを取得"""
        return self.hrf2_controller.get_adaptive_theta()

    def get_hrf2_adaptive_config(self) -> AdaptiveConfig:
        """HRF2の適応制御設定を取得"""
        return self.hrf2_controller.get_adaptive_config()

    def get_hrf2_gain_schedule_config(self):
        """HRF2のゲインスケジューリング設定を取得"""
        return self.hrf2_controller.get_gain_schedule_config()

    def set_hrf2_gain_schedule_thresholds(self, high: float, medium: float) -> None:
        """HRF2のゲインスケジューリング閾値を設定"""
        self.hrf2_controller.set_gain_schedule_thresholds(high, medium)

    def get_hrf2_gain_schedule_zone(self) -> str:
        """HRF2の現在のゲイン領域を取得"""
        return self.hrf2_controller.get_gain_schedule_zone()

    def set_hrf2_gain_schedule_types(self, high: GainType, medium: GainType, low: GainType) -> None:
        """HRF2のゲインスケジューリングのゲイン種類を設定"""
        self.hrf2_controller.set_gain_schedule_types(high, medium, low)

    def get_hrf2_gain_schedule_gain_type(self) -> str:
        """HRF2の現在使用中のゲイン種類を取得"""
        gain_type = self.hrf2_controller.get_gain_schedule_gain_type()
        return gain_type.value if gain_type else "---"

    # --- Robust制御用メソッド ---
    def get_hrf2_robust_adapted_gain(self) -> float:
        """HRF2のRobust制御の適応ゲインを取得"""
        return self.hrf2_controller.get_robust_adapted_gain()

    def get_hrf2_robust_config(self):
        """HRF2のRobust制御設定を取得"""
        return self.hrf2_controller.get_robust_config()


class VoicevoxManager:
    @staticmethod
    def check_server() -> bool:
        try:
            response = requests.get(f"{config.VOICEVOX_URL}/version", timeout=3)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"VOICEVOX server connection failed: {e}")
            return False

    @staticmethod
    def get_speakers() -> List[Dict[str, Any]]:
        speakers_list: List[Dict[str, Any]] = []
        try:
            response = requests.get(f"{config.VOICEVOX_URL}/speakers", timeout=5)
            response.raise_for_status()
            speakers_data = response.json()
            if not speakers_data:
                print("VOICEVOX speakers response is empty.")
                return []

            for speaker in speakers_data:
                speaker_name = speaker.get("name", "Unknown")
                styles = speaker.get("styles", [])
                if not styles:
                    print(f"Speaker '{speaker_name}' has no styles information.")
                    continue
                for style in styles:
                    style_id = style.get("id")
                    style_name = style.get("name", "Default")
                    if style_id is not None:
                        speakers_list.append({
                            'name': f"{speaker_name} - {style_name}",
                            'id': style_id
                        })
            return speakers_list
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Error retrieving speaker info from VOICEVOX: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error when getting VOICEVOX speakers: {e}")
            return []


class SpeakerSettings:
    def __init__(self, speakers_list: List[Dict[str, Any]]):
        self.speakers: List[Dict[str, Any]] = speakers_list
        self.current_style_id: int = speakers_list[0]['id'] if speakers_list else 0
        if self.current_style_id == 0:
            print("No speakers available or failed to load speakers. Default speaker ID is 0.")

    def get_speaker_name_by_id(self, style_id: int) -> Optional[str]:
        return next((s['name'] for s in self.speakers if s['id'] == style_id), None)

    def get_all_speaker_names(self) -> List[str]:
        return [s['name'] for s in self.speakers]

    def get_all_speaker_ids(self) -> List[int]:
        return [s['id'] for s in self.speakers]


class AudioProcessor:
    def __init__(self,
                 prosody_settings: ProsodySettings,
                 speaker_settings: SpeakerSettings,
                 hr_monitor: HeartRateMonitor,
                 h10_monitor: H10Monitor,
                 log_queue_ref: queue.Queue,
                 faster_whisper_model_instance: WhisperModel,
                 input_device_index: Optional[int] = None):
        self.prosody = prosody_settings
        self.speaker = speaker_settings
        self.hr_monitor = hr_monitor
        self.h10_monitor = h10_monitor
        self.log_queue = log_queue_ref
        self.whisper_model = faster_whisper_model_instance
        self.app: Optional[Any] = None

        self.stop_event = threading.Event()
        self.tts_lock = threading.Lock()

        self.sample_rate: int = 16000
        self.channels: int = 1
        # config.pyから設定を読み込み（Linux/Ubuntu向けの安定性設定）
        self.chunk_size: int = getattr(config, 'AUDIO_CHUNK_SIZE', 2048)
        self.silent_threshold: float = 0.02

        # Linux/Ubuntu向けの安定性設定
        # input overflow対策: バッファを大きくし、latencyを設定
        self.stream_latency = getattr(config, 'AUDIO_LATENCY', "high")
        self.min_record_seconds: float = 1.0 # Will be used for VAD
        self.required_silent_seconds: float = 1.8
        self.min_audio_length: float = 0.3

        # 入力デバイス（マイク）の設定
        # None = デフォルトデバイス、または明示的にインデックスを指定
        self.input_device_index = input_device_index

        self.last_hr_after_tts: Optional[int] = None
        self.hr_used_for_last_adjustment: Optional[int] = None

        # --- Real-time Transcription / VAD Attributes ---
        self.audio_q = queue.Queue() # Live audio chunks from stream callback
        self.stream_thread = threading.Thread(target=self._stream_input_loop, daemon=True)
        self.stream_stop_event = threading.Event()

        self.interim_transcription_queue = queue.Queue() # For GUI
        self.final_text_queue = queue.Queue() # For ConversationManager

        # トランスクリプション用キューとワーカー（順次処理でメモリ節約）
        self.transcription_queue = queue.Queue()
        self.transcription_worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.transcription_stop_event = threading.Event()

        self.vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
        self.vad_stop_event = threading.Event()
        self.vad_state: str = "WAITING"
        self.current_utterance_chunks: List[np.ndarray] = []
        
    def start_streaming_input(self):
        """Starts the continuous audio input stream."""
        if not self.stream_thread.is_alive():
            print("Starting continuous audio input stream...")
            self.stream_stop_event.clear()
            self.stream_thread.start()
        # トランスクリプションワーカーも開始
        if not self.transcription_worker_thread.is_alive():
            print("Starting transcription worker...")
            self.transcription_stop_event.clear()
            self.transcription_worker_thread.start()

    def stop_streaming_input(self):
        """Stops the continuous audio input stream."""
        if self.stream_thread.is_alive():
            print("Stopping continuous audio input stream...")
            self.stream_stop_event.set()
            self.stream_thread.join(timeout=2)

    def _stream_input_loop(self):
        """The main loop for the audio input stream.

        コールバック方式を使用。overflowはログに出さず無視する。
        """
        def _stream_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            # statusは無視（input overflowログを出さない）
            self.audio_q.put(indata.copy())

        try:
            # デバイス情報をログに出力
            device_info = ""
            if self.input_device_index is not None:
                try:
                    dev = sd.query_devices(self.input_device_index)
                    device_info = f" (デバイス: [{self.input_device_index}] {dev['name']})"
                except Exception:
                    device_info = f" (デバイスID: {self.input_device_index})"
            else:
                default_idx = sd.default.device[0]
                if default_idx is not None:
                    dev = sd.query_devices(default_idx)
                    device_info = f" (デフォルト: [{default_idx}] {dev['name']})"

            print(f"Audio stream thread started.{device_info} (blocksize: {self.chunk_size})")

            with sd.InputStream(samplerate=self.sample_rate,
                                channels=self.channels,
                                blocksize=self.chunk_size,
                                device=self.input_device_index,
                                callback=_stream_callback):
                while not self.stream_stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error in audio input stream: {e}", file=sys.stderr)
        finally:
            print("Audio stream thread finished.")

    def start_vad_loop(self):
        """Starts the real-time transcription and VAD loop."""
        if not self.vad_thread.is_alive():
            print("Starting VAD and transcription loop...")
            self.vad_stop_event.clear()
            self.vad_thread.start()

    def stop_vad_loop(self):
        """Stops the real-time transcription and VAD loop."""
        if self.vad_thread.is_alive():
            print("Stopping VAD and transcription loop...")
            self.vad_stop_event.set()
            self.vad_thread.join(timeout=2)

    def _vad_loop(self):
        """
        The main loop for Voice Activity Detection.
        Processes audio chunks to detect utterances and sends them for transcription.
        """
        print("VAD loop started.")
        silent_since = None
        
        while not self.vad_stop_event.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.1)
                rms = float(np.sqrt(np.mean(np.square(chunk))))
                is_loud = rms > self.silent_threshold

                if self.vad_state == "WAITING":
                    if is_loud:
                        print("Speech start detected.")
                        self.vad_state = "RECORDING"
                        self.current_utterance_chunks.append(chunk)
                        silent_since = None
                        if self.app:
                            self.app.after(0, lambda: self.app.set_status("Listening...", "cyan"))

                elif self.vad_state == "RECORDING":
                    self.current_utterance_chunks.append(chunk)
                    if is_loud:
                        silent_since = None
                    else:
                        if silent_since is None:
                            silent_since = time.time()
                        
                        silence_duration = time.time() - silent_since
                        total_duration = sum(len(c) for c in self.current_utterance_chunks) / self.sample_rate
                        
                        if silence_duration >= self.required_silent_seconds and total_duration >= self.min_record_seconds:
                            print("Utterance end detected.")

                            # キューに追加（ワーカースレッドが順次処理）
                            self.transcription_queue.put(list(self.current_utterance_chunks))

                            self.current_utterance_chunks = []
                            self.vad_state = "WAITING"
                            silent_since = None
                            if self.app:
                                self.app.after(0, lambda: self.app.set_status("Processing...", "orange"))

            except queue.Empty:
                # If the queue is empty, check if we were in the middle of an utterance
                if self.vad_state == "RECORDING" and silent_since:
                    silence_duration = time.time() - silent_since
                    total_duration = sum(len(c) for c in self.current_utterance_chunks) / self.sample_rate
                    if silence_duration >= self.required_silent_seconds and total_duration >= self.min_record_seconds:
                        print("Utterance end detected (timeout).")
                        self.transcription_queue.put(list(self.current_utterance_chunks))
                        self.current_utterance_chunks = []
                        self.vad_state = "WAITING"
                        if self.app:
                            self.app.after(0, lambda: self.app.set_status("Processing...", "orange"))
                continue
            except Exception as e:
                print(f"Error in VAD loop: {e}", file=sys.stderr)

        print("VAD loop finished.")

    def _transcription_worker(self):
        """トランスクリプションを順次処理するワーカースレッド"""
        print("Transcription worker started.")
        while not self.transcription_stop_event.is_set():
            try:
                # キューから音声データを取得（タイムアウト付き）
                audio_chunks = self.transcription_queue.get(timeout=0.5)
                self._process_utterance(audio_chunks)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in transcription worker: {e}")
        print("Transcription worker finished.")

    def _process_utterance(self, audio_chunks: List[np.ndarray]):
        """Transcribes a completed utterance and puts the final text in a queue."""
        if not audio_chunks:
            return

        audio_data = np.concatenate(audio_chunks)
        duration = len(audio_data) / self.sample_rate
        print(f"--- Processing utterance of {duration:.2f} seconds. ---")

        try:
            print("[_process_utterance] Starting transcription...")
            segments, info = self.whisper_model.transcribe(
                audio_data,  # Pass numpy array directly
                beam_size=config.WHISPER_TRANSCRIBE_BEAM_SIZE
            )
            print("[_process_utterance] Transcription finished.")

            final_text = "".join(segment.text for segment in segments).strip()
            print(f"[_process_utterance] Raw transcription result: '{final_text}'")

            if final_text:
                print(f"Final text: '{final_text}'")
                # Pass both text and audio data to the conversation manager
                self.final_text_queue.put((final_text, audio_data))
                # Also update the interim queue for immediate GUI feedback
                self.interim_transcription_queue.put(f"User: {final_text}")
            else:
                print("Transcription resulted in empty text.")
                self.interim_transcription_queue.put("")

        except Exception as e:
            print(f"Error during utterance transcription: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            if self.app:
                self.app.after(0, lambda: self.app.set_status("Ready", "green"))
            print("--- Utterance processing finished. ---")
    
    def record_audio(self, filename: str = "input.wav") -> Tuple[bool, Optional[datetime.datetime], Optional[datetime.datetime]]:
        """録音を行い、WAVファイルに保存する

        コールバック方式を使用。overflowはログに出さず無視する。
        """
        audio_q = queue.Queue()
        self.stop_event.clear()
        recording_success = False

        rec_start_dt: Optional[datetime.datetime] = None
        rec_end_dt: Optional[datetime.datetime] = None

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            # statusは無視（input overflowログを出さない）
            audio_q.put(indata.copy())

        try:
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

            # デバイスのネイティブサンプルレートを取得
            actual_samplerate = self.sample_rate
            if self.input_device_index is not None:
                try:
                    dev_info = sd.query_devices(self.input_device_index)
                    native_rate = int(dev_info['default_samplerate'])
                    # デバイスが16000Hzをサポートしていない場合、ネイティブレートを使用
                    if native_rate != self.sample_rate:
                        print(f"[record_audio] デバイスのネイティブレート({native_rate}Hz)を使用します")
                        actual_samplerate = native_rate
                except Exception as e:
                    print(f"[record_audio] デバイス情報取得エラー: {e}")

            print(f"[record_audio] Opening InputStream: device={self.input_device_index}, rate={actual_samplerate}, blocksize={self.chunk_size}")
            with sd.InputStream(samplerate=actual_samplerate, channels=self.channels,
                              callback=callback, blocksize=self.chunk_size, dtype='float32',
                              device=self.input_device_index):
                print("[record_audio] InputStream opened successfully, starting recording loop...")
                recorded_chunks: List[np.ndarray] = []
                silent_start_time: Optional[float] = None
                recording_start_time: Optional[float] = None
                is_recording_active = False
                logged_recording_start = False

                if self.app: self.app.after(0, lambda: self.app.set_status("Please speak...", "blue"))

                while not self.stop_event.is_set():
                    try:
                        audio_chunk = audio_q.get(timeout=0.1)
                    except queue.Empty:
                        if self.stop_event.is_set(): break
                        if is_recording_active and silent_start_time and recording_start_time:
                            current_time_for_silence_check = time.time()
                            if (current_time_for_silence_check - silent_start_time >= self.required_silent_seconds and
                                current_time_for_silence_check - recording_start_time >= self.min_record_seconds):
                                print(f"Silence detected for {self.required_silent_seconds:.1f}s. Stopping recording.")
                                break
                        continue

                    current_time = time.time()
                    rms_volume = np.sqrt(np.mean(audio_chunk**2))
                    is_currently_silent = rms_volume < self.silent_threshold

                    if not is_currently_silent and not is_recording_active:
                        is_recording_active = True
                        recording_start_time = current_time
                        rec_start_dt = datetime.datetime.now()
                        silent_start_time = None

                        # 話し始め時刻を記録（後でバッファからHRを取得するため）
                        # HRの取得は録音終了後に行う

                        if not logged_recording_start:
                            print("Recording started...")
                            logged_recording_start = True
                        if self.app: self.app.after(0, lambda: self.app.set_status("Recording...", "orange"))

                    if is_recording_active:
                        recorded_chunks.append(audio_chunk)
                        elapsed_recording_time = current_time - (recording_start_time or current_time)
                        if is_currently_silent:
                            if silent_start_time is None: silent_start_time = current_time
                            elif (current_time - silent_start_time >= self.required_silent_seconds and
                                  elapsed_recording_time >= self.min_record_seconds):
                                print(f"Silence detected for {self.required_silent_seconds:.1f}s. Stopping recording.")
                                break
                        else: silent_start_time = None

                if is_recording_active:
                    rec_end_dt = datetime.datetime.now()

                    # 話し始め時刻の前2秒＋開始時点＋後2秒（計5サンプル程度）のバッファからHRを取得
                    if self.hr_monitor and self.hr_monitor.is_connected and rec_start_dt:
                        buffered_hr = self.hr_monitor.get_buffered_hr(
                            target_time=rec_start_dt,
                            window_seconds=2.0
                        )
                        if buffered_hr is not None:
                            self.log_heartrate_at_recording_start(buffered_hr)
                            print(f"*** Recorded HR at recording start (Verity, buffered median): {buffered_hr} BPM ***")
                        else:
                            # バッファにデータがない場合は現在値を使用
                            hr_at_start = self.hr_monitor.get_current_hr()
                            self.log_heartrate_at_recording_start(hr_at_start)
                            print(f"*** Recorded HR at recording start (Verity, fallback to current): {hr_at_start} BPM ***")

                if recorded_chunks:
                    audio_data = np.concatenate(recorded_chunks, axis=0)
                    duration = len(audio_data) / actual_samplerate
                    if duration < self.min_audio_length or np.max(np.abs(audio_data)) < self.silent_threshold * 0.5:
                        print("Recording too short or nearly silent. Not saving.")
                        recording_success = False
                    else:
                        if not filename.lower().endswith(".wav"): filename += ".wav"
                        try:
                            sf.write(filename, audio_data.astype(np.float32), actual_samplerate, subtype='PCM_16')
                            print(f"Recording complete and saved: {filename} ({duration:.2f}s, {actual_samplerate}Hz)")
                            recording_success = True
                        except Exception as e_write: print(f"Error writing audio to file ({filename}): {e_write}")
                else:
                    if not self.stop_event.is_set(): print("No audio data was recorded.")
        except sd.PortAudioError as pae:
            print(f"Sounddevice PortAudio Error: {pae}")
        except Exception as e_rec:
            print(f"Unexpected error during recording: {e_rec}")
        finally:
            if not recording_success and self.app: self.app.after(0, lambda: self.app.set_status("Recording failed or cancelled", "orange"))
        return recording_success, rec_start_dt, rec_end_dt


    def text_to_speech(self, text: str, filename: str = "output.wav") -> Tuple[bool, Optional[datetime.datetime], Optional[datetime.datetime]]:
        """
        与えられた文章を VOICEVOX で音声合成し、再生まで行う。
        戻り値は (成功したかどうか, 再生開始時刻, 再生終了時刻)。
        """
        if not self.tts_lock.acquire(blocking=False):
            print("TTS/playback is already running. Skipping new request.")
            return False, None, None

        success: bool = False
        playback_start_dt: Optional[datetime.datetime] = None
        playback_end_dt: Optional[datetime.datetime] = None

        # 直前の抑揚値を取得しておく（ログなどで使う）
        applied_intonation_scale: float = self.prosody.get_parameter("intonation")
        hfb_type_for_log: str = "None"

        try:
            print("-" * 20)
            print(f"Starting speech synthesis for: '{text[:50]}...'")
            if self.app:
                self.app.after(0, lambda: self.app.set_status("抑揚などを計算中...", "orange"))

            # --------------------------------------------------------------
            # ここから HFB によるパラメータ調整
            # --------------------------------------------------------------
            applied_hfb_param_value: float = 0.0
            target_param_for_hfb: str = "intonation"

            # ProsodySettings 側で選ばれている対象パラメータ名を取得
            if hasattr(self.prosody, "get_hfb_target_param"):
                target_param_for_hfb = self.prosody.get_hfb_target_param()
            else:
                target_param_for_hfb = "intonation"

            # 基準値と感度をパラメータごとに用意
            base_value = 1.0
            sensitivity = 0.1
            if target_param_for_hfb == "pitch":
                base_value = 0.0
                sensitivity = 0.002
            elif target_param_for_hfb == "speed":
                base_value = 1.0
                sensitivity = 0.003
            elif target_param_for_hfb == "energy":
                base_value = 1.0
                sensitivity = 0.005
            elif target_param_for_hfb == "pause_duration":
                base_value = 1.0
                sensitivity = 0.003

            # 正弦波 HFB が有効な場合は、抑揚だけを正弦波で変化させる
            if hasattr(self.prosody, "is_sinusoidal_hfb_enabled") and self.prosody.is_sinusoidal_hfb_enabled():
                hfb_type_for_log = "Sinusoidal"
                if hasattr(self.prosody, "get_next_sinusoidal_intonation"):
                    applied_intonation_scale = self.prosody.get_next_sinusoidal_intonation()
                else:
                    applied_intonation_scale = 1.0
                self.prosody.set_parameter("intonation", applied_intonation_scale)
                applied_hfb_param_value = applied_intonation_scale
                target_param_for_hfb = "intonation"
                print(f"HFB (Sinusoidal) Intonation set to {applied_intonation_scale:.3f}")

            # HRF2: PID制御による心拍数追従モード
            elif hasattr(self.prosody, "is_hrf2_enabled") and self.prosody.is_hrf2_enabled():
                hfb_type_for_log = self.prosody.get_current_mode()
                target_param_for_hfb = "intonation"

                if self.hr_monitor is None or not getattr(self.hr_monitor, "is_connected", False):
                    print("HRF2 is enabled but heart rate monitor is not connected.")
                    applied_hfb_param_value = self.prosody.get_parameter("intonation")
                else:
                    current_hr = self.hr_monitor.get_current_hr()
                    if current_hr > 0:
                        hrf2_output, debug_info = self.prosody.get_hrf2_output(current_hr)
                        applied_hfb_param_value = hrf2_output
                        applied_intonation_scale = hrf2_output
                        self.prosody.set_parameter("intonation", hrf2_output)

                        print(
                            f"{hfb_type_for_log}: Target={debug_info.get('target_hr', 0):.0f} BPM, "
                            f"Current={current_hr} BPM, Error={debug_info.get('error', 0):.1f}, "
                            f"Output={hrf2_output:.3f}"
                        )
                    else:
                        print("HRF2: No valid HR data available.")
                        applied_hfb_param_value = self.prosody.get_parameter("intonation")

            # 直接 HR に応じて変化させるモード
            elif hasattr(self.prosody, "is_hfb_enabled") and self.prosody.is_hfb_enabled():
                hfb_type_for_log = f"Direct({target_param_for_hfb})"

                if self.hr_monitor is None or not getattr(self.hr_monitor, "is_connected", False):
                    print("HFB (Direct) is enabled but heart rate monitor is not connected.")
                    # HFB は論理的には有効だが、実際の調整は行わない
                    applied_hfb_param_value = self.prosody.get_parameter(target_param_for_hfb)
                else:
                    reference_hr = self.hr_monitor.get_reference_hr()
                    if reference_hr == 0:
                        # 基準心拍がまだ決まっていないときは基準値のみ
                        print("HFB (Direct): reference HR is not set yet. Using base value.")
                        applied_hfb_param_value = base_value
                        self.hr_used_for_last_adjustment = self.hr_monitor.get_current_hr()
                    elif self.last_hr_after_tts is None:
                        # 最初の TTS ではまだ前回 HR が無いので、基準値を使う
                        print("HFB (Direct): first TTS or HFB state reset. Using base value.")
                        applied_hfb_param_value = base_value
                        self.hr_used_for_last_adjustment = self.hr_monitor.get_current_hr()
                    else:
                        hr_for_adjustment_display = self.last_hr_after_tts
                        self.hr_used_for_last_adjustment = hr_for_adjustment_display
                        hr_diff = hr_for_adjustment_display - reference_hr
                        calculated_value = base_value + (hr_diff * sensitivity)

                        min_val, max_val = self.prosody.get_parameter_range(target_param_for_hfb)
                        applied_hfb_param_value = max(min_val, min(max_val, calculated_value))

                        print(
                            f"HFB (Direct) Calculation ({target_param_for_hfb}): "
                            f"Ref HR={reference_hr} BPM, HR after TTS={hr_for_adjustment_display} BPM "
                            f"(Diff={hr_diff:+d}), base={base_value:.3f}, "
                            f"calculated={calculated_value:.3f}, applied={applied_hfb_param_value:.3f}"
                        )

                    # 実際にパラメータへ反映
                    self.prosody.set_parameter(target_param_for_hfb, applied_hfb_param_value)
                    if target_param_for_hfb == "intonation":
                        applied_intonation_scale = applied_hfb_param_value

                # HR モニタ側にも、どのレベルを適用したか知らせておく
                if self.hr_monitor is not None:
                    try:
                        self.hr_monitor.update_prosody_level(applied_hfb_param_value)
                    except Exception as e_update:
                        print(f"Failed to update prosody level in HeartRateMonitor: {e_update}")

            # HFB 無効のときは手動設定値を使う
            else:
                hfb_type_for_log = "Manual"
                applied_hfb_param_value = self.prosody.get_parameter(target_param_for_hfb)
                print("HFB is disabled. Using manually set prosody values.")
                self.hr_used_for_last_adjustment = None
                if self.hr_monitor is not None:
                    try:
                        self.hr_monitor.update_prosody_level(applied_hfb_param_value)
                    except Exception as e_update:
                        print(f"Failed to update prosody level in HeartRateMonitor (manual): {e_update}")

            # --------------------------------------------------------------
            # VOICEVOX へのクエリ生成
            # --------------------------------------------------------------
            query_params = {"text": text, "speaker": self.speaker.current_style_id}
            if self.app:
                self.app.after(0, lambda: self.app.set_status("VOICEVOXに問い合わせ中...", "orange"))
            try:
                query_response = requests.post(
                    f"{config.VOICEVOX_URL}/audio_query",
                    params=query_params,
                    timeout=10
                )
                query_response.raise_for_status()
                audio_query = query_response.json()
            except Exception as e_query:
                print(f"VOICEVOX audio_query failed: {e_query}")
                return False, None, None

            # --------------------------------------------------------------
            # ProsodySettings から VOICEVOX パラメータに反映
            # --------------------------------------------------------------
            audio_query["intonationScale"] = self.prosody.get_parameter("intonation")
            audio_query["pitchScale"] = self.prosody.get_parameter("pitch")
            audio_query["speedScale"] = self.prosody.get_parameter("speed")
            audio_query["volumeScale"] = self.prosody.get_parameter("energy")

            # 休止時間 (前後の無音長)
            default_pause_s = float(audio_query.get("prePhonemeLength", 0.1))
            pause_scale = self.prosody.get_parameter("pause_duration")
            audio_query["prePhonemeLength"] = default_pause_s * pause_scale
            audio_query["postPhonemeLength"] = default_pause_s * pause_scale

            print(
                f"Synthesis params: Speaker={self.speaker.current_style_id}, "
                f"Intonation={audio_query['intonationScale']:.2f}, "
                f"Pitch={audio_query['pitchScale']:.2f}, "
                f"Speed={audio_query['speedScale']:.2f}, "
                f"Volume={audio_query['volumeScale']:.2f}, "
                f"HFB Mode={hfb_type_for_log}"
            )

            if self.app:
                self.app.after(0, lambda: self.app.set_status("音声を合成中...", "orange"))
            try:
                synthesis_response = requests.post(
                    f"{config.VOICEVOX_URL}/synthesis",
                    headers={"Content-Type": "application/json", "accept": "audio/wav"},
                    params=query_params,
                    data=json.dumps(audio_query),
                    timeout=20,
                )
                synthesis_response.raise_for_status()
            except Exception as e_syn:
                print(f"VOICEVOX synthesis failed: {e_syn}")
                return False, None, None

            # wav 保存
            try:
                with open(filename, "wb") as f:
                    f.write(synthesis_response.content)
            except Exception as e_write:
                print(f"Failed to write synthesized audio to file '{filename}': {e_write}")
                return False, None, None

            # 再生
            try:
                playback_start_dt = datetime.datetime.now()
                if self.app:
                    self.app.after(0, lambda: self.app.set_status("音声を再生中...", "green"))
                self._play_audio_file(filename)
                playback_end_dt = datetime.datetime.now()
                success = True
            except Exception as e_play:
                print(f"Audio playback failed: {e_play}")
                success = False

            # --------------------------------------------------------------
            # 再生後に HR を取得してログに残す
            # --------------------------------------------------------------
            if self.hr_monitor is not None and getattr(self.hr_monitor, "is_connected", False):
                try:
                    hr_after_this_tts = self.hr_monitor.get_current_hr()
                    self.last_hr_after_tts = hr_after_this_tts
                    print(
                        f"*** Recorded HR after current TTS (Verity): "
                        f"{self.last_hr_after_tts} BPM (HFB Mode: {hfb_type_for_log}) ***"
                    )
                    self.log_heartrate_after_tts(
                        hr_value=hr_after_this_tts,
                        applied_intonation=applied_intonation_scale,
                        hfb_enabled_during_tts=(hfb_type_for_log != "Manual" and hfb_type_for_log != "None"),
                        playback_start_time=playback_start_dt,
                        playback_end_time=playback_end_dt,
                    )
                except Exception as e_hrlog:
                    print(f"Failed to log heart rate after TTS: {e_hrlog}")
            else:
                self.last_hr_after_tts = None

        finally:
            if self.tts_lock.locked():
                self.tts_lock.release()
            print("-" * 20)

        return success, playback_start_dt, playback_end_dt

    def _play_audio_file(self, filename: str) -> None:
        """音声ファイルを再生する

        macOSではafplayを使用してノイズを回避。
        その他のOSではsounddeviceを使用。
        """
        import platform
        import subprocess

        try:
            if platform.system() == "Darwin":
                # macOS: afplayコマンドを使用（sounddeviceよりノイズが少ない）
                result = subprocess.run(
                    ["afplay", filename],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"afplay error: {result.stderr}")
                    raise RuntimeError(f"afplay failed: {result.stderr}")
            else:
                # その他のOS: sounddeviceを使用
                data, samplerate = sf.read(filename, dtype='float32')
                sd.stop()
                sd.play(data, samplerate)
                sd.wait()
        except FileNotFoundError:
            # afplayが見つからない場合はsounddeviceにフォールバック
            print("afplay not found, falling back to sounddevice")
            data, samplerate = sf.read(filename, dtype='float32')
            sd.stop()
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Error during audio playback: {e}")
            raise

    def log_heartrate_after_tts(self, hr_value: int, applied_intonation: float, hfb_enabled_during_tts: bool,
                                playback_start_time: Optional[datetime.datetime],
                                playback_end_time: Optional[datetime.datetime]):
        try:
            hr_used_for_adj_log_val = "N/A"
            if hfb_enabled_during_tts and hasattr(self.prosody, "is_hfb_enabled") and self.prosody.is_hfb_enabled():
                if self.hr_used_for_last_adjustment is not None:
                    hr_used_for_adj_log_val = str(self.hr_used_for_last_adjustment)

            start_time_str = playback_start_time.strftime('%Y-%m-%d %H:%M:%S.%f') if playback_start_time else ""
            end_time_str = playback_end_time.strftime('%Y-%m-%d %H:%M:%S.%f') if playback_end_time else ""

            payload = [
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                hr_value,
                self.hr_monitor.get_reference_hr() if self.hr_monitor is not None else 0,
                "Yes" if hfb_enabled_during_tts else "No",
                hr_used_for_adj_log_val,
                f"{applied_intonation:.3f}",
                start_time_str,
                end_time_str
            ]
            record = logging.LogRecord(
                name=config.LOGGER_HR_AFTER_TTS, level=logging.INFO, pathname="", lineno=0,
                msg="", args=(payload,), exc_info=None, func="")
            record.payload = payload  # type: ignore
            self.log_queue.put(record)
        except Exception as e:
            print(f"Failed to queue log for heart rate after TTS: {e}")

    def log_heartrate_at_recording_start(self, hr_value: int):
        try:
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            payload = [timestamp_str, hr_value]
            record = logging.LogRecord(
                name=config.LOGGER_HR_AT_RECORDING_START, level=logging.INFO, pathname="", lineno=0,
                msg="", args=(payload,), exc_info=None, func="")
            record.payload = payload  # type: ignore
            self.log_queue.put(record)
        except Exception as e:
            print(f"Failed to queue log for heart rate at recording start: {e}")

    def speech_to_text(self, audio_filename: str) -> Optional[str]:
        """音声ファイルをテキストに変換（ver3.10方式）"""
        if self.whisper_model is None:
            return None
        try:
            if not os.path.exists(audio_filename) or os.path.getsize(audio_filename) < 1024:
                return None

            # ファイルの長さをチェック
            with sf.SoundFile(audio_filename) as sf_file:
                duration = len(sf_file) / sf_file.samplerate
            if duration < self.min_audio_length:
                print(f"Audio too short ({duration:.2f}s). Skipping.")
                return None

            if self.app:
                self.app.after(0, lambda: self.app.set_status("音声認識中...", "orange"))

            print(f"Transcribing {audio_filename} ({duration:.2f}s)...")
            segments, info = self.whisper_model.transcribe(
                audio_filename,
                language="ja",
                beam_size=config.WHISPER_TRANSCRIBE_BEAM_SIZE,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            text = "".join(segment.text for segment in segments)
            cleaned_text = self._clean_text(text)

            if cleaned_text:
                print(f"Recognized text: {cleaned_text}")
            else:
                print("No valid text recognized.")

            return cleaned_text
        except Exception as e:
            print(f"Speech-to-text failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if self.app:
                self.app.after(0, lambda: self.app.set_status("Ready", "green"))

    def _clean_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        cleaned = text.strip()
        artifacts = ["[Music]", "(Music)", "Thanks for watching", "Subtitles by"]
        for art in artifacts:
            cleaned = cleaned.replace(art, "")
        cleaned = cleaned.strip()
        return cleaned if cleaned and not all(char in " ,.!?\"'()「」。、" for char in cleaned) else None

    def reset_hfb_state(self):
        """HFB 関連の状態を全て初期化する。"""
        self.last_hr_after_tts = None
        self.hr_used_for_last_adjustment = None
        if hasattr(self.prosody, 'reset_sinusoidal_hfb_state_values'):
            self.prosody.reset_sinusoidal_hfb_state_values()
        print("All HFB state variables (direct adjustment and sinusoidal modes) reset.")
