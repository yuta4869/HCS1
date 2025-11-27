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

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import tkinter as tk

from faster_whisper import WhisperModel

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
                 faster_whisper_model_instance: WhisperModel):
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
        self.silent_threshold: float = 0.03
        self.min_record_seconds: float = 1.0
        self.required_silent_seconds: float = 1.3
        self.min_audio_length: float = 0.3

        self.last_hr_after_tts: Optional[int] = None
        self.hr_used_for_last_adjustment: Optional[int] = None

    def record_audio(self, filename: str = "input.wav") -> Tuple[bool, Optional[datetime.datetime], Optional[datetime.datetime]]:
        audio_q = queue.Queue()
        self.stop_event.clear()
        recording_success = False

        rec_start_dt: Optional[datetime.datetime] = None
        rec_end_dt: Optional[datetime.datetime] = None

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                print(f"Recording status: {status}", file=sys.stderr)
            audio_q.put(indata.copy())

        try:
            with sd.InputStream(samplerate=self.sample_rate,
                                channels=self.channels,
                                callback=callback):
                recorded_chunks: List[np.ndarray] = []
                silent_start_time: Optional[float] = None
                recording_start_time: Optional[float] = None
                is_recording_active = False
                logged_recording_start = False

                if self.app:
                    self.app.after(0, lambda: self.app.set_status("Please speak...", "blue"))

                while not self.stop_event.is_set():
                    try:
                        audio_chunk = audio_q.get(timeout=0.1)
                    except queue.Empty:
                        if self.stop_event.is_set():
                            break
                        if is_recording_active and silent_start_time and recording_start_time:
                            elapsed_silence = time.time() - silent_start_time
                            total_record_time = time.time() - recording_start_time
                            if elapsed_silence >= self.required_silent_seconds and total_record_time >= self.min_record_seconds:
                                print("Silence detected. Stopping recording.")
                                break
                        continue

                    rms = float(np.sqrt(np.mean(np.square(audio_chunk))))
                    is_currently_silent = rms < self.silent_threshold
                    current_time = time.time()

                    if not is_recording_active:
                        if not is_currently_silent:
                            is_recording_active = True
                            recording_start_time = current_time
                            rec_start_dt = datetime.datetime.now()

                            if self.hr_monitor and getattr(self.hr_monitor, "is_connected", False):
                                hr_at_start = self.hr_monitor.get_current_hr()
                                self.log_heartrate_at_recording_start(hr_at_start)
                                print(f"*** Recorded HR at recording start (Verity): {hr_at_start} BPM ***")
                            else:
                                print("Verity Sense not connected at recording start. No HR logged for rec start.")

                            if not logged_recording_start:
                                print("Recording started...")
                                logged_recording_start = True
                            if self.app:
                                self.app.after(0, lambda: self.app.set_status("Recording...", "orange"))

                    if is_recording_active:
                        recorded_chunks.append(audio_chunk)
                        elapsed_recording_time = current_time - (recording_start_time or current_time)
                        if is_currently_silent:
                            if silent_start_time is None:
                                silent_start_time = current_time
                        else:
                            silent_start_time = None

                        if elapsed_recording_time >= 30.0:
                            print("Maximum recording time reached. Stopping recording.")
                            break

                if recorded_chunks:
                    recorded_audio = np.concatenate(recorded_chunks, axis=0)
                    duration = recorded_audio.shape[0] / self.sample_rate
                    if duration >= self.min_audio_length:
                        sf.write(filename, recorded_audio, self.sample_rate)
                        recording_success = True
                        rec_end_dt = datetime.datetime.now()
                        print(f"Recording saved to {filename}, duration: {duration:.2f} seconds.")
                    else:
                        print(f"Recorded audio too short ({duration:.2f} seconds). Discarded.")

        except Exception as e:
            print(f"Recording failed: {e}")
        finally:
            if self.app:
                status_text = "Recording complete." if recording_success else "Recording failed or canceled."
                status_color = "green" if recording_success else "red"
                self.app.after(0, lambda: self.app.set_status(status_text, status_color))

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
        try:
            data, samplerate = sf.read(filename, dtype='float32')
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
        if self.whisper_model is None:
            return None
        try:
            if not os.path.exists(audio_filename) or os.path.getsize(audio_filename) < 1024:
                return None

            if self.app:
                self.app.after(0, lambda: self.app.set_status("音声認識中...", "orange"))

            segments, info = self.whisper_model.transcribe(
                audio_filename,
                beam_size=config.WHISPER_TRANSCRIBE_BEAM_SIZE
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
