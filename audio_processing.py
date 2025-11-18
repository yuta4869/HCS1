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
    HeartRateMonitor = Any
    H10Monitor = Any


class ProsodySettings:
    """Manages prosody parameters for speech synthesis."""
    def __init__(self):
        self.intonation_scale: float = 1.0
        self.pitch_scale: float = 0.0
        self.speed_scale: float = 1.0
        self.energy_scale: float = 1.0  # Corresponds to volumeScale in Voicevox
        self.pause_duration_scale: float = 1.0 # Affects pre/postPhonemeLength
        self.hfb_enabled: bool = False # Heart Rate Feedback enabled
        self.min_intonation, self.max_intonation = 0.0, 2.0
        self.min_pitch, self.max_pitch = -0.15, 0.15
        self.min_speed, self.max_speed = 0.5, 2.0
        self.min_energy, self.max_energy = 0.0, 2.0
        self.min_pause_duration, self.max_pause_duration = 0.0, 2.0
        self.intonation_var: Optional[tk.DoubleVar] = None
        self.pitch_var: Optional[tk.DoubleVar] = None
        self.speed_var: Optional[tk.DoubleVar] = None
        self.energy_var: Optional[tk.DoubleVar] = None
        self.pause_duration_var: Optional[tk.DoubleVar] = None
        self.hfb_enabled_var: Optional[tk.BooleanVar] = None
        self.sinusoidal_hfb_enabled: bool = False
        self.sinusoidal_hfb_sequence: List[float] = self._generate_sinusoidal_sequence()
        self.sinusoidal_hfb_step_index: int = 0
        if self.sinusoidal_hfb_sequence:
            try:
                self.sinusoidal_hfb_step_index = self.sinusoidal_hfb_sequence.index(1.0)
            except ValueError:
                self.sinusoidal_hfb_step_index = 0
        self.sinusoidal_hfb_enabled_var: Optional[tk.BooleanVar] = None

    def _generate_sinusoidal_sequence(self) -> List[float]:
        seq = []
        for i in range(11):
            seq.append(round(1.0 + i * 0.1, 1))
        for i in range(20):
            seq.append(round(1.9 - i * 0.1, 1))
        for i in range(9):
            seq.append(round(0.1 + i * 0.1, 1))
        return seq if seq else [1.0]

    def set_parameter(self, param_name: str, value: float):
        min_val = getattr(self, f"min_{param_name}")
        max_val = getattr(self, f"max_{param_name}")
        clamped_value = max(min_val, min(max_val, value))
        setattr(self, f"{param_name}_scale", clamped_value)
        tk_var = getattr(self, f"{param_name}_var", None)
        if tk_var:
            try:
                tk_var.set(clamped_value)
            except tk.TclError:
                pass

    def get_parameter(self, param_name: str) -> float:
        return getattr(self, f"{param_name}_scale")

    def get_parameter_range(self, param_name: str) -> Tuple[float, float]:
        return getattr(self, f"min_{param_name}"), getattr(self, f"max_{param_name}")

    def enable_hfb(self, enable: bool):
        self.hfb_enabled = enable
        print(f"HFB (Direct Adjustment) set to {'enabled' if enable else 'disabled'}.")
        if self.hfb_enabled_var:
            try:
                self.hfb_enabled_var.set(enable)
            except tk.TclError:
                pass

    def is_hfb_enabled(self) -> bool:
        return self.hfb_enabled

    def enable_sinusoidal_hfb(self, enable: bool):
        self.sinusoidal_hfb_enabled = enable
        print(f"Sinusoidal HFB set to {'enabled' if enable else 'disabled'}.")
        if self.sinusoidal_hfb_enabled_var:
            try:
                self.sinusoidal_hfb_enabled_var.set(enable)
            except tk.TclError:
                pass
        if enable:
            self.reset_sinusoidal_hfb_state_values()

    def is_sinusoidal_hfb_enabled(self) -> bool:
        return self.sinusoidal_hfb_enabled

    def get_next_sinusoidal_intonation(self) -> float:
        if not self.sinusoidal_hfb_sequence:
            self.sinusoidal_hfb_step_index = 0
            return 1.0
        current_intonation = self.sinusoidal_hfb_sequence[self.sinusoidal_hfb_step_index]
        self.sinusoidal_hfb_step_index = (self.sinusoidal_hfb_step_index + 1) % len(self.sinusoidal_hfb_sequence)
        return max(self.min_intonation, min(self.max_intonation, current_intonation))

    def reset_sinusoidal_hfb_state_values(self):
        initial_value = 1.0
        try:
            self.sinusoidal_hfb_step_index = self.sinusoidal_hfb_sequence.index(initial_value)
        except ValueError:
            self.sinusoidal_hfb_step_index = 0
        current_val_display = 1.0
        if self.sinusoidal_hfb_sequence:
             current_val_display = self.sinusoidal_hfb_sequence[self.sinusoidal_hfb_step_index]
        print(f"Sinusoidal HFB state (step index) reset. Next intonation will be: {current_val_display:.1f}")


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
                return []
            for speaker in speakers_data:
                speaker_name = speaker.get('name', 'Unknown Speaker')
                for style in speaker.get('styles', []):
                    style_name = style.get('name', 'Default Style')
                    style_id = style.get('id')
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
            if status: print(f"Recording status: {status}", file=sys.stderr)
            audio_q.put(indata.copy())

        try:
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=callback, blocksize=1024, dtype='float32'):
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
                        
                        if self.hr_monitor and self.hr_monitor.is_connected:
                            hr_at_start = self.hr_monitor.get_current_hr()
                            self.log_heartrate_at_recording_start(hr_at_start)
                            print(f"*** Recorded HR at recording start (Verity): {hr_at_start} BPM ***")
                        else:
                            print("Verity Sense not connected at recording start. No HR logged for rec start.")

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

                if recorded_chunks:
                    audio_data = np.concatenate(recorded_chunks, axis=0)
                    duration = len(audio_data) / self.sample_rate
                    if duration < self.min_audio_length or np.max(np.abs(audio_data)) < self.silent_threshold * 0.5:
                        print("Recording too short or nearly silent. Not saving.")
                        recording_success = False
                    else:
                        if not filename.lower().endswith(".wav"): filename += ".wav"
                        try:
                            sf.write(filename, audio_data.astype(np.float32), self.sample_rate, subtype='PCM_16')
                            print(f"Recording complete and saved: {filename}")
                            recording_success = True
                        except Exception as e_write: print(f"Error writing audio to file ({filename}): {e_write}")
                else:
                    if not self.stop_event.is_set(): print("No audio data was recorded.")
        except sd.PortAudioError as pae:
            print(f"Sounddevice PortAudio Error: {pae}")
            if self.app and hasattr(self.app, 'messagebox'): self.app.after(0, lambda: self.app.messagebox.showerror("Recording Error", f"Audio device error:\n{pae}"))
        except Exception as e_rec:
            print(f"Unexpected error during recording: {e_rec}")
            if self.app and hasattr(self.app, 'messagebox'): self.app.after(0, lambda: self.app.messagebox.showerror("Recording Error", f"Unexpected recording error:\n{e_rec}"))
        finally:
            if not recording_success and self.app: self.app.after(0, lambda: self.app.set_status("Recording failed or cancelled", "orange"))
        return recording_success, rec_start_dt, rec_end_dt

    def text_to_speech(self, text: str, filename: str = "output.wav") -> Tuple[bool, Optional[datetime.datetime], Optional[datetime.datetime]]:
        if not self.tts_lock.acquire(blocking=False):
            print("TTS/playback is already running. Skipping new request.")
            return False, None, None

        success = False
        applied_intonation_scale = self.prosody.get_parameter("intonation")
        hfb_type_for_log = "None"
        
        playback_start_dt: Optional[datetime.datetime] = None
        playback_end_dt: Optional[datetime.datetime] = None

        try:
            print("-" * 20)
            print(f"Starting speech synthesis for: '{text[:50]}...'")
            if self.app: self.app.after(0, lambda: self.app.set_status("Calculating prosody...", "orange"))

            if self.prosody.is_sinusoidal_hfb_enabled():
                hfb_type_for_log = "Sinusoidal"
                applied_intonation_scale = self.prosody.get_next_sinusoidal_intonation()
                print(f"Sinusoidal HFB Mode: Set Intonation Scale to {applied_intonation_scale:.2f}")
                self.prosody.set_parameter("intonation", applied_intonation_scale)

            elif self.prosody.is_hfb_enabled():
                hfb_type_for_log = "Direct"
                reference_hr = self.hr_monitor.get_reference_hr()
                if not self.hr_monitor.is_connected:
                    print("HFB (Direct) is enabled but Verity Sense is not connected. Using current prosody settings.")
                    self.hr_used_for_last_adjustment = None
                elif self.last_hr_after_tts is None: 
                    print("HFB (Direct) enabled: First TTS run or HFB state reset. Using base intonation (1.0).")
                    applied_intonation_scale = 1.0
                    self.hr_used_for_last_adjustment = self.hr_monitor.get_current_hr()
                    self.prosody.set_parameter("intonation", applied_intonation_scale)
                else:
                    hr_for_adjustment_display = self.last_hr_after_tts
                    self.hr_used_for_last_adjustment = hr_for_adjustment_display 
                    hr_diff = hr_for_adjustment_display - reference_hr
                    sensitivity = 0.1
                    base_intonation_target = 1.0 
                    calculated_intonation = base_intonation_target + (hr_diff * sensitivity)
                    
                    min_int, max_int = self.prosody.get_parameter_range("intonation")
                    applied_intonation_scale = max(min_int, min(max_int, calculated_intonation))
                    print(f"HFB (Direct) Calculation: Ref HR={reference_hr}, Prev HR after TTS={hr_for_adjustment_display} (Diff:{hr_diff:+d})")
                    print(f"HFB (Direct) Adjusted Intonation Scale: {applied_intonation_scale:.3f} (from calculated {calculated_intonation:.3f})")
                    self.prosody.set_parameter("intonation", applied_intonation_scale)
            else:
                hfb_type_for_log = "Manual"
                print("HFB is disabled. Using manually set prosody.")
                self.hr_used_for_last_adjustment = None
            
            self.hr_monitor.update_prosody_level(applied_intonation_scale)

            query_params = {"text": text, "speaker": self.speaker.current_style_id}
            if self.app: self.app.after(0, lambda: self.app.set_status("Generating VOICEVOX query...", "orange"))
            try:
                query_response = requests.post(f"{config.VOICEVOX_URL}/audio_query", params=query_params, timeout=10)
                query_response.raise_for_status()
                audio_query = query_response.json()
            except (requests.RequestException, json.JSONDecodeError) as e_query:
                print(f"VOICEVOX Audio Query Error: {e_query}"); return False, None, None

            audio_query["intonationScale"] = applied_intonation_scale
            audio_query["pitchScale"] = self.prosody.get_parameter("pitch")
            audio_query["speedScale"] = self.prosody.get_parameter("speed")
            audio_query["volumeScale"] = self.prosody.get_parameter("energy")
            default_pause_s = 0.1 
            audio_query["prePhonemeLength"] = default_pause_s * self.prosody.get_parameter("pause_duration")
            audio_query["postPhonemeLength"] = default_pause_s * self.prosody.get_parameter("pause_duration")

            print(f"Synthesis params: Speaker={self.speaker.current_style_id}, "
                  f"Intonation={audio_query['intonationScale']:.2f}, Pitch={audio_query['pitchScale']:.2f}, "
                  f"Speed={audio_query['speedScale']:.2f}, Volume={audio_query['volumeScale']:.2f}, HFB Mode={hfb_type_for_log}")

            if self.app: self.app.after(0, lambda: self.app.set_status("Synthesizing audio...", "orange"))
            try:
                synthesis_response = requests.post(
                    f"{config.VOICEVOX_URL}/synthesis",
                    headers={"Content-Type": "application/json", "accept": "audio/wav"},
                    params=query_params, data=json.dumps(audio_query), timeout=20)
                synthesis_response.raise_for_status()
                if not synthesis_response.content or len(synthesis_response.content) < 1024:
                    print(f"VOICEVOX Synthesis Error: Response empty or too small."); return False, None, None
            except requests.RequestException as req_err:
                print(f"VOICEVOX Synthesis request error: {req_err}"); return False, None, None

            try:
                if not filename.lower().endswith(".wav"): filename += ".wav"
                os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
                with open(filename, "wb") as f: f.write(synthesis_response.content)
            except IOError as io_err:
                print(f"Audio file write error ({filename}): {io_err}"); return False, None, None

            if self.app: self.app.after(0, lambda: self.app.set_status("Playing audio...", "green"))
            try:
                audio_data, sr = sf.read(filename, dtype='float32')
                if audio_data is None or audio_data.size == 0:
                    print(f"Audio playback error: File data empty ({filename})"); return False, None, None
                
                playback_start_dt = datetime.datetime.now()
                sd.play(audio_data, sr); sd.wait()
                playback_end_dt = datetime.datetime.now()
                
                success = True
            except (sf.SoundFileError, sd.PortAudioError) as play_err:
                print(f"Audio playback error ({filename}): {play_err}"); return False, None, None
            except Exception as e_play_gen:
                print(f"Unexpected audio playback error ({filename}): {e_play_gen}"); return False, None, None
        except Exception as e_tts_main:
            print(f"Unexpected error in text_to_speech: {e_tts_main}"); import traceback; traceback.print_exc()
        finally:
            if self.hr_monitor.is_connected:
                time.sleep(0.1) 
                hr_after_this_tts = self.hr_monitor.get_current_hr()
                self.last_hr_after_tts = hr_after_this_tts
                print(f"*** Recorded HR after current TTS (Verity): {self.last_hr_after_tts} BPM (HFB Mode: {hfb_type_for_log}) ***")
                self.log_heartrate_after_tts(
                    hr_value=hr_after_this_tts,
                    applied_intonation=applied_intonation_scale,
                    hfb_enabled_during_tts=(hfb_type_for_log != "Manual" and hfb_type_for_log != "None"),
                    playback_start_time=playback_start_dt,
                    playback_end_time=playback_end_dt
                )
            else:
                self.last_hr_after_tts = None
            if self.tts_lock.locked(): self.tts_lock.release()
            print("-" * 20)
        return success, playback_start_dt, playback_end_dt

    def log_heartrate_after_tts(self, hr_value: int, applied_intonation: float, hfb_enabled_during_tts: bool,
                                playback_start_time: Optional[datetime.datetime],
                                playback_end_time: Optional[datetime.datetime]):
        try:
            hr_used_for_adj_log_val = "N/A"
            if hfb_enabled_during_tts and self.prosody.is_hfb_enabled():
                if self.hr_used_for_last_adjustment is not None:
                    hr_used_for_adj_log_val = str(self.hr_used_for_last_adjustment)
            
            start_time_str = playback_start_time.strftime('%Y-%m-%d %H:%M:%S.%f') if playback_start_time else ""
            end_time_str = playback_end_time.strftime('%Y-%m-%d %H:%M:%S.%f') if playback_end_time else ""

            payload = [
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                hr_value,
                self.hr_monitor.get_reference_hr(),
                "Yes" if hfb_enabled_during_tts else "No",
                hr_used_for_adj_log_val,
                f"{applied_intonation:.3f}",
                start_time_str,
                end_time_str
            ]
            record = logging.LogRecord(
                name=config.LOGGER_HR_AFTER_TTS, level=logging.INFO, pathname="", lineno=0,
                msg="", args=(payload,), exc_info=None, func="")
            record.payload = payload # type: ignore
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
            record.payload = payload # type: ignore
            self.log_queue.put(record)
        except Exception as e:
            print(f"Failed to queue log for heart rate at recording start: {e}")

    def speech_to_text(self, audio_filename: str) -> Optional[str]:
        if self.whisper_model is None: return None
        try:
            if not os.path.exists(audio_filename) or os.path.getsize(audio_filename) < 1024: return None
            with sf.SoundFile(audio_filename) as sf_file: duration = len(sf_file) / sf_file.samplerate
            if duration < self.min_audio_length: return None

            segments, info = self.whisper_model.transcribe(audio_filename, language="ja", beam_size=2, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
            full_text = "".join(segment.text for segment in segments)
            return self._clean_text(full_text)
        except Exception as e_stt:
            print(f"Speech recognition error ({audio_filename}): {e_stt}")
            if self.app: self.app.after(0, lambda: self.app.set_status("Speech recognition error", "red"))
            return None

    def _clean_text(self, text: str) -> Optional[str]:
        if not text: return None
        cleaned = text.strip()
        artifacts = ["[Music]", "(Music)", "Thanks for watching", "Subtitles by"]
        for art in artifacts: cleaned = cleaned.replace(art, "")
        cleaned = cleaned.strip()
        return cleaned if cleaned and not all(char in " ,.!?\"'()「」。、" for char in cleaned) else None

    def reset_hfb_state(self):
        """Resets all HFB related states."""
        self.last_hr_after_tts = None
        self.hr_used_for_last_adjustment = None
        if hasattr(self.prosody, 'reset_sinusoidal_hfb_state_values'):
            self.prosody.reset_sinusoidal_hfb_state_values()
        print("All HFB state variables (direct adjustment and sinusoidal modes) reset.")