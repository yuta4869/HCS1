# gui.py

import asyncio
import datetime
import json
import os
import queue
import signal
import sys
import threading
import time
from typing import Optional, Any, List, Dict

import tkinter as tk
from tkinter import font, messagebox, scrolledtext, ttk

import numpy as np

import config
from audio_processing import ProsodySettings, SpeakerSettings, AudioProcessor, VoicevoxManager
from logger_utils import LoggingThread
from polar_monitor import HeartRateMonitor, H10Monitor
from conversation_manager import ConversationManager


class Application(tk.Tk):
    """Main application class with Tkinter UI."""
    def __init__(self,
                 prosody_settings: ProsodySettings,
                 speaker_settings: SpeakerSettings,
                 audio_processor: AudioProcessor,
                 hr_monitor: HeartRateMonitor,
                 h10_monitor: H10Monitor,
                 log_queue_ref: queue.Queue):
        super().__init__()

        self.prosody = prosody_settings
        self.speaker = speaker_settings
        self.audio = audio_processor
        self.hr_monitor = hr_monitor
        self.h10_monitor = h10_monitor
        self.log_queue = log_queue_ref

        self.logger_thread: Optional[LoggingThread] = None

        self.conversation_manager = ConversationManager(
            audio_processor=self.audio,
            hr_monitor=self.hr_monitor,
            h10_monitor=self.h10_monitor,
            app_ref=self,
            log_queue_ref=self.log_queue
        )

        self.is_processing: bool = False
        self.is_conversing: bool = False
        self.is_measurement_stopped_by_user: bool = False
        self.is_measuring_baseline: bool = False
        self._closing: bool = False # Flag to indicate app is shutting down

        self.title("Heart-Linked Voice Assistant (HCS)")
        self.geometry("1200x900")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.default_font = font.Font(family="Helvetica", size=10)
        self.title_font = font.Font(family="Helvetica", size=14, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=10, weight="bold")
        self.button_font = font.Font(family="Helvetica", size=10)
        self.status_font = font.Font(family="Helvetica", size=9, slant="italic")
        self.check_font = font.Font(family="Helvetica", size=9)

        self.style = ttk.Style()
        self.style.configure('TButton', font=self.button_font, padding=5)
        self.style.configure('TLabel', font=self.default_font)
        self.style.configure('Bold.TLabel', font=self.label_font)
        self.style.configure('TCombobox', font=self.default_font)
        self.style.configure('Status.TLabel', font=self.status_font)
        self.style.configure('TCheckbutton', font=self.check_font)

        self._create_widgets()
        self._update_button_states()
        self._setup_logger_thread()
        # self.after(1000, self._init_hr_connection) # Auto-connection removed
        # self.after(1000, self._init_h10_connection) # Auto-connection removed

    def _setup_style(self):
        style = ttk.Style(self)
        available_themes = style.theme_names()
        if 'aqua' in available_themes:
            style.theme_use('aqua')
        elif 'clam' in available_themes:
            style.theme_use('clam')
        elif 'vista' in available_themes:
            style.theme_use('vista')
        else:
            print(f"Preferred themes not found, using default: {style.theme_use()}. Available: {available_themes}")

        style.configure('TButton', font=self.button_font, padding=5)
        style.configure('TLabel', font=self.default_font)
        style.configure('Bold.TLabel', font=self.label_font)
        style.configure('TCombobox', font=self.default_font)
        style.configure('Status.TLabel', font=self.status_font)
        style.configure('TCheckbutton', font=self.check_font)

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=0)
        main_frame.rowconfigure(1, weight=0)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(3, weight=0)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=2)
        main_frame.columnconfigure(3, weight=1)
        main_frame.columnconfigure(4, weight=1)

        row_idx = 0

        # --- Heart Rate Monitor Frame ---
        hr_frame = ttk.LabelFrame(main_frame, text="心拍数モニター (Polar Verity Sense / H10)", padding="10")
        hr_frame.grid(row=row_idx, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        hr_frame.columnconfigure(1, weight=1)
        hr_frame.columnconfigure(3, weight=1)

        ttk.Label(hr_frame, text="基準心拍数(HFB用):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.reference_hr_var = tk.StringVar(value=str(self.hr_monitor.get_reference_hr()))
        self.reference_hr_entry = ttk.Entry(hr_frame, width=8, textvariable=self.reference_hr_var)
        self.reference_hr_entry.grid(row=0, column=1, sticky=tk.W, pady=2)
        self.update_ref_hr_button = ttk.Button(hr_frame, text="基準心拍数を更新", command=self.update_reference_hr_button)
        self.update_ref_hr_button.grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)

        ttk.Label(hr_frame, text="Verity Sense 状態:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.hr_status_label = ttk.Label(hr_frame, text="未接続", style='Status.TLabel', foreground="red")
        self.hr_status_label.grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)

        ttk.Label(hr_frame, text="最新心拍数:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.current_hr_var = tk.StringVar(value="--")
        ttk.Label(hr_frame, textvariable=self.current_hr_var, style='Status.TLabel').grid(row=1, column=3, sticky=tk.W, pady=2, padx=5)

        ttk.Label(hr_frame, text="H10 状態:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.h10_status_label = ttk.Label(hr_frame, text="未接続", style='Status.TLabel', foreground="red")
        self.h10_status_label.grid(row=2, column=1, sticky=tk.W, pady=2, padx=5)

        ttk.Label(hr_frame, text="最新 RR間隔(ms):").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        self.current_rr_var = tk.StringVar(value="--")
        ttk.Label(hr_frame, textvariable=self.current_rr_var, style='Status.TLabel').grid(row=2, column=3, sticky=tk.W, pady=2, padx=5)

        # Unified Sensor Connection Control
        ttk.Label(hr_frame, text="センサー接続制御:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.connect_all_sensors_button = ttk.Button(hr_frame, text="全センサー接続", command=self.connect_all_sensors)
        self.connect_all_sensors_button.grid(row=3, column=1, sticky=tk.W, pady=2, padx=5)
        self.disconnect_all_sensors_button = ttk.Button(hr_frame, text="全センサー切断", command=self.disconnect_all_sensors)
        self.disconnect_all_sensors_button.grid(row=3, column=2, sticky=tk.W, pady=2, padx=5)

        row_idx += 1

        # --- Prosody Parameters Frame ---
        param_frame = ttk.LabelFrame(main_frame, text="音声パラメータ調整 (VOICEVOX)", padding="10")
        param_frame.grid(row=row_idx, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        param_frame.columnconfigure(2, weight=1)

        parameters_config = [
            ("intonation", "抑揚:", "intonation_var", "intonation_value_label"),
            ("pitch", "ピッチ:", "pitch_var", "pitch_value_label"),
            ("speed", "速度:", "speed_var", "speed_value_label"),
            ("energy", "音量 (エネルギー):", "energy_var", "energy_value_label"),
            ("pause_duration", "ポーズ長:", "pause_duration_var", "pause_duration_value_label")
        ]
        param_row_idx = 0
        for p_name, p_label, p_var_attr, p_lbl_attr in parameters_config:
            self.create_parameter_row(param_frame, p_name, p_label, p_var_attr, p_lbl_attr, param_row_idx)
            param_row_idx += 1

        hfb_frame = ttk.Frame(param_frame)
        hfb_frame.grid(row=param_row_idx, column=0, columnspan=5, pady=(5, 0), sticky='w')

        self.prosody.hfb_enabled_var = tk.BooleanVar(value=self.prosody.is_hfb_enabled())
        hfb_checkbox = ttk.Checkbutton(
            hfb_frame,
            text="心拍数による自動調整 (通常HFB - Verity Sense HR基準)",
            variable=self.prosody.hfb_enabled_var,
            command=self.toggle_hfb,
            style='TCheckbutton'
        )
        hfb_checkbox.pack(side=tk.LEFT, padx=5)

        self.prosody.sinusoidal_hfb_enabled_var = tk.BooleanVar(value=self.prosody.is_sinusoidal_hfb_enabled())
        sinusoidal_hfb_checkbox = ttk.Checkbutton(
            hfb_frame,
            text="抑揚正弦波モード (録音開始時トリガ)",
            variable=self.prosody.sinusoidal_hfb_enabled_var,
            command=self.toggle_sinusoidal_hfb,
            style='TCheckbutton'
        )
        sinusoidal_hfb_checkbox.pack(side=tk.LEFT, padx=15)

        # HFBで自動調整する対象パラメータの選択
        try:
            current_target = self.prosody.get_hfb_target_param()
        except Exception:
            current_target = "intonation"

        self.prosody.hfb_target_param_var = tk.StringVar(value=current_target)
        hfb_target_label = ttk.Label(hfb_frame, text="HFB対象パラメータ:")
        hfb_target_label.pack(side=tk.LEFT, padx=(10, 3))

        self.hfb_target_param_combo = ttk.Combobox(
            hfb_frame,
            textvariable=self.prosody.hfb_target_param_var,
            state="readonly",
            width=14,
            values=["intonation", "pitch", "speed", "energy", "pause_duration"],
        )
        self.hfb_target_param_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.hfb_target_param_combo.bind("<<ComboboxSelected>>", self.on_hfb_target_param_changed)

        row_idx += 1

        # --- Speaker Selection Frame ---
        speaker_frame = ttk.LabelFrame(main_frame, text="話者選択 (VOICEVOX)", padding="10")
        speaker_frame.grid(row=row_idx, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        speaker_frame.columnconfigure(0, weight=1)

        self.speaker_var = tk.StringVar()
        self.speaker_combo = ttk.Combobox(
            speaker_frame,
            textvariable=self.speaker_var,
            state="readonly",
            width=30,
            values=self.speaker.get_all_speaker_names()
        )
        self.speaker_combo.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.speaker_combo.bind("<<ComboboxSelected>>", self.on_speaker_selected)

        current_speaker_name = self.speaker.get_speaker_name_by_id(self.speaker.current_style_id)
        if current_speaker_name:
            self.speaker_var.set(current_speaker_name)

        row_idx += 1

        # --- System Prompt, Conversation, and Control Buttons ---
        main_frame.rowconfigure(row_idx, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=row_idx, column=0, columnspan=2, sticky="nsew", padx=(5, 2), pady=5)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)

        ttk.Label(left_frame, text="システムプロンプト (System Prompt):", style='Bold.TLabel').grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        self.system_prompt = scrolledtext.ScrolledText(left_frame, height=8, wrap=tk.WORD)
        self.system_prompt.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)
        self.system_prompt.insert(tk.END, "You are a kind and helpful AI assistant.")
        self.system_prompt.configure(font=self.default_font)

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=row_idx, column=2, columnspan=3, sticky="nsew", padx=(2, 5), pady=5)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        ttk.Label(right_frame, text="会話ログ:", style='Bold.TLabel').grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        self.conversation_log = scrolledtext.ScrolledText(right_frame, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.conversation_log.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)
        self.conversation_log.configure(font=self.default_font)

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row_idx + 1, column=0, columnspan=5, sticky="ew", padx=5, pady=(0, 5))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        button_frame.columnconfigure(3, weight=1)
        button_frame.columnconfigure(4, weight=1)
        button_frame.columnconfigure(5, weight=1)

        self.tts_test_button = ttk.Button(button_frame, text="TTSテスト (心拍連動)", command=self.on_tts_test_button_clicked)
        self.tts_test_button.grid(row=0, column=0, padx=5, pady=2, sticky="ew") # Shifted to column 0

        self.start_conversation_button = ttk.Button(button_frame, text="会話開始 (Start Conversation)", command=self.start_conversation)
        self.start_conversation_button.grid(row=0, column=1, padx=5, pady=2, sticky="ew") # Shifted to column 1

        self.stop_conversation_button = ttk.Button(button_frame, text="会話停止 (Stop Conversation)", command=self.stop_conversation)
        self.stop_conversation_button.grid(row=0, column=2, padx=5, pady=2, sticky="ew") # Shifted to column 2

        self.save_config_button = ttk.Button(button_frame, text="設定保存 (Save Config)", command=self.save_config)
        self.save_config_button.grid(row=0, column=3, padx=5, pady=2, sticky="ew") # Shifted to column 3

        self.baseline_button = ttk.Button(button_frame, text="安静時心拍測定開始", command=self.start_baseline_measurement)
        self.baseline_button.grid(row=0, column=4, padx=5, pady=2, sticky="ew") # Shifted to column 4

        baseline_frame = ttk.Frame(main_frame)
        baseline_frame.grid(row=row_idx + 2, column=0, columnspan=5, sticky="ew", padx=5, pady=(0, 5))
        baseline_frame.columnconfigure(0, weight=1)

        ttk.Label(baseline_frame, text="安静時心拍測定の時間 (秒):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.baseline_duration_var = tk.IntVar(value=config.BASELINE_MEASUREMENT_DURATION)
        self.baseline_duration_spinbox = ttk.Spinbox(
            baseline_frame,
            from_=10,
            to=600,
            textvariable=self.baseline_duration_var,
            width=6
        )
        self.baseline_duration_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=row_idx + 3, column=0, columnspan=5, sticky="ew", padx=5, pady=(0, 5))
        status_frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style='Status.TLabel',
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=5, pady=2)

        self.after(1000, self.update_hr_status_labels_periodically)

    def create_parameter_row(self, parent, param_name: str, label_text: str, tk_var_attr: str, value_label_attr: str, row_idx: int):
        tk_var = tk.DoubleVar(value=self.prosody.get_parameter(param_name))
        setattr(self.prosody, tk_var_attr, tk_var)

        row = ttk.Frame(parent)
        row.grid(row=row_idx, column=0, columnspan=5, sticky="ew", pady=2)
        row.columnconfigure(1, weight=1)

        ttk.Label(row, text=label_text, width=16).grid(row=0, column=0, sticky=tk.W, padx=(5, 2))
        self.update_parameter_display(param_name)

        scale = ttk.Scale(
            row,
            from_=self.prosody.get_parameter_range(param_name)[0],
            to=self.prosody.get_parameter_range(param_name)[1],
            orient=tk.HORIZONTAL,
            variable=tk_var,
            command=lambda v, p=param_name: self.update_parameter_from_scale(p, float(v))
        )
        scale.grid(row=0, column=1, sticky="ew", padx=5)

        value_label = ttk.Label(row, text=f"{self.prosody.get_parameter(param_name):.2f}", width=6)
        value_label.grid(row=0, column=2, sticky=tk.W, padx=(2, 5))
        setattr(self, value_label_attr, value_label)

        minus_button = ttk.Button(row, text="-", width=3, command=lambda p=param_name: self.adjust_parameter(p, -0.05))
        minus_button.grid(row=0, column=3, padx=2)
        plus_button = ttk.Button(row, text="+", width=3, command=lambda p=param_name: self.adjust_parameter(p, 0.05))
        plus_button.grid(row=0, column=4, padx=2)

    def set_status(self, text: str, color: str = "black") -> None:
        def _update_status():
            try:
                self.status_var.set(text)
                self.status_label.configure(foreground=color)
            except tk.TclError:
                pass
        self.after(0, _update_status)

    def append_log(self, text: str) -> None:
        def _append():
            try:
                self.conversation_log.configure(state=tk.NORMAL)
                self.conversation_log.insert(tk.END, text + "\n")
                self.conversation_log.see(tk.END)
                self.conversation_log.configure(state=tk.DISABLED)
            except tk.TclError:
                pass

        self.after(0, _append)

    def on_speaker_selected(self, event=None) -> None:
        selected_name = self.speaker_var.get()
        for s in self.speaker.speakers:
            if s['name'] == selected_name:
                self.speaker.current_style_id = s['id']
                print(f"Speaker style selected: {selected_name} (id={s['id']})")
                self.set_status(f"Speaker set to: {selected_name}", "blue")
                break

    def update_reference_hr(self) -> None:
        try:
            value_str = self.reference_hr_var.get().strip()
            if not value_str.isdigit():
                raise ValueError("Reference HR must be an integer.")
            value = int(value_str)
            if value <= 0:
                raise ValueError("Reference HR must be positive.")
            self.hr_monitor.set_reference_hr(value)
            print(f"Reference HR updated to: {value}")
            self.set_status(f"Reference HR updated to {value}", "green")
        except Exception as e:
            messagebox.showerror("Invalid HR", f"Failed to update reference HR: {e}")
            self.set_status(f"Failed to update reference HR: {e}", "red")
            self.reference_hr_var.set(str(self.hr_monitor.get_reference_hr()))

    def update_reference_hr_button(self):
        self.update_reference_hr()

    def on_hfb_target_param_changed(self, event=None):
        """HFBで自動調整する対象パラメータが変更されたときの処理。"""
        try:
            if not hasattr(self.prosody, "hfb_target_param_var") or self.prosody.hfb_target_param_var is None:
                return
            param_name = self.prosody.hfb_target_param_var.get()
            if hasattr(self.prosody, "set_hfb_target_param"):
                self.prosody.set_hfb_target_param(param_name)
            self.set_status(f"HFB自動調整パラメータを「{param_name}」に設定しました", "blue")
        except Exception as e:
            print(f"HFB target parameter change error: {e}")
            self.set_status("HFBターゲットパラメータの変更でエラーが発生しました", "red")

    def adjust_parameter(self, param_name: str, delta: float) -> None:
        is_any_hfb_active = (self.prosody.hfb_enabled_var and self.prosody.hfb_enabled_var.get()) or \
                            (self.prosody.sinusoidal_hfb_enabled_var and self.prosody.sinusoidal_hfb_enabled_var.get())

        # HFBが有効なときは、「HFB対象パラメータ」の手動変更を禁止
        target_param = "intonation"
        if hasattr(self.prosody, "get_hfb_target_param"):
            try:
                target_param = self.prosody.get_hfb_target_param()
            except Exception:
                pass

        if is_any_hfb_active and param_name == target_param:
            messagebox.showinfo(
                "HFB有効",
                "HFB（通常または正弦波）が有効なため、このパラメータは自動調整されています。\n"
                "手動変更はHFBを無効にしてから行ってください。"
            )
            current_val = self.prosody.get_parameter(param_name)
            tk_var = getattr(self.prosody, f"{param_name}_var", None)
            if tk_var:
                tk_var.set(current_val)
            self.update_parameter_display(param_name)
            return

        current_value = self.prosody.get_parameter(param_name)
        new_value = current_value + delta
        self.prosody.set_parameter(param_name, new_value)
        self.update_parameter_display(param_name)

    def update_parameter_from_scale(self, param_name: str, value: float) -> None:
        is_any_hfb_active = (self.prosody.hfb_enabled_var and self.prosody.hfb_enabled_var.get()) or \
                            (self.prosody.sinusoidal_hfb_enabled_var and self.prosody.sinusoidal_hfb_enabled_var.get())

        # HFBが有効なときは、「HFB対象パラメータ」のスライダー操作を禁止
        target_param = "intonation"
        if hasattr(self.prosody, "get_hfb_target_param"):
            try:
                target_param = self.prosody.get_hfb_target_param()
            except Exception:
                pass

        if is_any_hfb_active and param_name == target_param:
            current_hfb_val = self.prosody.get_parameter(param_name)
            tk_var = getattr(self.prosody, f"{param_name}_var", None)
            if tk_var and abs(tk_var.get() - current_hfb_val) > 0.001:
                messagebox.showinfo(
                    "HFB有効",
                    "HFB（通常または正弦波）が有効なため、このパラメータは自動調整されています。\n"
                    "手動変更はHFBを無効にしてください。"
                )
                tk_var.set(current_hfb_val)
            self.update_parameter_display(param_name)
            return

        self.prosody.set_parameter(param_name, value)
        self.update_parameter_display(param_name)

    def update_parameter_display(self, param_name: str) -> None:
        try:
            value = self.prosody.get_parameter(param_name)
            value_label_widget = getattr(self, f"{param_name}_value_label", None)
            if value_label_widget and value_label_widget.winfo_exists():
                self.after(0, lambda w=value_label_widget, v=value: w.config(text=f"{v:.2f}"))
        except tk.TclError:
            pass
        except Exception as e_upd_param_disp:
            print(f"Error updating parameter display for {param_name}: {e_upd_param_disp}")

    def toggle_hfb(self):
        try:
            if not hasattr(self.prosody, 'hfb_enabled_var') or self.prosody.hfb_enabled_var is None:
                return
            new_state = self.prosody.hfb_enabled_var.get()
            if new_state and not self.hr_monitor.is_connected:
                messagebox.showwarning("HFB注意", "HFBを有効にするには、まずVerity Senseを接続してください。")
                self.prosody.hfb_enabled_var.set(False)
                return

            self.prosody.enable_hfb(new_state)
            self.set_status(f"心拍数による自動調整(通常HFB)を「{'有効' if new_state else '無効'}」にしました", "blue")

            if new_state:
                if self.prosody.is_sinusoidal_hfb_enabled():
                    self.prosody.enable_sinusoidal_hfb(False)
                    if self.prosody.sinusoidal_hfb_enabled_var:
                        self.prosody.sinusoidal_hfb_enabled_var.set(False)
                    print("抑揚正弦波モードは無効化されました。")
                self.audio.reset_hfb_state()
                self.prosody.set_parameter("intonation", 1.0)
                self.update_parameter_display("intonation")
            else:
                if not self.prosody.is_sinusoidal_hfb_enabled():
                    self.prosody.set_parameter("intonation", 1.0)
                    self.update_parameter_display("intonation")
                    print("通常HFB無効。抑揚を1.0にリセット。")
            self._update_button_states()
        except Exception as e_toggle_hfb:
            print(f"HFB toggle error: {e_toggle_hfb}")
            self.set_status("通常HFB切り替えエラー", "red")

    def toggle_sinusoidal_hfb(self):
        try:
            if not hasattr(self.prosody, 'sinusoidal_hfb_enabled_var') or self.prosody.sinusoidal_hfb_enabled_var is None:
                return
            new_state = self.prosody.sinusoidal_hfb_enabled_var.get()

            self.prosody.enable_sinusoidal_hfb(new_state)
            self.set_status(f"抑揚正弦波モードを「{'有効' if new_state else '無効'}」にしました", "blue")

            if new_state:
                if self.prosody.is_hfb_enabled():
                    self.prosody.enable_hfb(False)
                    if self.prosody.hfb_enabled_var:
                        self.prosody.hfb_enabled_var.set(False)
                    print("通常HFBモードは無効化されました。")
                self.audio.reset_hfb_state()
                initial_intonation = 1.0
                if self.prosody.sinusoidal_hfb_sequence:
                    try:
                        initial_intonation_idx = self.prosody.sinusoidal_hfb_sequence.index(1.0)
                        initial_intonation = self.prosody.sinusoidal_hfb_sequence[initial_intonation_idx]
                    except ValueError:
                        pass
                self.prosody.set_parameter("intonation", initial_intonation)
                self.update_parameter_display("intonation")
            else:
                if not self.prosody.is_hfb_enabled():
                    self.prosody.set_parameter("intonation", 1.0)
                    self.update_parameter_display("intonation")
                    print("抑揚正弦波モード無効。抑揚を1.0にリセット。")
            self._update_button_states()
        except Exception as e_toggle_hfb:
            print(f"Sinusoidal HFB toggle error: {e_toggle_hfb}")
            self.set_status("抑揚正弦波モード切り替えエラー", "red")

    def _update_button_states(self):
        try:
            if self.is_processing or self.is_conversing or self.is_measuring_baseline:
                self.tts_test_button.config(state=tk.DISABLED)
                self.start_conversation_button.config(state=tk.DISABLED)
                self.stop_conversation_button.config(state=tk.NORMAL if self.is_conversing else tk.DISABLED)
                self.save_config_button.config(state=tk.DISABLED)
                self.baseline_button.config(state=tk.DISABLED if self.is_measuring_baseline else tk.NORMAL)
            else:
                self.tts_test_button.config(state=tk.NORMAL)
                self.start_conversation_button.config(state=tk.NORMAL)
                self.stop_conversation_button.config(state=tk.DISABLED)
                self.save_config_button.config(state=tk.NORMAL)
                self.baseline_button.config(state=tk.NORMAL)
        except tk.TclError:
            pass

    def on_tts_test_button_clicked(self):
        if self.is_processing or self.is_conversing or self.is_measuring_baseline:
            return
        self.is_processing = True
        self._update_button_states()
        threading.Thread(target=self._tts_test_procedure, daemon=True).start()

    def _tts_test_procedure(self):
        try:
            test_text = "これは心拍数と連動した抑揚や他のパラメータを確認するためのテストです。"
            self.append_log(f"[TTS Test] {test_text}")
            self.set_status("TTSテスト音声を生成中...", "orange")
            self.audio.text_to_speech(test_text, filename=config.OUTPUT_WAV_FILE)
            self.set_status("TTSテスト完了。", "green")
        finally:
            self.is_processing = False
            self._update_button_states()

    def start_conversation(self):
        if self.is_processing or self.is_conversing or self.is_measuring_baseline:
            return
        self.is_conversing = True
        self._update_button_states()
        threading.Thread(target=self.conversation_manager.conversation_loop, daemon=True).start()
        self.set_status("会話モードを開始しました。", "green")

    def stop_conversation(self):
        if not self.is_conversing:
            return
        self.conversation_manager.stop_conversation()
        self.is_conversing = False
        self._update_button_states()
        self.set_status("会話モードを停止しました。", "blue")

    def start_baseline_measurement(self):
        if self.is_processing or self.is_conversing or self.is_measuring_baseline:
            return
        if not self.hr_monitor.is_connected:
            messagebox.showwarning("安静時測定", "安静時測定にはVerity Senseの接続が必要です。")
            return
        self.is_measuring_baseline = True
        self.is_measurement_stopped_by_user = False
        self._update_button_states()
        threading.Thread(target=self._baseline_measurement_loop, daemon=True).start()

    def _baseline_measurement_loop(self):
        try:
            duration = self.baseline_duration_var.get()
            self.set_status(f"安静時心拍測定中... {duration}秒間、動かず安静にしてください。", "orange")
            start_time = time.time()
            hr_values = []
            while time.time() - start_time < duration:
                if self.is_measurement_stopped_by_user:
                    self.set_status("安静時測定はユーザーにより中断されました。", "blue")
                    return
                hr = self.hr_monitor.get_current_hr()
                if hr > 0:
                    hr_values.append(hr)
                time.sleep(1.0)
            if hr_values:
                avg_hr = int(np.mean(hr_values))
                self.hr_monitor.set_reference_hr(avg_hr)
                self.reference_hr_var.set(str(avg_hr))
                self.set_status(f"安静時心拍測定完了。基準心拍数を {avg_hr} BPM に設定しました。", "green")
            else:
                self.set_status("安静時心拍測定に失敗しました（有効な心拍データなし）。", "red")
        finally:
            self.is_measuring_baseline = False
            self._update_button_states()

    def load_config(self) -> None:
        print(f"Loading configuration from '{config.CONFIG_FILE}'...")
        try:
            if os.path.exists(config.CONFIG_FILE):
                with open(config.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    app_config = json.load(f)

                ref_hr = app_config.get('reference_hr', self.hr_monitor.get_reference_hr())
                self.hr_monitor.set_reference_hr(ref_hr)
                self.reference_hr_var.set(str(ref_hr))
                print(f"  Reference HR loaded: {ref_hr}")

                prosody_config = app_config.get('prosody', {})
                # 旧形式（キー名が 'intonation' など）の設定ファイルにも対応
                for param in ["intonation", "pitch", "speed", "energy", "pause_duration"]:
                    scale_key = f"{param}_scale"
                    if scale_key in prosody_config:
                        value = prosody_config.get(scale_key, getattr(self.prosody, scale_key))
                    elif param in prosody_config:
                        value = prosody_config.get(param, getattr(self.prosody, f"{param}_scale"))
                    else:
                        value = getattr(self.prosody, f"{param}_scale")
                    self.prosody.set_parameter(param, value)
                    self.update_parameter_display(param)
                    print(f"  Prosody ({param}) loaded: {value:.2f}")

                # HFB対象パラメータ（指定がなければ抑揚）
                hfb_target_param = prosody_config.get("hfb_target_param", getattr(self.prosody, "hfb_target_param", "intonation"))
                if hasattr(self.prosody, "set_hfb_target_param"):
                    self.prosody.set_hfb_target_param(hfb_target_param)
                if hasattr(self.prosody, "hfb_target_param_var") and self.prosody.hfb_target_param_var is not None:
                    try:
                        self.prosody.hfb_target_param_var.set(hfb_target_param)
                    except tk.TclError:
                        pass

                # 起動時はHFB系は常にOFFから開始
                self.prosody.enable_hfb(False)
                if self.prosody.hfb_enabled_var:
                    self.prosody.hfb_enabled_var.set(False)

                self.prosody.enable_sinusoidal_hfb(False)
                if self.prosody.sinusoidal_hfb_enabled_var:
                    self.prosody.sinusoidal_hfb_enabled_var.set(False)
                print(f"  HFB states initialized to: OFF (startup default)")

                prompt = app_config.get('system_prompt', "You are a kind and helpful AI assistant.")
                self.system_prompt.delete('1.0', tk.END)
                self.system_prompt.insert(tk.END, prompt)

                speaker_id = app_config.get('speaker_id', self.speaker.current_style_id)
                self.speaker.current_style_id = speaker_id
                current_speaker_name = self.speaker.get_speaker_name_by_id(speaker_id)
                if current_speaker_name:
                    self.speaker_var.set(current_speaker_name)
                    print(f"  Speaker loaded: {current_speaker_name} (id={speaker_id})")

                baseline_duration = app_config.get('baseline_measurement_duration', config.BASELINE_MEASUREMENT_DURATION)
                self.baseline_duration_var.set(baseline_duration)
                print(f"  Baseline measurement duration loaded: {baseline_duration} seconds")

                self.set_status("Configuration loaded", "green")
            else:
                print("No configuration file found. Using defaults.")
                self.set_status("No configuration file found. Using defaults.", "blue")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load configuration:\n{e}")
            print(f"Failed to load configuration: {e}")
            self.set_status(f"Failed to load config: {e}", "red")

    def save_config(self) -> None:
        if self.is_processing or self.is_conversing or self.is_measuring_baseline:
            messagebox.showwarning("Busy", "Cannot save configuration while processing, in conversation, or measuring baseline.")
            return

        print(f"Saving configuration to '{config.CONFIG_FILE}'...")
        try:
            app_config = {
                'reference_hr': self.hr_monitor.get_reference_hr(),
                'prosody': {
                    **{f"{p}_scale": self.prosody.get_parameter(p) for p in
                       ["intonation", "pitch", "speed", "energy", "pause_duration"]},
                    'hfb_target_param': getattr(self.prosody, "hfb_target_param", "intonation")
                },
                'system_prompt': self.system_prompt.get('1.0', tk.END).strip(),
                'speaker_id': self.speaker.current_style_id,
                'baseline_measurement_duration': self.baseline_duration_var.get()
            }
            with open(config.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(app_config, f, indent=4, ensure_ascii=False)
            print("Configuration saved successfully.")
            self.set_status("Configuration saved", "green")
        except Exception as e_save_cfg:
            messagebox.showerror("Save Error", f"Failed to save configuration:\n{e_save_cfg}")
            print(f"Failed to save configuration: {e_save_cfg}")
            self.set_status(f"Failed to save config: {e_save_cfg}", "red")

    def _setup_logger_thread(self):
        self.logger_thread = LoggingThread(self.log_queue)
        self.logger_thread.start()

    def connect_all_sensors(self):
        self.set_status("全てのセンサーを接続中...", "orange")
        threading.Thread(target=self._connect_all_sensors_thread, daemon=True).start()

    def _connect_all_sensors_thread(self):
        hr_connected = False
        h10_connected = False
        
        # Connect Verity Sense
        if not self.hr_monitor.is_connected:
            try:
                self.set_status("Verity Senseに接続中...", "blue")
                hr_connected = self.hr_monitor.start_monitoring()
                if hr_connected:
                    self.set_status("Verity Senseに接続しました。", "green")
                else:
                    self.set_status("Verity Senseの接続に失敗しました。", "red")
            except Exception as e:
                messagebox.showerror("接続エラー", f"Verity Senseの接続に失敗しました:\n{e}")
                self.set_status("Verity Senseの接続に失敗しました。", "red")
        else:
            hr_connected = True
            self.set_status("Verity Senseは既に接続されています。", "blue")

        # Connect H10
        if not self.h10_monitor.is_connected:
            try:
                self.set_status("H10に接続中...", "blue")
                h10_connected = self.h10_monitor.start_monitoring()
                if h10_connected:
                    self.set_status("H10に接続しました。", "green")
                else:
                    self.set_status("H10の接続に失敗しました。", "red")
            except Exception as e:
                messagebox.showerror("接続エラー", f"H10の接続に失敗しました:\n{e}")
                self.set_status("H10の接続に失敗しました。", "red")
        else:
            h10_connected = True
            self.set_status("H10は既に接続されています。", "blue")
        
        if hr_connected and h10_connected:
            self.set_status("全てのセンサーに接続しました。", "green")
        elif hr_connected or h10_connected:
            self.set_status("一部のセンサーに接続しました。", "orange")
        else:
            self.set_status("全てのセンサーの接続に失敗しました。", "red")
        self.update_hr_status_labels()

    def disconnect_all_sensors(self):
        self.set_status("全てのセンサーを切断中...", "orange")
        threading.Thread(target=self._disconnect_all_sensors_thread, daemon=True).start()

    def _disconnect_all_sensors_thread(self):
        hr_disconnected = False
        h10_disconnected = False
        
        if self.hr_monitor.is_connected:
            try:
                self.hr_monitor.stop_monitoring()
                self.set_status("Verity Senseを切断しました。", "blue")
                hr_disconnected = True
            except Exception as e:
                print(f"Verity Senseの切断中にエラーが発生しました: {e}")
                self.set_status("Verity Senseの切断に失敗しました。", "red")
        else:
            hr_disconnected = True # Already disconnected
            
        if self.h10_monitor.is_connected:
            try:
                self.h10_monitor.stop_monitoring()
                self.set_status("H10を切断しました。", "blue")
                h10_disconnected = True
            except Exception as e:
                print(f"H10の切断中にエラーが発生しました: {e}")
                self.set_status("H10の切断に失敗しました。", "red")
        else:
            h10_disconnected = True # Already disconnected

        if hr_disconnected and h10_disconnected:
            self.set_status("全てのセンサーを切断しました。", "blue")
        elif hr_disconnected or h10_disconnected:
            self.set_status("一部のセンサーを切断しました。", "orange")
        else:
            self.set_status("全てのセンサーの切断に失敗しました。", "red")
        self.update_hr_status_labels()

    def update_hr_status_labels(self):
        try:
            if self.hr_monitor.is_connected:
                self.hr_status_label.configure(text="接続中", foreground="green")
            else:
                self.hr_status_label.configure(text="未接続", foreground="red")

            current_hr = self.hr_monitor.get_current_hr()
            self.current_hr_var.set(str(current_hr) if current_hr > 0 else "--")

            if self.h10_monitor.is_connected:
                self.h10_status_label.configure(text="接続中", foreground="green")
            else:
                self.h10_status_label.configure(text="未接続", foreground="red")

            current_rr = self.h10_monitor.get_current_rr()
            self.current_rr_var.set(str(current_rr) if current_rr > 0 else "--")
        except tk.TclError:
            pass

    def update_hr_status_labels_periodically(self):
        if self._closing:
            return
        self.update_hr_status_labels()
        self.after(1000, self.update_hr_status_labels_periodically)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "アプリケーションを終了しますか？"):
            try:
                self._closing = True # Signal that we are shutting down
                self.is_conversing = False
                self.conversation_manager.stop_conversation()
                self.hr_monitor.stop_monitoring()
                self.h10_monitor.stop_monitoring()
                if self.logger_thread:
                    self.logger_thread.stop()
            except Exception as e:
                print(f"Error during shutdown: {e}")
            self.destroy()


def main():
    prosody = ProsodySettings()
    if not VoicevoxManager.check_server():
        print("VOICEVOX server is not reachable. Please start VOICEVOX engine.")
        sys.exit(1)

    speakers_list = VoicevoxManager.get_speakers()
    speaker_settings = SpeakerSettings(speakers_list)

    log_queue = queue.Queue()

    hr_monitor = HeartRateMonitor(log_queue_ref=log_queue)
    h10_monitor = H10Monitor(log_queue_ref=log_queue)

    audio_processor = AudioProcessor(
        prosody_settings=prosody,
        speaker_settings=speaker_settings,
        hr_monitor=hr_monitor,
        h10_monitor=h10_monitor,
        log_queue_ref=log_queue,
        faster_whisper_model_instance=config.load_whisper_model()
    )

    app = Application(
        prosody_settings=prosody,
        speaker_settings=speaker_settings,
        audio_processor=audio_processor,
        hr_monitor=hr_monitor,
        h10_monitor=h10_monitor,
        log_queue_ref=log_queue
    )
    audio_processor.app = app

    app.load_config()

    app.mainloop()


if __name__ == "__main__":
    main()
