# gui/application.py
"""メインアプリケーションクラス"""

import asyncio
import datetime
import json
import os
import queue
import signal
import sys
import threading
import time
import re
from typing import Optional, Any, List, Dict, Awaitable
from collections import defaultdict
from pathlib import Path

import tkinter as tk
from tkinter import font, messagebox, scrolledtext, ttk, filedialog
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rcParams
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

import openai

import config
from logger_utils import LoggingThread, get_timestamped_log_path, format_subject_id_for_filename
from polar_monitor import HeartRateMonitor, H10Monitor
from conversation_manager import ConversationManager
from audio_processing import ProsodySettings, SpeakerSettings, AudioProcessor, VoicevoxManager
from hrf2_controller import ControlMode, GainType
from video_recorder import VideoRecorder
from audio_device_utils import (
    get_conversation_mic_device,
    get_secondary_mic_device,
    list_input_devices,
)

from .status_window import StatusDisplayWindow
from .ecg_analysis import (
    FILENAME_PATTERN, ANALYS_CONDITION_MAP, ANALYS_CONDITION_ORDER,
    CONDITION_COLORS,
    run_batch_analysis as analys_run_batch_analysis,
    generate_box_plots as analys_generate_box_plots,
)
from .questionnaire_analysis import (
    Q_CONDITION_ORDER,
    Q_CONDITION_COLORS,
    generate_plots as analys_q_generate_plots,
)
from .panas_analysis import (
    generate_panas_plots,
    format_reliability_text,
)
from .realtime_monitor import RealtimeMonitorMixin
from .timeseries_analysis import TimeseriesAnalysisMixin


class Application(TimeseriesAnalysisMixin, RealtimeMonitorMixin, tk.Toplevel):
    """Main application class with Tkinter UI."""
    def __init__(self, master, # Added master parameter
                 prosody_settings: ProsodySettings,
                 speaker_settings: SpeakerSettings,
                 audio_processor: AudioProcessor,
                 hr_monitor: HeartRateMonitor,
                 h10_monitor: H10Monitor,
                 log_queue_ref: queue.Queue):
        super().__init__(master)

        self.master = master
        self.prosody = prosody_settings
        self.speaker = speaker_settings
        self.audio = audio_processor
        self.hr_monitor = hr_monitor
        self.h10_monitor = h10_monitor
        self.log_queue = log_queue_ref

        self.audio.app = self

        self.is_processing: bool = False
        self.is_conversing: bool = False
        self.is_measuring_baseline: bool = False
        self.processing_thread: Optional[threading.Thread] = None

        # --- New asyncio integration ---
        self.async_loop = asyncio.get_event_loop()
        
        self.logging_thread = LoggingThread(self.log_queue)
        self.logging_thread.start()

        # Initialize LLM client (supports both local LLM and OpenAI)
        if config.USE_LOCAL_LLM:
            self.openai_client = openai.OpenAI(
                base_url=config.LOCAL_LLM_BASE_URL,
                api_key=config.LOCAL_LLM_API_KEY
            )
            print(f"ローカルLLMに接続: {config.LOCAL_LLM_BASE_URL}")
        else:
            self.openai_client = openai.OpenAI() 
        
        self.conversation_manager = ConversationManager(
            audio_processor=self.audio,
            hr_monitor=self.hr_monitor,
            h10_monitor=self.h10_monitor,
            app_ref=self,
            log_queue_ref=self.log_queue
        )
        
        self._closing: bool = False
        self.conversation_start_time: Optional[datetime.datetime] = None
        self.current_session_timestamp: Optional[str] = None

        self.baseline_duration_var = tk.IntVar(value=config.DEFAULT_BASELINE_MEASUREMENT_DURATION)
        self.reference_hr_var = tk.StringVar(value=str(self.hr_monitor.get_reference_hr()))
        self.subject_id_var = tk.StringVar(value="")
        self.current_subject_id: Optional[str] = None

        # LLM設定用変数
        self.use_local_llm_var = tk.BooleanVar(value=config.USE_LOCAL_LLM)
        self.openai_api_key_var = tk.StringVar(value=os.getenv("OPENAI_API_KEY", ""))

        # マイク選択用の状態
        self.mic_device_var = tk.StringVar(value="")
        self._mic_label_to_index: Dict[str, int] = {}
        self._mic_devices_cache: List[Dict[str, Any]] = []

        # ビデオ録画機能の初期化
        # 会話システム優先: 2つ目のマイクがある場合のみ映像に音声を付ける
        self.video_recorder: Optional[VideoRecorder] = None
        self._configure_video_recorder_for_mic(self.audio.input_device_index, silent=True)
        self.video_recording_enabled = tk.BooleanVar(value=True)  # デフォルトで録画有効

        self.status_display_window = StatusDisplayWindow(self)
        self.status_display_window.withdraw()
        self.status_window_visible: bool = False

        self.setup_ui()
        self.load_config()

        # ver3.10方式: VADストリーミングは使用せず、会話ループ内で録音→認識を行う
        # self.audio.start_streaming_input()
        # self.audio.start_vad_loop()
        # self.after(100, self._check_interim_transcription_queue)

        self.update_ui_periodic()

        # Start the asyncio event loop polling
        self.after(100, self.poll_asyncio_loop)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
        except (ValueError, AttributeError):
            print("Warning: Cannot set SIGINT handler (e.g., not in main thread or unsupported OS).")

    def poll_asyncio_loop(self):
        """Poll the asyncio loop from the Tkinter main loop."""
        self.async_loop.call_soon(self.async_loop.stop)
        self.async_loop.run_forever()
        if not self._closing:
            self.after(50, self.poll_asyncio_loop)

    def run_coroutine(self, coro: Awaitable[Any]):
        """Schedule coroutine on the application's asyncio loop."""
        if self._closing:
            raise RuntimeError("Application is closing; cannot schedule coroutine.")
        return asyncio.ensure_future(coro, loop=self.async_loop)

    def setup_ui(self) -> None:
        self.title(f"心拍数連動AI音声アシスタント (v{datetime.datetime.now().strftime('%Y.%m.%d')})")
        self.geometry("950x1000")
        self.minsize(900, 950)

        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="Helvetica", size=11)
        self.label_font = font.Font(family="Helvetica", size=11, weight="bold")
        self.button_font = font.Font(family="Helvetica", size=11)
        self.status_font = font.Font(family="Helvetica", size=10)
        self.check_font = font.Font(family="Helvetica", size=11)

        style = ttk.Style(self)
        available_themes = style.theme_names()
        if 'aqua' in available_themes: style.theme_use('aqua')
        elif 'clam' in available_themes: style.theme_use('clam')
        elif 'vista' in available_themes: style.theme_use('vista')
        else: print(f"Preferred themes not found, using default: {style.theme_use()}. Available: {available_themes}")

        style.configure('TButton', font=self.button_font, padding=5)
        style.configure('TLabel', font=self.default_font)
        style.configure('Bold.TLabel', font=self.label_font)
        style.configure('TCombobox', font=self.default_font)
        style.configure('Status.TLabel', font=self.status_font)
        style.configure('TCheckbutton', font=self.check_font)
        style.configure('TLabelframe.Label', font=self.label_font)

        # タブ付きノートブックを作成
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # タブ1: 会話システム
        self.conversation_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.conversation_tab, text="会話システム")

        # タブ2: リアルタイムモニター
        self.realtime_monitor_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.realtime_monitor_tab, text="リアルタイムモニター")

        # タブ3: ECG/HRV解析 (Analys)
        self.ecg_analysis_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.ecg_analysis_tab, text="ECG/HRV解析")

        # タブ4: 個別時系列解析
        self.timeseries_analysis_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.timeseries_analysis_tab, text="時系列解析")

        # タブ5: アンケート解析 (Analys_Q)
        self.questionnaire_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.questionnaire_tab, text="アンケート解析")

        # 各タブのUIを構築
        self._setup_conversation_tab()
        self._setup_realtime_monitor_tab()
        self._setup_ecg_analysis_tab()
        self._setup_timeseries_analysis_tab()
        self._setup_questionnaire_tab()

    def _setup_conversation_tab(self) -> None:
        """会話システムタブのUIを構築"""
        # スクロール可能なフレームを作成
        tab_frame = self.conversation_tab
        tab_frame.columnconfigure(0, weight=1)
        tab_frame.rowconfigure(0, weight=1)

        # Canvas + Scrollbar
        canvas = tk.Canvas(tab_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # スクロール可能なメインフレーム
        main_frame = ttk.Frame(canvas, padding="10")
        canvas_window = canvas.create_window((0, 0), window=main_frame, anchor="nw")

        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def configure_canvas_width(event):
            canvas.itemconfig(canvas_window, width=event.width)

        main_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_width)

        # マウスホイールでスクロール
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

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

        ttk.Label(hr_frame, text="基準心拍数(HRF用):").grid(row=0, column=0, sticky=tk.W, pady=2, padx=5)
        reference_hr_entry = ttk.Entry(hr_frame, textvariable=self.reference_hr_var, width=5)
        reference_hr_entry.grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        reference_hr_entry.bind("<Return>", self.update_reference_hr)
        ttk.Label(hr_frame, text="BPM").grid(row=0, column=2, sticky=tk.W, pady=2, padx=0)
        update_hr_button = ttk.Button(hr_frame, text="設定", command=self.update_reference_hr_button, width=6)
        update_hr_button.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)

        ttk.Label(hr_frame, text="基準HR計測時間:").grid(row=1, column=0, sticky=tk.W, pady=2, padx=5)
        self.baseline_duration_spinbox = ttk.Spinbox(
            hr_frame, from_=10, to=300, increment=5,
            textvariable=self.baseline_duration_var, width=4
        )
        self.baseline_duration_spinbox.grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(hr_frame, text="秒").grid(row=1, column=2, sticky=tk.W, pady=2, padx=0)
        
        self.measure_baseline_button = ttk.Button(hr_frame, text="基準HR計測(Verity)", command=self.measure_baseline_hr, width=14)
        self.measure_baseline_button.grid(row=1, column=3, padx=(10,5), pady=2, sticky=tk.W)

        self.connect_button = ttk.Button(hr_frame, text="センサー類 接続", command=self.connect_devices, width=12)
        self.connect_button.grid(row=0, column=4, rowspan=2, padx=5, pady=5, sticky=tk.W)

        self.hr_label = ttk.Label(hr_frame, text="Verity Sense: -- BPM | H10: -- BPM", style='Bold.TLabel')
        self.hr_label.grid(row=2, column=0, columnspan=5, pady=(10,2), padx=5, sticky=tk.W)
        self.hr_status_label = ttk.Label(hr_frame, text="Verity Sense: 未接続", foreground="red", style='Status.TLabel')
        self.hr_status_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2, padx=5)
        self.h10_status_label = ttk.Label(hr_frame, text="H10: 未接続", foreground="red", style='Status.TLabel')
        self.h10_status_label.grid(row=3, column=2, columnspan=3, sticky=tk.W, pady=2, padx=5)
        row_idx += 1

        # --- Session Information Frame ---
        session_container = ttk.Frame(main_frame)
        session_container.grid(row=row_idx, column=0, columnspan=5, sticky="ew", padx=5, pady=5)

        session_left_column = ttk.Frame(session_container)
        session_left_column.pack(side=tk.LEFT, anchor="nw")

        session_frame = ttk.LabelFrame(session_left_column, text="セッション情報", padding="10")
        session_frame.pack(side=tk.TOP, anchor="nw", fill=tk.X)
        session_frame.columnconfigure(1, weight=0)

        ttk.Label(session_frame, text="被験者番号:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=5)
        self.subject_entry = ttk.Entry(session_frame, textvariable=self.subject_id_var, width=10)
        self.subject_entry.grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(session_frame, text="(半角英数字/-/_)").grid(row=0, column=2, sticky=tk.W, pady=2, padx=5)
        self.subject_hint_label = ttk.Label(session_frame, text="番号未設定", foreground="red", style='Status.TLabel')
        self.subject_hint_label.grid(row=0, column=3, sticky=tk.W, pady=2, padx=(10,5))
        self.subject_id_var.trace_add("write", self._on_subject_id_change)
        self._update_subject_id_hint()

        mic_frame = ttk.LabelFrame(session_left_column, text="会話マイク", padding="10")
        mic_frame.pack(side=tk.TOP, anchor="nw", fill=tk.X, pady=(10, 0))
        mic_frame.columnconfigure(1, weight=1)

        ttk.Label(mic_frame, text="マイクデバイス:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=5)
        self.mic_device_combo = ttk.Combobox(
            mic_frame,
            textvariable=self.mic_device_var,
            state="readonly",
            width=35
        )
        self.mic_device_combo.grid(row=0, column=1, sticky="ew", pady=2, padx=5)
        self.mic_device_combo.bind("<<ComboboxSelected>>", self._on_mic_device_selected)

        self.mic_refresh_button = ttk.Button(
            mic_frame,
            text="再読み込み",
            width=10,
            command=self._refresh_mic_devices
        )
        self.mic_refresh_button.grid(row=0, column=2, sticky=tk.W, pady=2, padx=5)

        self.mic_device_hint_label = ttk.Label(
            mic_frame,
            text="マイク未選択",
            foreground="red",
            style='Status.TLabel'
        )
        self.mic_device_hint_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5,0), padx=5)
        self._refresh_mic_devices()

        # --- LLM Settings Frame ---
        llm_frame = ttk.LabelFrame(session_container, text="LLM設定", padding="10")
        llm_frame.pack(side=tk.LEFT, anchor="nw", padx=(10, 0))

        # LLM選択ラジオボタン
        llm_select_frame = ttk.Frame(llm_frame)
        llm_select_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Radiobutton(
            llm_select_frame, text="ローカルLLM",
            variable=self.use_local_llm_var, value=True,
            command=self._on_llm_selection_change
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Radiobutton(
            llm_select_frame, text="OpenAI API",
            variable=self.use_local_llm_var, value=False,
            command=self._on_llm_selection_change
        ).pack(side=tk.LEFT)

        # OpenAI APIキー入力欄
        ttk.Label(llm_frame, text="OpenAI APIキー:").grid(row=1, column=0, sticky=tk.W, pady=2, padx=5)
        self.api_key_entry = ttk.Entry(llm_frame, textvariable=self.openai_api_key_var, width=35, show="*")
        self.api_key_entry.grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)

        # APIキー表示/非表示ボタン
        self.show_api_key_var = tk.BooleanVar(value=False)
        self.toggle_key_btn = ttk.Button(llm_frame, text="表示", width=5, command=self._toggle_api_key_visibility)
        self.toggle_key_btn.grid(row=1, column=2, sticky=tk.W, pady=2, padx=2)

        # LLMステータス表示
        self.llm_status_label = ttk.Label(llm_frame, text="", foreground="green", style='Status.TLabel')
        self.llm_status_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=2, padx=5)
        self._update_llm_status()

        # APIキー変更時に自動更新
        self.openai_api_key_var.trace_add("write", self._on_api_key_change)

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
        hfb_frame.grid(row=param_row_idx, column=0, columnspan=5, pady=(5,0), sticky='w')
        
        self.prosody.hfb_enabled_var = tk.BooleanVar(value=self.prosody.is_hfb_enabled())
        hfb_checkbox = ttk.Checkbutton(
            hfb_frame, text="心拍フィードバック (HRF - Verity Sense HR基準)",
            variable=self.prosody.hfb_enabled_var, command=self.toggle_hfb, style='TCheckbutton'
        )
        hfb_checkbox.pack(side=tk.LEFT, padx=5)

        self.prosody.sinusoidal_hfb_enabled_var = tk.BooleanVar(value=self.prosody.is_sinusoidal_hfb_enabled())
        sinusoidal_hfb_checkbox = ttk.Checkbutton(
            hfb_frame, text="正弦波モード (Sin)",
            variable=self.prosody.sinusoidal_hfb_enabled_var,
            command=self.toggle_sinusoidal_hfb,
            style='TCheckbutton'
        )
        sinusoidal_hfb_checkbox.pack(side=tk.LEFT, padx=15)

        # HRFで自動調整する対象パラメータの選択
        try:
            current_target = self.prosody.get_hfb_target_param()
        except Exception:
            current_target = "intonation"

        self.prosody.hfb_target_param_var = tk.StringVar(value=current_target)
        hfb_target_label = ttk.Label(hfb_frame, text="HRF対象パラメータ:")
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
        
        # --- Speaker Selection and HRF2 Frame (横に並べる) ---
        speaker_hrf2_container = ttk.Frame(main_frame)
        speaker_hrf2_container.grid(row=row_idx, column=0, columnspan=5, sticky="ew", padx=5, pady=5)

        # 話者選択フレーム（左側）- コンパクトレイアウト
        speaker_frame = ttk.LabelFrame(speaker_hrf2_container, text="話者 (VOICEVOX)", padding="5")
        speaker_frame.pack(side=tk.LEFT, fill=tk.X, expand=False, padx=(0, 5))

        # 上段: コンボボックスのみ
        self.speaker_var = tk.StringVar()
        self.speaker_combo = ttk.Combobox(speaker_frame, textvariable=self.speaker_var, state="readonly", width=25)
        self.speaker_combo.pack(fill=tk.X)
        self.speaker_combo.bind("<<ComboboxSelected>>", self.on_speaker_selected)

        # 下段: 選択中表示 + テストボタン
        speaker_row2 = ttk.Frame(speaker_frame)
        speaker_row2.pack(fill=tk.X, pady=(2, 0))

        self.speaker_status = ttk.Label(speaker_row2, text="未選択", foreground="orange", style='Status.TLabel')
        self.speaker_status.pack(side=tk.LEFT)

        test_button = ttk.Button(speaker_row2, text="テスト", command=self.test_speech, width=6)
        test_button.pack(side=tk.RIGHT)
        self.populate_speaker_list()

        # HRF2設定フレーム（右側）
        hrf2_frame = ttk.LabelFrame(speaker_hrf2_container, text="HRF2 (心拍追従モード)", padding="10")
        hrf2_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Row 0: HRF2有効/無効チェックボックス と 制御モード選択
        hrf2_row0_frame = ttk.Frame(hrf2_frame)
        hrf2_row0_frame.grid(row=0, column=0, columnspan=8, sticky=tk.W, pady=2, padx=5)

        self.prosody.hrf2_enabled_var = tk.BooleanVar(value=self.prosody.is_hrf2_enabled())
        hrf2_checkbox = ttk.Checkbutton(
            hrf2_row0_frame, text="HRF2有効",
            variable=self.prosody.hrf2_enabled_var, command=self.toggle_hrf2
        )
        hrf2_checkbox.pack(side=tk.LEFT, padx=(0, 10))

        # 制御モード選択（PID / Adaptive / GainScheduled / Robust）
        ttk.Label(hrf2_row0_frame, text="制御:").pack(side=tk.LEFT, padx=(0, 2))
        self.hrf2_control_mode_var = tk.StringVar(value=self.prosody.get_hrf2_control_mode().value)
        hrf2_mode_combo = ttk.Combobox(
            hrf2_row0_frame, textvariable=self.hrf2_control_mode_var,
            values=["PID", "Adaptive", "GainScheduled", "Robust"], state="readonly", width=12
        )
        hrf2_mode_combo.pack(side=tk.LEFT, padx=(0, 10))
        hrf2_mode_combo.bind("<<ComboboxSelected>>", self._on_hrf2_control_mode_change)

        # 目標心拍数
        ttk.Label(hrf2_row0_frame, text="目標BPM:").pack(side=tk.LEFT, padx=(0, 2))
        self.hrf2_target_hr_var = tk.DoubleVar(value=self.prosody.get_hrf2_target_hr())
        hrf2_target_spinbox = ttk.Spinbox(
            hrf2_row0_frame, from_=40, to=180, width=5,
            textvariable=self.hrf2_target_hr_var,
            command=self._on_hrf2_target_change
        )
        hrf2_target_spinbox.pack(side=tk.LEFT)
        hrf2_target_spinbox.bind("<Return>", lambda e: self._on_hrf2_target_change())
        hrf2_target_spinbox.bind("<FocusOut>", lambda e: self._on_hrf2_target_change())
        # 変数変更時にも反映
        self.hrf2_target_hr_var.trace_add("write", lambda *args: self._on_hrf2_target_change())

        # Row 1: PIDゲイン設定
        self.hrf2_pid_frame = ttk.Frame(hrf2_frame)
        self.hrf2_pid_frame.grid(row=1, column=0, columnspan=8, sticky=tk.W, pady=2, padx=5)

        ttk.Label(self.hrf2_pid_frame, text="PID:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(self.hrf2_pid_frame, text="Kp:").pack(side=tk.LEFT)
        self.hrf2_kp_var = tk.DoubleVar(value=0.02)
        ttk.Entry(self.hrf2_pid_frame, textvariable=self.hrf2_kp_var, width=6).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Label(self.hrf2_pid_frame, text="Ki:").pack(side=tk.LEFT)
        self.hrf2_ki_var = tk.DoubleVar(value=0.005)
        ttk.Entry(self.hrf2_pid_frame, textvariable=self.hrf2_ki_var, width=6).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Label(self.hrf2_pid_frame, text="Kd:").pack(side=tk.LEFT)
        self.hrf2_kd_var = tk.DoubleVar(value=0.01)
        ttk.Entry(self.hrf2_pid_frame, textvariable=self.hrf2_kd_var, width=6).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(self.hrf2_pid_frame, text="適用", command=self._apply_hrf2_pid_gains).pack(side=tk.LEFT)

        # Row 2: 適応制御パラメータ設定
        self.hrf2_adaptive_frame = ttk.Frame(hrf2_frame)
        self.hrf2_adaptive_frame.grid(row=2, column=0, columnspan=8, sticky=tk.W, pady=2, padx=5)

        ttk.Label(self.hrf2_adaptive_frame, text="適応:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(self.hrf2_adaptive_frame, text="γ:").pack(side=tk.LEFT)
        adaptive_config = self.prosody.get_hrf2_adaptive_config()
        self.hrf2_gamma_var = tk.DoubleVar(value=adaptive_config.gamma)
        ttk.Entry(self.hrf2_adaptive_frame, textvariable=self.hrf2_gamma_var, width=7).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Label(self.hrf2_adaptive_frame, text="τ:").pack(side=tk.LEFT)
        self.hrf2_tau_var = tk.DoubleVar(value=adaptive_config.reference_time_constant)
        ttk.Entry(self.hrf2_adaptive_frame, textvariable=self.hrf2_tau_var, width=5).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(self.hrf2_adaptive_frame, text="適用", command=self._apply_hrf2_adaptive_params).pack(side=tk.LEFT, padx=(0, 5))

        # 適応パラメータθの表示
        ttk.Label(self.hrf2_adaptive_frame, text="θ:").pack(side=tk.LEFT)
        self.hrf2_theta_label = ttk.Label(self.hrf2_adaptive_frame, text="0.0200", width=7)
        self.hrf2_theta_label.pack(side=tk.LEFT)

        # Row 3: ゲインスケジューリング設定
        self.hrf2_gainschedule_frame = ttk.Frame(hrf2_frame)
        self.hrf2_gainschedule_frame.grid(row=3, column=0, columnspan=8, sticky=tk.W, pady=2, padx=5)

        ttk.Label(self.hrf2_gainschedule_frame, text="GS:").pack(side=tk.LEFT, padx=(0, 5))

        # 閾値設定
        ttk.Label(self.hrf2_gainschedule_frame, text="高閾値:").pack(side=tk.LEFT)
        gs_config = self.prosody.get_hrf2_gain_schedule_config()
        self.hrf2_gs_high_var = tk.DoubleVar(value=gs_config.error_threshold_high)
        ttk.Entry(self.hrf2_gainschedule_frame, textvariable=self.hrf2_gs_high_var, width=4).pack(side=tk.LEFT, padx=(0, 3))

        ttk.Label(self.hrf2_gainschedule_frame, text="中閾値:").pack(side=tk.LEFT)
        self.hrf2_gs_med_var = tk.DoubleVar(value=gs_config.error_threshold_medium)
        ttk.Entry(self.hrf2_gainschedule_frame, textvariable=self.hrf2_gs_med_var, width=4).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(self.hrf2_gainschedule_frame, text="適用", command=self._apply_hrf2_gs_params).pack(side=tk.LEFT, padx=(0, 5))

        # 現在のゾーン表示
        ttk.Label(self.hrf2_gainschedule_frame, text="zone:").pack(side=tk.LEFT)
        self.hrf2_gs_zone_label = ttk.Label(self.hrf2_gainschedule_frame, text="---", width=6)
        self.hrf2_gs_zone_label.pack(side=tk.LEFT)

        # ゲインタイプ選択
        ttk.Label(self.hrf2_gainschedule_frame, text=" | 高:").pack(side=tk.LEFT)
        gain_types = ["P", "PI", "PD", "PID"]
        self.hrf2_gs_type_high_var = tk.StringVar(value=gs_config.gain_type_high.value)
        ttk.Combobox(self.hrf2_gainschedule_frame, textvariable=self.hrf2_gs_type_high_var,
                     values=gain_types, width=4, state="readonly").pack(side=tk.LEFT)

        ttk.Label(self.hrf2_gainschedule_frame, text="中:").pack(side=tk.LEFT)
        self.hrf2_gs_type_med_var = tk.StringVar(value=gs_config.gain_type_medium.value)
        ttk.Combobox(self.hrf2_gainschedule_frame, textvariable=self.hrf2_gs_type_med_var,
                     values=gain_types, width=4, state="readonly").pack(side=tk.LEFT)

        ttk.Label(self.hrf2_gainschedule_frame, text="低:").pack(side=tk.LEFT)
        self.hrf2_gs_type_low_var = tk.StringVar(value=gs_config.gain_type_low.value)
        ttk.Combobox(self.hrf2_gainschedule_frame, textvariable=self.hrf2_gs_type_low_var,
                     values=gain_types, width=4, state="readonly").pack(side=tk.LEFT)

        # 現在のゲインタイプ表示
        ttk.Label(self.hrf2_gainschedule_frame, text="type:").pack(side=tk.LEFT, padx=(5, 0))
        self.hrf2_gs_type_label = ttk.Label(self.hrf2_gainschedule_frame, text="---", width=4)
        self.hrf2_gs_type_label.pack(side=tk.LEFT)

        # 初期表示の切り替え
        self._update_hrf2_param_frames()

        # Row 4: HRF2ステータス表示
        self.hrf2_status_label = ttk.Label(hrf2_frame, text="HRF2: 無効", foreground="gray", style='Status.TLabel')
        self.hrf2_status_label.grid(row=4, column=0, columnspan=8, sticky=tk.W, pady=2, padx=5)

        row_idx += 1

        # --- AI Assistant Settings Frame ---
        prompt_frame = ttk.LabelFrame(main_frame, text="AIアシスタント設定 (OpenAI)", padding="10")
        prompt_frame.grid(row=row_idx, column=0, columnspan=5, sticky="nsew", padx=5, pady=5)
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(1, weight=1)

        ttk.Label(prompt_frame, text="システムプロンプト:").grid(row=0, column=0, sticky=tk.W, pady=(0,2), padx=5)
        self.system_prompt = scrolledtext.ScrolledText(
            prompt_frame, wrap=tk.WORD, height=5, relief=tk.SUNKEN, borderwidth=1,
            font=self.default_font
        )
        self.system_prompt.grid(row=1, column=0, sticky="nsew", pady=(0,5), padx=5)
        self.system_prompt.configure(
            height=8,
            background="white",
            foreground="black",
            insertbackground="black",
            highlightthickness=1,
            highlightbackground="#B0B0B0"
        )

        prompt_button_frame = ttk.Frame(prompt_frame)
        prompt_button_frame.grid(row=2, column=0, sticky='ew', padx=5)
        ttk.Button(prompt_button_frame, text="プロンプト保存", command=self.save_system_prompt).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(prompt_button_frame, text="会話履歴クリア", command=self.clear_conversation).pack(side=tk.LEFT, padx=5)
        row_idx += 1
        main_frame.rowconfigure(row_idx -1, weight=1)

        # --- Control Buttons Frame ---
        control_button_frame = ttk.Frame(main_frame, padding="10 0")
        control_button_frame.grid(row=row_idx, column=0, columnspan=5, pady=10, sticky="ew")
        buttons_config = [
            ("start_button", "音声対話 開始", self.start_conversation, tk.NORMAL),
            ("stop_button", "音声対話 停止", self.stop_conversation, tk.DISABLED),
            ("toggle_status_window_button", "AI発話表示/非表示", self.toggle_status_window, tk.NORMAL),
            ("save_config_button", "全設定 保存", self.save_config, tk.NORMAL)
        ]
        for i, (btn_attr, btn_text, btn_cmd, btn_state) in enumerate(buttons_config):
            button = ttk.Button(control_button_frame, text=btn_text, command=btn_cmd, state=btn_state)
            button.grid(row=0, column=i, padx=5, sticky='ew')
            setattr(self, btn_attr, button)
            control_button_frame.columnconfigure(i, weight=1)

        # ビデオ録画チェックボックス
        self.video_recording_checkbox = ttk.Checkbutton(
            control_button_frame,
            text="録画",
            variable=self.video_recording_enabled,
            style='TCheckbutton'
        )
        self.video_recording_checkbox.grid(row=0, column=len(buttons_config), padx=10, sticky='w')
        row_idx += 1

        # --- Status Bar ---
        status_bar_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, padding=0)
        status_bar_frame.grid(row=row_idx, column=0, columnspan=5, sticky="ew", padx=5, pady=(5,0))
        status_bar_frame.columnconfigure(0, weight=1)
        status_bar_frame.columnconfigure(1, weight=0)

        self.status_label = ttk.Label(status_bar_frame, text="準備完了", anchor=tk.W, style='Status.TLabel', padding="5 2")
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.elapsed_time_label = ttk.Label(status_bar_frame, text="", anchor=tk.E, style='Status.TLabel', padding="5 2")
        self.elapsed_time_label.grid(row=0, column=1, sticky="e")
        
        # --- Interim Transcription Display ---
        self.interim_text_var = tk.StringVar(value="")
        interim_label = ttk.Label(
            main_frame,
            textvariable=self.interim_text_var,
            style='Status.TLabel',
            anchor="w",
            foreground="gray"
        )
        interim_label.grid(row=row_idx + 1, column=0, columnspan=5, sticky="ew", padx=5, pady=2)

    # リアルタイムモニター関連のメソッドは RealtimeMonitorMixin から継承

    def _setup_ecg_analysis_tab(self) -> None:
        """ECG/HRV解析タブのUIを構築"""
        main_frame = self.ecg_analysis_tab
        main_frame.columnconfigure(0, weight=1)

        # --- 入力フォルダ選択セクション ---
        input_frame = ttk.LabelFrame(main_frame, text="入力設定", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        self.ecg_input_dir_var = tk.StringVar(value="入力フォルダが選択されていません。")

        ttk.Label(input_frame, text="入力フォルダ:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(input_frame, textvariable=self.ecg_input_dir_var, foreground="grey").grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(input_frame, text="参照...", command=self._browse_ecg_input_folder).grid(row=0, column=2, padx=5, pady=5)

        # --- 出力フォルダ選択 ---
        ttk.Label(input_frame, text="出力フォルダ:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.ecg_output_dir_var = tk.StringVar(value="出力フォルダが選択されていません。")
        ttk.Label(input_frame, textvariable=self.ecg_output_dir_var, foreground="grey").grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(input_frame, text="参照...", command=self._browse_ecg_output_folder).grid(row=1, column=2, padx=5, pady=5)

        # --- 解析区間設定 ---
        window_frame = ttk.LabelFrame(main_frame, text="解析区間設定（録画開始からの経過秒）", padding="10")
        window_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        window_frame.columnconfigure(1, weight=1)
        window_frame.columnconfigure(3, weight=1)

        self.ecg_window_start_var = tk.DoubleVar(value=30.0)
        self.ecg_window_end_var = tk.DoubleVar(value=330.0)

        ttk.Label(window_frame, text="開始 (秒):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(
            window_frame, from_=0, to=3600, increment=5,
            textvariable=self.ecg_window_start_var, width=8
        ).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(window_frame, text="終了 (秒):").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        ttk.Spinbox(
            window_frame, from_=10, to=7200, increment=5,
            textvariable=self.ecg_window_end_var, width=8
        ).grid(row=0, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(window_frame, text="※終了時刻は開始より大きい必要があります。").grid(
            row=1, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 5)
        )

        # --- 解析パラメータ設定 ---
        param_frame = ttk.LabelFrame(main_frame, text="解析パラメータ設定", padding="10")
        param_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        for col in range(4):
            param_frame.columnconfigure(col, weight=1)

        self.ecg_sensor_fs_var = tk.DoubleVar(value=130.0)
        self.ecg_resampling_freq_var = tk.DoubleVar(value=1.0)
        self.ecg_quantile_low_var = tk.DoubleVar(value=0.038)
        self.ecg_quantile_high_var = tk.DoubleVar(value=0.962)
        self.ecg_min_hr_var = tk.DoubleVar(value=45.0)
        self.ecg_max_hr_var = tk.DoubleVar(value=210.0)
        self.ecg_window_length_var = tk.DoubleVar(value=30.0)

        ttk.Label(param_frame, text="センサーサンプリング周波数 (Hz):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(
            param_frame, from_=10, to=1000, increment=5,
            textvariable=self.ecg_sensor_fs_var, width=8
        ).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(param_frame, text="リサンプリング周波数 (Hz):").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        ttk.Spinbox(
            param_frame, from_=0.2, to=10.0, increment=0.1,
            textvariable=self.ecg_resampling_freq_var, width=8
        ).grid(row=0, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(param_frame, text="外れ値除去 下位パーセンタイル:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(
            param_frame, from_=0.0, to=0.5, increment=0.001,
            textvariable=self.ecg_quantile_low_var, width=8
        ).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(param_frame, text="外れ値除去 上位パーセンタイル:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        ttk.Spinbox(
            param_frame, from_=0.5, to=1.0, increment=0.001,
            textvariable=self.ecg_quantile_high_var, width=8
        ).grid(row=1, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(param_frame, text="心拍数範囲 (最小BPM):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(
            param_frame, from_=20, to=100, increment=1,
            textvariable=self.ecg_min_hr_var, width=8
        ).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(param_frame, text="心拍数範囲 (最大BPM):").grid(row=2, column=2, sticky="w", padx=5, pady=5)
        ttk.Spinbox(
            param_frame, from_=80, to=250, increment=1,
            textvariable=self.ecg_max_hr_var, width=8
        ).grid(row=2, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(param_frame, text="LF/HF解析ウィンドウ (秒):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(
            param_frame, from_=5, to=120, increment=5,
            textvariable=self.ecg_window_length_var, width=8
        ).grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # --- 条件設定 ---
        condition_frame = ttk.LabelFrame(main_frame, text="条件設定", padding="10")
        condition_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        condition_frame.columnconfigure(0, weight=1)
        ttk.Label(condition_frame, text="解析・箱ひげ図に含める条件を選択してください。").grid(
            row=0, column=0, sticky="w", padx=5, pady=(0, 5)
        )

        self.ecg_condition_vars: Dict[str, tk.BooleanVar] = {}
        for idx, condition in enumerate(ANALYS_CONDITION_ORDER):
            var = tk.BooleanVar(value=True)
            self.ecg_condition_vars[condition] = var

            row_frame = ttk.Frame(condition_frame)
            row_frame.grid(row=1 + idx, column=0, sticky="ew", padx=5, pady=2)
            row_frame.columnconfigure(1, weight=1)

            color = CONDITION_COLORS.get(condition, "#CCCCCC")
            color_indicator = tk.Canvas(row_frame, width=16, height=16, highlightthickness=0)
            color_indicator.create_rectangle(0, 0, 16, 16, fill=color, outline="#666666")
            color_indicator.grid(row=0, column=0, padx=(0, 8))

            ttk.Checkbutton(
                row_frame,
                text=condition,
                variable=var
            ).grid(row=0, column=1, sticky="w")

        ecg_condition_action_frame = ttk.Frame(condition_frame)
        ecg_condition_action_frame.grid(row=1 + len(ANALYS_CONDITION_ORDER), column=0, sticky="w", padx=5, pady=(8, 0))
        ttk.Button(
            ecg_condition_action_frame,
            text="全て選択",
            command=lambda: self._set_all_ecg_conditions(True)
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            ecg_condition_action_frame,
            text="全て解除",
            command=lambda: self._set_all_ecg_conditions(False)
        ).pack(side=tk.LEFT)

        # --- 実行ボタン ---
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        self.ecg_run_button = ttk.Button(button_frame, text="HRV解析 実行", command=self._run_ecg_analysis)
        self.ecg_run_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.ecg_boxplot_button = ttk.Button(button_frame, text="箱ひげ図 生成", command=self._generate_ecg_boxplots)
        self.ecg_boxplot_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.ecg_combine_button = ttk.Button(button_frame, text="被験者統合", command=self._combine_ecg_subjects)
        self.ecg_combine_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # --- ステータス表示 ---
        self.ecg_status_var = tk.StringVar(value="フォルダを選択して解析を開始してください。")
        status_label = ttk.Label(main_frame, textvariable=self.ecg_status_var, foreground="blue", wraplength=800)
        status_label.grid(row=5, column=0, sticky="w", padx=10, pady=5)

        # --- 結果プレビュー用フレーム ---
        preview_frame = ttk.LabelFrame(main_frame, text="解析結果プレビュー", padding="10")
        preview_frame.grid(row=6, column=0, sticky="nsew", padx=5, pady=5)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)

        self.ecg_preview_canvas_frame = ttk.Frame(preview_frame)
        self.ecg_preview_canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.ecg_canvas_items = []

        # ECG解析用の内部変数
        self.ecg_input_dir = None
        self.ecg_output_dir = None

    def _setup_questionnaire_tab(self) -> None:
        """アンケート解析タブのUIを構築"""
        main_frame = self.questionnaire_tab
        main_frame.columnconfigure(0, weight=1)

        # --- ファイル選択セクション ---
        input_frame = ttk.LabelFrame(main_frame, text="アンケートデータ", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="実験後アンケート.xlsxを選択して箱ひげ図を生成してください。").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )

        self.questionnaire_file_var = tk.StringVar(value="")

        ttk.Label(input_frame, text="Excelファイル:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.questionnaire_file_var, width=50).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(input_frame, text="参照...", command=self._browse_questionnaire_file).grid(row=1, column=2, padx=5, pady=5)

        # --- 条件フィルタ ---
        condition_frame = ttk.LabelFrame(main_frame, text="条件フィルタ", padding="10")
        condition_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        condition_frame.columnconfigure(0, weight=1)
        ttk.Label(condition_frame, text="箱ひげ図に含める条件を選択してください。").grid(
            row=0, column=0, sticky="w", padx=5, pady=(0, 5)
        )

        self.questionnaire_condition_vars: Dict[str, tk.BooleanVar] = {}
        for idx, condition in enumerate(Q_CONDITION_ORDER):
            var = tk.BooleanVar(value=True)
            self.questionnaire_condition_vars[condition] = var

            row_frame = ttk.Frame(condition_frame)
            row_frame.grid(row=1 + idx, column=0, sticky="ew", padx=5, pady=2)
            row_frame.columnconfigure(1, weight=1)

            color = Q_CONDITION_COLORS.get(condition, "#CCCCCC")
            color_indicator = tk.Canvas(row_frame, width=16, height=16, highlightthickness=0)
            color_indicator.create_rectangle(0, 0, 16, 16, fill=color, outline="#666666")
            color_indicator.grid(row=0, column=0, padx=(0, 8))

            ttk.Checkbutton(
                row_frame,
                text=condition,
                variable=var
            ).grid(row=0, column=1, sticky="w")

        condition_action_frame = ttk.Frame(condition_frame)
        condition_action_frame.grid(row=1 + len(Q_CONDITION_ORDER), column=0, sticky="w", padx=5, pady=(8, 0))
        ttk.Button(
            condition_action_frame,
            text="全て選択",
            command=lambda: self._set_all_questionnaire_conditions(True)
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            condition_action_frame,
            text="全て解除",
            command=lambda: self._set_all_questionnaire_conditions(False)
        ).pack(side=tk.LEFT)

        # --- 実行ボタン ---
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=10)

        self.questionnaire_run_button = ttk.Button(button_frame, text="箱ひげ図を作成", command=self._run_questionnaire_analysis)
        self.questionnaire_run_button.pack(side=tk.LEFT, padx=5)

        self.panas_run_button = ttk.Button(button_frame, text="PANAS解析", command=self._run_panas_analysis)
        self.panas_run_button.pack(side=tk.LEFT, padx=5)

        # --- ステータス表示 ---
        self.questionnaire_status_var = tk.StringVar(value="ファイルを選択してください。")
        status_label = ttk.Label(main_frame, textvariable=self.questionnaire_status_var, foreground="blue", wraplength=800)
        status_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)

        # --- 結果プレビュー用フレーム ---
        preview_frame = ttk.LabelFrame(main_frame, text="プレビュー", padding="10")
        preview_frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        self.questionnaire_preview_frame = ttk.Frame(preview_frame)
        self.questionnaire_preview_frame.grid(row=0, column=0, sticky="nsew")
        self.questionnaire_canvas_items = []

    # =========================================================================
    # ECG解析タブのコールバック
    # =========================================================================

    def _browse_ecg_input_folder(self):
        folder_path = filedialog.askdirectory(title="解析対象のフォルダを選択してください")
        if folder_path:
            self.ecg_input_dir = folder_path
            self.ecg_input_dir_var.set(folder_path)
            self.ecg_status_var.set("HRV解析 実行 を押して解析を開始してください。")

    def _browse_ecg_output_folder(self):
        folder_path = filedialog.askdirectory(title="出力先フォルダを選択してください")
        if folder_path:
            self.ecg_output_dir = folder_path
            self.ecg_output_dir_var.set(folder_path)

    def _collect_subject_files(self, input_dir):
        """入力ディレクトリからファイルを収集"""
        subject_files = defaultdict(dict)
        for entry in os.listdir(input_dir):
            full_path = os.path.join(input_dir, entry)
            if not os.path.isfile(full_path) or not entry.lower().endswith('.csv'):
                continue
            match = FILENAME_PATTERN.match(entry)
            if not match:
                continue
            subject_id = f"No{match.group('subject')}"
            condition_raw = match.group('condition').lower()
            condition = ANALYS_CONDITION_MAP.get(condition_raw)
            if not condition:
                continue
            subject_files[subject_id][condition] = full_path
        return subject_files

    def _get_selected_ecg_conditions(self) -> List[str]:
        if not hasattr(self, "ecg_condition_vars"):
            return list(ANALYS_CONDITION_ORDER)
        selected = [cond for cond, var in self.ecg_condition_vars.items() if var.get()]
        return selected

    def _set_all_ecg_conditions(self, value: bool):
        if not hasattr(self, "ecg_condition_vars"):
            return
        for var in self.ecg_condition_vars.values():
            var.set(value)

    def _get_selected_questionnaire_conditions(self) -> List[str]:
        if not hasattr(self, "questionnaire_condition_vars"):
            return list(Q_CONDITION_ORDER)
        return [cond for cond, var in self.questionnaire_condition_vars.items() if var.get()]

    def _set_all_questionnaire_conditions(self, value: bool):
        if not hasattr(self, "questionnaire_condition_vars"):
            return
        for var in self.questionnaire_condition_vars.values():
            var.set(value)

    def _run_ecg_analysis(self):
        """ECG/HRV解析を実行"""
        if not self.ecg_input_dir or not os.path.isdir(self.ecg_input_dir):
            messagebox.showwarning("フォルダ未選択", "まず解析対象のフォルダを選択してください。")
            return

        try:
            start_offset = float(self.ecg_window_start_var.get())
            end_offset = float(self.ecg_window_end_var.get())
            sensor_fs = float(self.ecg_sensor_fs_var.get())
            resampling_freq = float(self.ecg_resampling_freq_var.get())
            quantile_low = float(self.ecg_quantile_low_var.get())
            quantile_high = float(self.ecg_quantile_high_var.get())
            min_hr = float(self.ecg_min_hr_var.get())
            max_hr = float(self.ecg_max_hr_var.get())
            window_length = float(self.ecg_window_length_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("入力エラー", "解析パラメータに無効な値があります。数値を入力してください。")
            return

        if start_offset < 0:
            start_offset = 0.0
        if end_offset <= start_offset:
            messagebox.showerror("入力エラー", "終了時刻は開始時刻より大きくしてください。")
            return
        if sensor_fs <= 0:
            messagebox.showerror("入力エラー", "センサーサンプリング周波数は0より大きい必要があります。")
            return
        if resampling_freq <= 0:
            messagebox.showerror("入力エラー", "リサンプリング周波数は0より大きい必要があります。")
            return
        if not (0 <= quantile_low < quantile_high <= 1):
            messagebox.showerror("入力エラー", "外れ値除去のパーセンタイルは 0〜1 の範囲で下限 < 上限 となるよう設定してください。")
            return
        if min_hr <= 0 or max_hr <= 0 or min_hr >= max_hr:
            messagebox.showerror("入力エラー", "心拍数範囲は正の値で、最小 < 最大 となるよう設定してください。")
            return
        if window_length <= 0:
            messagebox.showerror("入力エラー", "解析ウィンドウは正の値にしてください。")
            return

        self.ecg_run_button.config(state=tk.DISABLED)
        self.ecg_status_var.set("解析を実行中です。完了するまでお待ちください。")
        self.update_idletasks()

        def run_in_thread():
            try:
                subject_files = self._collect_subject_files(self.ecg_input_dir)
                if not subject_files:
                    self.after(0, lambda: messagebox.showwarning("解析不可", "指定フォルダに解析可能なファイルが見つかりません。"))
                    self.after(0, lambda: self.ecg_status_var.set("解析可能なファイルが見つかりませんでした。"))
                    self.after(0, lambda: self.ecg_run_button.config(state=tk.NORMAL))
                    return

                output_dir = self.ecg_output_dir if self.ecg_output_dir else os.path.join(os.path.dirname(self.ecg_input_dir), "result_batch")
                os.makedirs(output_dir, exist_ok=True)

                processed = 0
                skipped = {}

                selected_conditions = self._get_selected_ecg_conditions()
                if not selected_conditions:
                    selected_conditions = list(ANALYS_CONDITION_ORDER)

                for subject_id, files_by_condition in subject_files.items():
                    # 選択した条件のうち、存在するものだけを抽出
                    available_conditions = [cond for cond in selected_conditions if cond in files_by_condition]
                    missing = [cond for cond in selected_conditions if cond not in files_by_condition]

                    if not available_conditions:
                        # 選択した条件が1つも存在しない場合はスキップ
                        print(f"{subject_id}: 選択した条件のファイルが1つも存在しないためスキップします。")
                        skipped[subject_id] = missing
                        continue

                    if missing:
                        print(f"{subject_id}: {', '.join(missing)} は存在しないため、{', '.join(available_conditions)} のみ解析します。")

                    try:
                        subject_dir = os.path.join(output_dir, subject_id)
                        ordered_files = {condition: files_by_condition[condition] for condition in available_conditions}
                        print(f"\n=== {subject_id} の解析を開始します ===")
                        analys_run_batch_analysis(
                            ordered_files,
                            subject_dir,
                            analysis_start_offset=start_offset,
                            analysis_end_offset=end_offset,
                            sensor_sample_rate=sensor_fs,
                            resampling_freq=resampling_freq,
                            quantile_low=quantile_low,
                            quantile_high=quantile_high,
                            min_hr=min_hr,
                            max_hr=max_hr,
                            analysis_window_seconds=window_length,
                            subject_id=subject_id
                        )

                        combined_file = os.path.join(subject_dir, f"{subject_id}_Combined_HRV_Analysis.xlsx")
                        if os.path.exists(combined_file):
                            print(f"{subject_id}: 箱ひげ図を作成します")
                            analys_generate_box_plots(combined_file, subject_dir)

                        processed += 1
                    except Exception as exc:
                        skipped[subject_id] = [str(exc)]
                        print(f"{subject_id}: 解析中にエラーが発生しました -> {exc}")

                message_lines = [f"{processed}名の被験者を処理しました。"]
                if skipped:
                    detail_lines = []
                    for subject_id, reasons in skipped.items():
                        detail_lines.append(f"  - {subject_id}: {', '.join(reasons)}")
                    message_lines.append("以下の被験者はスキップされました:")
                    message_lines.extend(detail_lines)
                message_lines.append(f"出力フォルダ: {output_dir}")

                self.after(0, lambda: messagebox.showinfo("解析完了", "\n".join(message_lines)))
                self.after(0, lambda: self.ecg_status_var.set("解析が完了しました。"))

            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("解析エラー", f"解析中にエラーが発生しました。\n{exc}"))
                self.after(0, lambda: self.ecg_status_var.set("解析に失敗しました。"))

            finally:
                self.after(0, lambda: self.ecg_run_button.config(state=tk.NORMAL))

        threading.Thread(target=run_in_thread, daemon=True).start()

    def _generate_ecg_boxplots(self):
        """箱ひげ図を生成"""
        selected_conditions = self._get_selected_ecg_conditions()
        if not selected_conditions:
            selected_conditions = list(ANALYS_CONDITION_ORDER)

        output_dir = self.ecg_output_dir if self.ecg_output_dir else os.path.join(os.path.dirname(self.ecg_input_dir or "."), "result_batch")
        combined_file = os.path.join(output_dir, "Combined_HRV_Analysis.xlsx")

        if not os.path.exists(combined_file):
            subject_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))] if os.path.isdir(output_dir) else []
            if not subject_dirs:
                messagebox.showwarning("ファイル未発見", "Combined_HRV_Analysis.xlsx が見つかりません。先に解析を実行してください。")
                return

            for subdir in subject_dirs:
                combined_file = os.path.join(output_dir, subdir, "Combined_HRV_Analysis.xlsx")
                if os.path.exists(combined_file):
                    break
            else:
                messagebox.showwarning("ファイル未発見", "Combined_HRV_Analysis.xlsx が見つかりません。先に解析を実行してください。")
                return

        try:
            saved_files = analys_generate_box_plots(
                combined_file,
                os.path.dirname(combined_file),
                condition_order=selected_conditions
            )
            messagebox.showinfo("完了", f"箱ひげ図を作成しました:\n" + "\n".join(saved_files))
            self.ecg_status_var.set("箱ひげ図を生成しました。")
        except Exception as exc:
            messagebox.showerror("エラー", f"箱ひげ図の作成に失敗しました。\n{exc}")

    def _combine_ecg_subjects(self):
        """全被験者のデータを統合"""
        output_dir = self.ecg_output_dir if self.ecg_output_dir else os.path.join(os.path.dirname(self.ecg_input_dir or "."), "result_batch")

        if not os.path.isdir(output_dir):
            messagebox.showwarning("フォルダ未発見", "result_batch フォルダが見つかりません。先に解析を実行してください。")
            return

        self.ecg_combine_button.config(state=tk.DISABLED)
        self.ecg_status_var.set("全被験者の統合を実行中です...")
        self.update_idletasks()

        def run_in_thread():
            try:
                subject_files = []
                for entry in sorted(os.listdir(output_dir)):
                    subject_dir = os.path.join(output_dir, entry)
                    combined_path = os.path.join(subject_dir, "Combined_HRV_Analysis.xlsx")
                    if os.path.isfile(combined_path):
                        subject_files.append((entry, combined_path))

                if not subject_files:
                    self.after(0, lambda: messagebox.showwarning("統合不可", "統合対象の Combined_HRV_Analysis.xlsx が見つかりません。"))
                    return

                condition_priority = {cond: idx for idx, cond in enumerate(ANALYS_CONDITION_ORDER)}
                frames = []
                included_subjects = set()

                for subject_id, file_path in subject_files:
                    df = pd.read_excel(file_path)
                    subject_included = False
                    for condition in ANALYS_CONDITION_ORDER:
                        lf_col = f"{condition}_LF/HF"
                        rmssd_col = f"{condition}_RMSSD"
                        if lf_col not in df.columns or rmssd_col not in df.columns:
                            continue
                        subset = df[['Time', lf_col, rmssd_col]].copy()
                        subset.rename(columns={lf_col: 'LF/HF', rmssd_col: 'RMSSD'}, inplace=True)
                        subset['Subject'] = subject_id
                        subset['Condition'] = condition
                        subset['ConditionOrder'] = condition_priority[condition]
                        frames.append(subset[['Subject', 'Condition', 'ConditionOrder', 'Time', 'LF/HF', 'RMSSD']])
                        subject_included = True
                    if subject_included:
                        included_subjects.add(subject_id)

                if not frames:
                    self.after(0, lambda: messagebox.showerror("統合エラー", "統合に使用できるデータ列が見つかりませんでした。"))
                    return

                combined_df = pd.concat(frames, ignore_index=True)
                combined_df.sort_values(['ConditionOrder', 'Subject', 'Time'], inplace=True)
                combined_df.drop(columns=['ConditionOrder'], inplace=True)

                output_path = os.path.join(output_dir, "Combined_AllSubjects.xlsx")
                combined_df.to_excel(output_path, index=False)

                self.after(0, lambda: messagebox.showinfo("統合完了", f"全被験者データを統合しました。\n対象: {', '.join(sorted(included_subjects))}\n保存先: {output_path}"))
                self.after(0, lambda: self.ecg_status_var.set("全被験者統合が完了しました。"))

            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("統合エラー", f"統合処理中にエラーが発生しました。\n{exc}"))
                self.after(0, lambda: self.ecg_status_var.set("統合が失敗しました。"))

            finally:
                self.after(0, lambda: self.ecg_combine_button.config(state=tk.NORMAL))

        threading.Thread(target=run_in_thread, daemon=True).start()

    # =========================================================================
    # アンケート解析タブのコールバック
    # =========================================================================

    def _browse_questionnaire_file(self):
        file_path = filedialog.askopenfilename(
            title="Excelファイルを選択",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
        )
        if file_path:
            self.questionnaire_file_var.set(file_path)
            self.questionnaire_status_var.set(f"{os.path.basename(file_path)} を選択しました。")

    def _clear_questionnaire_canvas(self):
        for item in self.questionnaire_canvas_items:
            try:
                if hasattr(item, 'get_tk_widget'):
                    item.get_tk_widget().destroy()
                elif hasattr(item, 'destroy'):
                    item.destroy()
            except Exception:
                pass
        self.questionnaire_canvas_items.clear()
        # PhotoImageの参照を保持するリストもクリア
        if not hasattr(self, '_questionnaire_photo_refs'):
            self._questionnaire_photo_refs = []
        self._questionnaire_photo_refs.clear()

    def _display_questionnaire_image(self, image_path: str):
        """保存済みのPNG画像をGUIに表示する"""
        self._clear_questionnaire_canvas()
        try:
            from PIL import Image, ImageTk
            # 画像を読み込んでリサイズ
            img = Image.open(image_path)
            # プレビューフレームの幅に合わせてリサイズ
            max_width = 800
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            # PhotoImageの参照を保持（ガベージコレクション対策）
            if not hasattr(self, '_questionnaire_photo_refs'):
                self._questionnaire_photo_refs = []
            self._questionnaire_photo_refs.append(photo)

            label = tk.Label(self.questionnaire_preview_frame, image=photo)
            label.grid(row=0, column=0, sticky="nsew", pady=10)
            self.questionnaire_canvas_items.append(label)
            self.questionnaire_preview_frame.rowconfigure(0, weight=1)
            self.questionnaire_preview_frame.columnconfigure(0, weight=1)
        except ImportError:
            # PILがない場合はメッセージを表示
            label = tk.Label(
                self.questionnaire_preview_frame,
                text=f"画像プレビューにはPillowが必要です。\n保存先: {image_path}",
                wraplength=600
            )
            label.grid(row=0, column=0, sticky="nsew", pady=10)
            self.questionnaire_canvas_items.append(label)
        except Exception as e:
            label = tk.Label(
                self.questionnaire_preview_frame,
                text=f"画像の表示に失敗しました: {e}\n保存先: {image_path}",
                wraplength=600
            )
            label.grid(row=0, column=0, sticky="nsew", pady=10)
            self.questionnaire_canvas_items.append(label)

    def _run_questionnaire_analysis(self):
        """アンケート解析を実行"""
        file_path = self.questionnaire_file_var.get()
        if not file_path:
            messagebox.showwarning("ファイル未選択", "Excelファイルを選択してください。")
            return
        selected_conditions = self._get_selected_questionnaire_conditions()
        if not selected_conditions:
            messagebox.showwarning("条件未選択", "少なくとも1つの条件を選択してください。")
            return

        self.questionnaire_run_button.config(state=tk.DISABLED)
        self.questionnaire_status_var.set("箱ひげ図を作成中...")
        self.update_idletasks()

        def run_in_thread(conditions: List[str]):
            try:
                result = analys_q_generate_plots(file_path, condition_order=conditions)
                summary_path = str(result['summary_path'])
                # メインスレッドでPNG画像を表示
                self.after(0, lambda: self._display_questionnaire_image(summary_path))
                self.after(0, lambda: self.questionnaire_status_var.set(
                    f"サマリー: {result['summary_path'].name} / "
                    f"設問別: {len(result['per_question_paths'])}枚を question_boxplots フォルダに保存 "
                    f"(条件: {', '.join(conditions)})"
                ))
                self.after(0, lambda: messagebox.showinfo(
                    "完了",
                    "箱ひげ図を作成しました。\n"
                    "PNGファイルをExcelと同じフォルダに保存しました。",
                ))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: messagebox.showerror("エラー", f"箱ひげ図の作成に失敗しました。\n{exc}"))
                self.after(0, lambda: self.questionnaire_status_var.set("箱ひげ図の作成に失敗しました。"))
            finally:
                self.after(0, lambda: self.questionnaire_run_button.config(state=tk.NORMAL))

        threading.Thread(target=run_in_thread, args=(selected_conditions,), daemon=True).start()

    def _run_panas_analysis(self):
        """PANAS解析を実行"""
        file_path = self.questionnaire_file_var.get()
        if not file_path:
            messagebox.showwarning("ファイル未選択", "Excelファイルを選択してください。")
            return
        selected_conditions = self._get_selected_questionnaire_conditions()
        if not selected_conditions:
            messagebox.showwarning("条件未選択", "少なくとも1つの条件を選択してください。")
            return

        self.panas_run_button.config(state=tk.DISABLED)
        self.questionnaire_status_var.set("PANAS解析を実行中...")
        self.update_idletasks()

        def run_in_thread(conditions: List[str]):
            try:
                result = generate_panas_plots(file_path, condition_order=conditions)
                summary_path = str(result['summary_path'])

                # 信頼性係数のテキストを生成
                reliability_text = format_reliability_text(result['analysis']['reliability'])

                # メインスレッドでPNG画像を表示
                self.after(0, lambda: self._display_questionnaire_image(summary_path))
                self.after(0, lambda: self.questionnaire_status_var.set(
                    f"PANAS解析完了: {result['excel_path'].name} / "
                    f"グラフ: {len(result['figure_paths'])}枚を PANAS_analysis フォルダに保存"
                ))

                # 結果ダイアログ
                def show_result():
                    # 条件別統計のサマリー
                    stats_lines = ["=== 条件別PANAS得点 ===\n"]
                    for cond, stats in result['analysis']['results_by_condition'].items():
                        stats_lines.append(f"【{cond}】 n={stats['n']}")
                        stats_lines.append(f"  PA: {stats['PA_mean']:.1f} ± {stats['PA_std']:.1f}")
                        stats_lines.append(f"  NA: {stats['NA_mean']:.1f} ± {stats['NA_std']:.1f}")
                        stats_lines.append("")

                    full_text = "\n".join(stats_lines) + "\n" + reliability_text
                    messagebox.showinfo(
                        "PANAS解析完了",
                        f"解析結果を保存しました。\n\n{full_text}"
                    )

                self.after(0, show_result)

            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: messagebox.showerror("エラー", f"PANAS解析に失敗しました。\n{exc}"))
                self.after(0, lambda: self.questionnaire_status_var.set("PANAS解析に失敗しました。"))
            finally:
                self.after(0, lambda: self.panas_run_button.config(state=tk.NORMAL))

        threading.Thread(target=run_in_thread, args=(selected_conditions,), daemon=True).start()

    def _on_subject_id_change(self, *_) -> None:
        self._update_subject_id_hint()

    def _get_sanitized_subject_id(self) -> str:
        raw_value = self.subject_id_var.get().strip()
        return "".join(ch for ch in raw_value if ch.isalnum() or ch in ("-", "_"))

    def _update_subject_id_hint(self) -> None:
        sanitized = self._get_sanitized_subject_id()
        if hasattr(self, 'subject_hint_label'):
            if sanitized:
                subject_component = format_subject_id_for_filename(sanitized)
                self.subject_hint_label.config(
                    text=f"ファイル名に '{subject_component}' を使用", foreground="green"
                )
            else:
                self.subject_hint_label.config(text="番号未設定", foreground="red")
        subject_value = sanitized if sanitized else None
        self.conversation_manager.set_subject_id(subject_value)
        self.hr_monitor.set_subject_id(subject_value)

    def _on_llm_selection_change(self) -> None:
        """LLM選択が変更された時の処理"""
        use_local = self.use_local_llm_var.get()
        config.USE_LOCAL_LLM = use_local
        self._reinitialize_llm_client()
        self._update_llm_status()

    def _on_api_key_change(self, *_) -> None:
        """APIキーが変更された時の処理"""
        # OpenAI API選択時のみクライアントを再初期化
        if not self.use_local_llm_var.get():
            self._reinitialize_llm_client()
        self._update_llm_status()

    def _toggle_api_key_visibility(self) -> None:
        """APIキーの表示/非表示を切り替える"""
        if self.show_api_key_var.get():
            self.api_key_entry.config(show="*")
            self.toggle_key_btn.config(text="表示")
            self.show_api_key_var.set(False)
        else:
            self.api_key_entry.config(show="")
            self.toggle_key_btn.config(text="隠す")
            self.show_api_key_var.set(True)

    def _reinitialize_llm_client(self) -> None:
        """LLMクライアントを再初期化する"""
        use_local = self.use_local_llm_var.get()

        if use_local:
            self.openai_client = openai.OpenAI(
                base_url=config.LOCAL_LLM_BASE_URL,
                api_key=config.LOCAL_LLM_API_KEY
            )
            print(f"ローカルLLMに切り替え: {config.LOCAL_LLM_BASE_URL}")
            self._log_to_console(f"LLM: ローカルLLM ({config.LOCAL_LLM_BASE_URL})")
        else:
            api_key = self.openai_api_key_var.get().strip()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                self.openai_client = openai.OpenAI(api_key=api_key)
                print("OpenAI APIに切り替え (カスタムキー使用)")
                self._log_to_console("LLM: OpenAI API")
            else:
                env_key = os.getenv("OPENAI_API_KEY")
                if env_key:
                    self.openai_client = openai.OpenAI()
                    print("OpenAI APIに切り替え (環境変数キー使用)")
                    self._log_to_console("LLM: OpenAI API (環境変数)")
                else:
                    self.openai_client = openai.OpenAI(api_key="dummy")
                    print("警告: OpenAI APIキーが設定されていません")
                    self._log_to_console("警告: OpenAI APIキーが未設定です")

    def _update_llm_status(self) -> None:
        """LLMステータス表示を更新"""
        if not hasattr(self, 'llm_status_label'):
            return

        use_local = self.use_local_llm_var.get()

        if use_local:
            self.llm_status_label.config(
                text=f"ローカルLLM ({config.LOCAL_LLM_MODEL})",
                foreground="green"
            )
        else:
            api_key = self.openai_api_key_var.get().strip() or os.getenv("OPENAI_API_KEY", "")
            if api_key:
                masked_key = api_key[:8] + "..." if len(api_key) > 8 else "***"
                self.llm_status_label.config(
                    text=f"OpenAI API ({config.OPENAI_MODEL}) キー: {masked_key}",
                    foreground="green"
                )
            else:
                self.llm_status_label.config(
                    text="OpenAI API (APIキー未設定)",
                    foreground="red"
                )

    def _check_interim_transcription_queue(self):
        """Periodically check the interim transcription queue and update the UI."""
        try:
            while not self.audio.interim_transcription_queue.empty():
                interim_text = self.audio.interim_transcription_queue.get_nowait()
                self.interim_text_var.set(interim_text)
        except queue.Empty:
            pass
        finally:
            if not self._closing:
                self.after(200, self._check_interim_transcription_queue)

    def create_parameter_row(self, parent: ttk.Frame, param_name: str, label_text: str,
                             tk_var_attr: str, value_label_attr: str, row_index: int):
        min_val, max_val = self.prosody.get_parameter_range(param_name)
        
        ttk.Label(parent, text=label_text).grid(row=row_index, column=0, sticky=tk.W, padx=5, pady=2)
        tk_var = tk.DoubleVar(value=self.prosody.get_parameter(param_name))
        setattr(self.prosody, tk_var_attr, tk_var)

        slider = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                           variable=tk_var, length=250,
                           command=lambda v, p=param_name: self.update_parameter_from_scale(p, float(v)))
        slider.grid(row=row_index, column=2, sticky="ew", padx=5, pady=2)

        value_label = ttk.Label(parent, text=f"{tk_var.get():.2f}", width=5, anchor=tk.W)
        value_label.grid(row=row_index, column=4, sticky=tk.W, padx=5, pady=2)
        setattr(self, value_label_attr, value_label)

        value_entry = ttk.Entry(parent, textvariable=tk_var, width=6)
        value_entry.grid(row=row_index, column=1, sticky=tk.E, padx=(0,5))
        value_entry.bind("<Return>", lambda e, p=param_name, t_var=tk_var: self.update_parameter_from_entry(p, t_var))
        value_entry.bind("<FocusOut>", lambda e, p=param_name, t_var=tk_var: self.update_parameter_from_entry(p, t_var))

    def update_parameter_from_entry(self, param_name: str, tk_var: tk.DoubleVar):
        try:
            value = tk_var.get()
            self.update_parameter_from_scale(param_name, value)
        except (tk.TclError, ValueError):
            tk_var.set(self.prosody.get_parameter(param_name))

    def populate_speaker_list(self):
        def _populate():
            if VoicevoxManager.check_server():
                try:
                    speakers_list = VoicevoxManager.get_speakers()
                    if speakers_list:
                        self.speaker.speakers = speakers_list
                        speaker_names = self.speaker.get_all_speaker_names()
                        self.speaker_combo['values'] = speaker_names
                        current_id = self.speaker.current_style_id
                        current_name = self.speaker.get_speaker_name_by_id(current_id)
                        if current_name and current_name in speaker_names:
                            self.speaker_var.set(current_name)
                        elif speaker_names:
                            self.speaker_var.set(speaker_names[0])
                            self.speaker.current_style_id = self.speaker.get_all_speaker_ids()[0]
                        self.speaker_status.config(
                            text=f"選択中: {self.speaker_var.get()}", foreground="green"
                        )
                    else:
                        self.speaker_combo['values'] = ["話者取得失敗"]
                        self.speaker_var.set("話者取得失敗")
                        self.speaker_status.config(text="話者なし", foreground="red")
                except Exception as e_populate:
                    self.set_status(f"Speaker list population error: {e_populate}", "red")
                    self.speaker_combo['values'] = ["話者取得エラー"]
                    self.speaker_var.set("話者取得エラー")
                    self.speaker_status.config(text="エラー", foreground="red")
            else:
                self.set_status("VOICEVOX server not connected", "red")
                self.speaker_combo['values'] = ["VOICEVOX未接続"]
                self.speaker_var.set("VOICEVOX未接続")
                self.speaker_status.config(text="未接続", foreground="red")
        self.after(100, _populate)

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
                for param in ["intonation", "pitch", "speed", "energy", "pause_duration"]:
                    value = prosody_config.get(f"{param}_scale", getattr(self.prosody, f"{param}_scale"))
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

                self.prosody.enable_hfb(False)
                if self.prosody.hfb_enabled_var: self.prosody.hfb_enabled_var.set(False)
                
                self.prosody.enable_sinusoidal_hfb(False)
                if self.prosody.sinusoidal_hfb_enabled_var: self.prosody.sinusoidal_hfb_enabled_var.set(False)
                print(f"  HFB states initialized to: OFF (startup default)")

                prompt = app_config.get('system_prompt', "You are a kind and helpful AI assistant.")
                self.system_prompt.delete('1.0', tk.END)
                self.system_prompt.insert('1.0', prompt)
                self.conversation_manager.update_system_prompt(prompt)
                print(f"  System prompt loaded.")

                speaker_id_loaded = app_config.get('speaker_id')
                if speaker_id_loaded is not None:
                    self.speaker.current_style_id = speaker_id_loaded
                    print(f"  Speaker ID loaded: {speaker_id_loaded}")

                baseline_dur = app_config.get('baseline_measurement_duration', config.DEFAULT_BASELINE_MEASUREMENT_DURATION)
                self.baseline_duration_var.set(baseline_dur)
                print(f"  Baseline Measurement Duration loaded: {baseline_dur}s")

                # HRF2 設定の読み込み
                hrf2_config = app_config.get('hrf2', {})
                if hrf2_config:
                    controller = self.prosody.hrf2_controller
                    # PIDゲイン
                    kp = hrf2_config.get('kp', controller.config.kp)
                    ki = hrf2_config.get('ki', controller.config.ki)
                    kd = hrf2_config.get('kd', controller.config.kd)
                    controller.set_pid_gains(kp, ki, kd)
                    # GUI変数も更新
                    if hasattr(self, 'hrf2_kp_var'):
                        self.hrf2_kp_var.set(kp)
                    if hasattr(self, 'hrf2_ki_var'):
                        self.hrf2_ki_var.set(ki)
                    if hasattr(self, 'hrf2_kd_var'):
                        self.hrf2_kd_var.set(kd)
                    print(f"  HRF2 PID gains loaded: Kp={kp}, Ki={ki}, Kd={kd}")

                    # 目標心拍数
                    target_hr = hrf2_config.get('target_hr', controller.target_hr)
                    controller.target_hr = target_hr
                    if hasattr(self, 'hrf2_target_hr_var'):
                        self.hrf2_target_hr_var.set(target_hr)
                    print(f"  HRF2 Target HR loaded: {target_hr}")

                    # 制御モード
                    control_mode_str = hrf2_config.get('control_mode', 'PID')
                    try:
                        controller.control_mode = ControlMode(control_mode_str)
                        print(f"  HRF2 Control mode loaded: {control_mode_str}")
                    except ValueError:
                        print(f"  Warning: Unknown control mode '{control_mode_str}', using PID")

                    # ゲインスケジューリング設定
                    gs_config = hrf2_config.get('gain_schedule', {})
                    if gs_config:
                        gs = controller.gain_schedule_config
                        gs.error_threshold_high = gs_config.get('error_threshold_high', gs.error_threshold_high)
                        gs.error_threshold_medium = gs_config.get('error_threshold_medium', gs.error_threshold_medium)
                        gs.kp_high = gs_config.get('kp_high', gs.kp_high)
                        gs.ki_high = gs_config.get('ki_high', gs.ki_high)
                        gs.kd_high = gs_config.get('kd_high', gs.kd_high)
                        gs.kp_medium = gs_config.get('kp_medium', gs.kp_medium)
                        gs.ki_medium = gs_config.get('ki_medium', gs.ki_medium)
                        gs.kd_medium = gs_config.get('kd_medium', gs.kd_medium)
                        gs.kp_low = gs_config.get('kp_low', gs.kp_low)
                        gs.ki_low = gs_config.get('ki_low', gs.ki_low)
                        gs.kd_low = gs_config.get('kd_low', gs.kd_low)
                        # GUI変数も更新
                        if hasattr(self, 'hrf2_gs_high_var'):
                            self.hrf2_gs_high_var.set(gs.error_threshold_high)
                        if hasattr(self, 'hrf2_gs_med_var'):
                            self.hrf2_gs_med_var.set(gs.error_threshold_medium)
                        print(f"  HRF2 Gain schedule config loaded")

                self.set_status("Configuration loaded", "blue")
            else:
                print("Configuration file not found. Using default settings.")
                default_prompt = "You are a kind and helpful AI assistant. Please speak in Japanese."
                self.system_prompt.delete('1.0', tk.END)
                self.system_prompt.insert('1.0', default_prompt)
                self.conversation_manager.update_system_prompt(default_prompt)
                self.prosody.enable_hfb(False)
                if self.prosody.hfb_enabled_var: self.prosody.hfb_enabled_var.set(False)
                self.prosody.enable_sinusoidal_hfb(False)
                if self.prosody.sinusoidal_hfb_enabled_var: self.prosody.sinusoidal_hfb_enabled_var.set(False)

                for param in ["intonation", "pitch", "speed", "energy", "pause_duration"]:
                     self.update_parameter_display(param)
                self.set_status("Using default settings (config file not found)", "blue")

            self.populate_speaker_list()

        except (json.JSONDecodeError, Exception) as e_load_cfg:
            messagebox.showerror("Config Error", f"Error loading configuration file:\n{e_load_cfg}\n\nDefaults will be used.")
            print(f"Configuration file loading error: {e_load_cfg}")
            self.set_status(f"Config file error: {e_load_cfg}", "red")
            self.prosody.enable_hfb(False)
            if self.prosody.hfb_enabled_var: self.prosody.hfb_enabled_var.set(False)
            self.prosody.enable_sinusoidal_hfb(False)
            if self.prosody.sinusoidal_hfb_enabled_var: self.prosody.sinusoidal_hfb_enabled_var.set(False)
            self.populate_speaker_list()

    def save_config(self) -> None:
        if self.is_processing or self.is_conversing or self.is_measuring_baseline:
            messagebox.showwarning("Busy", "Cannot save configuration while processing, in conversation, or measuring baseline.")
            return

        print(f"Saving configuration to '{config.CONFIG_FILE}'...")
        try:
            controller = self.prosody.hrf2_controller
            gs = controller.gain_schedule_config

            app_config = {
                'reference_hr': self.hr_monitor.get_reference_hr(),
                'prosody': {
                    p: self.prosody.get_parameter(p) for p in
                    ["intonation", "pitch", "speed", "energy", "pause_duration"]
                },
                'hfb_target_param': getattr(self.prosody, "hfb_target_param", "intonation"),
                'system_prompt': self.system_prompt.get('1.0', tk.END).strip(),
                'speaker_id': self.speaker.current_style_id,
                'baseline_measurement_duration': self.baseline_duration_var.get(),
                'hrf2': {
                    'kp': controller.config.kp,
                    'ki': controller.config.ki,
                    'kd': controller.config.kd,
                    'target_hr': controller.target_hr,
                    'control_mode': controller.control_mode.value,
                    'gain_schedule': {
                        'error_threshold_high': gs.error_threshold_high,
                        'error_threshold_medium': gs.error_threshold_medium,
                        'kp_high': gs.kp_high,
                        'ki_high': gs.ki_high,
                        'kd_high': gs.kd_high,
                        'kp_medium': gs.kp_medium,
                        'ki_medium': gs.ki_medium,
                        'kd_medium': gs.kd_medium,
                        'kp_low': gs.kp_low,
                        'ki_low': gs.ki_low,
                        'kd_low': gs.kd_low,
                    }
                }
            }
            with open(config.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(app_config, f, indent=4, ensure_ascii=False)
            print("Configuration saved successfully.")
            self.set_status("Configuration saved", "green")
        except Exception as e_save_cfg:
            messagebox.showerror("Save Error", f"Failed to save configuration:\n{e_save_cfg}")
            print(f"Failed to save configuration: {e_save_cfg}")
            self.set_status(f"Failed to save config: {e_save_cfg}", "red")

    def set_status(self, message: str, color: str = "black") -> None:
        def _update():
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                self.status_label.config(text=message, foreground=color)
            action_prompts = ["Please speak...", "話してください"]
            thinking_prompts = ["Recording...", "Converting speech to text...", "Generating response...", "Synthesizing audio...", "思考中..."]

            if self.status_window_visible and self.status_display_window.winfo_exists():
                if message in action_prompts :
                    self._set_status_display_prompt("話してください")
                elif any(p in message for p in thinking_prompts):
                     self._set_status_display_prompt("AI 処理中...")
        if hasattr(self, 'after'):
            self.after(0, _update)
        else:
            print(f"[Status Update (No UI)]: {message}")

    def update_ui_periodic(self) -> None:
        def _update():
            if self._closing: return

            try:
                verity_hr_text = "-- BPM"
                h10_hr_text = "-- BPM"
                verity_status_text, verity_status_color = "Verity Sense: 未接続", "red"
                h10_status_text, h10_status_color = "H10: 未接続", "red"

                if self.hr_monitor.is_connected:
                    hr = self.hr_monitor.get_current_hr()
                    verity_stale = self.hr_monitor._is_verity_data_stale()
                    using_h10_fallback = getattr(self.hr_monitor, 'using_h10_fallback', False)
                    verity_battery = getattr(self.hr_monitor, 'battery_level', None)
                    battery_str = f" [{verity_battery}%]" if verity_battery is not None else ""

                    if verity_stale and hr > 0:
                        if using_h10_fallback:
                            verity_hr_text = f"{hr} BPM (H10代替){battery_str}"
                            verity_status_text, verity_status_color = "Verity Sense: 接続断 → H10代替中", "orange"
                        else:
                            verity_hr_text = f"{hr} BPM (接続断){battery_str}"
                            verity_status_text, verity_status_color = "Verity Sense: 接続断", "orange"
                    elif hr > 0:
                        verity_hr_text = f"{hr} BPM{battery_str}"
                        verity_status_text, verity_status_color = "Verity Sense: 接続中", "green"
                    else:
                        verity_hr_text = f"-- BPM{battery_str}"
                        verity_status_text, verity_status_color = "Verity Sense: 接続中 (データ待ち)", "orange"

                if self.h10_monitor.is_connected:
                    hr_h10 = self.h10_monitor.current_h10_hr
                    h10_hr_stale = self.h10_monitor.is_hr_data_stale()
                    h10_ecg_stale = self.h10_monitor.is_ecg_data_stale()
                    h10_battery = getattr(self.h10_monitor, 'battery_level', None)
                    battery_str = f" [{h10_battery}%]" if h10_battery is not None else ""

                    if h10_hr_stale and h10_ecg_stale:
                        h10_hr_text = f"{hr_h10} BPM (接続断){battery_str}" if hr_h10 > 0 else f"-- BPM (接続断){battery_str}"
                        h10_status_text, h10_status_color = "H10: 接続断", "orange"
                    elif h10_hr_stale:
                        h10_hr_text = f"{hr_h10} BPM (HR断){battery_str}" if hr_h10 > 0 else f"-- BPM{battery_str}"
                        h10_status_text, h10_status_color = "H10: ECGのみ", "orange"
                    elif h10_ecg_stale:
                        h10_hr_text = f"{hr_h10} BPM{battery_str}" if hr_h10 > 0 else f"-- BPM{battery_str}"
                        h10_status_text, h10_status_color = "H10: HRのみ (ECG断)", "orange"
                    elif hr_h10 > 0:
                        h10_hr_text = f"{hr_h10} BPM{battery_str}"
                        h10_status_text, h10_status_color = "H10: 接続中", "green"
                    else:
                        h10_hr_text = f"-- BPM{battery_str}"
                        h10_status_text, h10_status_color = "H10: 接続中 (データ待ち)", "orange"

                if hasattr(self, 'hr_label') and self.hr_label.winfo_exists():
                    self.hr_label.config(text=f"Verity Sense: {verity_hr_text} | H10: {h10_hr_text}")
                if hasattr(self, 'hr_status_label') and self.hr_status_label.winfo_exists():
                    self.hr_status_label.config(text=verity_status_text, foreground=verity_status_color)
                if hasattr(self, 'h10_status_label') and self.h10_status_label.winfo_exists():
                    self.h10_status_label.config(text=h10_status_text, foreground=h10_status_color)

                if self.is_conversing and self.conversation_start_time:
                    elapsed = datetime.datetime.now() - self.conversation_start_time
                    total_seconds = int(elapsed.total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    elapsed_str = f"会話時間: {hours:02}:{minutes:02}:{seconds:02}"
                    if hasattr(self, 'elapsed_time_label') and self.elapsed_time_label.winfo_exists():
                        self.elapsed_time_label.config(text=elapsed_str)
                elif hasattr(self, 'elapsed_time_label') and self.elapsed_time_label.winfo_exists() and self.elapsed_time_label.cget("text"):
                    self.elapsed_time_label.config(text="")

            except tk.TclError:
                pass
            except Exception as e_ui_update:
                print(f"Periodic UI update loop error: {e_ui_update}")

            if not self._closing:
                self.after(1000, _update)
        self.after(100, _update)

    def _initialize_session_logs(self):
        """Initializes logs specific to a conversation session."""
        if not self.current_session_timestamp:
            self._log_to_console("Error: Session timestamp not set for initializing session logs.")
            return

        # 現在のモードを取得（Sin/HRF/Fixed）
        self.current_session_mode = self.prosody.get_current_mode()
        self._log_to_console(f"セッションモード: {self.prosody.get_current_mode_display()}")
        subject_id = self.current_subject_id or self._get_sanitized_subject_id()

        log_files_initialized_names: List[str] = []
        try:
            self.conversation_manager.initialize_conversation_csv_log(
                self.current_session_timestamp,
                self.current_session_mode,
                subject_id=subject_id
            )
            if self.conversation_manager.csv_log_filepath:
                log_files_initialized_names.append("会話CSV")
        except Exception as e_conv_csv_init:
            print(f"会話CSVログの初期化エラー: {e_conv_csv_init}")

        # Verityモニターにセッション情報を設定
        if self.hr_monitor:
            self.hr_monitor.set_session_info(self.current_session_timestamp, self.current_session_mode)

        if self.hr_monitor.is_connected:
            try:
                verity_hr_path = get_timestamped_log_path(
                    config.VERITY_HR_SESSION_CSV_TEMPLATE,
                    self.current_session_timestamp,
                    self.current_session_mode,
                    subject_id=subject_id
                )
                self.hr_monitor.initialize_verity_hr_session_csv(verity_hr_path)
                log_files_initialized_names.append("Verity HRセッション")
            except Exception as e_vhr_init: print(f"Verity HRセッションログ初期化エラー: {e_vhr_init}")

        if self.h10_monitor.is_connected:
            try:
                h10_ecg_path = get_timestamped_log_path(
                    config.H10_ECG_SESSION_CSV_TEMPLATE,
                    self.current_session_timestamp,
                    self.current_session_mode,
                    subject_id=subject_id
                )
                self.h10_monitor.initialize_h10_ecg_session_csv(h10_ecg_path)
                log_files_initialized_names.append("H10 ECGセッション")
            except Exception as e_h10e_init: print(f"H10 ECGセッションログ初期化エラー: {e_h10e_init}")
            try:
                h10_hr_path = get_timestamped_log_path(
                    config.H10_HR_SESSION_CSV_TEMPLATE,
                    self.current_session_timestamp,
                    self.current_session_mode,
                    subject_id=subject_id
                )
                self.h10_monitor.initialize_h10_hr_session_csv(h10_hr_path)
                log_files_initialized_names.append("H10 HRセッション")
            except Exception as e_h10h_init: print(f"H10 HRセッションログ初期化エラー: {e_h10h_init}")

        try:
            hr_after_tts_filepath = get_timestamped_log_path(
                config.HEARTRATE_AFTER_TTS_CSV_TEMPLATE,
                self.current_session_timestamp,
                self.current_session_mode,
                subject_id=subject_id
            )
            os.makedirs(os.path.dirname(hr_after_tts_filepath), exist_ok=True)
            header_hr_tts = [
                "Timestamp", "HR After TTS (BPM)", "Reference HR",
                "Mode", "HR Used for Adjustment (BPM)",
                "Applied Intonation Scale",
                "Playback Start Time", "Playback End Time"
            ]
            self.log_queue.put(("add_handler", config.LOGGER_HR_AFTER_TTS, hr_after_tts_filepath, header_hr_tts))
            log_files_initialized_names.append("HR After TTS")
        except Exception as e_init_log_hr_tts:
            print(f"Error setting up session 'HR after TTS' log: {e_init_log_hr_tts}")

        try:
            hr_at_rec_start_filepath = get_timestamped_log_path(
                config.HEARTRATE_AT_RECORDING_START_CSV_TEMPLATE,
                self.current_session_timestamp,
                self.current_session_mode,
                subject_id=subject_id
            )
            os.makedirs(os.path.dirname(hr_at_rec_start_filepath), exist_ok=True)
            header_hr_rec_start = ["Timestamp", "HR at Recording Start (BPM)", "Mode"]
            self.log_queue.put(("add_handler", config.LOGGER_HR_AT_RECORDING_START, hr_at_rec_start_filepath, header_hr_rec_start))
            log_files_initialized_names.append("HR at Recording Start")
        except Exception as e_init_log_hr_rec:
            print(f"Error setting up session 'HR at Recording Start' log: {e_init_log_hr_rec}")

        if not log_files_initialized_names:
            self._log_to_console("警告: 接続デバイスがないかログタイプが選択されていないため、セッションログは開始されません。")
        else:
            self._log_to_console(f"セッションログ準備完了: {', '.join(log_files_initialized_names)}")


    def toggle_status_window(self):
        if not self.status_display_window or not self.status_display_window.winfo_exists():
            print("Status display window was destroyed or not found. Recreating...")
            self.status_display_window = StatusDisplayWindow(self)

        if self.status_window_visible:
            self.status_display_window.withdraw()
            self.status_window_visible = False
            if hasattr(self, 'toggle_status_window_button'):
                 self.toggle_status_window_button.config(text="AI発話表示")
        else:
            self.status_display_window.deiconify()
            self.status_display_window.lift()
            self.status_display_window.attributes('-topmost', True)
            self.after(100, lambda: self.status_display_window.attributes('-topmost', False))
            self.status_window_visible = True
            if hasattr(self, 'toggle_status_window_button'):
                 self.toggle_status_window_button.config(text="AI発話非表示")

        if self.status_window_visible:
            if self.is_conversing:
                last_ai_msg = next((m for m in reversed(self.conversation_manager.conversation_history) if m["role"] == "assistant"), None)
                if last_ai_msg: self._update_ai_speech_display(last_ai_msg['content'])
                else: self._clear_ai_speech_display()
                self.set_status(self.status_label.cget("text"))
            else:
                self._clear_ai_speech_display()
                self._clear_status_display_prompt()

    def connect_devices(self):
        """Handle device connection/disconnection asynchronously."""
        if self.is_conversing or self.is_processing or self.is_measuring_baseline:
            self.set_status("処理中はセンサーの接続/切断はできません", "orange")
            return

        is_any_connected = self.hr_monitor.is_connected or self.h10_monitor.is_connected

        try:
            if is_any_connected:
                future = self.run_coroutine(self._disconnect_devices_async())
            else:
                future = self.run_coroutine(self._connect_devices_async())
            # 例外はバックグラウンドで発生するためログに出す
            future.add_done_callback(
                lambda fut: fut.exception() and print(f"[AsyncLoop] Sensor task error: {fut.exception()}")
            )
        except RuntimeError as e:
            self.set_status(f"センサー制御のスケジュールに失敗しました: {e}", "red")
            print(f"[AsyncLoop] Failed to schedule sensor task: {e}")

    async def _connect_devices_async(self) -> None:
        import platform
        is_linux = platform.system() == "Linux"

        self.connect_button.config(text="接続中...", state=tk.DISABLED)

        if is_linux:
            # Linux: 順次接続（BLEスキャンの競合を回避）
            print("Linux detected: connecting devices sequentially...")
            verity_success = False
            h10_success = False

            try:
                self.set_status("Verity Sense 接続中...", "orange")
                verity_success = await self.hr_monitor.start_monitoring_async()
            except Exception as e:
                print(f"Verity Sense connection error: {e}")

            if verity_success:
                await asyncio.sleep(1.0)  # 接続完了後に待機

            try:
                self.set_status("H10 接続中...", "orange")
                h10_success = await self.h10_monitor.start_monitoring_async()
            except Exception as e:
                print(f"H10 connection error: {e}")
        else:
            # macOS/Windows: 同時接続
            results = await asyncio.gather(
                self.hr_monitor.start_monitoring_async(),
                self.h10_monitor.start_monitoring_async(),
                return_exceptions=True
            )
            verity_success = isinstance(results[0], bool) and results[0]
            h10_success = isinstance(results[1], bool) and results[1]

            if isinstance(results[0], Exception):
                print(f"Verity Sense connection error: {results[0]}")
            if isinstance(results[1], Exception):
                print(f"H10 connection error: {results[1]}")

        # Update UI based on results
        if verity_success and h10_success:
            self.set_status("両センサー接続完了", "green")
        elif verity_success:
            self.set_status("Verity Sense のみ接続", "orange")
        elif h10_success:
            self.set_status("H10 のみ接続", "orange")
        else:
            self.set_status("センサー接続失敗", "red")

        self._update_button_states()


    async def _disconnect_devices_async(self) -> None:
        self.connect_button.config(text="切断中...", state=tk.DISABLED)
        
        # Concurrently disconnect both devices
        await asyncio.gather(
            self.hr_monitor.stop_monitoring_async(),
            self.h10_monitor.stop_monitoring_async(),
            return_exceptions=True
        )
        self.set_status("全センサー切断完了", "green")
        self.audio.reset_hfb_state()
        self._update_button_states()

    def update_reference_hr(self, event=None):
        if self.is_conversing or self.is_processing or self.is_measuring_baseline:
            self.set_status("処理中は基準心拍数を変更できません", "orange")
            return
        try:
            hr_val_str = self.reference_hr_var.get()
            hr_val = int(hr_val_str)
            new_min_hr, new_max_hr = 30, 220
            if new_min_hr <= hr_val <= new_max_hr:
                self.hr_monitor.set_reference_hr(hr_val)
                self.reference_hr_var.set(str(hr_val))
                self.set_status(f"基準心拍数 (HFB用) を {hr_val} BPMに設定しました", "green")
                self._log_to_console(f"基準心拍数 更新: {hr_val} BPM")
            else:
                messagebox.showwarning("入力エラー", f"基準心拍数は{new_min_hr}から{new_max_hr}の間で設定してください。")
                self.reference_hr_var.set(str(self.hr_monitor.get_reference_hr()))
        except ValueError:
            messagebox.showerror("入力エラー", "基準心拍数は整数で入力してください。")
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
            self.set_status(f"心拍数による抑揚自動調整(通常HFB)を「{'有効' if new_state else '無効'}」にしました", "blue")
            
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
                if not self.prosody.is_hfb_enabled():
                    self.prosody.set_parameter("intonation", 1.0)
                    self.update_parameter_display("intonation")
                    print("抑揚正弦波モード無効。抑揚を1.0にリセット。")
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
                    except ValueError: pass 
                self.prosody.set_parameter("intonation", initial_intonation)
                self.update_parameter_display("intonation")
            else:
                if not self.prosody.is_hfb_enabled():
                    self.prosody.set_parameter("intonation", 1.0)
                    self.update_parameter_display("intonation")
                    print("抑揚正弦波モード無効。抑揚を1.0にリセット。")
            self._update_button_states()
        except Exception as e_toggle_shfb:
            print(f"Sinusoidal HFB toggle error: {e_toggle_shfb}")
            self.set_status("抑揚正弦波モード切り替えエラー", "red")

    def toggle_hrf2(self):
        """HRF2モードの切り替え"""
        try:
            if not hasattr(self.prosody, 'hrf2_enabled_var') or self.prosody.hrf2_enabled_var is None:
                return
            new_state = self.prosody.hrf2_enabled_var.get()

            self.prosody.enable_hrf2(new_state)
            self.set_status(f"HRF2モード（心拍追従）を「{'有効' if new_state else '無効'}」にしました", "blue")

            if new_state:
                # 他のモードを無効化
                if self.prosody.is_hfb_enabled():
                    self.prosody.enable_hfb(False)
                    if self.prosody.hfb_enabled_var:
                        self.prosody.hfb_enabled_var.set(False)
                    print("通常HFBモードは無効化されました。")
                if self.prosody.is_sinusoidal_hfb_enabled():
                    self.prosody.enable_sinusoidal_hfb(False)
                    if self.prosody.sinusoidal_hfb_enabled_var:
                        self.prosody.sinusoidal_hfb_enabled_var.set(False)
                    print("正弦波モードは無効化されました。")
                self._update_hrf2_status()
            else:
                self.prosody.set_parameter("intonation", 1.0)
                self.update_parameter_display("intonation")
                self._update_hrf2_status()

            self._update_button_states()
        except Exception as e_toggle_hrf2:
            print(f"HRF2 toggle error: {e_toggle_hrf2}")
            self.set_status("HRF2モード切り替えエラー", "red")

    def _on_hrf2_target_change(self):
        """HRF2目標心拍数の変更"""
        try:
            target = self.hrf2_target_hr_var.get()
            current_target = self.prosody.get_hrf2_target_hr()
            # 値が変わった場合のみ更新
            if abs(target - current_target) > 0.1:
                self.prosody.set_hrf2_target_hr(target)
                self._update_hrf2_status()
                print(f"HRF2目標心拍数を {target:.0f} BPM に設定")
        except (tk.TclError, ValueError):
            # 入力中の一時的なエラー（空文字など）は無視
            pass
        except Exception as e:
            print(f"HRF2 target change error: {e}")

    def _apply_hrf2_pid_gains(self):
        """HRF2のPIDゲインを適用"""
        try:
            kp = self.hrf2_kp_var.get()
            ki = self.hrf2_ki_var.get()
            kd = self.hrf2_kd_var.get()
            self.prosody.set_hrf2_pid_gains(kp, ki, kd)
            self.set_status(f"HRF2 PIDゲイン適用: Kp={kp}, Ki={ki}, Kd={kd}", "blue")
            self._log_to_console(f"HRF2 PIDゲイン: Kp={kp}, Ki={ki}, Kd={kd}")
        except Exception as e:
            print(f"HRF2 PID gains error: {e}")
            self.set_status("PIDゲイン設定エラー", "red")

    def _update_hrf2_status(self):
        """HRF2ステータス表示を更新"""
        if not hasattr(self, 'hrf2_status_label'):
            return
        if self.prosody.is_hrf2_enabled():
            target = self.prosody.get_hrf2_target_hr()
            current_hr = self.hr_monitor.get_current_hr() if self.hr_monitor.is_connected else 0
            mode = self.prosody.get_hrf2_control_mode().value
            if current_hr > 0:
                if mode == "Adaptive":
                    theta = self.prosody.get_hrf2_adaptive_theta()
                    status_text = f"HRF2({mode}): 目標{target:.0f}BPM / 現在{current_hr}BPM / θ={theta:.4f}"
                    # θラベルも更新
                    if hasattr(self, 'hrf2_theta_label'):
                        self.hrf2_theta_label.config(text=f"{theta:.4f}")
                elif mode == "GainScheduled":
                    zone = self.prosody.get_hrf2_gain_schedule_zone()
                    gain_type = self.prosody.get_hrf2_gain_schedule_gain_type()
                    status_text = f"HRF2({mode}): 目標{target:.0f}BPM / 現在{current_hr}BPM / {zone}({gain_type})"
                    # ゾーンとタイプラベルを更新
                    if hasattr(self, 'hrf2_gs_zone_label'):
                        self.hrf2_gs_zone_label.config(text=zone)
                    if hasattr(self, 'hrf2_gs_type_label'):
                        self.hrf2_gs_type_label.config(text=gain_type)
                elif mode == "Robust":
                    gain = self.prosody.get_hrf2_robust_adapted_gain()
                    status_text = f"HRF2({mode}): 目標{target:.0f}BPM / 現在{current_hr}BPM / gain={gain:.4f}"
                else:
                    status_text = f"HRF2({mode}): 目標{target:.0f}BPM / 現在{current_hr}BPM"
            else:
                status_text = f"HRF2({mode}): 目標{target:.0f}BPM / HR未取得"
            self.hrf2_status_label.config(text=status_text, foreground="green")
        else:
            self.hrf2_status_label.config(text="HRF2: 無効", foreground="gray")

    def _on_hrf2_control_mode_change(self, event=None):
        """HRF2制御モードの変更"""
        try:
            mode_str = self.hrf2_control_mode_var.get()
            if mode_str == "PID":
                mode = ControlMode.PID
            elif mode_str == "Adaptive":
                mode = ControlMode.ADAPTIVE
            elif mode_str == "Robust":
                mode = ControlMode.ROBUST
            else:  # GainScheduled
                mode = ControlMode.GAIN_SCHEDULED
            self.prosody.set_hrf2_control_mode(mode)
            self._update_hrf2_param_frames()
            self._update_hrf2_status()
            self.set_status(f"HRF2制御モードを「{mode_str}」に変更しました", "blue")
            self._log_to_console(f"HRF2制御モード: {mode_str}")
        except Exception as e:
            print(f"HRF2 control mode change error: {e}")
            self.set_status("制御モード変更エラー", "red")

    def _update_hrf2_param_frames(self):
        """制御モードに応じてパラメータフレームの表示/非表示を切り替え"""
        if not hasattr(self, 'hrf2_pid_frame') or not hasattr(self, 'hrf2_adaptive_frame'):
            return

        mode = self.prosody.get_hrf2_control_mode()

        # 各フレームの有効/無効を設定するヘルパー
        def set_frame_state(frame, enabled):
            for child in frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button)):
                    child.configure(state="normal" if enabled else "disabled")

        if mode == ControlMode.PID:
            set_frame_state(self.hrf2_pid_frame, True)
            set_frame_state(self.hrf2_adaptive_frame, False)
            if hasattr(self, 'hrf2_gainschedule_frame'):
                set_frame_state(self.hrf2_gainschedule_frame, False)
        elif mode == ControlMode.ADAPTIVE:
            set_frame_state(self.hrf2_pid_frame, False)
            set_frame_state(self.hrf2_adaptive_frame, True)
            if hasattr(self, 'hrf2_gainschedule_frame'):
                set_frame_state(self.hrf2_gainschedule_frame, False)
        elif mode == ControlMode.ROBUST:
            # Robustはデフォルトパラメータで動作（H∞ループ整形）
            set_frame_state(self.hrf2_pid_frame, False)
            set_frame_state(self.hrf2_adaptive_frame, False)
            if hasattr(self, 'hrf2_gainschedule_frame'):
                set_frame_state(self.hrf2_gainschedule_frame, False)
        else:  # GAIN_SCHEDULED
            set_frame_state(self.hrf2_pid_frame, False)
            set_frame_state(self.hrf2_adaptive_frame, False)
            if hasattr(self, 'hrf2_gainschedule_frame'):
                set_frame_state(self.hrf2_gainschedule_frame, True)

    def _apply_hrf2_adaptive_params(self):
        """HRF2の適応制御パラメータを適用"""
        try:
            gamma = self.hrf2_gamma_var.get()
            tau = self.hrf2_tau_var.get()
            self.prosody.set_hrf2_adaptive_params(gamma, tau)
            self.set_status(f"適応制御パラメータ適用: γ={gamma}, τ={tau}", "blue")
            self._log_to_console(f"適応制御パラメータ: γ={gamma}, τ={tau}")
        except Exception as e:
            print(f"HRF2 adaptive params error: {e}")
            self.set_status("適応制御パラメータ設定エラー", "red")

    def _apply_hrf2_gs_params(self):
        """HRF2のゲインスケジューリングパラメータを適用"""
        try:
            high = self.hrf2_gs_high_var.get()
            med = self.hrf2_gs_med_var.get()
            self.prosody.set_hrf2_gain_schedule_thresholds(high, med)

            # ゲインタイプを適用
            type_high = GainType(self.hrf2_gs_type_high_var.get())
            type_med = GainType(self.hrf2_gs_type_med_var.get())
            type_low = GainType(self.hrf2_gs_type_low_var.get())
            self.prosody.set_hrf2_gain_schedule_types(type_high, type_med, type_low)

            self.set_status(f"GS適用: 閾値(高={high},中={med}) タイプ(高={type_high.value},中={type_med.value},低={type_low.value})", "blue")
            self._log_to_console(f"GSパラメータ: 高閾値={high}, 中閾値={med}, タイプ(高={type_high.value},中={type_med.value},低={type_low.value})")
        except Exception as e:
            print(f"HRF2 GS params error: {e}")
            self.set_status("GS設定エラー", "red")

    def on_speaker_selected(self, event=None):
        try:
            selected_name = self.speaker_var.get()
            selected_id = next((s['id'] for s in self.speaker.speakers if s['name'] == selected_name), None)
            if selected_id is not None:
                self.speaker.current_style_id = selected_id
                if hasattr(self, 'speaker_status') and self.speaker_status.winfo_exists():
                    self.speaker_status.config(text=f"選択中: {selected_name}", foreground="green")
                self.set_status(f"話者を「{selected_name}」(ID: {selected_id})に設定しました", "blue")
                self._log_to_console(f"話者変更: {selected_name} (ID: {selected_id})")
            else:
                print(f"Error: Selected speaker '{selected_name}' ID not found.")
                if hasattr(self, 'speaker_status') and self.speaker_status.winfo_exists():
                    self.speaker_status.config(text="エラー", foreground="red")
        except Exception as e_spk_sel:
            print(f"Speaker selection error: {e_spk_sel}")
            if hasattr(self, 'speaker_status') and self.speaker_status.winfo_exists():
                self.speaker_status.config(text="選択エラー", foreground="red")

    def test_speech(self):
        if self.is_processing or self.is_conversing or self.is_measuring_baseline:
            self.set_status("他の処理中はテスト発話を実行できません", "orange")
            return
        if not VoicevoxManager.check_server():
             messagebox.showerror("VOICEVOX Error", "VOICEVOXサーバーが接続されていません。")
             return
        if self.speaker.current_style_id == 0 and not self.speaker.speakers:
             messagebox.showerror("VOICEVOX Error", "話者が選択されていません、または話者情報を取得できませんでした。")
             return
        if self.speaker.current_style_id == 0 :
             messagebox.showwarning("VOICEVOX Warning", "話者IDが0です。デフォルト話者で試行します。")

        self.is_processing = True
        self.set_status("テスト発話を実行中...", "orange")
        self._update_button_states()

        def _process_test_speech_thread():
            try:
                test_text = "こんにちは。これは、音声合成のテスト発話です。"
                self._log_to_console(f"テスト発話開始: {test_text}")
                self.after(0, lambda: self._update_ai_speech_display(test_text))
                self.after(0, lambda: self._clear_status_display_prompt())
                success, _, _ = self.audio.text_to_speech(test_text, "test_output.wav")
                status_msg, status_color = ("テスト発話 完了", "green") if success else ("テスト発話 失敗", "red")
                self.set_status(status_msg, status_color)
                self._log_to_console(f"テスト発話結果: {'成功' if success else '失敗'}")
                self.after(1000, self._clear_ai_speech_display)
            except Exception as e_test_speech:
                self.set_status(f"テスト発話エラー: {e_test_speech}", "red")
                print(f"Test speech thread error details: {e_test_speech}")
                self.after(0, self._clear_ai_speech_display)
            finally:
                self.is_processing = False
                self.after(0, self._update_button_states)
        threading.Thread(target=_process_test_speech_thread, daemon=True).start()

    def save_system_prompt(self):
        if self.is_conversing or self.is_measuring_baseline:
            messagebox.showwarning("Busy", "会話中または基準測定中はプロンプトを保存できません。")
            return
        try:
            new_prompt = self.system_prompt.get("1.0", tk.END).strip()
            if not new_prompt:
                messagebox.showerror("入力エラー", "システムプロンプトは空にできません。")
                return
            self.conversation_manager.update_system_prompt(new_prompt)
            self.save_config()
            self.set_status("システムプロンプトを更新・保存しました", "green")
            self._log_to_console("システムプロンプト 更新・保存完了")
        except Exception as e_save_prompt:
            messagebox.showerror("保存エラー", f"プロンプト保存エラー:\n{e_save_prompt}")
            self.set_status("プロンプト保存失敗", "red")

    def clear_conversation(self):
        if self.is_conversing or self.is_measuring_baseline:
            messagebox.showwarning("Busy", "会話中または基準測定中は履歴をクリアできません。")
            return
        if messagebox.askyesno("確認", "会話履歴をクリアしますか？\n（システムプロンプトは保持されます）"):
            try:
                self.conversation_manager.clear_history()
                self.audio.reset_hfb_state()
                self.conversation_start_time = None
                if hasattr(self, 'elapsed_time_label'): self.after(0, lambda: self.elapsed_time_label.config(text=""))
                self.set_status("会話履歴をクリアしました", "green")
                self.after(0, self._clear_ai_speech_display)
                self.after(0, self._clear_status_display_prompt)
                self._log_to_console("会話履歴クリア完了")
            except Exception as e_clear_conv:
                messagebox.showerror("エラー", f"履歴クリアエラー:\n{e_clear_conv}")
                self.set_status("履歴クリア失敗", "red")

    def _update_button_states(self):
        try:
            is_any_busy_state = self.is_conversing or self.is_processing or self.is_measuring_baseline
            start_conv_state = tk.DISABLED if is_any_busy_state else tk.NORMAL
            stop_conv_state = tk.NORMAL if self.is_conversing else tk.DISABLED
            if hasattr(self, 'start_button'): self.start_button.config(state=start_conv_state)
            if hasattr(self, 'stop_button'): self.stop_button.config(state=stop_conv_state)

            connect_btn_state = tk.DISABLED
            current_connect_text = ""
            if hasattr(self, 'connect_button'): current_connect_text = self.connect_button.cget('text')
            is_connecting_or_disconnecting = "中..." in current_connect_text
            if not is_any_busy_state and not is_connecting_or_disconnecting:
                connect_btn_state = tk.NORMAL
            is_any_sensor_connected = self.hr_monitor.is_connected or self.h10_monitor.is_connected
            connect_btn_text_final = "センサー類 切断" if is_any_sensor_connected else "センサー類 接続"
            if hasattr(self, 'connect_button'):
                if not is_connecting_or_disconnecting:
                    self.connect_button.config(state=connect_btn_state, text=connect_btn_text_final)
                else:
                    self.connect_button.config(state=tk.DISABLED)

            baseline_btn_state = tk.NORMAL if self.hr_monitor.is_connected and not is_any_busy_state else tk.DISABLED
            if hasattr(self, 'measure_baseline_button'): self.measure_baseline_button.config(state=baseline_btn_state)
            baseline_spin_state = tk.DISABLED if is_any_busy_state else tk.NORMAL
            if hasattr(self, 'baseline_duration_spinbox'): self.baseline_duration_spinbox.config(state=baseline_spin_state)
            save_cfg_btn_state = tk.DISABLED if is_any_busy_state else tk.NORMAL
            if hasattr(self, 'save_config_button'): self.save_config_button.config(state=save_cfg_btn_state)
            if hasattr(self, 'toggle_status_window_button'): self.toggle_status_window_button.config(state=tk.NORMAL)
        except tk.TclError:
            pass
        except Exception as e_upd_btn:
            print(f"Button state update error: {e_upd_btn}")

    def measure_baseline_hr(self):
        if self.is_conversing or self.is_processing or self.is_measuring_baseline:
            self.set_status("他の処理中は基準心拍数計測を開始できません", "orange")
            return
        if not self.hr_monitor.is_connected:
            messagebox.showerror("センサーエラー", "基準心拍数計測には Polar Verity Sense の接続が必要です。")
            return
        try:
            duration_s = self.baseline_duration_var.get()
            if not (10 <= duration_s <= 600):
                messagebox.showerror("入力エラー", "基準HR計測時間は10秒から600秒の間で設定してください。")
                return
        except tk.TclError:
            messagebox.showerror("入力エラー", "無効な基準HR計測時間です。")
            return

        self.is_measuring_baseline = True
        self._update_button_states()
        self.set_status(f"基準心拍数を計測中 ({duration_s}秒)... しばらくお待ちください。", "blue")
        self._log_to_console(f"基準心拍数計測開始 ({duration_s}秒)")
        if not self.hr_monitor.start_baseline_measurement(duration_s):
            self.set_status("基準心拍数計測の開始に失敗しました", "red")
            self._log_to_console("基準心拍数計測開始失敗")
            self.is_measuring_baseline = False
            self._update_button_states()
            return
        self.after(duration_s * 1000, self._finish_baseline_measurement)

    def _finish_baseline_measurement(self):
        if not self.is_measuring_baseline:
            return
        median_hr = self.hr_monitor.stop_baseline_measurement()
        if median_hr is not None:
            self.hr_monitor.set_reference_hr(median_hr)
            self.reference_hr_var.set(str(median_hr))
            self.set_status(f"基準心拍数計測完了。中央値: {median_hr} BPM。基準HRに設定しました。", "green")
            self._log_to_console(f"基準心拍数計測完了: 中央値 {median_hr} BPM")
        else:
            self.set_status("基準心拍数計測は終了しましたが、中央値の計算に失敗しました (データ不足など)。", "orange")
            self._log_to_console("基準心拍数計測完了、中央値計算失敗")
        self.is_measuring_baseline = False
        self._update_button_states()

    def _update_ai_speech_display(self, text: str):
        if self.status_display_window and self.status_display_window.winfo_exists() and self.status_window_visible:
            self.status_display_window.set_ai_speech(text)

    def _clear_ai_speech_display(self):
        if self.status_display_window and self.status_display_window.winfo_exists() and self.status_window_visible:
            self.status_display_window.clear_ai_speech_display()

    def _set_status_display_prompt(self, message: str):
        if self.status_display_window and self.status_display_window.winfo_exists() and self.status_window_visible:
            self.status_display_window.set_prompt_message(message)

    def _clear_status_display_prompt(self):
        if self.status_display_window and self.status_display_window.winfo_exists() and self.status_window_visible:
            self.status_display_window.clear_prompt_message()

    def _log_to_console(self, message: str):
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [App] {message}")

    def _configure_video_recorder_for_mic(self, primary_mic_index: Optional[int], silent: bool = False) -> None:
        """
        会話用マイクの選択に応じてVideoRecorderを構築し直す。
        映像録音中は設定を変更しない。
        """
        if self.video_recorder and getattr(self.video_recorder, "is_recording", False):
            print("[VideoRecorder] 録画中のためマイク設定を変更できません")
            return

        secondary_mic = get_secondary_mic_device(primary_mic_index)
        if secondary_mic is not None:
            self.video_recorder = VideoRecorder(
                record_audio=True,
                audio_device_index=secondary_mic,
                auto_detect_audio_device=False
            )
            if not silent:
                print(f"[VideoRecorder] 映像に音声を付けます（会話マイク: {primary_mic_index}, 2台目: {secondary_mic}）")
        else:
            self.video_recorder = VideoRecorder(record_audio=False)
            if not silent:
                print(f"[VideoRecorder] 2台目のマイクが見つからないため映像のみ録画になります（会話マイク: {primary_mic_index}）")

    def _format_mic_device_label(self, device: Dict[str, Any]) -> str:
        samplerate = int(device.get('sample_rate') or 0)
        sr_text = f"{samplerate:.0f}" if samplerate else "-"
        default_mark = " *" if device.get('is_default') else ""
        return f"[{device['index']}] {device['name']}{default_mark} ({sr_text}Hz/{device['channels']}ch)"

    def _refresh_mic_devices(self, keep_selection: bool = True):
        """マイクリストを再取得してコンボボックスを更新する"""
        try:
            devices = list_input_devices()
        except Exception as e:
            print(f"[AudioDevice] マイク一覧の取得に失敗しました: {e}")
            devices = []

        self._mic_devices_cache = devices
        values = [self._format_mic_device_label(dev) for dev in devices]
        self._mic_label_to_index = {label: dev['index'] for label, dev in zip(values, devices)}

        if not devices:
            self.mic_device_combo.configure(state="disabled")
            self.mic_device_var.set("")
            self.mic_device_hint_label.config(text="マイクが検出されませんでした", foreground="red")
            return

        self.mic_device_combo.configure(state="readonly")
        self.mic_device_combo['values'] = values

        target_index: Optional[int] = None
        if keep_selection and self.audio.input_device_index is not None:
            target_index = self.audio.input_device_index
        elif values:
            target_index = devices[0]['index']

        selected_label = None
        if target_index is not None:
            for label, idx in self._mic_label_to_index.items():
                if idx == target_index:
                    selected_label = label
                    break

        if selected_label:
            self.mic_device_var.set(selected_label)
        else:
            # デバイスが変わった/消えた場合は先頭を選択
            selected_label = values[0]
            self.mic_device_var.set(selected_label)
            self.audio.input_device_index = self._mic_label_to_index[selected_label]
            self._configure_video_recorder_for_mic(self.audio.input_device_index)

        self._update_mic_hint()

    def _update_mic_hint(self):
        idx = self.audio.input_device_index
        if idx is None:
            self.mic_device_hint_label.config(text="デフォルトマイクを使用します", foreground="orange")
            return

        device = next((dev for dev in self._mic_devices_cache if dev['index'] == idx), None)
        if device:
            text = f"使用中: [{idx}] {device['name']}"
            color = "green"
        else:
            text = f"使用中: [{idx}] (検出不可)"
            color = "orange"

        self.mic_device_hint_label.config(text=text, foreground=color)

    def _on_mic_device_selected(self, event=None):
        label = self.mic_device_var.get()
        target_index = self._mic_label_to_index.get(label)
        if target_index is None or target_index == self.audio.input_device_index:
            self._update_mic_hint()
            return

        if self.is_conversing:
            self._log_to_console("警告: 会話中にマイクを変更しました。必要に応じて会話を再開してください。")

        self.audio.input_device_index = target_index
        self._configure_video_recorder_for_mic(target_index)
        self._log_to_console(f"会話マイクを変更: [{target_index}] {label}")
        self._update_mic_hint()

    def start_conversation(self) -> None:
        if self.is_conversing or self.is_processing or self.is_measuring_baseline:
            self._log_to_console("Error: Conversation/processing/baseline measurement already active.")
            messagebox.showwarning("Busy", "他の処理が実行中です。")
            return
        subject_id = self._get_sanitized_subject_id()
        if not subject_id:
            messagebox.showwarning("被験者番号未設定", "会話を開始する前に被験者番号を入力してください。")
            if hasattr(self, 'subject_entry'):
                self.subject_entry.focus_set()
            return
        if not self.use_local_llm_var.get():
            api_key = self.openai_api_key_var.get().strip() or os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                messagebox.showerror(
                    "APIキーエラー",
                    "OpenAI APIキーが設定されていません。\n環境変数 'OPENAI_API_KEY' か入力欄を確認してください。"
                )
                if hasattr(self, "api_key_entry"):
                    self.api_key_entry.focus_set()
                return
            # keep env var in sync so他処理が参照するときに困らない
            os.environ["OPENAI_API_KEY"] = api_key
        if not VoicevoxManager.check_server():
            messagebox.showerror("VOICEVOXエラー", "VOICEVOXサーバーに接続できません。\n音声合成機能は利用できません。")
            return
        if self.speaker.current_style_id == 0 and not self.speaker.speakers:
             messagebox.showerror("話者エラー", "話者が選択されていないか、話者リストの読み込みに失敗しました。")
             return
        if self.speaker.current_style_id == 0:
             messagebox.showwarning("話者注意", "話者IDが0です。デフォルト話者で試行しますが、エラーになる可能性があります。")

        if self.status_display_window and not self.status_window_visible:
            self.toggle_status_window()

        self.current_subject_id = subject_id
        self.conversation_manager.set_subject_id(subject_id)
        self.hr_monitor.set_subject_id(subject_id)
        self._log_to_console(f"被験者番号: {subject_id}")

        self.current_session_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_to_console(f"--- 新しい会話セッション開始 (タイムスタンプ: {self.current_session_timestamp}) ---")
        self.after(0, self._clear_ai_speech_display)
        self.after(0, self._clear_status_display_prompt)

        self._initialize_session_logs()

        # ビデオ録画を開始（有効な場合）
        if self.video_recording_enabled.get():
            if self.video_recorder.start_recording(
                self.current_session_timestamp,
                self.current_session_mode,
                self.current_subject_id
            ):
                self._log_to_console("ビデオ録画を開始しました")
            else:
                self._log_to_console("警告: ビデオ録画の開始に失敗しました（カメラが見つからない可能性）")

        self.is_conversing = True
        self.is_processing = True
        self.audio.stop_event.clear()
        self.conversation_start_time = datetime.datetime.now()
        if hasattr(self, 'elapsed_time_label'): self.after(0, lambda: self.elapsed_time_label.config(text="会話時間: 00:00:00"))
        self.set_status("音声対話を開始しています...", "blue")
        self._update_button_states()
        self.audio.reset_hfb_state()
        print("HFB states (direct and sinusoidal) reset for new conversation.")

        self.processing_thread = threading.Thread(target=self.conversation_loop, daemon=True, name="ConversationThread")
        self.processing_thread.start()
        self.is_processing = False

        # 会話開始後にECGデータ受信を確認（3秒後）
        if self.h10_monitor and self.h10_monitor.is_connected:
            self.after(3000, self._check_ecg_data_after_start)

    def _check_ecg_data_after_start(self) -> None:
        """会話開始後にECGデータが正しく受信されているかを確認する。

        H10が接続されているにもかかわらずECGデータが受信されていない場合、
        警告ダイアログを表示してユーザーに通知する。
        """
        if not self.is_conversing:
            # 会話が既に停止している場合は確認不要
            return

        if not self.h10_monitor or not self.h10_monitor.is_connected:
            return

        if not self.h10_monitor.is_receiving_ecg_data():
            # ECGデータが受信されていない
            self._log_to_console("警告: H10からECGデータが受信されていません！")
            warning_msg = (
                "H10は接続されていますが、ECGデータが受信されていません。\n\n"
                "考えられる原因:\n"
                "• H10が胸に正しく装着されていない\n"
                "• H10のストラップが乾燥している（水で濡らしてください）\n"
                "• H10の電極部分が肌に密着していない\n\n"
                "ECGデータがないまま会話を続けると、\n"
                "ECGログファイルが空になります。\n\n"
                "会話を停止してH10を確認しますか？"
            )
            response = messagebox.askyesno(
                "ECGデータ受信エラー",
                warning_msg,
                icon="warning"
            )
            if response:
                # ユーザーが「はい」を選択した場合、会話を停止
                self.stop_conversation()
        else:
            ecg_buffer_size = len(self.h10_monitor.get_ecg_buffer())
            self._log_to_console(f"ECGデータ受信確認: OK ({ecg_buffer_size} サンプル)")

    def stop_conversation(self, called_from_close_signal: bool = False) -> None:
        if not self.is_conversing:
            if not called_from_close_signal: self._log_to_console("会話は実行されていません。")
            return
        self._log_to_console("会話停止リクエスト受信...")
        if not called_from_close_signal: self.set_status("音声対話を停止しています...", "orange")
        self.is_conversing = False
        self.audio.stop_event.set()
        if 'sd' in sys.modules:
            try:
                import sounddevice as sd
                sd.stop()
                print("sounddevice stream stopped.")
            except Exception as sd_err:
                print(f"Error stopping sounddevice (ignored): {sd_err}")

        thread_to_join = self.processing_thread
        if thread_to_join and thread_to_join.is_alive() and thread_to_join is not threading.current_thread():
            print("  会話処理スレッドの終了を待機中 (最大5秒)...")
            thread_to_join.join(timeout=5.0)
            if thread_to_join.is_alive():
                print("  警告: 会話処理スレッドが時間内に終了しませんでした。")
            else:
                print("  会話処理スレッドは正常に終了しました。")
        elif thread_to_join is threading.current_thread():
            print("  `stop_conversation` was called from within the conversation thread. Skipping self-join.")
        
        self.processing_thread = None

        # ビデオ録画を停止
        if self.video_recorder.is_recording:
            video_filepath = self.video_recorder.stop_recording()
            if video_filepath:
                self._log_to_console(f"ビデオ録画を停止しました: {video_filepath}")

        print("セッションログファイルのクローズを要求中...")
        if self.conversation_manager: self.conversation_manager.close_conversation_csv_log()
        if self.hr_monitor: self.hr_monitor.close_verity_hr_session_csv()
        if self.h10_monitor:
            self.h10_monitor.close_h10_ecg_session_csv()
            self.h10_monitor.close_h10_hr_session_csv()

        if self.current_session_timestamp:
            self.log_queue.put(("remove_handler", config.LOGGER_HR_AFTER_TTS))
            self.log_queue.put(("remove_handler", config.LOGGER_HR_AT_RECORDING_START))

        self.current_session_timestamp = None
        self.current_subject_id = None
        self.conversation_start_time = None
        self._log_to_console("--- 会話セッション終了 ---")
        self.after(0, self._clear_ai_speech_display)
        self.after(0, self._clear_status_display_prompt)
        if not called_from_close_signal:
            self.after(0, self._update_button_states)
            if hasattr(self, 'elapsed_time_label'): self.after(0, lambda: self.elapsed_time_label.config(text=""))
            self.set_status("音声対話が停止しました", "green")
        print("会話停止処理完了。")

    def conversation_loop(self) -> None:
        self._log_to_console("会話ループスレッド開始。")
        self.is_processing = True
        try:
            current_sys_prompt = self.system_prompt.get("1.0", tk.END).strip()
            if not any(m["role"] == "system" for m in self.conversation_manager.get_messages()):
                self.conversation_manager.update_system_prompt(current_sys_prompt)
            self._log_to_console(f"システムプロンプト確認/更新: {current_sys_prompt[:70]}{'...' if len(current_sys_prompt)>70 else ''}")
            self.set_status("挨拶を生成中...", "blue")
            greeting_text = "あなたの趣味について教えてください"
            self.after(0, lambda: self._update_ai_speech_display(greeting_text))
            if not self.audio.stop_event.is_set():
                _, greeting_start, greeting_end = self.audio.text_to_speech(greeting_text, "greeting_output.wav")
                self._log_to_console(f"初期挨拶 再生完了: {greeting_text}")
                # 定型文をconversation_logに記録
                self.conversation_manager.add_message(
                    "assistant", greeting_text,
                    start_time=greeting_start, end_time=greeting_end
                )
            self.after(0, lambda: self._set_status_display_prompt("話してください") ) 

            while self.is_conversing and not self.audio.stop_event.is_set():
                self.set_status("話してください...", "blue")

                # 録音（ver3.10方式: 一旦ファイルに保存）
                recorded_successfully, user_rec_start, user_rec_end = self.audio.record_audio("input.wav")

                if self.audio.stop_event.is_set() or not self.is_conversing: break
                if not recorded_successfully:
                    self.after(0, lambda: self._set_status_display_prompt("話してください"))
                    time.sleep(0.5)
                    continue

                self._log_to_console("ユーザー音声録音完了。")
                self.after(0, lambda: self._set_status_display_prompt("AI 処理中..."))
                self.set_status("音声をテキストに変換中...", "orange")

                # 音声認識（ver3.10方式: ファイルから直接）
                user_text = self.audio.speech_to_text("input.wav")

                if self.audio.stop_event.is_set() or not self.is_conversing: break
                if not user_text:
                    self.set_status("音声認識に失敗しました。もう一度どうぞ。", "orange")
                    self.after(0, lambda: self._set_status_display_prompt("話してください"))
                    time.sleep(0.5)
                    continue

                self.conversation_manager.add_message("user", user_text, start_time=user_rec_start, end_time=user_rec_end)
                
                print(f"User: {user_text}")
                self._log_to_console(f"ユーザー発話処理完了: {user_text[:70]}{'...' if len(user_text)>70 else ''}")
                self.set_status("AI応答を生成中...", "orange")
                current_model = config.LOCAL_LLM_MODEL if config.USE_LOCAL_LLM else config.OPENAI_MODEL
                self._log_to_console(f"LLM APIに問い合わせ開始 (モデル: {current_model}, ローカル: {config.USE_LOCAL_LLM})")
                assistant_response_text = "申し訳ありません、応答を処理できませんでした。"
                
                try:
                    messages_to_send = self.conversation_manager.get_messages()
                    llm_type = "LocalLLM" if config.USE_LOCAL_LLM else "OpenAI"
                    self._log_to_console(f"{llm_type}に送信するメッセージ数: {len(messages_to_send)}")

                    response = self.openai_client.chat.completions.create(
                        model=current_model,
                        messages=messages_to_send,
                        max_tokens=config.LLM_MAX_TOKENS,
                        temperature=config.LLM_TEMPERATURE,
                        timeout=config.LLM_TIMEOUT
                    )
                    
                    raw_response_content = response.choices[0].message.content
                    assistant_response_text = raw_response_content.strip() if raw_response_content else "AIからの応答が空でした。"
                    self._log_to_console(f"AI応答受信完了: {assistant_response_text[:100]}{'...' if len(assistant_response_text)>100 else ''}")

                except openai.APITimeoutError:
                    err_msg = "LLM API タイムアウト"
                    print(err_msg)
                    self.set_status("AI応答タイムアウト", "red")
                    self._log_to_console(f"エラー: {err_msg}")
                    assistant_response_text = "応答に時間がかかりすぎました。もう一度試してください。"
                except openai.AuthenticationError as auth_err:
                    err_msg = f"LLM API 認証エラー: {auth_err}"
                    print(err_msg)
                    self.set_status("AI API認証エラー", "red")
                    self._log_to_console(f"エラー: {err_msg}")
                    assistant_response_text = "AIとの通信で認証エラーが発生しました。APIキーを確認してください。"
                    self.stop_conversation()
                    break
                except openai.APIConnectionError as conn_err:
                    err_msg = f"LLM API 接続エラー: {conn_err}"
                    print(err_msg)
                    if config.USE_LOCAL_LLM:
                        self.set_status(f"ローカルLLM接続エラー ({config.LOCAL_LLM_BASE_URL})", "red")
                        assistant_response_text = f"ローカルLLMサーバーに接続できませんでした。サーバーが起動しているか確認してください。"
                    else:
                        self.set_status("OpenAI API接続エラー", "red")
                        assistant_response_text = "AIとの通信で接続エラーが発生しました。"
                    self._log_to_console(f"エラー: {err_msg}")
                except openai.APIError as api_err:
                    err_msg = f"LLM API エラー: {api_err}"
                    print(err_msg); self.set_status(f"AI応答エラー: {type(api_err).__name__}", "red")
                    self._log_to_console(f"エラー: {err_msg}")
                    assistant_response_text = "AIとの通信でエラーが発生しました。"
                except Exception as e_resp_gen:
                    if "APIRemovedInV1" in str(e_resp_gen):
                         err_msg = f"OpenAI API 互換性エラー: {e_resp_gen}"
                         print(err_msg)
                         self.set_status("OpenAI API 互換性エラー", "red")
                         self._log_to_console(f"エラー: {err_msg}")
                         assistant_response_text = "OpenAI APIのバージョンが古いコードと互換性がありません。"
                         self.stop_conversation() 
                         break
                    
                    err_msg = f"AI応答生成中の予期せぬエラー: {e_resp_gen}"
                    print(err_msg); self.set_status("AI応答生成エラー", "red")
                    self._log_to_console(f"予期せぬエラー: {err_msg}")
                    assistant_response_text = "応答処理中に問題が発生しました。"
                
                print(f"Assistant: {assistant_response_text}")
                if self.audio.stop_event.is_set() or not self.is_conversing: break
                self.set_status("音声を合成・再生中...", "green")
                self.after(0, lambda ai_resp=assistant_response_text: self._update_ai_speech_display(ai_resp))
                self._log_to_console("VOICEVOXでの音声合成開始。")
                
                tts_played_successfully, assistant_playback_start, assistant_playback_end = self.audio.text_to_speech(assistant_response_text, "output.wav")
                
                self.conversation_manager.add_message("assistant", assistant_response_text, start_time=assistant_playback_start, end_time=assistant_playback_end)

                if self.audio.stop_event.is_set() or not self.is_conversing: break
                if not tts_played_successfully:
                    self._log_to_console("エラー: AI応答の音声合成または再生に失敗しました。")
                else:
                    self._log_to_console("AI応答の音声合成・再生完了。")
                self.after(0, lambda: self._set_status_display_prompt("話してください") ) 
                time.sleep(0.1)
            self._log_to_console("会話ループが正常に終了しました。")
        except Exception as e_conv_loop:
            err_msg_loop = f"会話ループでの予期せぬエラー: {e_conv_loop}"
            print(err_msg_loop)
            self.set_status(f"会話ループエラー: {e_conv_loop}", "red")
            self._log_to_console(err_msg_loop)
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False
            if self.is_conversing:
                self.stop_conversation(called_from_close_signal=self._closing)
            self._log_to_console("会話ループスレッド クリーンアップ完了。")

    def signal_handler(self, sig, frame):
        print(f"\nシグナル {sig} 受信。アプリケーションを終了します...")
        self.on_close()

    def on_close(self) -> None:
        if self._closing: return
        print("シャットダウンプロセス開始...")
        self._closing = True
        self.set_status("アプリケーションを終了しています...", "orange")

        if self.is_conversing:
            print("  実行中の会話を停止中...")
            self.stop_conversation(called_from_close_signal=True)
        if self.is_measuring_baseline:
            print("  実行中の基準心拍数計測を停止中...")
            self.is_measuring_baseline = False
            self.hr_monitor.stop_baseline_measurement()

        # オーディオストリームを停止
        if 'sd' in sys.modules:
            print("  オーディオストリームを停止試行中 (sounddevice)...")
            try:
                import sounddevice as sd
                sd.stop(ignore_errors=True)
            except Exception as sd_err_close:
                print(f"  sounddevice停止エラー (無視): {sd_err_close}")

        print("  ロギングスレッドの停止を要求中...")
        if hasattr(self, 'logging_thread') and self.logging_thread.is_alive():
            self.logging_thread.stop()
            self.logging_thread.join(timeout=2.0)

        print("  最終設定を保存中...")
        self.save_config()

        print("アプリケーション終了。")

        # セグメンテーションフォルト回避のため、os._exit()で強制終了
        # bleakのCore Bluetoothバックエンドのクリーンアップ問題を回避
        import os
        os._exit(0)
