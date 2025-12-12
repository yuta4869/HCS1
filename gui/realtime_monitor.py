# gui/realtime_monitor.py
"""
リアルタイムモニター機能
心拍数・ECG・HRVのリアルタイム表示を行うモジュール
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import List, Optional, TYPE_CHECKING

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if TYPE_CHECKING:
    from polar_monitor import PolarVeritySenseMonitor, PolarH10Monitor


class RealtimeMonitorMixin:
    """リアルタイムモニター機能を提供するMixin クラス

    Application クラスにミックスインして使用する。
    必要な属性:
        - self.realtime_monitor_tab: ttk.Frame
        - self.hr_monitor: PolarVeritySenseMonitor
        - self.h10_monitor: PolarH10Monitor
        - self.after: tk.after メソッド
    """

    def _setup_realtime_monitor_tab(self) -> None:
        """リアルタイムモニタータブのUIを構築（心拍数・ECG・HRVのグラフ表示）"""
        main_frame = self.realtime_monitor_tab
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # --- コントロールパネル ---
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.monitor_running = False
        self.monitor_start_btn = ttk.Button(
            control_frame, text="モニター開始",
            command=self._toggle_realtime_monitor
        )
        self.monitor_start_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="表示時間(sec):").pack(side=tk.LEFT, padx=(20, 5))
        self.monitor_window_var = tk.IntVar(value=30)
        window_spinbox = ttk.Spinbox(
            control_frame, from_=10, to=120, width=5,
            textvariable=self.monitor_window_var
        )
        window_spinbox.pack(side=tk.LEFT, padx=5)

        # 現在値表示ラベル
        self.current_hr_label = ttk.Label(
            control_frame, text="心拍数: -- bpm",
            font=('Helvetica', 14, 'bold')
        )
        self.current_hr_label.pack(side=tk.LEFT, padx=(30, 10))

        self.current_hrv_label = ttk.Label(
            control_frame, text="HRV: -- ms",
            font=('Helvetica', 12)
        )
        self.current_hrv_label.pack(side=tk.LEFT, padx=10)

        # --- グラフ表示エリア ---
        graph_frame = ttk.Frame(main_frame)
        graph_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)
        graph_frame.rowconfigure(1, weight=1)
        graph_frame.rowconfigure(2, weight=1)

        # matplotlib Figure作成（3つのサブプロット）
        self.monitor_fig, (self.hr_ax, self.ecg_ax, self.hrv_ax) = plt.subplots(
            3, 1, figsize=(10, 8), tight_layout=True
        )

        # 心拍数グラフ設定
        self.hr_ax.set_title("心拍数 (Heart Rate)", fontsize=12)
        self.hr_ax.set_ylabel("BPM")
        self.hr_ax.set_xlabel("時間 (秒)")
        self.hr_ax.grid(True, alpha=0.3)
        self.hr_line, = self.hr_ax.plot([], [], 'r-', linewidth=1.5, label='HR')
        self.hr_ax.legend(loc='upper right')

        # ECGグラフ設定
        self.ecg_ax.set_title("心電図 (ECG)", fontsize=12)
        self.ecg_ax.set_ylabel("μV")
        self.ecg_ax.set_xlabel("時間 (秒)")
        self.ecg_ax.grid(True, alpha=0.3)
        self.ecg_line, = self.ecg_ax.plot([], [], 'b-', linewidth=0.8, label='ECG')
        self.ecg_ax.legend(loc='upper right')

        # HRVグラフ設定 (SDNN: RR間隔の標準偏差)
        self.hrv_ax.set_title("心拍変動 (HRV - SDNN)", fontsize=12)
        self.hrv_ax.set_ylabel("ms")
        self.hrv_ax.set_xlabel("時間 (秒)")
        self.hrv_ax.grid(True, alpha=0.3)
        self.hrv_line, = self.hrv_ax.plot([], [], 'g-', linewidth=1.5, label='SDNN')
        self.hrv_ax.legend(loc='upper right')

        # キャンバスをTkinterに埋め込み
        self.monitor_canvas = FigureCanvasTkAgg(self.monitor_fig, master=graph_frame)
        self.monitor_canvas.draw()
        self.monitor_canvas.get_tk_widget().grid(row=0, column=0, rowspan=3, sticky="nsew")

        # データバッファ初期化
        self._init_monitor_data_buffers()

    def _init_monitor_data_buffers(self) -> None:
        """モニター用データバッファの初期化"""
        self.hr_times: List[float] = []
        self.hr_values: List[float] = []
        self.ecg_times: List[float] = []
        self.ecg_values: List[float] = []
        self.hrv_times: List[float] = []
        self.hrv_values: List[float] = []
        self.rr_intervals: List[float] = []  # HRV計算用RR間隔バッファ
        self.monitor_start_time: Optional[float] = None
        self.last_hr_time: Optional[float] = None

    def _toggle_realtime_monitor(self) -> None:
        """リアルタイムモニターの開始/停止を切り替え"""
        if self.monitor_running:
            self._stop_realtime_monitor()
        else:
            self._start_realtime_monitor()

    def _start_realtime_monitor(self) -> None:
        """リアルタイムモニターを開始"""
        if not self.hr_monitor.is_connected and not self.h10_monitor.is_connected:
            messagebox.showwarning(
                "デバイス未接続",
                "Polar Verity SenseまたはH10が接続されていません。\n"
                "会話システムタブでデバイスを接続してください。"
            )
            return

        self.monitor_running = True
        self.monitor_start_btn.config(text="モニター停止")
        self._init_monitor_data_buffers()
        self.monitor_start_time = time.time()

        # グラフクリア
        self.hr_line.set_data([], [])
        self.ecg_line.set_data([], [])
        self.hrv_line.set_data([], [])
        self.monitor_canvas.draw()

        # 更新ループ開始
        self._update_realtime_monitor()

    def _stop_realtime_monitor(self) -> None:
        """リアルタイムモニターを停止"""
        self.monitor_running = False
        self.monitor_start_btn.config(text="モニター開始")

    def _update_realtime_monitor(self) -> None:
        """リアルタイムモニターのデータ更新"""
        if not self.monitor_running:
            return

        current_time = time.time()
        elapsed = current_time - self.monitor_start_time
        window_size = self.monitor_window_var.get()

        # 心拍数データ取得 (Verity SenseまたはH10から)
        hr_value = 0
        if self.hr_monitor.is_connected:
            hr_value = self.hr_monitor.get_current_hr()
        elif self.h10_monitor.is_connected:
            hr_value = self.h10_monitor.get_current_rr()

        if hr_value > 0:
            self.hr_times.append(elapsed)
            self.hr_values.append(hr_value)
            self.current_hr_label.config(text=f"心拍数: {hr_value} bpm")

            # HRからRR間隔を計算 (RR = 60000 / HR ms)
            rr_interval = 60000.0 / hr_value
            self.rr_intervals.append(rr_interval)
            self.last_hr_time = current_time

        # ECGデータ取得 (H10のみ) - 実データをバッファから取得
        if self.h10_monitor.is_connected:
            ecg_data = self.h10_monitor.get_ecg_buffer()
            if ecg_data:
                # バッファ全体を使用（直近5秒分）
                ecg_sample_rate = 130  # Hz
                num_samples = len(ecg_data)
                # 時間軸を生成（現在時刻から逆算）
                self.ecg_times = [elapsed - (num_samples - i - 1) / ecg_sample_rate for i in range(num_samples)]
                self.ecg_values = ecg_data

        # HRV (SDNN) 計算: RR間隔の標準偏差
        if len(self.rr_intervals) >= 2:
            # 直近30拍分のRR間隔でSDNN計算
            recent_rr = self.rr_intervals[-30:]
            if len(recent_rr) >= 2:
                mean_rr = sum(recent_rr) / len(recent_rr)
                squared_diffs = [(rr - mean_rr) ** 2 for rr in recent_rr]
                sdnn = (sum(squared_diffs) / len(squared_diffs)) ** 0.5
                self.hrv_times.append(elapsed)
                self.hrv_values.append(sdnn)
                self.current_hrv_label.config(text=f"HRV: {sdnn:.1f} ms")

        # 古いデータを削除（表示ウィンドウ外）
        min_time = elapsed - window_size
        self._trim_buffer(self.hr_times, self.hr_values, min_time)
        self._trim_buffer(self.ecg_times, self.ecg_values, min_time)
        self._trim_buffer(self.hrv_times, self.hrv_values, min_time)

        # グラフ更新
        self._update_hr_graph(elapsed, window_size)
        self._update_ecg_graph(elapsed)
        self._update_hrv_graph(elapsed, window_size)

        self.monitor_canvas.draw_idle()

        # 次の更新をスケジュール（約7.7ms間隔 = 130Hz）
        if self.monitor_running:
            self.after(8, self._update_realtime_monitor)

    def _update_hr_graph(self, elapsed: float, window_size: int) -> None:
        """心拍数グラフを更新"""
        if self.hr_times:
            self.hr_line.set_data(self.hr_times, self.hr_values)
            self.hr_ax.set_xlim(max(0, elapsed - window_size), elapsed + 1)
            hr_min = min(self.hr_values) - 10 if self.hr_values else 50
            hr_max = max(self.hr_values) + 10 if self.hr_values else 120
            self.hr_ax.set_ylim(hr_min, hr_max)

    def _update_ecg_graph(self, elapsed: float) -> None:
        """ECGグラフを更新"""
        if self.ecg_times and self.ecg_values:
            # ECGデータをそのまま表示（H10バッファから直接取得済み）
            ecg_window = 5
            self.ecg_line.set_data(self.ecg_times, self.ecg_values)
            self.ecg_ax.set_xlim(max(0, elapsed - ecg_window), elapsed + 0.5)
            if self.ecg_values:
                ecg_min = min(self.ecg_values)
                ecg_max = max(self.ecg_values)
                margin = max(100, (ecg_max - ecg_min) * 0.1)
                self.ecg_ax.set_ylim(ecg_min - margin, ecg_max + margin)

    def _update_hrv_graph(self, elapsed: float, window_size: int) -> None:
        """HRVグラフを更新"""
        if self.hrv_times:
            self.hrv_line.set_data(self.hrv_times, self.hrv_values)
            self.hrv_ax.set_xlim(max(0, elapsed - window_size), elapsed + 1)
            hrv_min = max(0, min(self.hrv_values) - 10) if self.hrv_values else 0
            hrv_max = max(self.hrv_values) + 20 if self.hrv_values else 100
            self.hrv_ax.set_ylim(hrv_min, hrv_max)

    def _trim_buffer(self, times: List[float], values: List[float], min_time: float) -> None:
        """バッファから古いデータを削除"""
        while times and times[0] < min_time:
            times.pop(0)
            values.pop(0)
