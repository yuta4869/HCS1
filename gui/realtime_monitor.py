# gui/realtime_monitor.py
"""
リアルタイムモニター機能
心拍数・ECG・HRVのリアルタイム表示を行うモジュール
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
import datetime
import csv
import os
from typing import List, Optional, Tuple, TYPE_CHECKING

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config

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
            control_frame, text="HRV(SDNN): -- ms",
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
        self.hrv_ax.set_title("心拍変動 (HRV - SDNN from ECG)", fontsize=12)
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
        # ECGからのRR間隔検出用（130Hzリアルタイム検出）
        self.rr_intervals_from_ecg: List[Tuple[float, float]] = []  # (timestamp, rr_ms)
        self.monitor_start_time: Optional[float] = None
        self.last_hr_time: Optional[float] = None
        self.last_hrv_update_time: Optional[float] = None
        # R波検出用の状態
        self.last_r_peak_time: Optional[float] = None  # 最後に検出したR波の時刻（秒）
        # SDNN CSV保存用
        self.sdnn_csv_filepath: Optional[str] = None
        self.sdnn_csv_file = None
        self.sdnn_csv_writer = None

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

        # グラフクリア＆初期範囲設定
        self.hr_line.set_data([], [])
        self.ecg_line.set_data([], [])
        self.hrv_line.set_data([], [])

        # 初期X軸範囲を0から設定
        window_size = self.monitor_window_var.get()
        self.hr_ax.set_xlim(0, window_size)
        self.hr_ax.set_ylim(50, 120)
        self.ecg_ax.set_xlim(0, 5)
        self.ecg_ax.set_ylim(-500, 500)
        self.hrv_ax.set_xlim(0, window_size)
        self.hrv_ax.set_ylim(0, 100)

        self.monitor_canvas.draw()

        # SDNN CSV初期化
        self._init_sdnn_csv()

        # 更新ループ開始
        self._update_realtime_monitor()

    def _init_sdnn_csv(self) -> None:
        """SDNN CSV保存用ファイルを初期化"""
        try:
            # ログディレクトリ確認
            os.makedirs(config.LOG_DIR, exist_ok=True)

            # subject_idを取得（Applicationクラスから）
            subject_id = getattr(self, 'subject_id_entry', None)
            if subject_id:
                subject_id = subject_id.get().strip() or "unknown"
            else:
                subject_id = "unknown"

            # タイムスタンプ
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # ファイルパス生成
            self.sdnn_csv_filepath = config.H10_SDNN_SESSION_CSV_TEMPLATE.format(
                subject_id=subject_id,
                session_timestamp=timestamp,
                mode="monitor"
            )

            # CSVファイルを開く
            self.sdnn_csv_file = open(self.sdnn_csv_filepath, 'w', newline='', encoding='utf-8')
            self.sdnn_csv_writer = csv.writer(self.sdnn_csv_file)
            self.sdnn_csv_writer.writerow(["Timestamp", "Elapsed (sec)", "SDNN (ms)", "RR Count"])
            print(f"SDNN CSV log ready: {self.sdnn_csv_filepath}")
        except Exception as e:
            print(f"Failed to initialize SDNN CSV: {e}")
            self.sdnn_csv_filepath = None
            self.sdnn_csv_file = None
            self.sdnn_csv_writer = None

    def _close_sdnn_csv(self) -> None:
        """SDNN CSVファイルを閉じる"""
        if self.sdnn_csv_file:
            try:
                self.sdnn_csv_file.close()
                print(f"SDNN CSV log closed: {self.sdnn_csv_filepath}")
            except Exception as e:
                print(f"Error closing SDNN CSV: {e}")
            finally:
                self.sdnn_csv_file = None
                self.sdnn_csv_writer = None
                self.sdnn_csv_filepath = None

    def _stop_realtime_monitor(self) -> None:
        """リアルタイムモニターを停止"""
        self.monitor_running = False
        self.monitor_start_btn.config(text="モニター開始")
        self._close_sdnn_csv()

    def _detect_r_peaks_realtime(self, ecg_data: List[float], elapsed: float, sample_rate: int = 130) -> None:
        """ECGデータからR波ピークをリアルタイム検出し、RR間隔をバッファに追加"""
        if len(ecg_data) < sample_rate // 2:  # 最低0.5秒分のデータが必要
            return

        # 動的閾値: データの標準偏差を使用
        mean_val = sum(ecg_data) / len(ecg_data)
        std_val = (sum((x - mean_val) ** 2 for x in ecg_data) / len(ecg_data)) ** 0.5

        # R波は正の方向（上向き）に大きいピーク
        # 閾値を平均から1.5標準偏差上に設定
        threshold = mean_val + 1.5 * std_val

        min_rr_sec = 0.3  # 最小RR間隔 300ms（200 bpm相当）
        num_samples = len(ecg_data)

        # R波検出（正のピーク）
        for i in range(1, num_samples - 1):
            if ecg_data[i] > threshold:
                if ecg_data[i] > ecg_data[i-1] and ecg_data[i] > ecg_data[i+1]:
                    # このピークの絶対時刻を計算
                    peak_time = elapsed - (num_samples - i - 1) / sample_rate

                    # 最小距離をチェック（時刻ベース）
                    if self.last_r_peak_time is None or (peak_time - self.last_r_peak_time) >= min_rr_sec:
                        # RR間隔を計算
                        if self.last_r_peak_time is not None:
                            rr_ms = (peak_time - self.last_r_peak_time) * 1000  # msに変換

                            # 妥当なRR間隔のみ追加（300ms-2000ms = 30-200 bpm）
                            if 300 <= rr_ms <= 2000:
                                self.rr_intervals_from_ecg.append((peak_time, rr_ms))

                        self.last_r_peak_time = peak_time

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
            # 前回と同じ時間でなければ追加
            if not self.hr_times or elapsed - self.hr_times[-1] >= 0.5:
                self.hr_times.append(elapsed)
                self.hr_values.append(hr_value)
            self.current_hr_label.config(text=f"心拍数: {hr_value} bpm")

        # ECGデータ取得 (H10のみ) - 実データをバッファから取得
        if self.h10_monitor.is_connected:
            ecg_data = self.h10_monitor.get_ecg_buffer()
            if ecg_data:
                ecg_sample_rate = 130  # Hz
                num_samples = len(ecg_data)

                # 時間軸を生成（現在時刻から逆算）
                self.ecg_times = [elapsed - (num_samples - i - 1) / ecg_sample_rate for i in range(num_samples)]
                self.ecg_values = ecg_data

                # 130HzでリアルタイムR波検出
                self._detect_r_peaks_realtime(ecg_data, elapsed, ecg_sample_rate)

        # HRV (SDNN) 計算: 1秒ごとに更新、30秒バッファ
        if self.last_hrv_update_time is None or (current_time - self.last_hrv_update_time) >= 1.0:
            self.last_hrv_update_time = current_time

            # 30秒以内のRR間隔を取得
            rr_buffer_duration = 30.0  # 30秒
            cutoff_time = elapsed - rr_buffer_duration

            # 古いRR間隔を削除
            self.rr_intervals_from_ecg = [
                (t, rr) for t, rr in self.rr_intervals_from_ecg if t > cutoff_time
            ]

            # SDNNを計算
            if len(self.rr_intervals_from_ecg) >= 2:
                rr_values = [rr for _, rr in self.rr_intervals_from_ecg]
                mean_rr = sum(rr_values) / len(rr_values)
                squared_diffs = [(rr - mean_rr) ** 2 for rr in rr_values]
                sdnn = (sum(squared_diffs) / len(squared_diffs)) ** 0.5

                self.hrv_times.append(elapsed)
                self.hrv_values.append(sdnn)
                self.current_hrv_label.config(text=f"HRV(SDNN): {sdnn:.1f} ms")

                # CSVに保存
                if self.sdnn_csv_writer:
                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self.sdnn_csv_writer.writerow([timestamp_str, f"{elapsed:.2f}", f"{sdnn:.2f}", len(rr_values)])
                    self.sdnn_csv_file.flush()

        # 古いデータを削除（表示ウィンドウ外）
        min_time = elapsed - window_size
        self._trim_buffer(self.hr_times, self.hr_values, min_time)
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
