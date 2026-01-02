# gui/timeseries_analysis.py
"""個別時系列解析モジュール

被験者ごとに条件別・全条件統合の時系列グラフを生成する。
入力:
  - ECG/HRV解析で出力された {subject_id}_{条件名}_result.xlsx ファイル
  - 会話システムで出力された h10_hr_session_*.csv / verity_hr_session_*.csv ファイル
出力: HR, RMSSD, SDNN の時系列グラフ画像
"""

import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

from .ecg_analysis import CONDITION_COLORS, ANALYS_CONDITION_ORDER


class TimeseriesAnalysisMixin:
    """個別時系列解析機能を提供するMixinクラス

    Applicationクラスにミックスインして使用する。
    """

    def _setup_timeseries_analysis_tab(self) -> None:
        """個別時系列解析タブのUIを構築"""
        main_frame = self.timeseries_analysis_tab
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # --- 入力フォルダ選択 ---
        input_frame = ttk.LabelFrame(main_frame, text="入力設定", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="解析結果フォルダ:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.ts_input_folder_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.ts_input_folder_var, width=50).grid(
            row=0, column=1, sticky="ew", padx=5, pady=5
        )
        ttk.Button(input_frame, text="参照...", command=self._ts_select_input_folder).grid(
            row=0, column=2, padx=5, pady=5
        )

        ttk.Label(input_frame, text="出力フォルダ:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.ts_output_folder_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.ts_output_folder_var, width=50).grid(
            row=1, column=1, sticky="ew", padx=5, pady=5
        )
        ttk.Button(input_frame, text="参照...", command=self._ts_select_output_folder).grid(
            row=1, column=2, padx=5, pady=5
        )

        # --- パラメータ設定 ---
        param_frame = ttk.LabelFrame(main_frame, text="グラフ設定", padding="10")
        param_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # 基準心拍数
        ttk.Label(param_frame, text="基準心拍数 (BPM):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.ts_reference_hr_var = tk.DoubleVar(value=70.0)
        ttk.Entry(param_frame, textvariable=self.ts_reference_hr_var, width=10).grid(
            row=0, column=1, sticky="w", padx=5, pady=5
        )

        # 目標心拍数
        ttk.Label(param_frame, text="目標心拍数 (BPM):").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.ts_target_hr_var = tk.DoubleVar(value=0.0)
        ttk.Entry(param_frame, textvariable=self.ts_target_hr_var, width=10).grid(
            row=0, column=3, sticky="w", padx=5, pady=5
        )
        ttk.Label(param_frame, text="(0で非表示)").grid(row=0, column=4, sticky="w", padx=5, pady=5)

        # 解析区間
        ttk.Label(param_frame, text="解析区間:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        time_range_frame = ttk.Frame(param_frame)
        time_range_frame.grid(row=1, column=1, columnspan=4, sticky="w", padx=5, pady=5)

        ttk.Label(time_range_frame, text="開始").pack(side=tk.LEFT)
        self.ts_start_time_var = tk.StringVar(value="")
        ttk.Entry(time_range_frame, textvariable=self.ts_start_time_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(time_range_frame, text="秒 〜 終了").pack(side=tk.LEFT)
        self.ts_end_time_var = tk.StringVar(value="")
        ttk.Entry(time_range_frame, textvariable=self.ts_end_time_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(time_range_frame, text="秒 (空欄で全区間)").pack(side=tk.LEFT)

        # グラフオプション
        self.ts_show_reference_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="基準心拍数ラインを表示",
                        variable=self.ts_show_reference_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        self.ts_show_target_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="目標心拍数ラインを表示",
                        variable=self.ts_show_target_var).grid(row=2, column=2, columnspan=2, sticky="w", padx=5, pady=5)

        self.ts_generate_combined_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="全条件統合グラフも生成",
                        variable=self.ts_generate_combined_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        self.ts_show_speech_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="発言時間帯を色付け表示",
                        variable=self.ts_show_speech_var).grid(row=3, column=2, columnspan=2, sticky="w", padx=5, pady=5)

        # --- 実行ボタン ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=10)

        self.ts_run_btn = ttk.Button(btn_frame, text="時系列グラフ生成", command=self._ts_run_analysis)
        self.ts_run_btn.pack(side=tk.LEFT, padx=10)

        self.ts_progress_label = ttk.Label(btn_frame, text="")
        self.ts_progress_label.pack(side=tk.LEFT, padx=10)

        # --- ログ表示エリア ---
        log_frame = ttk.LabelFrame(main_frame, text="ログ", padding="5")
        log_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.ts_log_text = tk.Text(log_frame, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.ts_log_text.grid(row=0, column=0, sticky="nsew")

        ts_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.ts_log_text.yview)
        ts_scrollbar.grid(row=0, column=1, sticky="ns")
        self.ts_log_text.config(yscrollcommand=ts_scrollbar.set)

    def _ts_select_input_folder(self) -> None:
        """入力フォルダを選択"""
        folder = filedialog.askdirectory(title="解析結果フォルダを選択")
        if folder:
            self.ts_input_folder_var.set(folder)
            # 出力フォルダも同じに設定
            if not self.ts_output_folder_var.get():
                self.ts_output_folder_var.set(folder)

    def _ts_select_output_folder(self) -> None:
        """出力フォルダを選択"""
        folder = filedialog.askdirectory(title="出力フォルダを選択")
        if folder:
            self.ts_output_folder_var.set(folder)

    def _ts_log(self, message: str) -> None:
        """ログにメッセージを追加"""
        self.ts_log_text.config(state=tk.NORMAL)
        self.ts_log_text.insert(tk.END, message + "\n")
        self.ts_log_text.see(tk.END)
        self.ts_log_text.config(state=tk.DISABLED)
        self.update_idletasks()

    def _ts_run_analysis(self) -> None:
        """時系列解析を実行"""
        input_folder = self.ts_input_folder_var.get()
        output_folder = self.ts_output_folder_var.get()

        if not input_folder or not os.path.isdir(input_folder):
            messagebox.showerror("エラー", "有効な入力フォルダを選択してください。")
            return

        if not output_folder:
            output_folder = input_folder
            self.ts_output_folder_var.set(output_folder)

        os.makedirs(output_folder, exist_ok=True)

        # ログクリア
        self.ts_log_text.config(state=tk.NORMAL)
        self.ts_log_text.delete(1.0, tk.END)
        self.ts_log_text.config(state=tk.DISABLED)

        self._ts_log(f"=== 時系列解析開始 ===")
        self._ts_log(f"入力フォルダ: {input_folder}")
        self._ts_log(f"出力フォルダ: {output_folder}")

        # パラメータ取得
        reference_hr = self.ts_reference_hr_var.get()
        target_hr = self.ts_target_hr_var.get()
        show_reference = self.ts_show_reference_var.get()
        show_target = self.ts_show_target_var.get() and target_hr > 0
        generate_combined = self.ts_generate_combined_var.get()

        # 解析区間パラメータ取得
        start_time_str = self.ts_start_time_var.get().strip()
        end_time_str = self.ts_end_time_var.get().strip()
        start_time: Optional[float] = None
        end_time: Optional[float] = None

        if start_time_str:
            try:
                start_time = float(start_time_str)
                self._ts_log(f"解析開始時間: {start_time}秒")
            except ValueError:
                messagebox.showerror("エラー", "開始時間には数値を入力してください。")
                return

        if end_time_str:
            try:
                end_time = float(end_time_str)
                self._ts_log(f"解析終了時間: {end_time}秒")
            except ValueError:
                messagebox.showerror("エラー", "終了時間には数値を入力してください。")
                return

        if start_time is not None and end_time is not None and start_time >= end_time:
            messagebox.showerror("エラー", "開始時間は終了時間より小さくしてください。")
            return

        # 解析結果ファイルを検索
        result_files = self._ts_find_result_files(input_folder)

        if not result_files:
            self._ts_log("エラー: *_result.xlsx ファイルが見つかりません。")
            messagebox.showerror("エラー", "解析結果ファイル (*_result.xlsx) が見つかりません。")
            return

        self._ts_log(f"検出したファイル数: {len(result_files)}")

        # 被験者IDを取得（最初のファイルから）
        subject_id = result_files[0][1] if result_files else "Unknown"
        self._ts_log(f"被験者ID: {subject_id}")

        # 条件ごとのデータを読み込み
        condition_data: Dict[str, pd.DataFrame] = {}
        for filepath, subj_id, condition in result_files:
            self._ts_log(f"読み込み中: {os.path.basename(filepath)} ({condition})")
            try:
                df = pd.read_excel(filepath)
                if 'Time' in df.columns:
                    # 時間範囲でフィルタリング
                    df_filtered = self._ts_filter_by_time(df, 'Time', start_time, end_time)
                    if len(df_filtered) > 0:
                        condition_data[condition] = df_filtered
                        self._ts_log(f"  データ点数: {len(df_filtered)} (元: {len(df)})")
                    else:
                        self._ts_log(f"  警告: フィルタリング後のデータが空です。スキップします。")
                else:
                    self._ts_log(f"  警告: Time列がありません。スキップします。")
            except Exception as e:
                self._ts_log(f"  エラー: {e}")

        if not condition_data:
            self._ts_log("エラー: 有効なデータが読み込めませんでした。")
            return

        # 発言時間帯の色付けオプション
        show_speech = self.ts_show_speech_var.get()

        # グラフ生成
        self._ts_log("\n=== グラフ生成 ===")

        # 条件別グラフ（HR, RMSSD, SDNN）
        for condition, df in condition_data.items():
            # 発言時間帯を取得
            speech_intervals = None
            if show_speech:
                conv_log_path = self._ts_find_conversation_log(input_folder, subject_id, condition)
                if conv_log_path:
                    self._ts_log(f"  発言ログ: {os.path.basename(conv_log_path)}")
                    speech_intervals = self._ts_load_speech_intervals(conv_log_path)
                    self._ts_log(f"    -> {len(speech_intervals)}件の発言を検出")

            self._ts_generate_condition_graphs(
                subject_id, condition, df, output_folder,
                reference_hr, target_hr, show_reference, show_target,
                speech_intervals=speech_intervals
            )

        # 全条件統合グラフ
        if generate_combined and len(condition_data) > 1:
            self._ts_generate_combined_graphs(
                subject_id, condition_data, output_folder,
                reference_hr, target_hr, show_reference, show_target
            )

        # H10/Verity HRセッションファイルからのHRグラフ生成
        self._ts_log("\n=== HRセッションファイル処理 ===")
        hr_session_files = self._ts_find_hr_session_files(input_folder)
        if hr_session_files:
            self._ts_log(f"検出したHRセッションファイル数: {len(hr_session_files)}")
            self._ts_generate_hr_session_graphs(
                hr_session_files, output_folder,
                reference_hr, target_hr, show_reference, show_target,
                show_speech=show_speech,
                start_time=start_time, end_time=end_time
            )
        else:
            self._ts_log("HRセッションファイル (h10_hr_session_*.csv, verity_hr_session_*.csv) は見つかりませんでした。")

        self._ts_log("\n=== 解析完了 ===")
        self.ts_progress_label.config(text="完了")
        messagebox.showinfo("完了", "時系列グラフの生成が完了しました。")

    def _ts_find_result_files(self, folder: str) -> List[Tuple[str, str, str]]:
        """解析結果ファイルを検索して (ファイルパス, 被験者ID, 条件名) のリストを返す"""
        result_files = []
        # パターン1: {subject_id}_{condition}_result.xlsx (新形式)
        pattern_new = re.compile(r"(.+?)_(.+)_result\.xlsx$", re.IGNORECASE)
        # パターン2: {condition}_result.xlsx (旧形式)
        pattern_old = re.compile(r"(.+)_result\.xlsx$", re.IGNORECASE)

        for filename in os.listdir(folder):
            if filename.endswith("_result.xlsx") and not filename.startswith("Combined"):
                filepath = os.path.join(folder, filename)

                # 新形式を試す
                match_new = pattern_new.match(filename)
                if match_new:
                    subject_id = match_new.group(1)
                    condition = match_new.group(2)
                    # 条件名を正規化
                    condition_normalized = self._ts_normalize_condition(condition)
                    result_files.append((filepath, subject_id, condition_normalized))
                else:
                    # 旧形式（条件名のみ）
                    match_old = pattern_old.match(filename)
                    if match_old:
                        condition = match_old.group(1)
                        condition_normalized = self._ts_normalize_condition(condition)
                        # 被験者IDはフォルダ名から推測
                        folder_name = os.path.basename(folder.rstrip('/\\'))
                        result_files.append((filepath, folder_name, condition_normalized))

        return result_files

    def _ts_normalize_condition(self, condition: str) -> str:
        """条件名を正規化"""
        condition_lower = condition.lower()
        for standard_cond in ANALYS_CONDITION_ORDER:
            if condition_lower == standard_cond.lower():
                return standard_cond
        return condition

    def _ts_filter_by_time(
        self,
        df: pd.DataFrame,
        time_column: str,
        start_time: Optional[float],
        end_time: Optional[float]
    ) -> pd.DataFrame:
        """DataFrameを時間範囲でフィルタリング

        Args:
            df: 元のDataFrame
            time_column: 時間列の名前
            start_time: 開始時間（秒）。Noneの場合は最初から
            end_time: 終了時間（秒）。Noneの場合は最後まで

        Returns:
            フィルタリング後のDataFrame
        """
        if start_time is None and end_time is None:
            return df

        mask = pd.Series([True] * len(df))
        if start_time is not None:
            mask &= df[time_column] >= start_time
        if end_time is not None:
            mask &= df[time_column] <= end_time

        return df[mask].copy()

    def _ts_generate_condition_graphs(
        self,
        subject_id: str,
        condition: str,
        df: pd.DataFrame,
        output_folder: str,
        reference_hr: float,
        target_hr: float,
        show_reference: bool,
        show_target: bool,
        speech_intervals: Optional[List[Tuple[float, float, str]]] = None
    ) -> None:
        """条件別のグラフを生成"""
        color = CONDITION_COLORS.get(condition, '#666666')
        time_col = df['Time'].values

        # HR グラフ（HRデータがある場合）
        if 'HR' in df.columns:
            filename = f"{subject_id}_{condition}_HR.png"
            self._ts_plot_single(
                time_col, df['HR'].values,
                f"{subject_id} {condition} - 心拍数 (HR)",
                "Time (sec)", "HR (BPM)",
                color, output_folder, filename,
                reference_hr if show_reference else None,
                target_hr if show_target else None,
                speech_intervals=speech_intervals
            )
            self._ts_log(f"  生成: {filename}")

        # RMSSD グラフ
        if 'RMSSD' in df.columns:
            filename = f"{subject_id}_{condition}_RMSSD.png"
            self._ts_plot_single(
                time_col, df['RMSSD'].values,
                f"{subject_id} {condition} - RMSSD",
                "Time (sec)", "RMSSD (ms)",
                color, output_folder, filename,
                speech_intervals=speech_intervals
            )
            self._ts_log(f"  生成: {filename}")

        # SDNN グラフ
        if 'SDNN' in df.columns:
            filename = f"{subject_id}_{condition}_SDNN.png"
            self._ts_plot_single(
                time_col, df['SDNN'].values,
                f"{subject_id} {condition} - SDNN",
                "Time (sec)", "SDNN (ms)",
                color, output_folder, filename,
                speech_intervals=speech_intervals
            )
            self._ts_log(f"  生成: {filename}")

    def _ts_plot_single(
        self,
        time_data,
        value_data,
        title: str,
        xlabel: str,
        ylabel: str,
        color: str,
        output_folder: str,
        filename: str,
        reference_line: Optional[float] = None,
        target_line: Optional[float] = None,
        speech_intervals: Optional[List[Tuple[float, float, str]]] = None
    ) -> None:
        """単一グラフを生成"""
        fig, ax = plt.subplots(figsize=(10, 5))

        # 発言時間帯を先にプロット（データラインの後ろに表示）
        if speech_intervals:
            y_min = np.nanmin(value_data) if len(value_data) > 0 else 0
            y_max = np.nanmax(value_data) if len(value_data) > 0 else 100
            self._ts_plot_speech_intervals(ax, speech_intervals, y_min, y_max)

        ax.plot(time_data, value_data, color=color, linewidth=1.5, label=ylabel)

        # X軸範囲をデータの範囲に設定
        if len(time_data) > 0:
            ax.set_xlim(np.nanmin(time_data), np.nanmax(time_data))

        # 基準心拍数ライン
        if reference_line is not None and reference_line > 0:
            ax.axhline(y=reference_line, color='green', linestyle='--',
                       linewidth=2, label=f'基準HR ({reference_line:.0f} BPM)')

        # 目標心拍数ライン
        if target_line is not None and target_line > 0:
            ax.axhline(y=target_line, color='blue', linestyle='-.',
                       linewidth=2, label=f'目標HR ({target_line:.0f} BPM)')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        plt.tight_layout()
        output_path = os.path.join(output_folder, filename)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _ts_generate_combined_graphs(
        self,
        subject_id: str,
        condition_data: Dict[str, pd.DataFrame],
        output_folder: str,
        reference_hr: float,
        target_hr: float,
        show_reference: bool,
        show_target: bool
    ) -> None:
        """全条件統合グラフを生成"""
        self._ts_log("\n--- 全条件統合グラフ ---")

        # HR 統合グラフ
        has_hr = any('HR' in df.columns for df in condition_data.values())
        if has_hr:
            filename = f"{subject_id}_Combined_HR.png"
            self._ts_plot_combined(
                condition_data, 'HR',
                f"{subject_id} 全条件 - 心拍数 (HR)",
                "Time (sec)", "HR (BPM)",
                output_folder, filename,
                reference_hr if show_reference else None,
                target_hr if show_target else None
            )
            self._ts_log(f"  生成: {filename}")

        # RMSSD 統合グラフ
        has_rmssd = any('RMSSD' in df.columns for df in condition_data.values())
        if has_rmssd:
            filename = f"{subject_id}_Combined_RMSSD.png"
            self._ts_plot_combined(
                condition_data, 'RMSSD',
                f"{subject_id} 全条件 - RMSSD",
                "Time (sec)", "RMSSD (ms)",
                output_folder, filename
            )
            self._ts_log(f"  生成: {filename}")

        # SDNN 統合グラフ
        has_sdnn = any('SDNN' in df.columns for df in condition_data.values())
        if has_sdnn:
            filename = f"{subject_id}_Combined_SDNN.png"
            self._ts_plot_combined(
                condition_data, 'SDNN',
                f"{subject_id} 全条件 - SDNN",
                "Time (sec)", "SDNN (ms)",
                output_folder, filename
            )
            self._ts_log(f"  生成: {filename}")

    def _ts_plot_combined(
        self,
        condition_data: Dict[str, pd.DataFrame],
        column: str,
        title: str,
        xlabel: str,
        ylabel: str,
        output_folder: str,
        filename: str,
        reference_line: Optional[float] = None,
        target_line: Optional[float] = None
    ) -> None:
        """統合グラフを生成"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 全条件の時間範囲を収集
        all_times = []

        # 条件順序に従ってプロット
        for condition in ANALYS_CONDITION_ORDER:
            if condition in condition_data:
                df = condition_data[condition]
                if column in df.columns:
                    color = CONDITION_COLORS.get(condition, '#666666')
                    time_vals = df['Time'].values
                    ax.plot(time_vals, df[column].values,
                            color=color, linewidth=1.5, label=condition)
                    all_times.extend(time_vals)

        # 残りの条件（標準順序にないもの）
        for condition, df in condition_data.items():
            if condition not in ANALYS_CONDITION_ORDER and column in df.columns:
                time_vals = df['Time'].values
                ax.plot(time_vals, df[column].values,
                        linewidth=1.5, label=condition)
                all_times.extend(time_vals)

        # X軸範囲をデータの範囲に設定
        if all_times:
            ax.set_xlim(np.nanmin(all_times), np.nanmax(all_times))

        # 基準心拍数ライン
        if reference_line is not None and reference_line > 0:
            ax.axhline(y=reference_line, color='green', linestyle='--',
                       linewidth=2, label=f'基準HR ({reference_line:.0f})')

        # 目標心拍数ライン
        if target_line is not None and target_line > 0:
            ax.axhline(y=target_line, color='blue', linestyle='-.',
                       linewidth=2, label=f'目標HR ({target_line:.0f})')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', ncol=2)

        plt.tight_layout()
        output_path = os.path.join(output_folder, filename)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _ts_find_hr_session_files(self, folder: str) -> List[Tuple[str, str, str, str, str]]:
        """HRセッションファイルを検索

        Returns:
            List of tuples: (filepath, device_type, subject_id, timestamp, mode)
            device_type: 'H10' or 'Verity'
        """
        hr_files = []

        # パターン: h10_hr_session_{subject_id}_{timestamp}_{mode}.csv
        # パターン: verity_hr_session_{subject_id}_{timestamp}_{mode}.csv
        h10_pattern = re.compile(
            r"h10_hr_session_(.+?)_(\d{8}_\d{6})_(.+)\.csv$",
            re.IGNORECASE
        )
        verity_pattern = re.compile(
            r"verity_hr_session_(.+?)_(\d{8}_\d{6})_(.+)\.csv$",
            re.IGNORECASE
        )

        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)

            # H10ファイルをチェック
            match = h10_pattern.match(filename)
            if match:
                subject_id = match.group(1)
                timestamp = match.group(2)
                mode = match.group(3)
                hr_files.append((filepath, 'H10', subject_id, timestamp, mode))
                continue

            # Verityファイルをチェック
            match = verity_pattern.match(filename)
            if match:
                subject_id = match.group(1)
                timestamp = match.group(2)
                mode = match.group(3)
                hr_files.append((filepath, 'Verity', subject_id, timestamp, mode))

        return hr_files

    def _ts_generate_hr_session_graphs(
        self,
        hr_files: List[Tuple[str, str, str, str, str]],
        output_folder: str,
        reference_hr: float,
        target_hr: float,
        show_reference: bool,
        show_target: bool,
        show_speech: bool = False,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> None:
        """HRセッションファイルからHRグラフを生成"""
        for filepath, device_type, subject_id, timestamp, mode in hr_files:
            self._ts_log(f"読み込み中: {os.path.basename(filepath)}")
            try:
                df = pd.read_csv(filepath)

                # カラム名を確認
                # 期待: Timestamp, Heart Rate (BPM)
                if 'Heart Rate (BPM)' not in df.columns:
                    self._ts_log(f"  警告: 'Heart Rate (BPM)'列がありません。スキップします。")
                    continue

                if 'Timestamp' not in df.columns:
                    self._ts_log(f"  警告: 'Timestamp'列がありません。スキップします。")
                    continue

                # タイムスタンプから経過時間（秒）を計算
                try:
                    timestamps = pd.to_datetime(df['Timestamp'])
                    data_start_time = timestamps.iloc[0]
                    elapsed_seconds = (timestamps - data_start_time).dt.total_seconds().values
                except Exception as e:
                    self._ts_log(f"  警告: タイムスタンプ解析失敗: {e}")
                    # インデックスを使用
                    elapsed_seconds = np.arange(len(df))
                    data_start_time = None

                hr_values = df['Heart Rate (BPM)'].values

                # 時間範囲でフィルタリング
                if start_time is not None or end_time is not None:
                    mask = np.ones(len(elapsed_seconds), dtype=bool)
                    if start_time is not None:
                        mask &= elapsed_seconds >= start_time
                    if end_time is not None:
                        mask &= elapsed_seconds <= end_time
                    elapsed_seconds = elapsed_seconds[mask]
                    hr_values = hr_values[mask]
                    self._ts_log(f"  データ点数: {len(elapsed_seconds)} (フィルタリング後)")

                    if len(elapsed_seconds) == 0:
                        self._ts_log(f"  警告: フィルタリング後のデータが空です。スキップします。")
                        continue

                # 条件名（mode）を正規化
                mode_normalized = self._ts_normalize_condition(mode)

                # 発言時間帯を取得
                speech_intervals = None
                if show_speech:
                    input_folder = os.path.dirname(filepath)
                    conv_log_path = self._ts_find_conversation_log(input_folder, subject_id, mode_normalized)
                    if conv_log_path:
                        self._ts_log(f"  発言ログ: {os.path.basename(conv_log_path)}")
                        speech_intervals = self._ts_load_speech_intervals(conv_log_path, data_start_time)
                        self._ts_log(f"    -> {len(speech_intervals)}件の発言を検出")

                # グラフ生成
                color = CONDITION_COLORS.get(mode_normalized, '#666666')
                filename = f"{subject_id}_{mode_normalized}_{device_type}_HR.png"
                title = f"{subject_id} {mode_normalized} - 心拍数 ({device_type})"

                self._ts_plot_single(
                    elapsed_seconds, hr_values,
                    title,
                    "Time (sec)", "HR (BPM)",
                    color, output_folder, filename,
                    reference_hr if show_reference else None,
                    target_hr if show_target else None,
                    speech_intervals=speech_intervals
                )
                self._ts_log(f"  生成: {filename}")

            except Exception as e:
                self._ts_log(f"  エラー: {e}")

    def _ts_find_conversation_log(
        self,
        folder: str,
        subject_id: str,
        condition: str
    ) -> Optional[str]:
        """対応するconversation_logファイルを検索

        Args:
            folder: 検索対象フォルダ
            subject_id: 被験者ID (例: "No1")
            condition: 条件名 (例: "Fixed", "HRF2_Robust")

        Returns:
            見つかった場合はファイルパス、見つからない場合はNone
        """
        # パターン: conversation_log_{subject_id}_{timestamp}_{condition}.csv
        pattern = re.compile(
            rf"conversation_log_{re.escape(subject_id)}_\d{{8}}_\d{{6}}_{re.escape(condition)}\.csv$",
            re.IGNORECASE
        )

        for filename in os.listdir(folder):
            if pattern.match(filename):
                return os.path.join(folder, filename)

        return None

    def _ts_load_speech_intervals(
        self,
        conversation_log_path: str,
        data_start_time: Optional[datetime] = None
    ) -> List[Tuple[float, float, str]]:
        """conversation_logから発言時間帯を読み込む

        Args:
            conversation_log_path: conversation_logファイルのパス
            data_start_time: データの開始時刻（指定しない場合は最初の発言から計算）

        Returns:
            List of (start_sec, end_sec, role) - role is 'user' or 'assistant'
        """
        try:
            df = pd.read_csv(conversation_log_path, encoding='utf-8')
        except Exception as e:
            self._ts_log(f"  conversation_log読み込みエラー: {e}")
            return []

        required_cols = ['start_time', 'end_time', 'role']
        if not all(col in df.columns for col in required_cols):
            self._ts_log(f"  警告: conversation_logに必要な列がありません")
            return []

        intervals = []

        # データ開始時刻を特定（最初の発言のstart_timeを使用）
        try:
            first_start = pd.to_datetime(df['start_time'].iloc[0])
            if data_start_time is None:
                data_start_time = first_start
        except Exception:
            self._ts_log(f"  警告: start_timeの解析に失敗")
            return []

        for _, row in df.iterrows():
            try:
                start_dt = pd.to_datetime(row['start_time'])
                end_dt = pd.to_datetime(row['end_time'])
                role = row['role'].lower()

                # 開始時刻からの経過秒数に変換
                start_sec = (start_dt - data_start_time).total_seconds()
                end_sec = (end_dt - data_start_time).total_seconds()

                # 負の値はスキップ
                if start_sec < 0:
                    continue

                intervals.append((start_sec, end_sec, role))
            except Exception:
                continue

        return intervals

    def _ts_plot_speech_intervals(
        self,
        ax,
        intervals: List[Tuple[float, float, str]],
        y_min: float,
        y_max: float
    ) -> None:
        """発言時間帯を色付けしてプロット

        Args:
            ax: matplotlib axes
            intervals: List of (start_sec, end_sec, role)
            y_min: y軸の最小値
            y_max: y軸の最大値
        """
        # 色設定
        USER_COLOR = '#FFCCCC'  # 薄い赤
        ASSISTANT_COLOR = '#CCCCFF'  # 薄い青

        user_plotted = False
        assistant_plotted = False

        for start_sec, end_sec, role in intervals:
            if role == 'user':
                color = USER_COLOR
                if not user_plotted:
                    ax.axvspan(start_sec, end_sec, alpha=0.5, color=color, label='人の発言')
                    user_plotted = True
                else:
                    ax.axvspan(start_sec, end_sec, alpha=0.5, color=color)
            elif role == 'assistant':
                color = ASSISTANT_COLOR
                if not assistant_plotted:
                    ax.axvspan(start_sec, end_sec, alpha=0.5, color=color, label='ボットの発言')
                    assistant_plotted = True
                else:
                    ax.axvspan(start_sec, end_sec, alpha=0.5, color=color)
