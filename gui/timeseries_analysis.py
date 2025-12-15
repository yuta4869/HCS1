# gui/timeseries_analysis.py
"""個別時系列解析モジュール

被験者ごとに条件別・全条件統合の時系列グラフを生成する。
入力: ECG/HRV解析で出力された {条件名}_result.xlsx ファイル
出力: HR, RMSSD, SDNN の時系列グラフ画像
"""

import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
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

        # グラフオプション
        self.ts_show_reference_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="基準心拍数ラインを表示",
                        variable=self.ts_show_reference_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        self.ts_show_target_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="目標心拍数ラインを表示",
                        variable=self.ts_show_target_var).grid(row=1, column=2, columnspan=2, sticky="w", padx=5, pady=5)

        self.ts_generate_combined_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="全条件統合グラフも生成",
                        variable=self.ts_generate_combined_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)

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

        # 解析結果ファイルを検索
        result_files = self._ts_find_result_files(input_folder)

        if not result_files:
            self._ts_log("エラー: *_result.xlsx ファイルが見つかりません。")
            messagebox.showerror("エラー", "解析結果ファイル (*_result.xlsx) が見つかりません。")
            return

        self._ts_log(f"検出したファイル数: {len(result_files)}")

        # 条件ごとのデータを読み込み
        condition_data: Dict[str, pd.DataFrame] = {}
        for filepath, condition in result_files:
            self._ts_log(f"読み込み中: {os.path.basename(filepath)} ({condition})")
            try:
                df = pd.read_excel(filepath)
                if 'Time' in df.columns:
                    condition_data[condition] = df
                else:
                    self._ts_log(f"  警告: Time列がありません。スキップします。")
            except Exception as e:
                self._ts_log(f"  エラー: {e}")

        if not condition_data:
            self._ts_log("エラー: 有効なデータが読み込めませんでした。")
            return

        # グラフ生成
        self._ts_log("\n=== グラフ生成 ===")

        # 条件別グラフ（HR, RMSSD, SDNN）
        for condition, df in condition_data.items():
            self._ts_generate_condition_graphs(
                condition, df, output_folder,
                reference_hr, target_hr, show_reference, show_target
            )

        # 全条件統合グラフ
        if generate_combined and len(condition_data) > 1:
            self._ts_generate_combined_graphs(
                condition_data, output_folder,
                reference_hr, target_hr, show_reference, show_target
            )

        self._ts_log("\n=== 解析完了 ===")
        self.ts_progress_label.config(text="完了")
        messagebox.showinfo("完了", "時系列グラフの生成が完了しました。")

    def _ts_find_result_files(self, folder: str) -> List[Tuple[str, str]]:
        """解析結果ファイルを検索して (ファイルパス, 条件名) のリストを返す"""
        result_files = []
        pattern = re.compile(r"(.+)_result\.xlsx$", re.IGNORECASE)

        for filename in os.listdir(folder):
            if filename.endswith("_result.xlsx"):
                match = pattern.match(filename)
                if match:
                    condition = match.group(1)
                    # 標準条件名に正規化
                    condition_normalized = self._ts_normalize_condition(condition)
                    filepath = os.path.join(folder, filename)
                    result_files.append((filepath, condition_normalized))

        return result_files

    def _ts_normalize_condition(self, condition: str) -> str:
        """条件名を正規化"""
        condition_lower = condition.lower()
        for standard_cond in ANALYS_CONDITION_ORDER:
            if condition_lower == standard_cond.lower():
                return standard_cond
        return condition

    def _ts_generate_condition_graphs(
        self,
        condition: str,
        df: pd.DataFrame,
        output_folder: str,
        reference_hr: float,
        target_hr: float,
        show_reference: bool,
        show_target: bool
    ) -> None:
        """条件別のグラフを生成"""
        color = CONDITION_COLORS.get(condition, '#666666')
        time_col = df['Time'].values

        # HR グラフ（HRデータがある場合）
        if 'HR' in df.columns:
            self._ts_plot_single(
                time_col, df['HR'].values,
                f"{condition} - 心拍数 (HR)",
                "Time (sec)", "HR (BPM)",
                color, output_folder, f"{condition}_HR.png",
                reference_hr if show_reference else None,
                target_hr if show_target else None
            )
            self._ts_log(f"  生成: {condition}_HR.png")

        # RMSSD グラフ
        if 'RMSSD' in df.columns:
            self._ts_plot_single(
                time_col, df['RMSSD'].values,
                f"{condition} - RMSSD",
                "Time (sec)", "RMSSD (ms)",
                color, output_folder, f"{condition}_RMSSD.png"
            )
            self._ts_log(f"  生成: {condition}_RMSSD.png")

        # SDNN グラフ
        if 'SDNN' in df.columns:
            self._ts_plot_single(
                time_col, df['SDNN'].values,
                f"{condition} - SDNN",
                "Time (sec)", "SDNN (ms)",
                color, output_folder, f"{condition}_SDNN.png"
            )
            self._ts_log(f"  生成: {condition}_SDNN.png")

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
        target_line: Optional[float] = None
    ) -> None:
        """単一グラフを生成"""
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(time_data, value_data, color=color, linewidth=1.5, label=ylabel)

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
            self._ts_plot_combined(
                condition_data, 'HR',
                "全条件 - 心拍数 (HR)",
                "Time (sec)", "HR (BPM)",
                output_folder, "Combined_HR.png",
                reference_hr if show_reference else None,
                target_hr if show_target else None
            )
            self._ts_log(f"  生成: Combined_HR.png")

        # RMSSD 統合グラフ
        has_rmssd = any('RMSSD' in df.columns for df in condition_data.values())
        if has_rmssd:
            self._ts_plot_combined(
                condition_data, 'RMSSD',
                "全条件 - RMSSD",
                "Time (sec)", "RMSSD (ms)",
                output_folder, "Combined_RMSSD.png"
            )
            self._ts_log(f"  生成: Combined_RMSSD.png")

        # SDNN 統合グラフ
        has_sdnn = any('SDNN' in df.columns for df in condition_data.values())
        if has_sdnn:
            self._ts_plot_combined(
                condition_data, 'SDNN',
                "全条件 - SDNN",
                "Time (sec)", "SDNN (ms)",
                output_folder, "Combined_SDNN.png"
            )
            self._ts_log(f"  生成: Combined_SDNN.png")

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

        # 条件順序に従ってプロット
        for condition in ANALYS_CONDITION_ORDER:
            if condition in condition_data:
                df = condition_data[condition]
                if column in df.columns:
                    color = CONDITION_COLORS.get(condition, '#666666')
                    ax.plot(df['Time'].values, df[column].values,
                            color=color, linewidth=1.5, label=condition)

        # 残りの条件（標準順序にないもの）
        for condition, df in condition_data.items():
            if condition not in ANALYS_CONDITION_ORDER and column in df.columns:
                ax.plot(df['Time'].values, df[column].values,
                        linewidth=1.5, label=condition)

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
