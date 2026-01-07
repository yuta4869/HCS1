# gui/advanced_analysis_tab.py
"""高度解析タブのUIと機能を提供するMixinクラス"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np

from .advanced_analysis import (
    AdvancedHRVAnalyzer, AdvancedHRVMetrics,
    StatisticalAnalyzer, ANOVAResult, FriedmanResult,
    CorrelationAnalyzer, CorrelationResult
)


class AdvancedAnalysisMixin:
    """高度解析タブ機能を提供するMixinクラス"""

    def _setup_advanced_analysis_tab(self) -> None:
        """高度解析タブのUIを構築"""
        main_frame = self.advanced_analysis_tab
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # --- 解析タイプ選択 ---
        type_frame = ttk.LabelFrame(main_frame, text="解析タイプ", padding="10")
        type_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.adv_analysis_type_var = tk.StringVar(value="advanced_hrv")
        ttk.Radiobutton(
            type_frame, text="高度HRV解析（非線形指標）",
            variable=self.adv_analysis_type_var, value="advanced_hrv",
            command=self._adv_switch_analysis_type
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Radiobutton(
            type_frame, text="統計検定（ANOVA/Friedman）",
            variable=self.adv_analysis_type_var, value="statistics",
            command=self._adv_switch_analysis_type
        ).grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Radiobutton(
            type_frame, text="相関分析",
            variable=self.adv_analysis_type_var, value="correlation",
            command=self._adv_switch_analysis_type
        ).grid(row=0, column=2, padx=10, pady=5, sticky="w")

        # --- 解析パネル用コンテナ ---
        self.adv_panel_container = ttk.Frame(main_frame)
        self.adv_panel_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.adv_panel_container.columnconfigure(0, weight=1)
        self.adv_panel_container.rowconfigure(0, weight=1)

        # 各パネルを作成
        self._create_advanced_hrv_panel()
        self._create_statistics_panel()
        self._create_correlation_panel()

        # 初期表示
        self._adv_switch_analysis_type()

    def _adv_switch_analysis_type(self) -> None:
        """解析タイプに応じてパネルを切り替え"""
        analysis_type = self.adv_analysis_type_var.get()

        # 全パネルを非表示
        self.adv_hrv_panel.grid_remove()
        self.adv_stats_panel.grid_remove()
        self.adv_corr_panel.grid_remove()

        # 選択されたパネルを表示
        if analysis_type == "advanced_hrv":
            self.adv_hrv_panel.grid(row=0, column=0, sticky="nsew")
        elif analysis_type == "statistics":
            self.adv_stats_panel.grid(row=0, column=0, sticky="nsew")
        elif analysis_type == "correlation":
            self.adv_corr_panel.grid(row=0, column=0, sticky="nsew")

    # =========================================================================
    # 高度HRV解析パネル
    # =========================================================================

    def _create_advanced_hrv_panel(self) -> None:
        """高度HRV解析パネルを作成"""
        self.adv_hrv_panel = ttk.Frame(self.adv_panel_container)
        self.adv_hrv_panel.columnconfigure(0, weight=1)
        self.adv_hrv_panel.rowconfigure(2, weight=1)

        # 入力設定
        input_frame = ttk.LabelFrame(self.adv_hrv_panel, text="入力データ", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="RRデータファイル:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.adv_hrv_file_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.adv_hrv_file_var, width=50).grid(
            row=0, column=1, sticky="ew", padx=5, pady=5
        )
        ttk.Button(input_frame, text="参照...", command=self._adv_hrv_browse_file).grid(
            row=0, column=2, padx=5, pady=5
        )

        ttk.Label(input_frame, text="RR列名:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.adv_hrv_rr_col_var = tk.StringVar(value="RR")
        ttk.Entry(input_frame, textvariable=self.adv_hrv_rr_col_var, width=20).grid(
            row=1, column=1, sticky="w", padx=5, pady=5
        )
        ttk.Label(input_frame, text="（CSVの場合はRR間隔のミリ秒データを含む列名）").grid(
            row=1, column=2, sticky="w", padx=5, pady=5
        )

        # オプション
        option_frame = ttk.LabelFrame(self.adv_hrv_panel, text="オプション", padding="10")
        option_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.adv_hrv_nonlinear_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            option_frame, text="非線形指標を計算（Sample Entropy, DFA）",
            variable=self.adv_hrv_nonlinear_var
        ).grid(row=0, column=0, sticky="w", padx=5, pady=5)

        ttk.Button(option_frame, text="解析実行", command=self._adv_hrv_run_analysis).grid(
            row=0, column=1, padx=20, pady=5
        )

        # 結果表示
        result_frame = ttk.LabelFrame(self.adv_hrv_panel, text="解析結果", padding="10")
        result_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.adv_hrv_result_text = tk.Text(result_frame, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.adv_hrv_result_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.adv_hrv_result_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.adv_hrv_result_text.config(yscrollcommand=scrollbar.set)

    def _adv_hrv_browse_file(self) -> None:
        """RRデータファイルを選択"""
        filepath = filedialog.askopenfilename(
            title="RRデータファイルを選択",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filepath:
            self.adv_hrv_file_var.set(filepath)

    def _adv_hrv_run_analysis(self) -> None:
        """高度HRV解析を実行"""
        filepath = self.adv_hrv_file_var.get()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror("エラー", "有効なファイルを選択してください。")
            return

        rr_col = self.adv_hrv_rr_col_var.get().strip()
        compute_nonlinear = self.adv_hrv_nonlinear_var.get()

        try:
            # データ読み込み
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)

            if rr_col not in df.columns:
                messagebox.showerror("エラー", f"'{rr_col}' 列が見つかりません。")
                return

            rr_intervals = df[rr_col].dropna().values

            # 解析実行
            analyzer = AdvancedHRVAnalyzer()
            metrics = analyzer.calculate_metrics(rr_intervals, compute_nonlinear)

            # 結果表示
            self._adv_hrv_display_results(metrics)

            # CSV出力
            output_path = Path(filepath).with_suffix('_advanced_hrv.csv')
            metrics_df = pd.DataFrame([metrics.to_dict()])
            metrics_df.to_csv(output_path, index=False, encoding='utf-8-sig')

            messagebox.showinfo("完了", f"解析結果を保存しました:\n{output_path}")

        except Exception as e:
            messagebox.showerror("エラー", f"解析に失敗しました:\n{e}")

    def _adv_hrv_display_results(self, metrics: AdvancedHRVMetrics) -> None:
        """HRV解析結果を表示"""
        self.adv_hrv_result_text.config(state=tk.NORMAL)
        self.adv_hrv_result_text.delete(1.0, tk.END)

        text = "=== 高度HRV解析結果 ===\n\n"

        text += "【時間領域指標】\n"
        text += f"  Mean HR: {metrics.mean_hr:.1f} BPM\n"
        text += f"  Mean RR: {metrics.mean_rr:.1f} ms\n"
        text += f"  SDNN: {metrics.sdnn:.2f} ms\n"
        text += f"  RMSSD: {metrics.rmssd:.2f} ms\n"
        text += f"  pNN50: {metrics.pnn50:.1f} %\n"
        text += f"  pNN20: {metrics.pnn20:.1f} %\n\n"

        text += "【周波数領域指標】\n"
        text += f"  Total Power: {metrics.total_power:.1f} ms²\n"
        text += f"  VLF Power: {metrics.vlf_power:.1f} ms²\n"
        text += f"  LF Power: {metrics.lf_power:.1f} ms²\n"
        text += f"  HF Power: {metrics.hf_power:.1f} ms²\n"
        text += f"  LF/HF: {metrics.lf_hf_ratio:.3f}\n"
        text += f"  LF nu: {metrics.lf_nu:.1f}\n"
        text += f"  HF nu: {metrics.hf_nu:.1f}\n\n"

        text += "【非線形指標】\n"
        text += f"  SD1: {metrics.sd1:.2f} ms\n"
        text += f"  SD2: {metrics.sd2:.2f} ms\n"
        text += f"  SD2/SD1: {metrics.sd_ratio:.3f}\n"
        if metrics.sample_entropy is not None:
            text += f"  Sample Entropy: {metrics.sample_entropy:.4f}\n"
        else:
            text += "  Sample Entropy: 計算不可\n"
        if metrics.dfa_alpha1 is not None:
            text += f"  DFA α1: {metrics.dfa_alpha1:.4f}\n"
        else:
            text += "  DFA α1: 計算不可\n"
        if metrics.dfa_alpha2 is not None:
            text += f"  DFA α2: {metrics.dfa_alpha2:.4f}\n"
        else:
            text += "  DFA α2: 計算不可\n"

        self.adv_hrv_result_text.insert(tk.END, text)
        self.adv_hrv_result_text.config(state=tk.DISABLED)

    # =========================================================================
    # 統計検定パネル
    # =========================================================================

    def _create_statistics_panel(self) -> None:
        """統計検定パネルを作成"""
        self.adv_stats_panel = ttk.Frame(self.adv_panel_container)
        self.adv_stats_panel.columnconfigure(0, weight=1)
        self.adv_stats_panel.rowconfigure(2, weight=1)

        # 入力設定
        input_frame = ttk.LabelFrame(self.adv_stats_panel, text="入力データ", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="データファイル:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.adv_stats_file_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.adv_stats_file_var, width=50).grid(
            row=0, column=1, sticky="ew", padx=5, pady=5
        )
        ttk.Button(input_frame, text="参照...", command=self._adv_stats_browse_file).grid(
            row=0, column=2, padx=5, pady=5
        )

        ttk.Label(input_frame, text="※ ロング形式: Subject, Condition, Value 列を含むCSV/Excel").grid(
            row=1, column=0, columnspan=3, sticky="w", padx=5, pady=2
        )

        # 列名設定
        col_frame = ttk.Frame(input_frame)
        col_frame.grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        ttk.Label(col_frame, text="被験者列:").pack(side=tk.LEFT, padx=5)
        self.adv_stats_subject_col_var = tk.StringVar(value="Subject")
        ttk.Entry(col_frame, textvariable=self.adv_stats_subject_col_var, width=15).pack(side=tk.LEFT, padx=5)

        ttk.Label(col_frame, text="条件列:").pack(side=tk.LEFT, padx=5)
        self.adv_stats_condition_col_var = tk.StringVar(value="Condition")
        ttk.Entry(col_frame, textvariable=self.adv_stats_condition_col_var, width=15).pack(side=tk.LEFT, padx=5)

        ttk.Label(col_frame, text="値列:").pack(side=tk.LEFT, padx=5)
        self.adv_stats_value_col_var = tk.StringVar(value="Value")
        ttk.Entry(col_frame, textvariable=self.adv_stats_value_col_var, width=15).pack(side=tk.LEFT, padx=5)

        # 検定タイプ
        option_frame = ttk.LabelFrame(self.adv_stats_panel, text="検定タイプ", padding="10")
        option_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.adv_stats_test_type_var = tk.StringVar(value="repeated_anova")
        ttk.Radiobutton(
            option_frame, text="反復測定ANOVA（パラメトリック）",
            variable=self.adv_stats_test_type_var, value="repeated_anova"
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Radiobutton(
            option_frame, text="Friedman検定（ノンパラメトリック）",
            variable=self.adv_stats_test_type_var, value="friedman"
        ).grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Radiobutton(
            option_frame, text="一元配置ANOVA（独立群）",
            variable=self.adv_stats_test_type_var, value="one_way_anova"
        ).grid(row=0, column=2, padx=10, pady=5, sticky="w")

        ttk.Button(option_frame, text="検定実行", command=self._adv_stats_run_test).grid(
            row=1, column=0, columnspan=3, pady=10
        )

        # 結果表示
        result_frame = ttk.LabelFrame(self.adv_stats_panel, text="検定結果", padding="10")
        result_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.adv_stats_result_text = tk.Text(result_frame, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.adv_stats_result_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.adv_stats_result_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.adv_stats_result_text.config(yscrollcommand=scrollbar.set)

    def _adv_stats_browse_file(self) -> None:
        """統計検定用データファイルを選択"""
        filepath = filedialog.askopenfilename(
            title="データファイルを選択",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filepath:
            self.adv_stats_file_var.set(filepath)

    def _adv_stats_run_test(self) -> None:
        """統計検定を実行"""
        filepath = self.adv_stats_file_var.get()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror("エラー", "有効なファイルを選択してください。")
            return

        subject_col = self.adv_stats_subject_col_var.get().strip()
        condition_col = self.adv_stats_condition_col_var.get().strip()
        value_col = self.adv_stats_value_col_var.get().strip()
        test_type = self.adv_stats_test_type_var.get()

        try:
            # データ読み込み
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)

            for col in [subject_col, condition_col, value_col]:
                if col not in df.columns:
                    messagebox.showerror("エラー", f"'{col}' 列が見つかりません。")
                    return

            # 検定実行
            analyzer = StatisticalAnalyzer()

            if test_type == "repeated_anova":
                result = analyzer.repeated_measures_anova(df, subject_col, condition_col, value_col)
            elif test_type == "friedman":
                result = analyzer.friedman_test(df, subject_col, condition_col, value_col)
            elif test_type == "one_way_anova":
                groups = {
                    cond: df[df[condition_col] == cond][value_col].values
                    for cond in df[condition_col].unique()
                }
                result = analyzer.one_way_anova(groups)
            else:
                messagebox.showerror("エラー", "不明な検定タイプです。")
                return

            # 結果表示
            self._adv_stats_display_results(result, test_type, analyzer)

            # CSV出力
            output_path = Path(filepath).with_suffix('_stats_result.csv')
            result_df = pd.DataFrame([result.to_dict()])
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

            messagebox.showinfo("完了", f"検定結果を保存しました:\n{output_path}")

        except Exception as e:
            messagebox.showerror("エラー", f"検定に失敗しました:\n{e}")

    def _adv_stats_display_results(self, result, test_type: str, analyzer: StatisticalAnalyzer) -> None:
        """統計検定結果を表示"""
        self.adv_stats_result_text.config(state=tk.NORMAL)
        self.adv_stats_result_text.delete(1.0, tk.END)

        if test_type in ["repeated_anova", "one_way_anova"]:
            text = "=== ANOVA検定結果 ===\n\n"
            text += f"F統計量: {result.f_statistic:.4f}\n"
            text += f"p値: {result.p_value:.6f}\n"
            text += f"自由度 (条件間): {result.df_between}\n"
            text += f"自由度 (条件内): {result.df_within}\n"
            text += f"η²: {result.eta_squared:.4f} ({analyzer.effect_size_interpretation(result.eta_squared, 'eta_squared')})\n"
            text += f"偏η²: {result.partial_eta_squared:.4f}\n"
            text += f"有意判定: {'有意 (p<0.05)' if result.is_significant else '非有意'}\n"
        else:  # friedman
            text = "=== Friedman検定結果 ===\n\n"
            text += f"統計量: {result.statistic:.4f}\n"
            text += f"p値: {result.p_value:.6f}\n"
            text += f"有意判定: {'有意 (p<0.05)' if result.is_significant else '非有意'}\n"

        if result.post_hoc:
            text += "\n【事後検定（多重比較）】\n"
            for comparison, res in result.post_hoc.items():
                text += f"\n  {comparison}:\n"
                text += f"    p値: {res['p_value']:.6f}\n"
                text += f"    調整p値: {res['p_adjusted']:.6f}\n"
                if 'cohens_d' in res:
                    text += f"    Cohen's d: {res['cohens_d']:.4f} ({analyzer.effect_size_interpretation(res['cohens_d'], 'cohens_d')})\n"
                if 'effect_r' in res:
                    text += f"    効果量 r: {res['effect_r']:.4f} ({analyzer.effect_size_interpretation(res['effect_r'], 'r')})\n"
                text += f"    有意: {'*' if res['significant'] else ''}\n"

        self.adv_stats_result_text.insert(tk.END, text)
        self.adv_stats_result_text.config(state=tk.DISABLED)

    # =========================================================================
    # 相関分析パネル
    # =========================================================================

    def _create_correlation_panel(self) -> None:
        """相関分析パネルを作成"""
        self.adv_corr_panel = ttk.Frame(self.adv_panel_container)
        self.adv_corr_panel.columnconfigure(0, weight=1)
        self.adv_corr_panel.rowconfigure(2, weight=1)

        # 入力設定
        input_frame = ttk.LabelFrame(self.adv_corr_panel, text="入力データ", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="データファイル:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.adv_corr_file_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.adv_corr_file_var, width=50).grid(
            row=0, column=1, sticky="ew", padx=5, pady=5
        )
        ttk.Button(input_frame, text="参照...", command=self._adv_corr_browse_file).grid(
            row=0, column=2, padx=5, pady=5
        )

        ttk.Label(input_frame, text="※ 数値列を含むCSV/Excel。相関行列を計算します。").grid(
            row=1, column=0, columnspan=3, sticky="w", padx=5, pady=2
        )

        # オプション
        option_frame = ttk.LabelFrame(self.adv_corr_panel, text="オプション", padding="10")
        option_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.adv_corr_method_var = tk.StringVar(value="pearson")
        ttk.Radiobutton(
            option_frame, text="Pearson相関",
            variable=self.adv_corr_method_var, value="pearson"
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Radiobutton(
            option_frame, text="Spearman順位相関",
            variable=self.adv_corr_method_var, value="spearman"
        ).grid(row=0, column=1, padx=10, pady=5, sticky="w")

        ttk.Button(option_frame, text="相関分析実行", command=self._adv_corr_run_analysis).grid(
            row=0, column=2, padx=20, pady=5
        )

        # 結果表示
        result_frame = ttk.LabelFrame(self.adv_corr_panel, text="相関行列", padding="10")
        result_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.adv_corr_result_text = tk.Text(result_frame, height=20, wrap=tk.NONE, state=tk.DISABLED)
        self.adv_corr_result_text.grid(row=0, column=0, sticky="nsew")

        scrollbar_y = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.adv_corr_result_text.yview)
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x = ttk.Scrollbar(result_frame, orient=tk.HORIZONTAL, command=self.adv_corr_result_text.xview)
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.adv_corr_result_text.config(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    def _adv_corr_browse_file(self) -> None:
        """相関分析用データファイルを選択"""
        filepath = filedialog.askopenfilename(
            title="データファイルを選択",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filepath:
            self.adv_corr_file_var.set(filepath)

    def _adv_corr_run_analysis(self) -> None:
        """相関分析を実行"""
        filepath = self.adv_corr_file_var.get()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror("エラー", "有効なファイルを選択してください。")
            return

        method = self.adv_corr_method_var.get()

        try:
            # データ読み込み
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)

            # 数値列のみを抽出
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                messagebox.showerror("エラー", "数値列が2つ以上必要です。")
                return

            # 相関分析実行
            analyzer = CorrelationAnalyzer()
            corr_matrix, p_matrix = analyzer.correlation_matrix(numeric_df, method)

            # 結果表示
            self._adv_corr_display_results(corr_matrix, p_matrix, method)

            # CSV出力
            output_path = Path(filepath).with_suffix(f'_correlation_{method}.csv')
            corr_matrix.to_csv(output_path, encoding='utf-8-sig')

            p_output_path = Path(filepath).with_suffix(f'_correlation_{method}_pvalues.csv')
            p_matrix.to_csv(p_output_path, encoding='utf-8-sig')

            messagebox.showinfo("完了", f"相関行列を保存しました:\n{output_path}\n{p_output_path}")

        except Exception as e:
            messagebox.showerror("エラー", f"相関分析に失敗しました:\n{e}")

    def _adv_corr_display_results(
        self, corr_matrix: pd.DataFrame, p_matrix: pd.DataFrame, method: str
    ) -> None:
        """相関分析結果を表示"""
        self.adv_corr_result_text.config(state=tk.NORMAL)
        self.adv_corr_result_text.delete(1.0, tk.END)

        method_name = "Pearson" if method == "pearson" else "Spearman"
        text = f"=== {method_name}相関行列 ===\n\n"

        # 相関行列を整形して表示
        text += "【相関係数】\n"
        text += corr_matrix.round(3).to_string()
        text += "\n\n"

        text += "【p値】（* p<0.05, ** p<0.01, *** p<0.001）\n"

        # p値に有意性マークを付加
        p_formatted = p_matrix.copy()
        for col in p_formatted.columns:
            for idx in p_formatted.index:
                p = p_formatted.loc[idx, col]
                if pd.isna(p):
                    p_formatted.loc[idx, col] = "N/A"
                elif p < 0.001:
                    p_formatted.loc[idx, col] = f"{p:.4f}***"
                elif p < 0.01:
                    p_formatted.loc[idx, col] = f"{p:.4f}**"
                elif p < 0.05:
                    p_formatted.loc[idx, col] = f"{p:.4f}*"
                else:
                    p_formatted.loc[idx, col] = f"{p:.4f}"

        text += p_formatted.to_string()

        self.adv_corr_result_text.insert(tk.END, text)
        self.adv_corr_result_text.config(state=tk.DISABLED)
