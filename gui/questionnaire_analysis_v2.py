# gui/questionnaire_analysis_v2.py
"""アンケート解析 V2 モジュール

新フォーマット対応版（2026年1月～）
- 会話番号はファイル名から取得（例: 実験後アンケート0（回答）.xlsx の "0"）
- C列: 設問1開始（会話番号列なし）
- 設問: C〜T列（18問）
- PANAS: U〜AK列（16項目）
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

from .ecg_analysis import CONDITION_COLORS

# V2用の設定（番号→条件マッピングは同じ）
V2_CONDITION_MAP = {
    0: "Test",  # テスト番号
    1: "Fixed",
    2: "HRF",
    3: "Sin",
    4: "HRF2_PID",
    5: "HRF2_Adaptive",
    6: "HRF2_GS",
    7: "HRF2_Robust",
}

V2_CONDITION_ORDER = [
    "Fixed",
    "HRF",
    "Sin",
    "HRF2_PID",
    "HRF2_Adaptive",
    "HRF2_GS",
    "HRF2_Robust",
]

V2_CONDITION_COLORS = {
    name: CONDITION_COLORS.get(name, "#999999") for name in V2_CONDITION_ORDER
}

# PANAS項目定義（V1と同じ）
PA_ITEMS = ["活気のある", "誇らしい", "強気な", "きっぱりとした",
            "気合いの入った", "わくわくした", "機敏な", "熱狂した"]
NA_ITEMS = ["びくびくした", "おびえた", "うろたえた", "心配した",
            "ぴりぴりした", "苦悩した", "恥じた", "いらだった"]


def extract_condition_from_filename(filename: str) -> Optional[str]:
    """ファイル名から会話条件番号を抽出し、条件名に変換する。

    Args:
        filename: ファイル名（例: "実験後アンケート0（回答）.xlsx"）

    Returns:
        条件名（例: "Test"）またはNone
    """
    # パターン: 「実験後アンケート」の後の数字を抽出
    # 例: 実験後アンケート0（回答）.xlsx -> 0
    #     実験後アンケート1（回答）.xlsx -> 1
    patterns = [
        r'実験後アンケート(\d+)',  # 実験後アンケート0（回答）.xlsx
        r'アンケート(\d+)',        # アンケート0.xlsx
        r'_(\d+)\(',              # xxx_0（回答）.xlsx
        r'(\d+)（回答）',          # 0（回答）.xlsx
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            num = int(match.group(1))
            return V2_CONDITION_MAP.get(num)

    return None


def column_letter_to_index(letter: str) -> int:
    """Excel列文字（A, B, ..., Z, AA, AB, ...）を0ベースのインデックスに変換"""
    letter = letter.upper().strip()
    result = 0
    for char in letter:
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result - 1


def scan_folder_for_surveys(folder_path: str) -> Dict[str, List[str]]:
    """フォルダ内のアンケートファイルをスキャンし、条件ごとにグループ化する。

    Args:
        folder_path: スキャン対象のフォルダパス

    Returns:
        {条件名: [ファイルパスリスト]} の辞書
    """
    result = {}
    folder = Path(folder_path)

    for file_path in folder.glob("*.xlsx"):
        if file_path.name.startswith("~$"):  # 一時ファイルをスキップ
            continue

        condition = extract_condition_from_filename(file_path.name)
        if condition:
            if condition not in result:
                result[condition] = []
            result[condition].append(str(file_path))

    return result


def load_v2_data(
    file_paths: List[str],
    start_col: str = "C",
    end_col: str = "T"
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """V2フォーマットのExcelファイルを読み込み、ロング形式DataFrameを返す。

    Args:
        file_paths: Excelファイルパスのリスト（同じ条件のファイル群）
        start_col: 開始列（デフォルト: C）
        end_col: 終了列（デフォルト: T）

    Returns:
        (tidy_df, question_labels)
    """
    all_data = []
    question_labels = {}

    start_idx = column_letter_to_index(start_col)
    end_idx = column_letter_to_index(end_col) + 1

    for file_path in file_paths:
        condition = extract_condition_from_filename(os.path.basename(file_path))
        if not condition or condition == "Test":
            continue

        df = pd.read_excel(file_path)
        top_row = pd.read_excel(file_path, header=None, nrows=1)

        # 被験者番号（B列、インデックス1）
        subject_series = df.iloc[:, 1].astype(str)

        # 設問データ抽出
        if end_idx > df.shape[1]:
            end_idx = df.shape[1]

        question_df = df.iloc[:, start_idx:end_idx].apply(pd.to_numeric, errors="coerce")

        # ラベルの取得（最初のファイルから）
        if not question_labels:
            question_columns = question_df.columns.tolist()
            question_titles = top_row.iloc[0, start_idx:end_idx]
            for col, title in zip(question_columns, question_titles):
                title_str = str(title).strip()
                question_labels[col] = title_str if title_str and title_str != 'nan' else col

        # データフレームに条件と被験者を追加
        tidy_df = question_df.copy()
        tidy_df["Subject"] = subject_series
        tidy_df["Condition"] = condition
        tidy_df = tidy_df.melt(
            id_vars=["Subject", "Condition"],
            var_name="Question",
            value_name="Score",
        )
        all_data.append(tidy_df)

    if not all_data:
        raise ValueError("有効なデータが見つかりませんでした。")

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=["Score", "Condition"])

    # カテゴリ順序を設定
    question_columns = list(question_labels.keys())
    combined_df["Question"] = pd.Categorical(
        combined_df["Question"], categories=question_columns, ordered=True
    )

    return combined_df, question_labels


def load_v2_data_from_folder(
    folder_path: str,
    selected_conditions: Optional[List[str]] = None,
    start_col: str = "C",
    end_col: str = "T"
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """フォルダから全ファイルを読み込み、選択された条件のデータを返す。

    Args:
        folder_path: フォルダパス
        selected_conditions: 解析対象の条件リスト（Noneなら全条件）
        start_col: 開始列
        end_col: 終了列

    Returns:
        (tidy_df, question_labels, active_conditions)
    """
    files_by_condition = scan_folder_for_surveys(folder_path)

    if not files_by_condition:
        raise ValueError("フォルダ内にアンケートファイルが見つかりませんでした。")

    # 条件フィルタ
    if selected_conditions:
        target_conditions = selected_conditions
    else:
        target_conditions = V2_CONDITION_ORDER

    all_files = []
    for cond in target_conditions:
        if cond in files_by_condition:
            all_files.extend(files_by_condition[cond])

    if not all_files:
        raise ValueError("選択された条件のファイルが見つかりませんでした。")

    tidy_df, question_labels = load_v2_data(all_files, start_col, end_col)

    # 条件でフィルタ
    tidy_df = tidy_df[tidy_df["Condition"].isin(target_conditions)].copy()

    if tidy_df.empty:
        raise ValueError("有効なデータが見つかりませんでした。")

    # アクティブな条件（実際にデータがある条件）を順序付きで取得
    active_conditions = [c for c in target_conditions if c in tidy_df["Condition"].unique()]

    tidy_df["Condition"] = pd.Categorical(
        tidy_df["Condition"], categories=active_conditions, ordered=True
    )

    return tidy_df, question_labels, active_conditions


def generate_v2_plots(
    folder_path: str,
    selected_conditions: Optional[List[str]] = None,
    start_col: str = "C",
    end_col: str = "T",
    output_folder: Optional[str] = None
) -> Dict[str, Any]:
    """V2フォーマットの箱ひげ図を作成する。

    Args:
        folder_path: フォルダパス
        selected_conditions: 解析対象の条件リスト
        start_col: 開始列
        end_col: 終了列
        output_folder: 出力フォルダ（Noneの場合はfolder_pathを使用）

    Returns:
        結果辞書
    """
    tidy_df, question_labels, active_conditions = load_v2_data_from_folder(
        folder_path, selected_conditions, start_col, end_col
    )

    output_dir = Path(output_folder) if output_folder else Path(folder_path)
    palette = [V2_CONDITION_COLORS.get(name, "#999999") for name in active_conditions]

    # サマリーグリッド図を作成
    questions = list(tidy_df["Question"].cat.categories)
    n_questions = len(questions)
    n_cols = 3
    n_rows = (n_questions + n_cols - 1) // n_cols

    fig_summary = Figure(figsize=(n_cols * 3.8, n_rows * 3.8))
    canvas_summary = FigureCanvasAgg(fig_summary)

    for idx, question in enumerate(questions):
        ax = fig_summary.add_subplot(n_rows, n_cols, idx + 1)
        question_data = tidy_df[tidy_df["Question"] == question]
        if not question_data.empty:
            sns.boxplot(
                data=question_data,
                x="Condition",
                y="Score",
                order=active_conditions,
                hue="Condition",
                palette=palette,
                legend=False,
                ax=ax,
            )
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=45)

    fig_summary.tight_layout()
    output_path = output_dir / "question_boxplots_grid_v2.png"
    fig_summary.savefig(output_path, dpi=300, bbox_inches="tight")

    # 個別の設問ごとの図を作成
    question_dir = output_dir / "question_boxplots_v2"
    question_dir.mkdir(exist_ok=True)

    figure_paths = []
    for question in questions:
        question_data = tidy_df[tidy_df["Question"] == question]
        if question_data.empty:
            continue
        title = question_labels.get(question, question) or question

        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        sns.boxplot(
            data=question_data,
            x="Condition",
            y="Score",
            order=active_conditions,
            hue="Condition",
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()

        filename = str(title).replace("/", "_").replace("\\", "_").replace(" ", "_")[:50]
        path = question_dir / f"{filename}_boxplot.png"
        fig.savefig(path, dpi=300)
        figure_paths.append(path)

    return {
        "summary_path": output_path,
        "per_question_paths": figure_paths,
        "active_conditions": active_conditions,
        "n_subjects": len(tidy_df["Subject"].unique()),
    }


# ===== PANAS V2 解析 =====

def calculate_cronbach_alpha(data: np.ndarray) -> float:
    """クロンバックのα係数を計算する。"""
    n_items = data.shape[1]
    if n_items < 2:
        return np.nan

    item_variances = np.var(data, axis=0, ddof=1)
    total_scores = np.sum(data, axis=1)
    total_variance = np.var(total_scores, ddof=1)

    if total_variance == 0:
        return np.nan

    alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
    return alpha


def calculate_omega(data: np.ndarray) -> Optional[float]:
    """McDonald's ω係数を計算する（簡易版）。"""
    try:
        n_items = data.shape[1]
        if n_items < 2 or data.shape[0] < 3:
            return None

        corr_matrix = np.corrcoef(data, rowvar=False)
        mask = ~np.eye(n_items, dtype=bool)
        mean_corr = np.mean(corr_matrix[mask])

        if mean_corr <= 0:
            return None

        omega = (n_items * mean_corr) / (1 + (n_items - 1) * mean_corr)
        return omega
    except Exception:
        return None


def load_v2_panas_data(
    file_paths: List[str],
    start_col: str = "U",
    end_col: str = "AK"
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """V2フォーマットのPANASデータを読み込む。

    Args:
        file_paths: Excelファイルパスのリスト
        start_col: PANAS開始列（デフォルト: U）
        end_col: PANAS終了列（デフォルト: AK）

    Returns:
        (DataFrame, PA項目リスト, NA項目リスト)
    """
    all_data = []
    pa_cols_found = []
    na_cols_found = []

    for file_path in file_paths:
        condition = extract_condition_from_filename(os.path.basename(file_path))
        if not condition or condition == "Test":
            continue

        df = pd.read_excel(file_path)

        # PANAS項目を探す（列名がPA_ITEMS/NA_ITEMSに一致するもの）
        pa_cols = [col for col in df.columns if col in PA_ITEMS]
        na_cols = [col for col in df.columns if col in NA_ITEMS]

        if not pa_cols or not na_cols:
            # 列名が見つからない場合は指定範囲から探す
            start_idx = column_letter_to_index(start_col)
            end_idx = column_letter_to_index(end_col) + 1
            if end_idx > df.shape[1]:
                end_idx = df.shape[1]

            # ヘッダー行からPANAS項目を探す
            top_row = pd.read_excel(file_path, header=None, nrows=1)
            for i in range(start_idx, end_idx):
                if i < len(top_row.columns):
                    item = str(top_row.iloc[0, i]).strip()
                    col_name = df.columns[i] if i < len(df.columns) else None
                    if item in PA_ITEMS and col_name is not None:
                        pa_cols.append(col_name)
                    elif item in NA_ITEMS and col_name is not None:
                        na_cols.append(col_name)

        if not pa_cols or not na_cols:
            continue

        if not pa_cols_found:
            pa_cols_found = pa_cols
            na_cols_found = na_cols

        # 被験者番号（B列、インデックス1）
        subject_series = df.iloc[:, 1].astype(str)

        # PA/NA得点を計算
        df_copy = df.copy()
        df_copy["PA_Score"] = df[pa_cols].sum(axis=1)
        df_copy["NA_Score"] = df[na_cols].sum(axis=1)
        df_copy["Subject"] = subject_series
        df_copy["Condition"] = condition

        # PA/NA項目の生データも保持
        for col in pa_cols + na_cols:
            if col in df.columns:
                df_copy[col] = df[col]

        all_data.append(df_copy)

    if not all_data:
        raise ValueError("有効なPANASデータが見つかりませんでした。")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df, pa_cols_found, na_cols_found


def analyze_v2_panas(
    folder_path: str,
    selected_conditions: Optional[List[str]] = None,
    start_col: str = "U",
    end_col: str = "AK"
) -> Dict[str, Any]:
    """V2フォーマットのPANAS解析を実行する。"""
    files_by_condition = scan_folder_for_surveys(folder_path)

    if not files_by_condition:
        raise ValueError("フォルダ内にアンケートファイルが見つかりませんでした。")

    # 条件フィルタ
    target_conditions = selected_conditions or V2_CONDITION_ORDER

    all_files = []
    for cond in target_conditions:
        if cond in files_by_condition:
            all_files.extend(files_by_condition[cond])

    if not all_files:
        raise ValueError("選択された条件のファイルが見つかりませんでした。")

    df, pa_cols, na_cols = load_v2_panas_data(all_files, start_col, end_col)

    # 条件でフィルタ
    df_filtered = df[df["Condition"].isin(target_conditions)].copy()

    if df_filtered.empty:
        raise ValueError("選択された条件のデータがありません。")

    # 条件ごとの統計
    results_by_condition = {}
    reliability_results = {}

    active_conditions = [c for c in target_conditions if c in df_filtered["Condition"].unique()]

    for condition in active_conditions:
        cond_data = df_filtered[df_filtered["Condition"] == condition]

        pa_scores = cond_data["PA_Score"].values
        na_scores = cond_data["NA_Score"].values

        results_by_condition[condition] = {
            "n": len(cond_data),
            "PA_mean": np.mean(pa_scores),
            "PA_std": np.std(pa_scores, ddof=1) if len(pa_scores) > 1 else 0,
            "PA_median": np.median(pa_scores),
            "NA_mean": np.mean(na_scores),
            "NA_std": np.std(na_scores, ddof=1) if len(na_scores) > 1 else 0,
            "NA_median": np.median(na_scores),
        }

        # 内的一貫性
        if len(cond_data) >= 3 and pa_cols and na_cols:
            pa_data = cond_data[pa_cols].values
            na_data = cond_data[na_cols].values

            reliability_results[condition] = {
                "PA_alpha": calculate_cronbach_alpha(pa_data),
                "NA_alpha": calculate_cronbach_alpha(na_data),
                "PA_omega": calculate_omega(pa_data),
                "NA_omega": calculate_omega(na_data),
            }

    # 全体の内的一貫性
    if len(df_filtered) >= 3 and pa_cols and na_cols:
        pa_data_all = df_filtered[pa_cols].values
        na_data_all = df_filtered[na_cols].values

        reliability_results["全体"] = {
            "PA_alpha": calculate_cronbach_alpha(pa_data_all),
            "NA_alpha": calculate_cronbach_alpha(na_data_all),
            "PA_omega": calculate_omega(pa_data_all),
            "NA_omega": calculate_omega(na_data_all),
        }

    return {
        "df": df_filtered,
        "results_by_condition": results_by_condition,
        "reliability": reliability_results,
        "active_conditions": active_conditions,
        "pa_cols": pa_cols,
        "na_cols": na_cols,
    }


def generate_v2_panas_plots(
    folder_path: str,
    selected_conditions: Optional[List[str]] = None,
    start_col: str = "U",
    end_col: str = "AK",
    output_folder: Optional[str] = None
) -> Dict[str, Any]:
    """V2フォーマットのPANAS解析を実行し、グラフを生成する。

    Args:
        folder_path: 入力フォルダパス
        selected_conditions: 解析対象の条件リスト
        start_col: PANAS開始列
        end_col: PANAS終了列
        output_folder: 出力フォルダ（Noneの場合はfolder_pathを使用）
    """
    analysis = analyze_v2_panas(folder_path, selected_conditions, start_col, end_col)
    base_output_dir = Path(output_folder) if output_folder else Path(folder_path)
    output_dir = base_output_dir / "PANAS_analysis_v2"
    output_dir.mkdir(exist_ok=True)

    df = analysis["df"]
    active_conditions = analysis["active_conditions"]
    palette = [V2_CONDITION_COLORS.get(name, "#999999") for name in active_conditions]

    figure_paths = []

    # 1. PA/NA得点の箱ひげ図
    fig1 = Figure(figsize=(10, 5))
    canvas1 = FigureCanvasAgg(fig1)

    ax1 = fig1.add_subplot(121)
    sns.boxplot(
        data=df, x="Condition", y="PA_Score",
        order=active_conditions, hue="Condition",
        palette=palette, legend=False, ax=ax1
    )
    ax1.set_title("PA得点（ポジティブ情動）")
    ax1.set_xlabel("")
    ax1.set_ylabel("得点")
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(8, 48)

    ax2 = fig1.add_subplot(122)
    sns.boxplot(
        data=df, x="Condition", y="NA_Score",
        order=active_conditions, hue="Condition",
        palette=palette, legend=False, ax=ax2
    )
    ax2.set_title("NA得点（ネガティブ情動）")
    ax2.set_xlabel("")
    ax2.set_ylabel("得点")
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(8, 48)

    fig1.tight_layout()
    path1 = output_dir / "PANAS_boxplot_v2.png"
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    figure_paths.append(path1)

    # 2. PA/NA得点の棒グラフ
    fig2 = Figure(figsize=(10, 5))
    canvas2 = FigureCanvasAgg(fig2)

    ax3 = fig2.add_subplot(111)
    x = np.arange(len(active_conditions))
    width = 0.35

    pa_means = [analysis["results_by_condition"][c]["PA_mean"] for c in active_conditions]
    pa_sds = [analysis["results_by_condition"][c]["PA_std"] for c in active_conditions]
    na_means = [analysis["results_by_condition"][c]["NA_mean"] for c in active_conditions]
    na_sds = [analysis["results_by_condition"][c]["NA_std"] for c in active_conditions]

    ax3.bar(x - width/2, pa_means, width, yerr=pa_sds,
            label='PA', color='#4CAF50', capsize=5)
    ax3.bar(x + width/2, na_means, width, yerr=na_sds,
            label='NA', color='#F44336', capsize=5)

    ax3.set_ylabel('得点')
    ax3.set_title('PANAS得点（平均±SD）')
    ax3.set_xticks(x)
    ax3.set_xticklabels(active_conditions, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 48)
    ax3.axhline(y=8, color='gray', linestyle='--', alpha=0.5)

    fig2.tight_layout()
    path2 = output_dir / "PANAS_barplot_v2.png"
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    figure_paths.append(path2)

    # 3. 結果をExcelに保存
    excel_path = output_dir / "PANAS_results_v2.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 条件別統計
        stats_data = []
        for cond in active_conditions:
            stats = analysis["results_by_condition"][cond]
            stats_data.append({
                "条件": cond,
                "n": stats["n"],
                "PA平均": round(stats["PA_mean"], 2),
                "PA標準偏差": round(stats["PA_std"], 2),
                "PA中央値": round(stats["PA_median"], 2),
                "NA平均": round(stats["NA_mean"], 2),
                "NA標準偏差": round(stats["NA_std"], 2),
                "NA中央値": round(stats["NA_median"], 2),
            })
        pd.DataFrame(stats_data).to_excel(writer, sheet_name="条件別統計", index=False)

        # 内的一貫性
        reliability_data = []
        for cond, rel in analysis["reliability"].items():
            reliability_data.append({
                "条件": cond,
                "PA_α係数": round(rel["PA_alpha"], 3) if not np.isnan(rel["PA_alpha"]) else "-",
                "PA_ω係数": round(rel["PA_omega"], 3) if rel["PA_omega"] is not None else "-",
                "NA_α係数": round(rel["NA_alpha"], 3) if not np.isnan(rel["NA_alpha"]) else "-",
                "NA_ω係数": round(rel["NA_omega"], 3) if rel["NA_omega"] is not None else "-",
            })
        pd.DataFrame(reliability_data).to_excel(writer, sheet_name="内的一貫性", index=False)

        # 個人データ
        individual_data = df[["Subject", "Condition", "PA_Score", "NA_Score"]].copy()
        individual_data.to_excel(writer, sheet_name="個人データ", index=False)

    return {
        "analysis": analysis,
        "figure_paths": figure_paths,
        "excel_path": excel_path,
        "summary_path": path1,
    }


def format_v2_reliability_text(reliability: Dict[str, Dict[str, Any]]) -> str:
    """内的一貫性の結果をテキストフォーマットする。"""
    lines = ["=== 内的一貫性（信頼性係数） ===\n"]

    for cond, rel in reliability.items():
        lines.append(f"【{cond}】")
        pa_alpha = rel["PA_alpha"]
        na_alpha = rel["NA_alpha"]
        pa_omega = rel["PA_omega"]
        na_omega = rel["NA_omega"]

        pa_alpha_str = f"{pa_alpha:.3f}" if not np.isnan(pa_alpha) else "計算不可"
        na_alpha_str = f"{na_alpha:.3f}" if not np.isnan(na_alpha) else "計算不可"
        pa_omega_str = f"{pa_omega:.3f}" if pa_omega is not None else "計算不可"
        na_omega_str = f"{na_omega:.3f}" if na_omega is not None else "計算不可"

        lines.append(f"  PA: α={pa_alpha_str}, ω={pa_omega_str}")
        lines.append(f"  NA: α={na_alpha_str}, ω={na_omega_str}")
        lines.append("")

    return "\n".join(lines)
