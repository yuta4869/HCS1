# gui/questionnaire_analysis.py
"""アンケート解析関連のヘルパー関数"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

from .ecg_analysis import CONDITION_COLORS

# Analys_Q用の設定
Q_CONDITION_MAP = {
    1: "Fixed",
    2: "HRF",
    3: "Sin",
    4: "HRF2_PID",
    5: "HRF2_Adaptive",
    6: "HRF2_GS",
    7: "HRF2_Robust",
}
Q_CONDITION_ORDER = [
    "Fixed",
    "HRF",
    "Sin",
    "HRF2_PID",
    "HRF2_Adaptive",
    "HRF2_GS",
    "HRF2_Robust",
]
Q_CONDITION_COLORS = {
    name: CONDITION_COLORS.get(name, "#999999") for name in Q_CONDITION_ORDER
}


def normalize_condition(value):
    """Excelの条件列の値をラベルへ変換する。"""
    if pd.isna(value):
        return None
    if value in Q_CONDITION_MAP:
        return Q_CONDITION_MAP[value]
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        for label in Q_CONDITION_ORDER:
            if text.lower() == label.lower():
                return label
        return text or None
    return Q_CONDITION_MAP.get(as_int, str(value))


def load_and_melt_data(file_path: str, condition_order: Optional[List[str]] = None):
    """Excelを読み込み、ロング形式DataFrameと設問タイトルを返す。"""
    df = pd.read_excel(file_path)
    if df.shape[1] < 4:
        raise ValueError("必要な列（B〜W列）が足りません。")

    top_row = pd.read_excel(file_path, header=None, nrows=1)

    subject_series = df.iloc[:, 1].astype(str)
    condition_series = df.iloc[:, 2].apply(normalize_condition)
    question_df_raw = df.iloc[:, 3:]
    if question_df_raw.shape[1] <= 1:
        raise ValueError("設問列が不足しています。")
    question_df = question_df_raw.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    question_columns = question_df.columns.tolist()
    title_end = 3 + len(question_columns)
    question_titles = top_row.iloc[0, 3:title_end]
    question_labels = {}
    for col, title in zip(question_columns, question_titles):
        title_str = str(title).strip()
        question_labels[col] = title_str if title_str else col

    tidy_df = question_df.copy()
    tidy_df["Subject"] = subject_series
    tidy_df["Condition"] = condition_series
    tidy_df = tidy_df.melt(
        id_vars=["Subject", "Condition"],
        var_name="Question",
        value_name="Score",
    )
    tidy_df = tidy_df.dropna(subset=["Score", "Condition"])
    tidy_df["Question"] = pd.Categorical(tidy_df["Question"], categories=question_columns, ordered=True)

    selected_order = condition_order or Q_CONDITION_ORDER
    tidy_df = tidy_df[tidy_df["Condition"].isin(selected_order)].copy()
    if tidy_df.empty:
        raise ValueError("有効な回答データが見つかりませんでした（選択された条件のデータがありません）。")
    tidy_df["Condition"] = pd.Categorical(
        tidy_df["Condition"], categories=selected_order, ordered=True
    ).remove_unused_categories()

    return tidy_df, question_labels


def generate_plots(file_path: str, condition_order: Optional[List[str]] = None):
    """Excelを読み込み、箱ひげ図を作成する。

    スレッドセーフ: matplotlib.figure.Figureを直接使用してバックエンドに依存しない。
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    tidy_df, question_labels = load_and_melt_data(file_path, condition_order=condition_order)
    output_dir = Path(file_path).parent

    active_conditions = list(tidy_df["Condition"].cat.categories)
    palette = [Q_CONDITION_COLORS.get(name, "#999999") for name in active_conditions]

    # サマリーグリッド図を作成（seabornのcatplotは使わず手動で作成）
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
    output_path = output_dir / "question_boxplots_grid.png"
    fig_summary.savefig(output_path, dpi=300, bbox_inches="tight")

    # 個別の設問ごとの図を作成
    question_dir = output_dir / "question_boxplots"
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

        filename = str(title).replace("/", "_").replace("\\", "_").replace(" ", "_")
        path = question_dir / f"{filename}_boxplot.png"
        fig.savefig(path, dpi=300)
        figure_paths.append(path)

    return {
        "summary_path": output_path,
        "per_question_paths": figure_paths,
    }


# 後方互換性のためのエイリアス（analys_q_プレフィックス付き）
analys_q_normalize_condition = normalize_condition
analys_q_load_and_melt_data = load_and_melt_data
analys_q_generate_plots = generate_plots
