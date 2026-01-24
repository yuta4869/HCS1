# gui/panas_analysis.py
"""日本語版PANAS解析モジュール

PA（ポジティブ情動）8項目：「活気のある」「誇らしい」「強気な」「きっぱりとした」
                          「気合いの入った」「わくわくした」「機敏な」「熱狂した」
NA（ネガティブ情動）8項目：「びくびくした」「おびえた」「うろたえた」「心配した」
                          「ぴりぴりした」「苦悩した」「恥じた」「いらだった」

参考文献: 佐藤 徳・安田朝子 2001 日本語版PANASの作成 性格心理学研究, 9, 138-139.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

from .questionnaire_analysis import normalize_condition, Q_CONDITION_ORDER, Q_CONDITION_COLORS

# PANAS項目定義
PA_ITEMS = ["活気のある", "誇らしい", "強気な", "きっぱりとした",
            "気合いの入った", "わくわくした", "機敏な", "熱狂した"]
NA_ITEMS = ["びくびくした", "おびえた", "うろたえた", "心配した",
            "ぴりぴりした", "苦悩した", "恥じた", "いらだった"]

# 全16項目
ALL_PANAS_ITEMS = PA_ITEMS + NA_ITEMS


def calculate_cronbach_alpha(data: np.ndarray) -> Tuple[float, int]:
    """クロンバックのα係数を計算する。

    欠損値（NaN）を含む行は自動的に除外して計算する。

    Args:
        data: 項目×被験者の2次元配列 (shape: n_subjects x n_items)

    Returns:
        (クロンバックのα係数, 有効サンプル数) のタプル
    """
    # DataFrameの場合はnumpy配列に変換
    if isinstance(data, pd.DataFrame):
        data = data.values

    # 欠損値を含む行を除外
    mask = ~np.isnan(data).any(axis=1)
    data_clean = data[mask]

    n_valid = data_clean.shape[0]
    n_items = data_clean.shape[1]

    if n_items < 2 or n_valid < 3:
        return np.nan, n_valid

    # 各項目の分散
    item_variances = np.var(data_clean, axis=0, ddof=1)
    # 合計得点の分散
    total_scores = np.sum(data_clean, axis=1)
    total_variance = np.var(total_scores, ddof=1)

    if total_variance == 0:
        return np.nan, n_valid

    alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
    return alpha, n_valid


def calculate_standardized_alpha(data: np.ndarray) -> Tuple[Optional[float], int]:
    """標準化α係数（相関行列ベース）を計算する。

    相関行列から計算される標準化されたクロンバックα係数。
    Spearman-Brown公式を使用し、項目間の平均相関から算出する。
    欠損値（NaN）を含む行は自動的に除外して計算する。

    注: これは真のMcDonald's ω係数ではない。真のωには因子分析が必要。

    Args:
        data: 項目×被験者の2次元配列 (shape: n_subjects x n_items)

    Returns:
        (標準化α係数, 有効サンプル数) のタプル（計算できない場合はNone）
    """
    try:
        # DataFrameの場合はnumpy配列に変換
        if isinstance(data, pd.DataFrame):
            data = data.values

        # 欠損値を含む行を除外
        mask = ~np.isnan(data).any(axis=1)
        data_clean = data[mask]

        n_valid = data_clean.shape[0]
        n_items = data_clean.shape[1]

        if n_items < 2 or n_valid < 3:
            return None, n_valid

        # 相関行列を計算
        corr_matrix = np.corrcoef(data_clean, rowvar=False)

        # 対角成分を除いた相関の平均
        eye_mask = ~np.eye(n_items, dtype=bool)
        mean_corr = np.mean(corr_matrix[eye_mask])

        if mean_corr <= 0:
            return None, n_valid

        # 標準化α（Spearman-Brown公式）
        standardized_alpha = (n_items * mean_corr) / (1 + (n_items - 1) * mean_corr)
        return standardized_alpha, n_valid
    except Exception:
        return None, 0


def load_panas_data(file_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """ExcelファイルからPANASデータを読み込む。

    Args:
        file_path: Excelファイルのパス

    Returns:
        (DataFrame, PA項目リスト, NA項目リスト)
    """
    df = pd.read_excel(file_path)

    # PANAS項目を探す
    pa_cols = [col for col in df.columns if col in PA_ITEMS]
    na_cols = [col for col in df.columns if col in NA_ITEMS]

    if not pa_cols or not na_cols:
        raise ValueError("PANASの項目が見つかりません。")

    return df, pa_cols, na_cols


def analyze_panas(file_path: str, condition_order: Optional[List[str]] = None) -> Dict[str, Any]:
    """PANAS解析を実行する。

    Args:
        file_path: Excelファイルのパス
        condition_order: 条件の順序（Noneの場合はデフォルト）

    Returns:
        解析結果の辞書
    """
    df, pa_cols, na_cols = load_panas_data(file_path)

    # 被験者番号と条件を取得
    subject_col = df.iloc[:, 1].astype(str)
    condition_col = df.iloc[:, 2].apply(normalize_condition)

    # PA/NA得点を計算（欠損値がある場合は除外して計算し、項目数で調整）
    # 欠損がない場合はそのまま合計、欠損がある場合は回答した項目の平均×8で計算
    def calc_score_with_missing(row, cols, n_items=8):
        valid_values = row[cols].dropna()
        if len(valid_values) == 0:
            return np.nan
        elif len(valid_values) == n_items:
            return valid_values.sum()
        else:
            # 欠損がある場合：回答した項目の平均 × 総項目数で補完
            return valid_values.mean() * n_items

    df["PA_Score"] = df.apply(lambda row: calc_score_with_missing(row, pa_cols, 8), axis=1)
    df["NA_Score"] = df.apply(lambda row: calc_score_with_missing(row, na_cols, 8), axis=1)

    # 欠損数を記録
    df["PA_Missing"] = df[pa_cols].isna().sum(axis=1)
    df["NA_Missing"] = df[na_cols].isna().sum(axis=1)
    df["Subject"] = subject_col
    df["Condition"] = condition_col

    # 条件フィルタ
    selected_order = condition_order or Q_CONDITION_ORDER
    df_filtered = df[df["Condition"].isin(selected_order)].copy()

    if df_filtered.empty:
        raise ValueError("選択された条件のデータがありません。")

    # 条件ごとの統計
    results_by_condition = {}
    reliability_results = {}

    active_conditions = [c for c in selected_order if c in df_filtered["Condition"].unique()]

    for condition in active_conditions:
        cond_data = df_filtered[df_filtered["Condition"] == condition]

        # PA/NA得点の統計
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

        # 内的一貫性（α係数）- 欠損値は自動除外
        if len(cond_data) >= 3:
            pa_data = cond_data[pa_cols].values
            na_data = cond_data[na_cols].values

            pa_alpha, pa_n = calculate_cronbach_alpha(pa_data)
            na_alpha, na_n = calculate_cronbach_alpha(na_data)
            pa_std_alpha, _ = calculate_standardized_alpha(pa_data)
            na_std_alpha, _ = calculate_standardized_alpha(na_data)

            reliability_results[condition] = {
                "PA_alpha": pa_alpha,
                "NA_alpha": na_alpha,
                "PA_std_alpha": pa_std_alpha,
                "NA_std_alpha": na_std_alpha,
                "PA_n": pa_n,
                "NA_n": na_n,
            }

    # 全体の内的一貫性
    if len(df_filtered) >= 3:
        pa_data_all = df_filtered[pa_cols].values
        na_data_all = df_filtered[na_cols].values

        pa_alpha_all, pa_n_all = calculate_cronbach_alpha(pa_data_all)
        na_alpha_all, na_n_all = calculate_cronbach_alpha(na_data_all)
        pa_std_alpha_all, _ = calculate_standardized_alpha(pa_data_all)
        na_std_alpha_all, _ = calculate_standardized_alpha(na_data_all)

        reliability_results["全体"] = {
            "PA_alpha": pa_alpha_all,
            "NA_alpha": na_alpha_all,
            "PA_std_alpha": pa_std_alpha_all,
            "NA_std_alpha": na_std_alpha_all,
            "PA_n": pa_n_all,
            "NA_n": na_n_all,
        }

    return {
        "df": df_filtered,
        "results_by_condition": results_by_condition,
        "reliability": reliability_results,
        "active_conditions": active_conditions,
        "pa_cols": pa_cols,
        "na_cols": na_cols,
    }


def generate_panas_plots(file_path: str, condition_order: Optional[List[str]] = None) -> Dict[str, Any]:
    """PANAS解析を実行し、グラフを生成する。

    Args:
        file_path: Excelファイルのパス
        condition_order: 条件の順序

    Returns:
        解析結果とグラフパスの辞書
    """
    analysis = analyze_panas(file_path, condition_order)
    output_dir = Path(file_path).parent / "PANAS_analysis"
    output_dir.mkdir(exist_ok=True)

    df = analysis["df"]
    active_conditions = analysis["active_conditions"]
    palette = [Q_CONDITION_COLORS.get(name, "#999999") for name in active_conditions]

    figure_paths = []

    # 1. PA/NA得点の箱ひげ図（並列表示）
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
    ax1.set_ylim(8, 48)  # 理論範囲: 8-48

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
    ax2.set_ylim(8, 48)  # 理論範囲: 8-48

    fig1.tight_layout()
    path1 = output_dir / "PANAS_boxplot.png"
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    figure_paths.append(path1)

    # 2. PA/NA得点の棒グラフ（平均±SD）
    fig2 = Figure(figsize=(10, 5))
    canvas2 = FigureCanvasAgg(fig2)

    # データ準備
    bar_data = []
    for cond in active_conditions:
        stats = analysis["results_by_condition"][cond]
        bar_data.append({
            "Condition": cond,
            "Score_Type": "PA",
            "Mean": stats["PA_mean"],
            "SD": stats["PA_std"],
        })
        bar_data.append({
            "Condition": cond,
            "Score_Type": "NA",
            "Mean": stats["NA_mean"],
            "SD": stats["NA_std"],
        })
    bar_df = pd.DataFrame(bar_data)

    ax3 = fig2.add_subplot(111)
    x = np.arange(len(active_conditions))
    width = 0.35

    pa_means = [analysis["results_by_condition"][c]["PA_mean"] for c in active_conditions]
    pa_sds = [analysis["results_by_condition"][c]["PA_std"] for c in active_conditions]
    na_means = [analysis["results_by_condition"][c]["NA_mean"] for c in active_conditions]
    na_sds = [analysis["results_by_condition"][c]["NA_std"] for c in active_conditions]

    bars1 = ax3.bar(x - width/2, pa_means, width, yerr=pa_sds,
                    label='PA', color='#4CAF50', capsize=5)
    bars2 = ax3.bar(x + width/2, na_means, width, yerr=na_sds,
                    label='NA', color='#F44336', capsize=5)

    ax3.set_ylabel('得点')
    ax3.set_title('PANAS得点（平均±SD）')
    ax3.set_xticks(x)
    ax3.set_xticklabels(active_conditions, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 48)
    ax3.axhline(y=8, color='gray', linestyle='--', alpha=0.5, label='最低点')

    fig2.tight_layout()
    path2 = output_dir / "PANAS_barplot.png"
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    figure_paths.append(path2)

    # 3. 結果をExcelに保存
    excel_path = output_dir / "PANAS_results.xlsx"
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
                "PA_標準化α": round(rel["PA_std_alpha"], 3) if rel["PA_std_alpha"] is not None else "-",
                "NA_α係数": round(rel["NA_alpha"], 3) if not np.isnan(rel["NA_alpha"]) else "-",
                "NA_標準化α": round(rel["NA_std_alpha"], 3) if rel["NA_std_alpha"] is not None else "-",
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


def format_reliability_text(reliability: Dict[str, Dict[str, Any]]) -> str:
    """内的一貫性の結果をテキストフォーマットする。"""
    lines = ["=== 内的一貫性（信頼性係数） ===\n"]

    for cond, rel in reliability.items():
        lines.append(f"【{cond}】")
        pa_alpha = rel["PA_alpha"]
        na_alpha = rel["NA_alpha"]
        pa_std_alpha = rel["PA_std_alpha"]
        na_std_alpha = rel["NA_std_alpha"]
        pa_n = rel.get("PA_n", "?")
        na_n = rel.get("NA_n", "?")

        pa_alpha_str = f"{pa_alpha:.3f}" if not np.isnan(pa_alpha) else "計算不可"
        na_alpha_str = f"{na_alpha:.3f}" if not np.isnan(na_alpha) else "計算不可"
        pa_std_alpha_str = f"{pa_std_alpha:.3f}" if pa_std_alpha is not None else "計算不可"
        na_std_alpha_str = f"{na_std_alpha:.3f}" if na_std_alpha is not None else "計算不可"

        lines.append(f"  PA: α={pa_alpha_str}, 標準化α={pa_std_alpha_str} (n={pa_n})")
        lines.append(f"  NA: α={na_alpha_str}, 標準化α={na_std_alpha_str} (n={na_n})")
        lines.append("")

    return "\n".join(lines)
