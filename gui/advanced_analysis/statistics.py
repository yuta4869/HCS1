# gui/advanced_analysis/statistics.py
"""統計検定モジュール

条件間比較のための統計検定と効果量計算を行う。
参考: /Users/user/Research/Analys/advanced_analysis/statistics.py
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ANOVAResult:
    """ANOVA検定結果"""
    f_statistic: float
    p_value: float
    df_between: int
    df_within: int
    eta_squared: float
    partial_eta_squared: float
    is_significant: bool
    post_hoc: Optional[Dict[str, Dict]] = None

    def to_dict(self) -> Dict:
        return {
            'F統計量': self.f_statistic,
            'p値': self.p_value,
            '自由度(条件間)': self.df_between,
            '自由度(条件内)': self.df_within,
            'η²': self.eta_squared,
            '偏η²': self.partial_eta_squared,
            '有意 (p<0.05)': self.is_significant,
        }


@dataclass
class TTestResult:
    """t検定結果"""
    t_statistic: float
    p_value: float
    df: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    is_significant: bool

    def to_dict(self) -> Dict:
        return {
            't統計量': self.t_statistic,
            'p値': self.p_value,
            '自由度': self.df,
            "Cohen's d": self.cohens_d,
            '95%CI下限': self.ci_lower,
            '95%CI上限': self.ci_upper,
            '有意 (p<0.05)': self.is_significant,
        }


@dataclass
class FriedmanResult:
    """Friedman検定結果"""
    statistic: float
    p_value: float
    is_significant: bool
    post_hoc: Optional[Dict[str, Dict]] = None

    def to_dict(self) -> Dict:
        return {
            'Friedman統計量': self.statistic,
            'p値': self.p_value,
            '有意 (p<0.05)': self.is_significant,
        }


class StatisticalAnalyzer:
    """統計解析クラス"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def one_way_anova(self, groups: Dict[str, np.ndarray], post_hoc: bool = True) -> ANOVAResult:
        """一元配置分散分析"""
        group_names = list(groups.keys())
        group_data = [np.array(groups[name]) for name in group_names]

        f_stat, p_value = stats.f_oneway(*group_data)

        all_data = np.concatenate(group_data)
        grand_mean = np.mean(all_data)
        n_total = len(all_data)
        k = len(group_data)

        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data)
        ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in group_data)
        ss_total = np.sum((all_data - grand_mean) ** 2)

        df_between = k - 1
        df_within = n_total - k

        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        partial_eta_squared = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else 0

        post_hoc_results = None
        if post_hoc and p_value < self.alpha and k > 2:
            post_hoc_results = self._tukey_hsd(groups)

        return ANOVAResult(
            f_statistic=f_stat, p_value=p_value,
            df_between=df_between, df_within=df_within,
            eta_squared=eta_squared, partial_eta_squared=partial_eta_squared,
            is_significant=p_value < self.alpha, post_hoc=post_hoc_results
        )

    def repeated_measures_anova(
        self, data: pd.DataFrame,
        subject_col: str, condition_col: str, value_col: str
    ) -> ANOVAResult:
        """反復測定一元配置分散分析"""
        pivot = data.pivot(index=subject_col, columns=condition_col, values=value_col)
        pivot = pivot.dropna()

        if pivot.shape[0] < 3:
            raise ValueError("被験者数が不足しています（最低3名必要）")

        conditions = pivot.columns.tolist()
        k = len(conditions)
        n = len(pivot)

        grand_mean = pivot.values.mean()
        ss_total = np.sum((pivot.values - grand_mean) ** 2)

        condition_means = pivot.mean(axis=0)
        ss_condition = n * np.sum((condition_means - grand_mean) ** 2)

        subject_means = pivot.mean(axis=1)
        ss_subject = k * np.sum((subject_means - grand_mean) ** 2)

        ss_error = ss_total - ss_condition - ss_subject

        df_condition = k - 1
        df_error = (k - 1) * (n - 1)

        ms_condition = ss_condition / df_condition if df_condition > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 1e-10

        f_stat = ms_condition / ms_error
        p_value = 1 - stats.f.cdf(f_stat, df_condition, df_error)

        eta_squared = ss_condition / ss_total if ss_total > 0 else 0
        partial_eta_squared = ss_condition / (ss_condition + ss_error) if (ss_condition + ss_error) > 0 else 0

        post_hoc_results = None
        if p_value < self.alpha and k > 2:
            groups = {cond: pivot[cond].values for cond in conditions}
            post_hoc_results = self._paired_post_hoc(groups)

        return ANOVAResult(
            f_statistic=f_stat, p_value=p_value,
            df_between=df_condition, df_within=df_error,
            eta_squared=eta_squared, partial_eta_squared=partial_eta_squared,
            is_significant=p_value < self.alpha, post_hoc=post_hoc_results
        )

    def _tukey_hsd(self, groups: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Tukey HSD事後検定"""
        from itertools import combinations

        results = {}
        group_names = list(groups.keys())
        n_comparisons = len(list(combinations(group_names, 2)))

        for g1, g2 in combinations(group_names, 2):
            data1 = groups[g1]
            data2 = groups[g2]

            t_stat, p_value = stats.ttest_ind(data1, data2)
            p_adjusted = min(p_value * n_comparisons, 1.0)

            pooled_std = np.sqrt(
                ((len(data1) - 1) * np.var(data1, ddof=1) +
                 (len(data2) - 1) * np.var(data2, ddof=1)) /
                (len(data1) + len(data2) - 2)
            )
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

            key = f"{g1} vs {g2}"
            results[key] = {
                'p_value': p_value,
                'p_adjusted': p_adjusted,
                'cohens_d': cohens_d,
                'significant': p_adjusted < self.alpha,
                'mean_diff': np.mean(data1) - np.mean(data2),
            }

        return results

    def _paired_post_hoc(self, groups: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """対応ありデータの事後検定（Bonferroni補正付きWilcoxon）"""
        from itertools import combinations

        results = {}
        group_names = list(groups.keys())
        n_comparisons = len(list(combinations(group_names, 2)))

        for g1, g2 in combinations(group_names, 2):
            data1 = groups[g1]
            data2 = groups[g2]

            try:
                stat, p_value = stats.wilcoxon(data1, data2)
            except Exception:
                stat, p_value = np.nan, 1.0

            p_adjusted = min(p_value * n_comparisons, 1.0)

            n = len(data1)
            z = stats.norm.ppf(1 - p_value / 2) if p_value < 1 else 0
            effect_r = abs(z) / np.sqrt(n) if n > 0 else 0

            key = f"{g1} vs {g2}"
            results[key] = {
                'p_value': p_value,
                'p_adjusted': p_adjusted,
                'effect_r': effect_r,
                'significant': p_adjusted < self.alpha,
                'median_diff': np.median(data1) - np.median(data2),
            }

        return results

    def independent_ttest(
        self, group1: np.ndarray, group2: np.ndarray, equal_var: bool = True
    ) -> TTestResult:
        """独立2群t検定"""
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)

        n1, n2 = len(group1), len(group2)
        if equal_var:
            df = n1 + n2 - 2
        else:
            v1 = np.var(group1, ddof=1) / n1
            v2 = np.var(group2, ddof=1) / n2
            df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))

        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(group1, ddof=1) +
             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2)
        )
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        mean_diff = np.mean(group1) - np.mean(group2)
        se = pooled_std * np.sqrt(1 / n1 + 1 / n2)
        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se

        return TTestResult(
            t_statistic=t_stat, p_value=p_value, df=df,
            cohens_d=cohens_d, ci_lower=ci_lower, ci_upper=ci_upper,
            is_significant=p_value < self.alpha
        )

    def paired_ttest(self, before: np.ndarray, after: np.ndarray) -> TTestResult:
        """対応ありt検定"""
        t_stat, p_value = stats.ttest_rel(before, after)

        n = len(before)
        df = n - 1

        diff = after - before
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

        mean_diff = np.mean(diff)
        se = np.std(diff, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se

        return TTestResult(
            t_statistic=t_stat, p_value=p_value, df=df,
            cohens_d=cohens_d, ci_lower=ci_lower, ci_upper=ci_upper,
            is_significant=p_value < self.alpha
        )

    def friedman_test(
        self, data: pd.DataFrame,
        subject_col: str, condition_col: str, value_col: str,
        post_hoc: bool = True
    ) -> FriedmanResult:
        """Friedman検定（ノンパラメトリック反復測定）"""
        pivot = data.pivot(index=subject_col, columns=condition_col, values=value_col)
        pivot = pivot.dropna()

        conditions = pivot.columns.tolist()
        matrices = [pivot[cond].values for cond in conditions]

        stat, p_value = stats.friedmanchisquare(*matrices)

        post_hoc_results = None
        if post_hoc and p_value < self.alpha:
            groups = {cond: pivot[cond].values for cond in conditions}
            post_hoc_results = self._paired_post_hoc(groups)

        return FriedmanResult(
            statistic=stat, p_value=p_value,
            is_significant=p_value < self.alpha, post_hoc=post_hoc_results
        )

    def normality_test(self, data: np.ndarray) -> Tuple[float, float, bool]:
        """正規性検定（Shapiro-Wilk）"""
        if len(data) < 3:
            return np.nan, np.nan, False
        stat, p_value = stats.shapiro(data)
        return stat, p_value, p_value >= self.alpha

    def levene_test(self, *groups) -> Tuple[float, float, bool]:
        """等分散性検定（Levene）"""
        stat, p_value = stats.levene(*groups)
        return stat, p_value, p_value >= self.alpha

    def effect_size_interpretation(self, d: float, metric: str = 'cohens_d') -> str:
        """効果量の解釈"""
        d = abs(d)
        if metric == 'cohens_d':
            if d < 0.2:
                return '無視できる'
            elif d < 0.5:
                return '小'
            elif d < 0.8:
                return '中'
            else:
                return '大'
        elif metric == 'eta_squared':
            if d < 0.01:
                return '無視できる'
            elif d < 0.06:
                return '小'
            elif d < 0.14:
                return '中'
            else:
                return '大'
        elif metric == 'r':
            if d < 0.1:
                return '無視できる'
            elif d < 0.3:
                return '小'
            elif d < 0.5:
                return '中'
            else:
                return '大'
        return '不明'

    def summary_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """記述統計量を計算"""
        return {
            'n': len(data),
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'se': np.std(data, ddof=1) / np.sqrt(len(data)),
            'median': np.median(data),
            'min': np.min(data),
            'max': np.max(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        }
