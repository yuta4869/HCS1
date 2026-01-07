# gui/advanced_analysis/correlation.py
"""相関分析モジュール

HRV指標と主観評価の関連性を分析する。
参考: /Users/user/Research/Analys/advanced_analysis/correlation.py
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CorrelationResult:
    """相関分析結果"""
    r: float
    p_value: float
    n: int
    ci_lower: float
    ci_upper: float
    r_squared: float
    is_significant: bool

    def to_dict(self) -> Dict:
        return {
            '相関係数 (r)': self.r,
            'p値': self.p_value,
            'n': self.n,
            '95%CI下限': self.ci_lower,
            '95%CI上限': self.ci_upper,
            'R²': self.r_squared,
            '有意 (p<0.05)': self.is_significant,
        }


@dataclass
class RegressionResult:
    """回帰分析結果"""
    slope: float
    intercept: float
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    p_value: float
    se_slope: float
    coefficients: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            '傾き': self.slope,
            '切片': self.intercept,
            'R²': self.r_squared,
            '調整済みR²': self.adjusted_r_squared,
            'F統計量': self.f_statistic,
            'p値': self.p_value,
            '傾きSE': self.se_slope,
        }


class CorrelationAnalyzer:
    """相関分析クラス"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> CorrelationResult:
        """Pearson相関係数"""
        x = np.array(x)
        y = np.array(y)

        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        n = len(x)
        if n < 3:
            raise ValueError("データ数が不足しています（最低3点必要）")

        r, p_value = stats.pearsonr(x, y)

        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = np.tanh(z - z_crit * se)
        ci_upper = np.tanh(z + z_crit * se)

        return CorrelationResult(
            r=r, p_value=p_value, n=n,
            ci_lower=ci_lower, ci_upper=ci_upper,
            r_squared=r ** 2, is_significant=p_value < self.alpha
        )

    def spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> CorrelationResult:
        """Spearman順位相関係数"""
        x = np.array(x)
        y = np.array(y)

        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        n = len(x)
        if n < 3:
            raise ValueError("データ数が不足しています")

        r, p_value = stats.spearmanr(x, y)

        se = 1 / np.sqrt(n - 3)
        z = np.arctanh(r)
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = np.tanh(z - z_crit * se)
        ci_upper = np.tanh(z + z_crit * se)

        return CorrelationResult(
            r=r, p_value=p_value, n=n,
            ci_lower=ci_lower, ci_upper=ci_upper,
            r_squared=r ** 2, is_significant=p_value < self.alpha
        )

    def partial_correlation(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> CorrelationResult:
        """偏相関係数（zの影響を制御）"""
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]

        n = len(x)
        if n < 4:
            raise ValueError("データ数が不足しています")

        slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
        slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)

        residual_x = x - (slope_xz * z + intercept_xz)
        residual_y = y - (slope_yz * z + intercept_yz)

        r, p_value = stats.pearsonr(residual_x, residual_y)

        df = n - 3
        se = 1 / np.sqrt(df)
        z_val = np.arctanh(r)
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = np.tanh(z_val - z_crit * se)
        ci_upper = np.tanh(z_val + z_crit * se)

        return CorrelationResult(
            r=r, p_value=p_value, n=n,
            ci_lower=ci_lower, ci_upper=ci_upper,
            r_squared=r ** 2, is_significant=p_value < self.alpha
        )

    def correlation_matrix(
        self, data: pd.DataFrame, method: str = 'pearson'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """相関行列とp値行列を計算"""
        columns = data.columns
        n = len(columns)
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                else:
                    x = data.iloc[:, i].values
                    y = data.iloc[:, j].values

                    mask = ~(np.isnan(x) | np.isnan(y))
                    if np.sum(mask) < 3:
                        corr_matrix[i, j] = np.nan
                        p_matrix[i, j] = np.nan
                    else:
                        if method == 'pearson':
                            r, p = stats.pearsonr(x[mask], y[mask])
                        else:
                            r, p = stats.spearmanr(x[mask], y[mask])
                        corr_matrix[i, j] = r
                        p_matrix[i, j] = p

        corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
        p_df = pd.DataFrame(p_matrix, index=columns, columns=columns)

        return corr_df, p_df

    def simple_regression(self, x: np.ndarray, y: np.ndarray) -> RegressionResult:
        """単回帰分析"""
        x = np.array(x)
        y = np.array(y)

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        n = len(x)
        if n < 3:
            raise ValueError("データ数が不足しています")

        slope, intercept, r, p_value, se = stats.linregress(x, y)

        r_squared = r ** 2
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)

        ss_reg = np.sum((slope * x + intercept - np.mean(y)) ** 2)
        ss_res = np.sum((y - (slope * x + intercept)) ** 2)
        f_stat = (ss_reg / 1) / (ss_res / (n - 2)) if ss_res > 0 else 0

        return RegressionResult(
            slope=slope, intercept=intercept,
            r_squared=r_squared, adjusted_r_squared=adjusted_r_squared,
            f_statistic=f_stat, p_value=p_value, se_slope=se,
            coefficients={'slope': slope, 'intercept': intercept}
        )

    def correlation_interpretation(self, r: float) -> str:
        """相関係数の解釈"""
        r = abs(r)
        if r < 0.1:
            return 'ほぼ無相関'
        elif r < 0.3:
            return '弱い相関'
        elif r < 0.5:
            return '中程度の相関'
        elif r < 0.7:
            return 'やや強い相関'
        elif r < 0.9:
            return '強い相関'
        else:
            return '非常に強い相関'

    def analyze_hrv_subjective_relation(
        self,
        hrv_data: pd.DataFrame,
        subjective_data: pd.DataFrame,
        hrv_cols: List[str],
        subjective_cols: List[str],
        merge_on: str = 'Subject'
    ) -> pd.DataFrame:
        """HRV指標と主観評価の関連性を分析"""
        merged = pd.merge(hrv_data, subjective_data, on=merge_on, how='inner')

        results = []
        for hrv_col in hrv_cols:
            for subj_col in subjective_cols:
                if hrv_col not in merged.columns or subj_col not in merged.columns:
                    continue

                try:
                    pearson = self.pearson_correlation(
                        merged[hrv_col].values,
                        merged[subj_col].values
                    )
                    spearman = self.spearman_correlation(
                        merged[hrv_col].values,
                        merged[subj_col].values
                    )

                    results.append({
                        'HRV指標': hrv_col,
                        '主観評価': subj_col,
                        'Pearson r': pearson.r,
                        'Pearson p': pearson.p_value,
                        'Spearman ρ': spearman.r,
                        'Spearman p': spearman.p_value,
                        'n': pearson.n,
                        '解釈': self.correlation_interpretation(pearson.r),
                    })
                except Exception as e:
                    print(f"Warning: {hrv_col} vs {subj_col} の分析に失敗: {e}")

        return pd.DataFrame(results)
