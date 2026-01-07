# gui/advanced_analysis/advanced_hrv.py
"""高度HRV解析モジュール

非線形HRV指標（Sample Entropy, DFA, Poincaréプロット）を計算する。
参考: /Users/user/Research/Analys/advanced_analysis/advanced_hrv.py
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d


@dataclass
class AdvancedHRVMetrics:
    """高度HRV指標"""
    # 時間領域（追加）
    pnn50: float  # 連続RR間隔差>50msの割合
    pnn20: float  # 連続RR間隔差>20msの割合
    rmssd: float  # RMSSD
    sdnn: float  # SDNN
    mean_rr: float  # 平均RR間隔
    mean_hr: float  # 平均心拍数

    # 周波数領域
    total_power: float  # 総パワー
    vlf_power: float  # VLF (0.003-0.04Hz)
    lf_power: float  # LF (0.04-0.15Hz)
    hf_power: float  # HF (0.15-0.4Hz)
    lf_hf_ratio: float  # LF/HF比
    lf_nu: float  # 正規化LF
    hf_nu: float  # 正規化HF

    # 非線形指標
    sd1: float  # ポアンカレプロットSD1（短期変動）
    sd2: float  # ポアンカレプロットSD2（長期変動）
    sd_ratio: float  # SD2/SD1比
    sample_entropy: Optional[float]  # サンプルエントロピー
    dfa_alpha1: Optional[float]  # DFA短期スケーリング指数
    dfa_alpha2: Optional[float]  # DFA長期スケーリング指数

    def to_dict(self) -> Dict:
        return {
            # 時間領域
            'pNN50 (%)': self.pnn50,
            'pNN20 (%)': self.pnn20,
            'RMSSD (ms)': self.rmssd,
            'SDNN (ms)': self.sdnn,
            'Mean RR (ms)': self.mean_rr,
            'Mean HR (BPM)': self.mean_hr,
            # 周波数領域
            'Total Power (ms²)': self.total_power,
            'VLF Power (ms²)': self.vlf_power,
            'LF Power (ms²)': self.lf_power,
            'HF Power (ms²)': self.hf_power,
            'LF/HF': self.lf_hf_ratio,
            'LF nu': self.lf_nu,
            'HF nu': self.hf_nu,
            # 非線形指標
            'SD1 (ms)': self.sd1,
            'SD2 (ms)': self.sd2,
            'SD2/SD1': self.sd_ratio,
            'Sample Entropy': self.sample_entropy,
            'DFA α1': self.dfa_alpha1,
            'DFA α2': self.dfa_alpha2,
        }


class AdvancedHRVAnalyzer:
    """高度HRV解析クラス"""

    def __init__(
        self,
        fs_resample: float = 4.0,
        vlf_band: Tuple[float, float] = (0.003, 0.04),
        lf_band: Tuple[float, float] = (0.04, 0.15),
        hf_band: Tuple[float, float] = (0.15, 0.4),
    ):
        self.fs_resample = fs_resample
        self.vlf_band = vlf_band
        self.lf_band = lf_band
        self.hf_band = hf_band

    def calculate_metrics(
        self,
        rr_intervals: np.ndarray,
        compute_nonlinear: bool = True
    ) -> AdvancedHRVMetrics:
        """HRV指標を計算

        Args:
            rr_intervals: RR間隔配列（ミリ秒）
            compute_nonlinear: 非線形指標を計算するか

        Returns:
            AdvancedHRVMetrics: HRV指標
        """
        rr = np.array(rr_intervals, dtype=float)
        rr = self._remove_ectopic_beats(rr)

        if len(rr) < 10:
            raise ValueError("有効なRR間隔が不足しています")

        time_domain = self._calculate_time_domain(rr)
        freq_domain = self._calculate_frequency_domain(rr)

        if compute_nonlinear:
            nonlinear = self._calculate_nonlinear(rr)
        else:
            nonlinear = {
                'sd1': 0.0, 'sd2': 0.0, 'sd_ratio': 0.0,
                'sample_entropy': None, 'dfa_alpha1': None, 'dfa_alpha2': None
            }

        return AdvancedHRVMetrics(
            pnn50=time_domain['pnn50'],
            pnn20=time_domain['pnn20'],
            rmssd=time_domain['rmssd'],
            sdnn=time_domain['sdnn'],
            mean_rr=time_domain['mean_rr'],
            mean_hr=time_domain['mean_hr'],
            total_power=freq_domain['total_power'],
            vlf_power=freq_domain['vlf_power'],
            lf_power=freq_domain['lf_power'],
            hf_power=freq_domain['hf_power'],
            lf_hf_ratio=freq_domain['lf_hf_ratio'],
            lf_nu=freq_domain['lf_nu'],
            hf_nu=freq_domain['hf_nu'],
            sd1=nonlinear['sd1'],
            sd2=nonlinear['sd2'],
            sd_ratio=nonlinear['sd_ratio'],
            sample_entropy=nonlinear['sample_entropy'],
            dfa_alpha1=nonlinear['dfa_alpha1'],
            dfa_alpha2=nonlinear['dfa_alpha2'],
        )

    def _remove_ectopic_beats(self, rr: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """異常拍を除去（20%以上の変化）"""
        if len(rr) < 2:
            return rr

        rr_diff = np.abs(np.diff(rr))
        rr_mean = np.mean(rr)
        mask = np.ones(len(rr), dtype=bool)

        for i in range(1, len(rr)):
            if rr_diff[i - 1] > rr_mean * threshold:
                mask[i] = False

        return rr[mask]

    def _calculate_time_domain(self, rr: np.ndarray) -> Dict:
        """時間領域指標を計算"""
        mean_rr = np.mean(rr)
        sdnn = np.std(rr, ddof=1)
        mean_hr = 60000 / mean_rr

        diff_rr = np.abs(np.diff(rr))
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        pnn50 = np.sum(diff_rr > 50) / len(diff_rr) * 100
        pnn20 = np.sum(diff_rr > 20) / len(diff_rr) * 100

        return {
            'mean_rr': mean_rr,
            'sdnn': sdnn,
            'mean_hr': mean_hr,
            'rmssd': rmssd,
            'pnn50': pnn50,
            'pnn20': pnn20,
        }

    def _calculate_frequency_domain(self, rr: np.ndarray) -> Dict:
        """周波数領域指標を計算（Welch法）"""
        time = np.cumsum(rr) / 1000
        time = time - time[0]

        if len(time) < 4:
            return self._empty_freq_domain()

        f_interp = interp1d(time, rr, kind='cubic', fill_value='extrapolate')
        t_resample = np.arange(time[0], time[-1], 1 / self.fs_resample)
        rr_resample = f_interp(t_resample)
        rr_resample = signal.detrend(rr_resample)

        nperseg = min(256, len(rr_resample) // 2)
        if nperseg < 16:
            return self._empty_freq_domain()

        freqs, psd = signal.welch(
            rr_resample,
            fs=self.fs_resample,
            nperseg=nperseg,
            noverlap=nperseg // 2
        )

        vlf_power = self._band_power(freqs, psd, self.vlf_band)
        lf_power = self._band_power(freqs, psd, self.lf_band)
        hf_power = self._band_power(freqs, psd, self.hf_band)
        total_power = vlf_power + lf_power + hf_power

        lf_hf_sum = lf_power + hf_power
        lf_nu = (lf_power / lf_hf_sum * 100) if lf_hf_sum > 0 else 0
        hf_nu = (hf_power / lf_hf_sum * 100) if lf_hf_sum > 0 else 0
        lf_hf_ratio = (lf_power / hf_power) if hf_power > 0 else 0

        return {
            'total_power': total_power,
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_hf_ratio': lf_hf_ratio,
            'lf_nu': lf_nu,
            'hf_nu': hf_nu,
        }

    def _empty_freq_domain(self) -> Dict:
        return {
            'total_power': 0.0, 'vlf_power': 0.0, 'lf_power': 0.0,
            'hf_power': 0.0, 'lf_hf_ratio': 0.0, 'lf_nu': 0.0, 'hf_nu': 0.0,
        }

    def _band_power(self, freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if np.sum(mask) == 0:
            return 0.0
        return np.trapz(psd[mask], freqs[mask])

    def _calculate_nonlinear(self, rr: np.ndarray) -> Dict:
        """非線形指標を計算"""
        result = {
            'sd1': 0.0, 'sd2': 0.0, 'sd_ratio': 0.0,
            'sample_entropy': None, 'dfa_alpha1': None, 'dfa_alpha2': None,
        }

        # ポアンカレプロット解析
        if len(rr) > 1:
            rr_n = rr[:-1]
            rr_n1 = rr[1:]
            sd1 = np.std(rr_n1 - rr_n, ddof=1) / np.sqrt(2)
            sd2 = np.std(rr_n1 + rr_n, ddof=1) / np.sqrt(2)
            result['sd1'] = sd1
            result['sd2'] = sd2
            result['sd_ratio'] = sd2 / sd1 if sd1 > 0 else 0

        # サンプルエントロピー
        try:
            result['sample_entropy'] = self._sample_entropy(rr, m=2, r=0.2)
        except Exception:
            result['sample_entropy'] = None

        # DFA
        try:
            alpha1, alpha2 = self._dfa(rr)
            result['dfa_alpha1'] = alpha1
            result['dfa_alpha2'] = alpha2
        except Exception:
            result['dfa_alpha1'] = None
            result['dfa_alpha2'] = None

        return result

    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """サンプルエントロピーを計算"""
        n = len(data)
        if n < m + 2:
            return np.nan

        r_val = r * np.std(data, ddof=1)

        def _count_matches(template_length):
            count = 0
            templates = np.array([
                data[i:i + template_length] for i in range(n - template_length)
            ])
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) <= r_val:
                        count += 1
            return count

        a = _count_matches(m + 1)
        b = _count_matches(m)

        if b == 0 or a == 0:
            return np.nan

        return -np.log(a / b)

    def _dfa(
        self, data: np.ndarray,
        scale_range1: Tuple[int, int] = (4, 16),
        scale_range2: Tuple[int, int] = (16, 64)
    ) -> Tuple[Optional[float], Optional[float]]:
        """デトレンド変動解析（DFA）"""
        n = len(data)
        if n < scale_range2[1]:
            return None, None

        y = np.cumsum(data - np.mean(data))

        def _calculate_fluctuation(scale):
            n_segments = n // scale
            if n_segments < 2:
                return np.nan
            fluctuations = []
            for i in range(n_segments):
                segment = y[i * scale:(i + 1) * scale]
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                fluctuations.append(np.sqrt(np.mean((segment - trend) ** 2)))
            return np.mean(fluctuations)

        scales1 = np.arange(scale_range1[0], min(scale_range1[1], n // 4) + 1)
        scales2 = np.arange(scale_range2[0], min(scale_range2[1], n // 4) + 1)

        def _fit_alpha(scales):
            if len(scales) < 3:
                return None
            f_values = np.array([_calculate_fluctuation(s) for s in scales])
            valid = ~np.isnan(f_values) & (f_values > 0)
            if np.sum(valid) < 3:
                return None
            coeffs = np.polyfit(np.log(scales[valid]), np.log(f_values[valid]), 1)
            return coeffs[0]

        return _fit_alpha(scales1), _fit_alpha(scales2)

    def analyze_from_dataframe(
        self,
        df: pd.DataFrame,
        rr_col: str = 'RR',
        compute_nonlinear: bool = True
    ) -> AdvancedHRVMetrics:
        """DataFrameからHRV指標を計算"""
        rr_intervals = df[rr_col].dropna().values
        return self.calculate_metrics(rr_intervals, compute_nonlinear)

    def compare_conditions(
        self,
        condition_rr: Dict[str, np.ndarray],
        compute_nonlinear: bool = True
    ) -> pd.DataFrame:
        """複数条件のHRV指標を比較"""
        results = []
        for condition, rr in condition_rr.items():
            try:
                metrics = self.calculate_metrics(rr, compute_nonlinear)
                metrics_dict = metrics.to_dict()
                metrics_dict['Condition'] = condition
                results.append(metrics_dict)
            except Exception as e:
                print(f"Warning: {condition} の解析に失敗: {e}")

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        cols = ['Condition'] + [c for c in result_df.columns if c != 'Condition']
        return result_df[cols]
