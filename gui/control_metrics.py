# gui/control_metrics.py
"""制御性能評価モジュール

心拍フィードバック制御の性能を定量的に評価するための指標を計算する。
参考: /Users/user/Research/Analys/advanced_analysis/control_metrics.py
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class ControlMetrics:
    """制御性能指標"""
    rmse: float  # Root Mean Square Error
    mae: float  # Mean Absolute Error
    rise_time: Optional[float]  # 立ち上がり時間（秒）
    settling_time: Optional[float]  # 整定時間（秒）
    overshoot: Optional[float]  # オーバーシュート（%）
    control_rate: float  # 制御率（目標±tolerance_bpm以内の割合）
    convergence_rate: float  # 収束率

    def to_dict(self, include_metrics: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """指標を辞書形式で返す

        Args:
            include_metrics: 含める指標を指定する辞書。Noneの場合は全て含める。
                           キー: 'rmse', 'mae', 'control_rate', 'convergence_rate',
                                'rise_time', 'settling_time', 'overshoot'
        """
        all_metrics = {
            'RMSE (BPM)': self.rmse,
            'MAE (BPM)': self.mae,
            '立ち上がり時間 (秒)': self.rise_time,
            '整定時間 (秒)': self.settling_time,
            'オーバーシュート (%)': self.overshoot,
            '制御率 (%)': self.control_rate * 100,
            '収束率 (%)': self.convergence_rate * 100,
        }

        if include_metrics is None:
            return all_metrics

        # 指定された指標のみを含める
        metric_key_map = {
            'rmse': 'RMSE (BPM)',
            'mae': 'MAE (BPM)',
            'rise_time': '立ち上がり時間 (秒)',
            'settling_time': '整定時間 (秒)',
            'overshoot': 'オーバーシュート (%)',
            'control_rate': '制御率 (%)',
            'convergence_rate': '収束率 (%)',
        }

        filtered = {}
        for key, should_include in include_metrics.items():
            if should_include and key in metric_key_map:
                dict_key = metric_key_map[key]
                filtered[dict_key] = all_metrics[dict_key]

        return filtered


class ControlMetricsAnalyzer:
    """制御性能解析クラス"""

    def __init__(
        self,
        tolerance_bpm: float = 5.0,
        settling_threshold: float = 0.05,
        rise_threshold: float = 0.9,
        steady_state_ratio: float = 0.2
    ):
        """
        Args:
            tolerance_bpm: 制御率計算時の許容誤差 (BPM)
            settling_threshold: 整定判定の閾値（目標値に対する割合）
            rise_threshold: 立ち上がり時間の閾値（90%到達）
            steady_state_ratio: 定常偏差計算に使用する終盤の割合
        """
        self.tolerance_bpm = tolerance_bpm
        self.settling_threshold = settling_threshold
        self.rise_threshold = rise_threshold
        self.steady_state_ratio = steady_state_ratio

    def calculate_metrics(
        self,
        time: np.ndarray,
        hr_actual: np.ndarray,
        hr_target: float,
        hr_initial: Optional[float] = None
    ) -> ControlMetrics:
        """制御性能指標を計算

        Args:
            time: 時間配列（秒）
            hr_actual: 実測心拍数配列
            hr_target: 目標心拍数
            hr_initial: 初期心拍数（Noneの場合は最初の値を使用）

        Returns:
            ControlMetrics: 制御性能指標
        """
        if len(hr_actual) == 0:
            return ControlMetrics(
                rmse=np.nan, mae=np.nan, rise_time=None,
                settling_time=None, overshoot=None,
                control_rate=0.0, convergence_rate=0.0
            )

        if hr_initial is None:
            hr_initial = hr_actual[0]

        # 基本誤差指標
        error = hr_actual - hr_target
        rmse = np.sqrt(np.mean(error ** 2))
        mae = np.mean(np.abs(error))

        # 定常偏差（終盤の平均誤差）
        n_steady = max(1, int(len(hr_actual) * self.steady_state_ratio))

        # 制御率（目標±tolerance_bpm以内の割合）
        within_tolerance = np.abs(error) <= self.tolerance_bpm
        control_rate = np.mean(within_tolerance)

        # 収束率（初期値から目標への到達度）
        # 終盤20%区間の平均心拍数を使用（瞬間的なノイズの影響を軽減）
        if abs(hr_initial - hr_target) > 1.0:  # 初期値と目標の差が1BPM以上ある場合
            # 終盤20%区間の平均心拍数を計算
            n_final = max(1, int(len(hr_actual) * 0.2))
            final_hr_mean = np.mean(hr_actual[-n_final:])
            convergence_rate = 1 - abs(final_hr_mean - hr_target) / abs(hr_initial - hr_target)
            convergence_rate = max(0, min(1, convergence_rate))
        else:
            # 初期値が既に目標に近い場合は、目標±tolerance_bpm内の維持率で評価
            convergence_rate = control_rate  # 制御率と同じ値を使用

        # 立ち上がり時間
        rise_time = self._calculate_rise_time(time, hr_actual, hr_target, hr_initial)

        # 整定時間
        settling_time = self._calculate_settling_time(time, hr_actual, hr_target)

        # オーバーシュート
        overshoot = self._calculate_overshoot(hr_actual, hr_target, hr_initial)

        return ControlMetrics(
            rmse=rmse,
            mae=mae,
            rise_time=rise_time,
            settling_time=settling_time,
            overshoot=overshoot,
            control_rate=control_rate,
            convergence_rate=convergence_rate
        )

    def _calculate_rise_time(
        self,
        time: np.ndarray,
        hr_actual: np.ndarray,
        hr_target: float,
        hr_initial: float
    ) -> Optional[float]:
        """立ち上がり時間を計算（目標値の90%到達時間）"""
        if hr_initial == hr_target:
            return 0.0

        threshold = hr_initial + (hr_target - hr_initial) * self.rise_threshold

        if hr_target > hr_initial:
            # 心拍数を上げる場合
            idx = np.where(hr_actual >= threshold)[0]
        else:
            # 心拍数を下げる場合
            idx = np.where(hr_actual <= threshold)[0]

        if len(idx) > 0:
            return time[idx[0]] - time[0]
        return None

    def _calculate_settling_time(
        self,
        time: np.ndarray,
        hr_actual: np.ndarray,
        hr_target: float
    ) -> Optional[float]:
        """整定時間を計算（目標値±5%以内に収束する時間）"""
        tolerance = hr_target * self.settling_threshold
        within_band = np.abs(hr_actual - hr_target) <= tolerance

        # 最後から連続して収束している区間を探す
        settled_idx = None
        for i in range(len(within_band) - 1, -1, -1):
            if within_band[i]:
                settled_idx = i
            else:
                break

        if settled_idx is not None and settled_idx < len(within_band) - 1:
            # 最初に収束した時点を探す
            for i in range(settled_idx + 1):
                if all(within_band[i:settled_idx + 1]):
                    return time[i] - time[0]

        return None

    def _calculate_overshoot(
        self,
        hr_actual: np.ndarray,
        hr_target: float,
        hr_initial: float
    ) -> Optional[float]:
        """オーバーシュートを計算"""
        if hr_initial == hr_target:
            return 0.0

        if hr_target > hr_initial:
            # 心拍数を上げる場合
            max_hr = np.max(hr_actual)
            if max_hr > hr_target:
                return (max_hr - hr_target) / (hr_target - hr_initial) * 100
        else:
            # 心拍数を下げる場合
            min_hr = np.min(hr_actual)
            if min_hr < hr_target:
                return (hr_target - min_hr) / (hr_initial - hr_target) * 100

        return 0.0

    def analyze_from_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str = 'Time',
        hr_col: str = 'HR',
        target_hr: float = 70.0
    ) -> ControlMetrics:
        """DataFrameから制御性能を解析

        Args:
            df: 解析対象のDataFrame
            time_col: 時間列名
            hr_col: 心拍数列名
            target_hr: 目標心拍数

        Returns:
            ControlMetrics: 制御性能指標
        """
        time = df[time_col].values
        hr_actual = df[hr_col].values

        return self.calculate_metrics(time, hr_actual, target_hr)


def save_metrics_to_csv(
    metrics_list: List[Dict[str, Any]],
    output_path: str
) -> None:
    """メトリクスをCSVファイルに保存

    Args:
        metrics_list: メトリクス辞書のリスト
                     各辞書には 'subject_id', 'condition', その他のメトリクスが含まれる
        output_path: 出力ファイルパス
    """
    if not metrics_list:
        return

    df = pd.DataFrame(metrics_list)

    # 列の順序を整理（subject_id, condition を先頭に）
    cols = df.columns.tolist()
    priority_cols = ['subject_id', 'condition', 'device_type']
    ordered_cols = [c for c in priority_cols if c in cols]
    ordered_cols += [c for c in cols if c not in priority_cols]
    df = df[ordered_cols]

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
