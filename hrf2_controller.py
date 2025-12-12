# hrf2_controller.py
"""
HRF2 Controller - PID制御/適応制御による心拍数追従モード

目標心拍数に向けて抑揚レベルを自動調整し、
被験者の心拍数を目標値に追従させる。

制御方式:
1. PID制御 - 従来の比例・積分・微分制御
2. 適応制御 (MRAC) - Model Reference Adaptive Control with MIT rule

- BPMが目標より低い → 抑揚を上げて興奮させる
- BPMが目標より高い → 抑揚を下げて落ち着かせる
"""

import time
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class ControlMode(Enum):
    """制御モードの列挙型"""
    PID = "PID"
    ADAPTIVE = "Adaptive"
    GAIN_SCHEDULED = "GainScheduled"


@dataclass
class HRF2Config:
    """HRF2制御のパラメータ設定"""
    # 制御モード
    control_mode: ControlMode = ControlMode.PID

    # PIDゲイン
    kp: float = 0.02       # 比例ゲイン
    ki: float = 0.005      # 積分ゲイン
    kd: float = 0.01       # 微分ゲイン

    # 目標心拍数
    target_hr: float = 70.0

    # 抑揚の出力範囲
    min_output: float = 0.0
    max_output: float = 2.0

    # 積分項のアンチワインドアップ
    integral_max: float = 20.0

    # デッドバンド（目標値付近で制御を緩める）
    deadband: float = 3.0  # BPM


@dataclass
class AdaptiveConfig:
    """適応制御(MRAC)のパラメータ設定"""
    # 参照モデルパラメータ（一次遅れ系: τ * dy/dt + y = r）
    # τが小さいほど応答が速い（心拍数の生理的応答を考慮して設定）
    reference_time_constant: float = 30.0  # 秒（心拍数の応答遅れを考慮）

    # MIT則の適応ゲイン
    gamma: float = 0.001  # 適応速度（大きいほど速く適応するが不安定になりやすい）

    # 適応パラメータの制限
    theta_min: float = 0.005  # 適応パラメータの最小値（正のゲインを維持）
    theta_max: float = 0.1    # 適応パラメータの最大値

    # 忘却係数（パラメータドリフト防止）
    forgetting_factor: float = 0.995

    # デッドゾーン（小さい誤差では適応しない）
    deadzone: float = 2.0  # BPM

    # 正規化ゲイン（数値安定性のため）
    normalization_gain: float = 0.01


class GainType(Enum):
    """ゲインの種類"""
    P = "P"      # 比例のみ
    PI = "PI"    # 比例 + 積分
    PD = "PD"    # 比例 + 微分
    PID = "PID"  # 比例 + 積分 + 微分


@dataclass
class GainScheduleConfig:
    """
    ゲインスケジューリング制御のパラメータ設定

    誤差の大きさに応じてPIDゲインを切り替える：
    - 大誤差: 積極的に追従（高ゲイン）
    - 中誤差: 通常追従
    - 小誤差: 微調整（低ゲイン）
    """
    # 誤差領域の閾値 (BPM)
    error_threshold_high: float = 15.0   # これ以上は大誤差
    error_threshold_medium: float = 7.0  # これ以上は中誤差、以下は小誤差

    # 大誤差時のゲイン（積極的追従）
    kp_high: float = 0.04
    ki_high: float = 0.008
    kd_high: float = 0.015
    gain_type_high: GainType = GainType.PID

    # 中誤差時のゲイン（通常追従）
    kp_medium: float = 0.02
    ki_medium: float = 0.005
    kd_medium: float = 0.01
    gain_type_medium: GainType = GainType.PI

    # 小誤差時のゲイン（微調整）
    kp_low: float = 0.01
    ki_low: float = 0.002
    kd_low: float = 0.005
    gain_type_low: GainType = GainType.P

    # デッドバンド（目標値付近で制御を緩める）
    deadband: float = 2.0  # BPM

    # 積分項のアンチワインドアップ
    integral_max: float = 15.0

    # ゲイン切替時の平滑化係数（0-1, 1で即時切替）
    smoothing_factor: float = 0.3


class AdaptiveController:
    """
    適応制御コントローラー (MRAC with MIT Rule)

    Model Reference Adaptive Control (MRAC) を用いて、
    心拍数を目標値に追従させる。

    MITルール:
    - 参照モデル: 理想的な心拍数応答を定義（一次遅れ系）
    - 追従誤差: 実際の心拍数と参照モデル出力の差
    - 適応則: 追従誤差を最小化するようにパラメータを調整

    心拍数フィードバックの特性:
    - 人体の心拍数応答には遅れがある（数十秒オーダー）
    - 個人差が大きい
    - 非線形性がある（興奮と鎮静で応答が異なる）
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()

        # 適応パラメータ（MIT則で更新される）
        self._theta: float = 0.02  # 初期ゲイン（PIDのKpと同程度）

        # 参照モデルの状態
        self._reference_hr: float = 70.0  # 参照モデル出力
        self._target_hr: float = 70.0

        # 時間管理
        self._last_time: Optional[float] = None

        # 履歴（適応のため）
        self._hr_history: List[Tuple[float, float]] = []  # (time, hr)
        self._output_history: List[Tuple[float, float]] = []  # (time, output)
        self._max_history_length: int = 100

        # 状態
        self._last_output: float = 1.0
        self._last_error: float = 0.0

    def reset(self) -> None:
        """適応制御の状態をリセット"""
        self._theta = 0.02
        self._reference_hr = self._target_hr
        self._last_time = None
        self._hr_history.clear()
        self._output_history.clear()
        self._last_output = 1.0
        self._last_error = 0.0
        print("Adaptive Controller reset")

    def set_target_hr(self, target_hr: float) -> None:
        """目標心拍数を設定"""
        self._target_hr = max(40.0, min(180.0, target_hr))

    def get_target_hr(self) -> float:
        return self._target_hr

    def update(self, current_hr: float, min_output: float, max_output: float) -> Tuple[float, dict]:
        """
        適応制御による出力計算

        Args:
            current_hr: 現在の心拍数 (BPM)
            min_output: 出力の最小値
            max_output: 出力の最大値

        Returns:
            Tuple[float, dict]: (抑揚レベル, デバッグ情報)
        """
        current_time = time.time()

        # 時間差分計算
        dt = 0.0
        if self._last_time is not None:
            dt = current_time - self._last_time
            if dt > 10.0 or dt <= 0:
                dt = 0.0

        # --- 参照モデルの更新 ---
        # 一次遅れ系: τ * d(ref)/dt + ref = target
        # 離散化: ref[k+1] = ref[k] + (dt/τ) * (target - ref[k])
        if dt > 0:
            tau = self.config.reference_time_constant
            alpha = dt / tau
            alpha = min(alpha, 1.0)  # 安定性のため
            self._reference_hr += alpha * (self._target_hr - self._reference_hr)

        # --- 追従誤差の計算 ---
        # 追従誤差 = 参照モデル出力 - 実際の心拍数
        tracking_error = self._reference_hr - current_hr

        # --- MIT則による適応パラメータ更新 ---
        # dθ/dt = -γ * e * (∂e/∂θ)
        # 簡略化: θ[k+1] = θ[k] + γ * e * sign(感度)
        # 心拍数が低い → 抑揚を上げる → 正のθ変化
        if dt > 0 and abs(tracking_error) > self.config.deadzone:
            # 正規化された適応則（数値安定性のため）
            sensitivity = self._last_output  # 出力に対する感度
            norm_factor = 1.0 + self.config.normalization_gain * sensitivity * sensitivity

            # MIT則: 誤差と感度の積に基づいて更新
            delta_theta = self.config.gamma * tracking_error * sensitivity / norm_factor

            # 忘却係数の適用（ゼロへのドリフト防止）
            self._theta = self.config.forgetting_factor * self._theta + delta_theta

            # パラメータのクランプ
            self._theta = max(self.config.theta_min, min(self.config.theta_max, self._theta))

        # --- 制御出力の計算 ---
        # 制御誤差 = 目標 - 現在
        control_error = self._target_hr - current_hr

        # デッドゾーン処理
        if abs(control_error) < self.config.deadzone:
            effective_error = 0.0
        else:
            effective_error = control_error

        # 適応ゲインを用いた出力計算
        raw_output = 1.0 + self._theta * effective_error

        # 出力範囲にクランプ
        output = max(min_output, min(max_output, raw_output))

        # --- 履歴の更新 ---
        self._hr_history.append((current_time, current_hr))
        self._output_history.append((current_time, output))

        # 履歴の長さ制限
        if len(self._hr_history) > self._max_history_length:
            self._hr_history.pop(0)
        if len(self._output_history) > self._max_history_length:
            self._output_history.pop(0)

        # 状態更新
        self._last_time = current_time
        self._last_output = output
        self._last_error = tracking_error

        debug_info = {
            "control_mode": "Adaptive",
            "target_hr": self._target_hr,
            "current_hr": current_hr,
            "reference_hr": self._reference_hr,
            "tracking_error": tracking_error,
            "control_error": control_error,
            "theta": self._theta,
            "raw_output": raw_output,
            "output": output
        }

        return output, debug_info

    def get_theta(self) -> float:
        """現在の適応パラメータを取得"""
        return self._theta

    def set_gamma(self, gamma: float) -> None:
        """適応ゲインを設定"""
        self.config.gamma = max(0.0001, min(0.01, gamma))
        print(f"Adaptive gamma set to {self.config.gamma}")

    def set_time_constant(self, tau: float) -> None:
        """参照モデルの時定数を設定"""
        self.config.reference_time_constant = max(5.0, min(120.0, tau))
        print(f"Reference time constant set to {self.config.reference_time_constant}")


class GainScheduledController:
    """
    ゲインスケジューリング制御コントローラー

    誤差の大きさに応じてPIDゲインを動的に切り替える制御方式。
    - 大きな誤差: 高ゲインで積極的に追従
    - 中程度の誤差: 標準的なゲインで安定追従
    - 小さな誤差: 低ゲインで微調整（オーバーシュート抑制）

    特徴:
    - 収束速度と安定性のバランスを自動調整
    - ゲイン切替時の平滑化でチャタリング防止
    """

    def __init__(self, config: Optional[GainScheduleConfig] = None):
        self.config = config or GainScheduleConfig()

        # PID状態
        self._integral: float = 0.0
        self._last_error: Optional[float] = None
        self._last_time: Optional[float] = None

        # 現在の平滑化されたゲイン
        self._current_kp: float = self.config.kp_medium
        self._current_ki: float = self.config.ki_medium
        self._current_kd: float = self.config.kd_medium

        # 目標心拍数
        self._target_hr: float = 70.0

        # 現在のゲイン領域（デバッグ用）
        self._current_zone: str = "medium"
        self._current_gain_type: GainType = self.config.gain_type_medium

        # 最後の出力
        self._last_output: float = 1.0

    def reset(self) -> None:
        """制御状態をリセット"""
        self._integral = 0.0
        self._last_error = None
        self._last_time = None
        self._current_kp = self.config.kp_medium
        self._current_ki = self.config.ki_medium
        self._current_kd = self.config.kd_medium
        self._current_zone = "medium"
        self._current_gain_type = self.config.gain_type_medium
        self._last_output = 1.0
        print("GainScheduled Controller reset")

    def set_target_hr(self, target_hr: float) -> None:
        """目標心拍数を設定"""
        self._target_hr = max(40.0, min(180.0, target_hr))

    def get_target_hr(self) -> float:
        return self._target_hr

    def _get_target_gains(self, abs_error: float) -> Tuple[float, float, float, str, GainType]:
        """
        誤差の大きさに基づいて目標ゲインを決定

        Returns:
            Tuple[kp, ki, kd, zone_name, gain_type]
        """
        if abs_error >= self.config.error_threshold_high:
            gt = self.config.gain_type_high
            kp = self.config.kp_high
            ki = self.config.ki_high if gt in (GainType.PI, GainType.PID) else 0.0
            kd = self.config.kd_high if gt in (GainType.PD, GainType.PID) else 0.0
            return (kp, ki, kd, "high", gt)
        elif abs_error >= self.config.error_threshold_medium:
            gt = self.config.gain_type_medium
            kp = self.config.kp_medium
            ki = self.config.ki_medium if gt in (GainType.PI, GainType.PID) else 0.0
            kd = self.config.kd_medium if gt in (GainType.PD, GainType.PID) else 0.0
            return (kp, ki, kd, "medium", gt)
        else:
            gt = self.config.gain_type_low
            kp = self.config.kp_low
            ki = self.config.ki_low if gt in (GainType.PI, GainType.PID) else 0.0
            kd = self.config.kd_low if gt in (GainType.PD, GainType.PID) else 0.0
            return (kp, ki, kd, "low", gt)

    def update(self, current_hr: float, target_hr: float,
               min_output: float, max_output: float) -> Tuple[float, dict]:
        """
        ゲインスケジューリング制御による出力計算

        Args:
            current_hr: 現在の心拍数 (BPM)
            target_hr: 目標心拍数 (BPM)
            min_output: 出力の最小値
            max_output: 出力の最大値

        Returns:
            Tuple[float, dict]: (抑揚レベル, デバッグ情報)
        """
        current_time = time.time()
        self._target_hr = target_hr

        # 誤差計算
        error = target_hr - current_hr
        abs_error = abs(error)

        # デッドバンド処理
        effective_error = 0.0 if abs_error < self.config.deadband else error

        # 目標ゲインの決定
        target_kp, target_ki, target_kd, zone, gain_type = self._get_target_gains(abs_error)

        # ゲインの平滑化（急激な切替を防ぐ）
        alpha = self.config.smoothing_factor
        self._current_kp += alpha * (target_kp - self._current_kp)
        self._current_ki += alpha * (target_ki - self._current_ki)
        self._current_kd += alpha * (target_kd - self._current_kd)
        self._current_zone = zone
        self._current_gain_type = gain_type

        # 時間差分
        dt = 0.0
        if self._last_time is not None:
            dt = current_time - self._last_time
            if dt > 10.0 or dt <= 0:
                dt = 0.0

        # 比例項
        p_term = self._current_kp * effective_error

        # 積分項
        if dt > 0:
            self._integral += effective_error * dt
            # アンチワインドアップ
            self._integral = max(-self.config.integral_max,
                                 min(self.config.integral_max, self._integral))
        i_term = self._current_ki * self._integral

        # 微分項
        d_term = 0.0
        if self._last_error is not None and dt > 0:
            derivative = (effective_error - self._last_error) / dt
            d_term = self._current_kd * derivative

        # PID出力（ベース1.0からの調整）
        raw_output = 1.0 + p_term + i_term + d_term

        # 出力範囲にクランプ
        output = max(min_output, min(max_output, raw_output))

        # 状態更新
        self._last_error = effective_error
        self._last_time = current_time
        self._last_output = output

        debug_info = {
            "control_mode": "GainScheduled",
            "target_hr": target_hr,
            "current_hr": current_hr,
            "error": error,
            "effective_error": effective_error,
            "zone": zone,
            "gain_type": gain_type.value,
            "kp": self._current_kp,
            "ki": self._current_ki,
            "kd": self._current_kd,
            "p_term": p_term,
            "i_term": i_term,
            "d_term": d_term,
            "raw_output": raw_output,
            "output": output,
            "integral": self._integral
        }

        return output, debug_info

    def get_current_gains(self) -> Tuple[float, float, float]:
        """現在のゲインを取得"""
        return self._current_kp, self._current_ki, self._current_kd

    def get_current_zone(self) -> str:
        """現在のゲイン領域を取得"""
        return self._current_zone

    def get_current_gain_type(self) -> GainType:
        """現在のゲインタイプを取得"""
        return self._current_gain_type

    def set_thresholds(self, high: float, medium: float) -> None:
        """誤差閾値を設定"""
        self.config.error_threshold_high = max(5.0, high)
        self.config.error_threshold_medium = max(2.0, min(high - 1.0, medium))
        print(f"GainSchedule thresholds: high={self.config.error_threshold_high}, "
              f"medium={self.config.error_threshold_medium}")

    def set_gain_types(self, high: GainType, medium: GainType, low: GainType) -> None:
        """各ゾーンのゲインタイプを設定"""
        self.config.gain_type_high = high
        self.config.gain_type_medium = medium
        self.config.gain_type_low = low
        print(f"GainSchedule types: high={high.value}, medium={medium.value}, low={low.value}")


class HRF2Controller:
    """
    HRF2 - PID制御/適応制御/ゲインスケジューリング制御による心拍数追従コントローラー

    目標心拍数に対して現在の心拍数を追従させるため、
    抑揚パラメータを制御する。

    制御モード:
    - PID: 従来のPID制御
    - Adaptive: MRAC（モデル規範型適応制御）
    - GainScheduled: 誤差ベースのゲインスケジューリング
    """

    def __init__(self, config: Optional[HRF2Config] = None,
                 adaptive_config: Optional[AdaptiveConfig] = None,
                 gain_schedule_config: Optional[GainScheduleConfig] = None):
        self.config = config or HRF2Config()
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        self.gain_schedule_config = gain_schedule_config or GainScheduleConfig()

        # PID制御の内部状態
        self._integral: float = 0.0
        self._last_error: Optional[float] = None
        self._last_time: Optional[float] = None

        # 適応制御コントローラー
        self._adaptive_controller = AdaptiveController(self.adaptive_config)

        # ゲインスケジューリングコントローラー
        self._gain_scheduled_controller = GainScheduledController(self.gain_schedule_config)

        # 有効/無効フラグ
        self._enabled: bool = False

        # 最後の出力値
        self._last_output: float = 1.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if value and not self._enabled:
            # 有効化時にリセット
            self.reset()
        self._enabled = value
        print(f"HRF2 Controller {'enabled' if value else 'disabled'}")

    @property
    def control_mode(self) -> ControlMode:
        return self.config.control_mode

    @control_mode.setter
    def control_mode(self, value: ControlMode) -> None:
        self.config.control_mode = value
        print(f"HRF2 control mode set to {value.value}")

    @property
    def target_hr(self) -> float:
        return self.config.target_hr

    @target_hr.setter
    def target_hr(self, value: float) -> None:
        self.config.target_hr = max(40.0, min(180.0, value))
        self._adaptive_controller.set_target_hr(self.config.target_hr)
        self._gain_scheduled_controller.set_target_hr(self.config.target_hr)
        print(f"HRF2 target HR set to {self.config.target_hr} BPM")

    def reset(self) -> None:
        """制御状態をリセット（全モード）"""
        # PID状態リセット
        self._integral = 0.0
        self._last_error = None
        self._last_time = None
        self._last_output = 1.0

        # 適応制御状態リセット
        self._adaptive_controller.reset()
        self._adaptive_controller.set_target_hr(self.config.target_hr)

        # ゲインスケジューリング状態リセット
        self._gain_scheduled_controller.reset()
        self._gain_scheduled_controller.set_target_hr(self.config.target_hr)
        print("HRF2 Controller reset (all modes)")

    def update(self, current_hr: float) -> Tuple[float, dict]:
        """
        現在の心拍数から抑揚レベルを計算

        Args:
            current_hr: 現在の心拍数 (BPM)

        Returns:
            Tuple[float, dict]: (抑揚レベル, デバッグ情報)
        """
        if not self._enabled or current_hr <= 0:
            return self._last_output, {"enabled": False, "reason": "disabled or invalid HR"}

        # 制御モードに応じて処理を分岐
        if self.config.control_mode == ControlMode.ADAPTIVE:
            return self._update_adaptive(current_hr)
        elif self.config.control_mode == ControlMode.GAIN_SCHEDULED:
            return self._update_gain_scheduled(current_hr)
        else:
            return self._update_pid(current_hr)

    def _update_pid(self, current_hr: float) -> Tuple[float, dict]:
        """PID制御による出力計算"""
        current_time = time.time()

        # 誤差計算（目標 - 現在）
        # 正の誤差 = BPMが低い → 抑揚を上げる
        # 負の誤差 = BPMが高い → 抑揚を下げる
        error = self.config.target_hr - current_hr

        # デッドバンド処理
        if abs(error) < self.config.deadband:
            error = 0.0

        # 時間差分
        dt = 0.0
        if self._last_time is not None:
            dt = current_time - self._last_time
            # 異常な時間差は無視
            if dt > 10.0 or dt <= 0:
                dt = 0.0

        # 比例項
        p_term = self.config.kp * error

        # 積分項
        if dt > 0:
            self._integral += error * dt
            # アンチワインドアップ
            self._integral = max(-self.config.integral_max,
                                 min(self.config.integral_max, self._integral))
        i_term = self.config.ki * self._integral

        # 微分項
        d_term = 0.0
        if self._last_error is not None and dt > 0:
            derivative = (error - self._last_error) / dt
            d_term = self.config.kd * derivative

        # PID出力（ベース1.0からの調整）
        raw_output = 1.0 + p_term + i_term + d_term

        # 出力範囲にクランプ
        output = max(self.config.min_output,
                     min(self.config.max_output, raw_output))

        # 状態更新
        self._last_error = error
        self._last_time = current_time
        self._last_output = output

        debug_info = {
            "enabled": True,
            "control_mode": "PID",
            "target_hr": self.config.target_hr,
            "current_hr": current_hr,
            "error": error,
            "p_term": p_term,
            "i_term": i_term,
            "d_term": d_term,
            "raw_output": raw_output,
            "output": output,
            "integral": self._integral
        }

        return output, debug_info

    def _update_adaptive(self, current_hr: float) -> Tuple[float, dict]:
        """適応制御による出力計算"""
        output, debug_info = self._adaptive_controller.update(
            current_hr,
            self.config.min_output,
            self.config.max_output
        )
        self._last_output = output
        debug_info["enabled"] = True
        return output, debug_info

    def _update_gain_scheduled(self, current_hr: float) -> Tuple[float, dict]:
        """ゲインスケジューリング制御による出力計算"""
        output, debug_info = self._gain_scheduled_controller.update(
            current_hr,
            self.config.target_hr,
            self.config.min_output,
            self.config.max_output
        )
        self._last_output = output
        debug_info["enabled"] = True
        return output, debug_info

    def get_status_text(self, current_hr: float) -> str:
        """現在の状態をテキストで返す"""
        if not self._enabled:
            return "HRF2: 無効"

        error = self.config.target_hr - current_hr
        direction = "↑" if error > 0 else "↓" if error < 0 else "→"
        mode_str = self.config.control_mode.value

        if self.config.control_mode == ControlMode.ADAPTIVE:
            theta = self._adaptive_controller.get_theta()
            return (f"HRF2({mode_str}): 目標{self.config.target_hr:.0f}BPM "
                    f"現在{current_hr:.0f}BPM {direction} "
                    f"θ={theta:.4f} 抑揚{self._last_output:.2f}")
        elif self.config.control_mode == ControlMode.GAIN_SCHEDULED:
            zone = self._gain_scheduled_controller.get_current_zone()
            kp, ki, kd = self._gain_scheduled_controller.get_current_gains()
            return (f"HRF2({mode_str}): 目標{self.config.target_hr:.0f}BPM "
                    f"現在{current_hr:.0f}BPM {direction} "
                    f"zone={zone} Kp={kp:.3f} 抑揚{self._last_output:.2f}")
        else:
            return (f"HRF2({mode_str}): 目標{self.config.target_hr:.0f}BPM "
                    f"現在{current_hr:.0f}BPM {direction} "
                    f"抑揚{self._last_output:.2f}")

    def set_pid_gains(self, kp: float, ki: float, kd: float) -> None:
        """PIDゲインを設定"""
        self.config.kp = max(0.0, kp)
        self.config.ki = max(0.0, ki)
        self.config.kd = max(0.0, kd)
        print(f"HRF2 PID gains: Kp={self.config.kp}, Ki={self.config.ki}, Kd={self.config.kd}")

    def set_adaptive_params(self, gamma: float, tau: float) -> None:
        """適応制御パラメータを設定"""
        self._adaptive_controller.set_gamma(gamma)
        self._adaptive_controller.set_time_constant(tau)

    def get_adaptive_theta(self) -> float:
        """現在の適応パラメータθを取得"""
        return self._adaptive_controller.get_theta()

    def get_adaptive_config(self) -> AdaptiveConfig:
        """適応制御設定を取得"""
        return self._adaptive_controller.config

    def set_output_range(self, min_output: float, max_output: float) -> None:
        """出力範囲を設定"""
        self.config.min_output = max(0.0, min(3.0, min_output))
        self.config.max_output = max(0.0, min(3.0, max_output))
        print(f"HRF2 output range: {self.config.min_output} - {self.config.max_output}")

    def set_gain_schedule_thresholds(self, high: float, medium: float) -> None:
        """ゲインスケジューリングの閾値を設定"""
        self._gain_scheduled_controller.set_thresholds(high, medium)

    def get_gain_schedule_config(self) -> GainScheduleConfig:
        """ゲインスケジューリング設定を取得"""
        return self._gain_scheduled_controller.config

    def get_gain_schedule_zone(self) -> str:
        """現在のゲイン領域を取得"""
        return self._gain_scheduled_controller.get_current_zone()

    def get_gain_schedule_gains(self) -> Tuple[float, float, float]:
        """現在のゲインスケジューリングゲインを取得"""
        return self._gain_scheduled_controller.get_current_gains()

    def set_gain_schedule_types(self, high: GainType, medium: GainType, low: GainType) -> None:
        """ゲインスケジューリングの各ゾーンのゲインタイプを設定"""
        self._gain_scheduled_controller.set_gain_types(high, medium, low)

    def get_gain_schedule_gain_type(self) -> GainType:
        """現在のゲインタイプを取得"""
        return self._gain_scheduled_controller.get_current_gain_type()
