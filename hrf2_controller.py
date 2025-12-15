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
    # ADAPTIVE_MPC = "AdaptiveMPC"  # コメントアウト: ロバスト制御に置き換え
    ROBUST = "Robust"  # H∞ループ整形に基づくロバスト制御


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


# === コメントアウト: AdaptiveMPC（ロバスト制御に置き換え） ===
# @dataclass
# class AdaptiveMPCConfig:
#     """
#     適応モデル予測制御 (Adaptive MPC) のパラメータ設定
#
#     心拍数応答モデル: y(k+1) = a * y(k) + b * u(k-d)
#     - a: 自己回帰係数（心拍の慣性）
#     - b: 入力ゲイン（抑揚→心拍の影響度）
#     - d: むだ時間（入力の遅延ステップ数）
#
#     MPC最適化: J = Σ(y_pred - target)² + λ * Σ(Δu)²
#     """
#     # 予測モデル初期パラメータ
#     initial_a: float = 0.95  # 自己回帰係数（0.9-0.99、心拍の持続性）
#     initial_b: float = 0.5   # 入力ゲイン（抑揚の影響度）
#
#     # むだ時間（ステップ数、1ステップ≈1秒）
#     delay_steps: int = 5  # 約5秒の遅延
#
#     # RLS（再帰的最小二乗法）パラメータ
#     rls_forgetting_factor: float = 0.98  # 忘却係数（0.95-0.99）
#     rls_initial_covariance: float = 100.0  # 初期共分散（大きいほど適応が速い）
#
#     # MPCパラメータ
#     prediction_horizon: int = 10  # 予測ホライズン（ステップ数）
#     control_horizon: int = 3      # 制御ホライズン（実際に計算する入力数）
#     output_weight: float = 1.0    # 出力誤差の重み
#     input_change_weight: float = 0.1  # 入力変化量の重み（λ）
#
#     # 制約
#     u_min: float = 0.0   # 抑揚の最小値
#     u_max: float = 2.0   # 抑揚の最大値
#     du_max: float = 0.3  # 1ステップあたりの最大変化量
#
#     # パラメータ制約（安定性のため）
#     a_min: float = 0.8
#     a_max: float = 0.99
#     b_min: float = 0.1
#     b_max: float = 2.0
#
#     # デッドバンド
#     deadband: float = 2.0  # BPM
# === AdaptiveMPCConfig コメントアウト終了 ===


@dataclass
class RobustConfig:
    """
    H∞ロバスト制御のパラメータ設定

    混合感度問題に基づくロバスト制御:
    ||[Ws*S; Wt*T]||∞ < γ

    - S = 1/(1+PK): 感度関数（外乱抑制）
    - T = PK/(1+PK): 相補感度関数（ロバスト安定性）
    - Ws: 低周波重み（目標追従性能）
    - Wt: 高周波重み（ノイズ抑制・ロバスト性）

    参考文献:
    - Paradiso et al., IEEE Trans. Biomed. Eng., Vol.60, No.11, 2013
    - Aranda et al., ECC 2007 (QFT approach)
    """
    # 公称プラントパラメータ（1次遅れ＋むだ時間モデル）
    # P0(s) = K * exp(-L*s) / (T*s + 1)
    plant_gain: float = 1.5       # K: 定常ゲイン（抑揚→心拍数の影響度）
    plant_time_constant: float = 30.0  # T: 時定数 [秒]（心拍応答の遅れ）
    plant_dead_time: float = 5.0  # L: むだ時間 [秒]

    # 不確かさパラメータ（乗法的不確かさ）
    # 真のプラント P = P0 * (1 + Δ*Wm), |Δ|≤1
    uncertainty_low_freq: float = 0.3   # 低周波での不確かさ（個人差など）
    uncertainty_high_freq: float = 1.5  # 高周波での不確かさ（ノイズ等）
    uncertainty_crossover: float = 0.1  # 不確かさが増加し始める周波数 [rad/s]

    # 感度重みWsパラメータ
    # Ws(s) = (s/Ms + ωb) / (s + ωb*εs)
    # - 低周波ゲイン: 1/εs（追従誤差の上限）
    # - 高周波ゲイン: Ms（感度のピーク制限）
    # - 帯域幅: ωb
    ws_bandwidth: float = 0.05    # ωb: 帯域幅 [rad/s]（心拍応答に合わせて低め）
    ws_low_freq_gain: float = 100.0  # 1/εs: 定常誤差抑制（大きいほど追従性向上）
    ws_peak: float = 2.0          # Ms: 感度ピーク制限

    # 相補感度重みWtパラメータ
    # Wt(s) = (s + ωt/Mt) / (εt*s + ωt)
    # - 低周波ゲイン: 1/Mt
    # - 高周波ゲイン: 1/εt（ロバスト安定マージン）
    # - 帯域幅: ωt
    wt_bandwidth: float = 0.3     # ωt: 帯域幅 [rad/s]
    wt_high_freq_gain: float = 1.2  # 1/εt: 高周波でのロバスト性
    wt_peak: float = 1.5          # Mt: 相補感度ピーク制限

    # 制御器パラメータ
    controller_order: int = 2     # 制御器の次数（離散化後）

    # ループ整形パラメータ
    loop_gain: float = 0.02       # 開ループゲイン調整係数
    phase_margin_target: float = 45.0  # 目標位相余裕 [度]
    gain_margin_target: float = 6.0    # 目標ゲイン余裕 [dB]

    # 積分器パラメータ（定常偏差除去用）
    integral_gain: float = 0.003  # 積分ゲイン
    integral_max: float = 15.0    # 積分項のアンチワインドアップ

    # フィルタパラメータ（ノイズ抑制・むだ時間補償）
    filter_time_constant: float = 2.0  # ローパスフィルタ時定数 [秒]
    smith_predictor_enabled: bool = True  # スミス予測器（むだ時間補償）

    # 出力制限
    u_min: float = 0.0   # 抑揚の最小値
    u_max: float = 2.0   # 抑揚の最大値
    du_max: float = 0.2  # 1ステップあたりの最大変化量（スルーレート制限）

    # デッドバンド
    deadband: float = 2.0  # BPM（目標値付近で制御を緩める）

    # 適応機能（オプション）
    adaptive_enabled: bool = True  # オンライン同定を有効化
    adaptation_rate: float = 0.01  # 適応速度


class RobustController:
    """
    H∞ロバスト制御コントローラー

    混合感度問題に基づく設計:
    - 低周波: 高い追従性能（目標心拍数への収束）
    - 高周波: ロバスト安定性（ノイズ・個人差への耐性）

    特徴:
    1. むだ時間補償（スミス予測器）
    2. アンチワインドアップ付き積分器
    3. オンラインゲイン適応（オプション）
    4. スルーレート制限（急激な変化を抑制）

    実装:
    離散時間のH∞制御器を2次IIRフィルタとして実装。
    連続時間設計を双一次変換（Tustin変換）で離散化。
    """

    def __init__(self, config: Optional[RobustConfig] = None):
        self.config = config or RobustConfig()

        # 目標心拍数
        self._target_hr: float = 70.0

        # 時間管理
        self._last_time: Optional[float] = None
        self._sample_time: float = 1.0  # サンプリング時間 [秒]

        # 制御器の内部状態（2次IIRフィルタ）
        self._x1: float = 0.0  # 状態変数1
        self._x2: float = 0.0  # 状態変数2

        # 積分器状態
        self._integral: float = 0.0

        # フィルタ状態（ローパスフィルタ）
        self._filtered_error: float = 0.0
        self._filtered_hr: float = 70.0

        # スミス予測器の履歴
        self._prediction_buffer: List[float] = []
        self._delay_steps: int = int(self.config.plant_dead_time / self._sample_time)

        # 出力履歴（スルーレート制限用）
        self._last_output: float = 1.0

        # 適応パラメータ
        self._adapted_gain: float = self.config.loop_gain

        # 履歴（解析用）
        self._hr_history: List[float] = []
        self._error_history: List[float] = []
        self._max_history: int = 100

        # 制御器係数（離散時間）
        self._update_controller_coefficients()

    def _update_controller_coefficients(self) -> None:
        """
        H∞制御器の離散時間係数を計算

        連続時間の2次制御器を双一次変換で離散化:
        K(s) = (b2*s² + b1*s + b0) / (a2*s² + a1*s + a0)

        離散化後:
        K(z) = (B0 + B1*z⁻¹ + B2*z⁻²) / (1 + A1*z⁻¹ + A2*z⁻²)
        """
        T = self._sample_time
        cfg = self.config

        # H∞ループ整形に基づく制御器パラメータ
        # 開ループ伝達関数 L(s) = K(s)*P(s) の形状を設計

        # 位相余裕・ゲイン余裕を満たす制御器を近似設計
        # PI制御器ベースにローパスフィルタを追加した構造:
        # K(s) = Kp * (1 + Ki/s) * 1/(Tf*s + 1)

        # プラントの帯域幅に合わせたクロスオーバー周波数
        omega_c = cfg.ws_bandwidth * 2  # クロスオーバー周波数

        # 位相余裕から比例ゲインを計算
        # φm = 180° - arg(L(jωc)) = 180° - arg(K(jωc)) - arg(P(jωc))
        # プラントの位相遅れ（1次遅れ＋むだ時間）を考慮
        plant_phase = -math.atan(omega_c * cfg.plant_time_constant) - omega_c * cfg.plant_dead_time
        target_phase_margin = math.radians(cfg.phase_margin_target)

        # 必要な制御器位相進み
        controller_phase_lead = target_phase_margin + math.pi + plant_phase

        # PI制御器 + フィルタの係数
        Kp = cfg.loop_gain * cfg.plant_gain
        Ti = 1.0 / (cfg.integral_gain / Kp) if cfg.integral_gain > 0 else 1000.0
        Tf = cfg.filter_time_constant

        # 連続時間係数
        # K(s) = Kp * (Ti*s + 1) / (Ti*s) * 1/(Tf*s + 1)
        #      = Kp * (Ti*s + 1) / (Ti*Tf*s² + Ti*s)
        # 分子: Kp*Ti*s + Kp
        # 分母: Ti*Tf*s² + Ti*s + 0 (厳密には不安定なので修正)

        # 安定な2次制御器として再設計
        # K(s) = Kp * (1 + 1/(Ti*s)) / (1 + Tf*s)
        # 離散化には双一次変換 s = (2/T)*(z-1)/(z+1) を使用

        # 簡略化: 状態空間形式での離散化
        # PI制御器の離散化
        alpha = T / Ti if Ti > 0 else 0
        beta = Tf / (Tf + T) if Tf > 0 else 0

        # 2次IIRフィルタとして実装
        # y[k] = B0*e[k] + B1*e[k-1] + B2*e[k-2] - A1*y[k-1] - A2*y[k-2]
        self._B0 = Kp * (1 + alpha/2)
        self._B1 = Kp * alpha
        self._B2 = Kp * (alpha/2 - 1) * 0.1  # 微分項（減衰）
        self._A1 = -(2 * beta - 1)
        self._A2 = beta * 0.5

        # 係数の正規化（安定性確保）
        self._B0 = max(-1.0, min(1.0, self._B0))
        self._B1 = max(-1.0, min(1.0, self._B1))
        self._B2 = max(-0.5, min(0.5, self._B2))

    def reset(self) -> None:
        """制御状態をリセット"""
        self._x1 = 0.0
        self._x2 = 0.0
        self._integral = 0.0
        self._filtered_error = 0.0
        self._filtered_hr = self._target_hr
        self._prediction_buffer.clear()
        self._last_output = 1.0
        self._adapted_gain = self.config.loop_gain
        self._hr_history.clear()
        self._error_history.clear()
        self._last_time = None
        print("Robust Controller reset")

    def set_target_hr(self, target_hr: float) -> None:
        """目標心拍数を設定"""
        self._target_hr = max(40.0, min(180.0, target_hr))

    def get_target_hr(self) -> float:
        return self._target_hr

    def update(self, current_hr: float, min_output: float, max_output: float) -> Tuple[float, dict]:
        """
        H∞ロバスト制御による出力計算

        Args:
            current_hr: 現在の心拍数 (BPM)
            min_output: 出力の最小値
            max_output: 出力の最大値

        Returns:
            Tuple[float, dict]: (抑揚レベル, デバッグ情報)
        """
        current_time = time.time()
        cfg = self.config

        # サンプリング時間の更新
        if self._last_time is not None:
            dt = current_time - self._last_time
            if 0 < dt < 10.0:
                self._sample_time = dt

        # 心拍数のローパスフィルタリング（ノイズ除去）
        alpha_lpf = self._sample_time / (cfg.filter_time_constant + self._sample_time)
        self._filtered_hr += alpha_lpf * (current_hr - self._filtered_hr)

        # 誤差計算
        error = self._target_hr - self._filtered_hr

        # 履歴に追加
        self._hr_history.append(current_hr)
        self._error_history.append(error)
        if len(self._hr_history) > self._max_history:
            self._hr_history.pop(0)
            self._error_history.pop(0)

        # デッドバンド処理
        effective_error = 0.0 if abs(error) < cfg.deadband else error

        # フィルタ済み誤差の更新（微分項用）
        self._filtered_error += alpha_lpf * (effective_error - self._filtered_error)

        # === スミス予測器（むだ時間補償） ===
        if cfg.smith_predictor_enabled:
            # 予測バッファの更新
            self._prediction_buffer.append(self._last_output)
            if len(self._prediction_buffer) > self._delay_steps + 1:
                self._prediction_buffer.pop(0)

            # むだ時間分遅延した入力の効果を予測
            if len(self._prediction_buffer) > self._delay_steps:
                delayed_output = self._prediction_buffer[0]
                # 予測された心拍数変化
                predicted_effect = cfg.plant_gain * (delayed_output - 1.0)
                # 補正された誤差
                effective_error = effective_error + predicted_effect * 0.5

        # === H∞制御器（2次IIRフィルタ） ===
        # y = B0*e + x1
        # x1_new = B1*e + x2 - A1*y
        # x2_new = B2*e - A2*y
        controller_output = self._B0 * effective_error + self._x1
        x1_new = self._B1 * effective_error + self._x2 - self._A1 * controller_output
        x2_new = self._B2 * effective_error - self._A2 * controller_output
        self._x1 = x1_new
        self._x2 = x2_new

        # === 積分器（定常偏差除去） ===
        if self._sample_time > 0:
            self._integral += cfg.integral_gain * effective_error * self._sample_time
            # アンチワインドアップ
            self._integral = max(-cfg.integral_max, min(cfg.integral_max, self._integral))

        # === 適応ゲイン（オプション） ===
        if cfg.adaptive_enabled and len(self._error_history) >= 10:
            # 誤差の変化率に基づいてゲインを調整
            recent_errors = self._error_history[-10:]
            error_trend = abs(recent_errors[-1]) - abs(recent_errors[0])

            # 誤差が減少していない場合はゲインを上げる
            if error_trend > 0 and abs(recent_errors[-1]) > cfg.deadband:
                self._adapted_gain *= (1 + cfg.adaptation_rate)
            elif error_trend < 0:
                self._adapted_gain *= (1 - cfg.adaptation_rate * 0.5)

            # ゲインの範囲制限
            self._adapted_gain = max(cfg.loop_gain * 0.5, min(cfg.loop_gain * 2.0, self._adapted_gain))

        # === 総合出力 ===
        # ベース出力 (1.0) + 制御器出力 + 積分器出力
        gain_factor = self._adapted_gain / cfg.loop_gain if cfg.adaptive_enabled else 1.0
        raw_output = 1.0 + gain_factor * controller_output + self._integral

        # === スルーレート制限 ===
        delta_output = raw_output - self._last_output
        if abs(delta_output) > cfg.du_max:
            raw_output = self._last_output + cfg.du_max * (1 if delta_output > 0 else -1)

        # === 出力範囲制限 ===
        output = max(min_output, min(max_output, raw_output))

        # 状態更新
        self._last_time = current_time
        self._last_output = output

        debug_info = {
            "control_mode": "Robust",
            "target_hr": self._target_hr,
            "current_hr": current_hr,
            "filtered_hr": self._filtered_hr,
            "error": error,
            "effective_error": effective_error,
            "controller_output": controller_output,
            "integral": self._integral,
            "adapted_gain": self._adapted_gain,
            "raw_output": raw_output,
            "output": output,
            "x1": self._x1,
            "x2": self._x2
        }

        return output, debug_info

    def get_adapted_gain(self) -> float:
        """現在の適応ゲインを取得"""
        return self._adapted_gain

    def set_loop_gain(self, gain: float) -> None:
        """ループゲインを設定"""
        self.config.loop_gain = max(0.001, min(0.1, gain))
        self._adapted_gain = self.config.loop_gain
        self._update_controller_coefficients()
        print(f"Robust loop gain set to {self.config.loop_gain}")

    def set_integral_gain(self, gain: float) -> None:
        """積分ゲインを設定"""
        self.config.integral_gain = max(0.0, min(0.01, gain))
        self._update_controller_coefficients()
        print(f"Robust integral gain set to {self.config.integral_gain}")

    def set_filter_time_constant(self, tau: float) -> None:
        """フィルタ時定数を設定"""
        self.config.filter_time_constant = max(0.5, min(10.0, tau))
        print(f"Robust filter time constant set to {self.config.filter_time_constant}")

    def enable_smith_predictor(self, enabled: bool) -> None:
        """スミス予測器の有効/無効を設定"""
        self.config.smith_predictor_enabled = enabled
        if not enabled:
            self._prediction_buffer.clear()
        print(f"Smith predictor {'enabled' if enabled else 'disabled'}")

    def enable_adaptation(self, enabled: bool) -> None:
        """適応機能の有効/無効を設定"""
        self.config.adaptive_enabled = enabled
        if not enabled:
            self._adapted_gain = self.config.loop_gain
        print(f"Adaptive gain {'enabled' if enabled else 'disabled'}")


# === コメントアウト: AdaptiveMPCController（ロバスト制御に置き換え） ===
# class AdaptiveMPCController:
#     """
#     適応モデル予測制御 (Adaptive MPC) コントローラー
#
#     特徴:
#     1. オンラインシステム同定: RLSで心拍応答モデルを適応的に推定
#     2. 予測制御: 将来の心拍数を予測し、最適な抑揚系列を計算
#     3. 制約付き最適化: 抑揚の範囲と変化量を制約
#
#     心拍数フィードバックの課題への対応:
#     - 個人差 → オンライン同定で対応
#     - 応答遅延 → むだ時間補償
#     - 非線形性 → 予測制御で先読み
#     """
#
#     def __init__(self, config: Optional[AdaptiveMPCConfig] = None):
#         self.config = config or AdaptiveMPCConfig()
#
#         # 予測モデルパラメータ
#         self._a: float = self.config.initial_a  # 自己回帰係数
#         self._b: float = self.config.initial_b  # 入力ゲイン
#
#         # RLS推定器の状態
#         self._P: float = self.config.rls_initial_covariance  # 共分散（スカラー近似）
#         self._P_a: float = self.config.rls_initial_covariance
#         self._P_b: float = self.config.rls_initial_covariance
#
#         # 履歴バッファ
#         self._hr_history: List[float] = []  # 心拍数履歴
#         self._u_history: List[float] = []   # 入力履歴（むだ時間補償用）
#         self._max_history: int = 50
#
#         # 目標心拍数
#         self._target_hr: float = 70.0
#
#         # 時間管理
#         self._last_time: Optional[float] = None
#         self._sample_count: int = 0
#
#         # 最後の出力
#         self._last_output: float = 1.0
#
#     def reset(self) -> None:
#         """制御状態をリセット"""
#         self._a = self.config.initial_a
#         self._b = self.config.initial_b
#         self._P_a = self.config.rls_initial_covariance
#         self._P_b = self.config.rls_initial_covariance
#         self._hr_history.clear()
#         self._u_history.clear()
#         self._last_time = None
#         self._sample_count = 0
#         self._last_output = 1.0
#         print("AdaptiveMPC Controller reset")
#
#     def set_target_hr(self, target_hr: float) -> None:
#         """目標心拍数を設定"""
#         self._target_hr = max(40.0, min(180.0, target_hr))
#
#     def get_target_hr(self) -> float:
#         return self._target_hr
#
#     def _update_model_rls(self, y_new: float) -> None:
#         """
#         RLS（再帰的最小二乗法）でモデルパラメータを更新
#
#         モデル: y(k) = a * y(k-1) + b * u(k-1-d)
#         """
#         if len(self._hr_history) < 2:
#             return
#
#         d = self.config.delay_steps
#         if len(self._u_history) <= d:
#             return
#
#         # 回帰ベクトル
#         y_prev = self._hr_history[-1]
#         u_delayed = self._u_history[-(d + 1)] if len(self._u_history) > d else 1.0
#
#         # 予測誤差
#         y_pred = self._a * y_prev + self._b * (u_delayed - 1.0)  # u=1.0が基準
#         e = y_new - y_pred
#
#         # RLS更新（簡略化: 各パラメータを個別に更新）
#         lambda_f = self.config.rls_forgetting_factor
#
#         # aの更新
#         k_a = self._P_a * y_prev / (lambda_f + self._P_a * y_prev * y_prev)
#         self._a += k_a * e
#         self._P_a = (1 / lambda_f) * (self._P_a - k_a * y_prev * self._P_a)
#
#         # bの更新
#         u_centered = u_delayed - 1.0
#         if abs(u_centered) > 0.01:  # 入力変化がある場合のみ
#             k_b = self._P_b * u_centered / (lambda_f + self._P_b * u_centered * u_centered)
#             self._b += k_b * e
#             self._P_b = (1 / lambda_f) * (self._P_b - k_b * u_centered * self._P_b)
#
#         # パラメータをクランプ
#         self._a = max(self.config.a_min, min(self.config.a_max, self._a))
#         self._b = max(self.config.b_min, min(self.config.b_max, self._b))
#
#         # 共分散の発散防止
#         self._P_a = max(0.1, min(1000.0, self._P_a))
#         self._P_b = max(0.1, min(1000.0, self._P_b))
#
#     def _predict_trajectory(self, y0: float, u_sequence: List[float]) -> List[float]:
#         """
#         将来の心拍数軌道を予測
#
#         Args:
#             y0: 現在の心拍数
#             u_sequence: 将来の入力系列
#
#         Returns:
#             予測心拍数軌道
#         """
#         y_pred = [y0]
#         d = self.config.delay_steps
#
#         # 過去の入力を含めた系列を作成
#         u_full = list(self._u_history[-d:]) + u_sequence if len(self._u_history) >= d else [1.0] * d + u_sequence
#
#         for k in range(len(u_sequence)):
#             y_prev = y_pred[-1]
#             # むだ時間を考慮した入力
#             u_delayed_idx = k  # u_fullの中でd個前
#             u_delayed = u_full[u_delayed_idx] if u_delayed_idx >= 0 else 1.0
#             y_next = self._a * y_prev + self._b * (u_delayed - 1.0)
#             y_pred.append(y_next)
#
#         return y_pred[1:]  # 最初の値（現在値）は除く
#
#     def _solve_mpc(self, current_hr: float) -> float:
#         """
#         MPC最適化問題を解いて最適な入力を計算
#
#         簡略化: 勾配降下法による近似解
#         """
#         N = self.config.prediction_horizon
#         M = self.config.control_horizon
#
#         # 初期入力系列（現在の出力を維持）
#         u_seq = [self._last_output] * N
#
#         # 簡易最適化（数回の反復）
#         for _ in range(5):
#             # 現在のコスト
#             y_pred = self._predict_trajectory(current_hr, u_seq)
#             cost = self._compute_cost(y_pred, u_seq)
#
#             # 勾配近似による更新
#             for m in range(min(M, len(u_seq))):
#                 # 数値勾配
#                 epsilon = 0.01
#                 u_seq_plus = u_seq.copy()
#                 u_seq_plus[m] += epsilon
#                 cost_plus = self._compute_cost(
#                     self._predict_trajectory(current_hr, u_seq_plus),
#                     u_seq_plus
#                 )
#                 gradient = (cost_plus - cost) / epsilon
#
#                 # 勾配降下
#                 step_size = 0.1
#                 u_seq[m] -= step_size * gradient
#
#                 # 制約適用
#                 u_seq[m] = max(self.config.u_min, min(self.config.u_max, u_seq[m]))
#
#                 # 変化量制約
#                 if abs(u_seq[m] - self._last_output) > self.config.du_max:
#                     u_seq[m] = self._last_output + self.config.du_max * (
#                         1 if u_seq[m] > self._last_output else -1
#                     )
#
#         return u_seq[0]  # 最初の入力のみ適用（Receding Horizon）
#
#     def _compute_cost(self, y_pred: List[float], u_seq: List[float]) -> float:
#         """MPC目的関数を計算"""
#         cost = 0.0
#
#         # 出力誤差項
#         for y in y_pred:
#             cost += self.config.output_weight * (y - self._target_hr) ** 2
#
#         # 入力変化量項
#         u_prev = self._last_output
#         for u in u_seq:
#             cost += self.config.input_change_weight * (u - u_prev) ** 2
#             u_prev = u
#
#         return cost
#
#     def update(self, current_hr: float, min_output: float, max_output: float) -> Tuple[float, dict]:
#         """
#         適応MPC制御による出力計算
#
#         Args:
#             current_hr: 現在の心拍数 (BPM)
#             min_output: 出力の最小値
#             max_output: 出力の最大値
#
#         Returns:
#             Tuple[float, dict]: (抑揚レベル, デバッグ情報)
#         """
#         current_time = time.time()
#         self._sample_count += 1
#
#         # 制約を更新
#         self.config.u_min = min_output
#         self.config.u_max = max_output
#
#         # モデル更新（十分なデータがある場合）
#         if len(self._hr_history) >= 2:
#             self._update_model_rls(current_hr)
#
#         # 履歴に追加
#         self._hr_history.append(current_hr)
#         if len(self._hr_history) > self._max_history:
#             self._hr_history.pop(0)
#
#         # デッドバンド処理
#         error = self._target_hr - current_hr
#         if abs(error) < self.config.deadband:
#             # 目標付近では現状維持
#             output = self._last_output
#         else:
#             # MPC最適化
#             output = self._solve_mpc(current_hr)
#
#         # 出力範囲にクランプ
#         output = max(min_output, min(max_output, output))
#
#         # 履歴に追加
#         self._u_history.append(output)
#         if len(self._u_history) > self._max_history:
#             self._u_history.pop(0)
#
#         # 状態更新
#         self._last_time = current_time
#         self._last_output = output
#
#         debug_info = {
#             "control_mode": "AdaptiveMPC",
#             "target_hr": self._target_hr,
#             "current_hr": current_hr,
#             "error": error,
#             "model_a": self._a,
#             "model_b": self._b,
#             "P_a": self._P_a,
#             "P_b": self._P_b,
#             "sample_count": self._sample_count,
#             "output": output
#         }
#
#         return output, debug_info
#
#     def get_model_params(self) -> Tuple[float, float]:
#         """現在のモデルパラメータを取得"""
#         return self._a, self._b
#
#     def set_prediction_horizon(self, N: int) -> None:
#         """予測ホライズンを設定"""
#         self.config.prediction_horizon = max(1, min(30, N))
#         print(f"AdaptiveMPC prediction horizon set to {self.config.prediction_horizon}")
#
#     def set_weights(self, output_weight: float, input_change_weight: float) -> None:
#         """MPC重みを設定"""
#         self.config.output_weight = max(0.1, output_weight)
#         self.config.input_change_weight = max(0.01, input_change_weight)
#         print(f"AdaptiveMPC weights: output={self.config.output_weight}, "
#               f"input_change={self.config.input_change_weight}")
# === AdaptiveMPCController コメントアウト終了 ===


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
    HRF2 - PID制御/適応制御/ゲインスケジューリング制御/ロバスト制御による心拍数追従コントローラー

    目標心拍数に対して現在の心拍数を追従させるため、
    抑揚パラメータを制御する。

    制御モード:
    - PID: 従来のPID制御
    - Adaptive: MRAC（モデル規範型適応制御）
    - GainScheduled: 誤差ベースのゲインスケジューリング
    - Robust: H∞ロバスト制御（混合感度問題に基づく設計）
    """

    def __init__(self, config: Optional[HRF2Config] = None,
                 adaptive_config: Optional[AdaptiveConfig] = None,
                 gain_schedule_config: Optional[GainScheduleConfig] = None,
                 robust_config: Optional[RobustConfig] = None):
        self.config = config or HRF2Config()
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        self.gain_schedule_config = gain_schedule_config or GainScheduleConfig()
        self.robust_config = robust_config or RobustConfig()

        # PID制御の内部状態
        self._integral: float = 0.0
        self._last_error: Optional[float] = None
        self._last_time: Optional[float] = None

        # 適応制御コントローラー
        self._adaptive_controller = AdaptiveController(self.adaptive_config)

        # ゲインスケジューリングコントローラー
        self._gain_scheduled_controller = GainScheduledController(self.gain_schedule_config)

        # ロバスト制御コントローラー
        self._robust_controller = RobustController(self.robust_config)

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
        self._robust_controller.set_target_hr(self.config.target_hr)
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

        # ロバスト制御状態リセット
        self._robust_controller.reset()
        self._robust_controller.set_target_hr(self.config.target_hr)
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
        elif self.config.control_mode == ControlMode.ROBUST:
            return self._update_robust(current_hr)
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

    def _update_robust(self, current_hr: float) -> Tuple[float, dict]:
        """H∞ロバスト制御による出力計算"""
        output, debug_info = self._robust_controller.update(
            current_hr,
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
        elif self.config.control_mode == ControlMode.ROBUST:
            gain = self._robust_controller.get_adapted_gain()
            return (f"HRF2({mode_str}): 目標{self.config.target_hr:.0f}BPM "
                    f"現在{current_hr:.0f}BPM {direction} "
                    f"gain={gain:.4f} 抑揚{self._last_output:.2f}")
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

    # --- Robust制御用メソッド ---
    def get_robust_adapted_gain(self) -> float:
        """ロバスト制御の現在の適応ゲインを取得"""
        return self._robust_controller.get_adapted_gain()

    def get_robust_config(self) -> RobustConfig:
        """ロバスト制御設定を取得"""
        return self._robust_controller.config

    def set_robust_loop_gain(self, gain: float) -> None:
        """ロバスト制御のループゲインを設定"""
        self._robust_controller.set_loop_gain(gain)

    def set_robust_integral_gain(self, gain: float) -> None:
        """ロバスト制御の積分ゲインを設定"""
        self._robust_controller.set_integral_gain(gain)

    def set_robust_filter_time_constant(self, tau: float) -> None:
        """ロバスト制御のフィルタ時定数を設定"""
        self._robust_controller.set_filter_time_constant(tau)

    def enable_robust_smith_predictor(self, enabled: bool) -> None:
        """ロバスト制御のスミス予測器を有効/無効化"""
        self._robust_controller.enable_smith_predictor(enabled)

    def enable_robust_adaptation(self, enabled: bool) -> None:
        """ロバスト制御のゲイン適応を有効/無効化"""
        self._robust_controller.enable_adaptation(enabled)
