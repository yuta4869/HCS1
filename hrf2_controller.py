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
    頑健制御のパラメータ設定（MEC: モデル誤差抑制補償器）

    構成:
    - 基本制御器 K0: PI制御で目標追従
    - MEC: 名目モデルとの誤差を補償して頑健化

    ブロック図:
        r ──> (+) ─ e ─> [PI制御器K0] ─ u0 ──┐
               ▲                              │
               │                              ▼
               │                          (+) ─ u ──> [人体G] ──> y
               │                              ▲
               │                              │ u_mec
               │                          [MEC: Cε]
               │                              ▲
               │                              │ ε = y - ŷ
               └─────── y ◄──── [名目モデルĜ] ◄─ u

    MEC（モデル誤差抑制補償器）:
    - ε = y - ŷ （実出力と名目モデル出力の差）
    - u_mec = -Cε(s) * ε （誤差を高利得+低域ろ波で返す）
    - Cε(s) = k / (1 + s/ωc) （1次ローパスフィルタ付き利得）

    名目モデル Ĝ(s):
    - 一次遅れ + むだ時間: Ĝ(s) = K * exp(-Ls) / (τs + 1)
    - 完璧でなくてよい（MECが誤差を補償するため）
    """
    # === 基本PI制御器パラメータ ===
    kp: float = 0.02              # 比例ゲイン
    ki: float = 0.003             # 積分ゲイン
    integral_max: float = 0.5     # 積分項のアンチワインドアップ

    # === 名目モデルパラメータ ===
    # Ĝ(s) = model_gain / (model_tau * s + 1)
    # むだ時間は離散実装でバッファで近似
    model_gain: float = 5.0       # 名目モデルの定常ゲイン（抑揚1.0からの変化→HR変化）
    model_tau: float = 30.0       # 名目モデルの時定数 [秒]
    model_delay: float = 5.0      # 名目モデルのむだ時間 [秒]

    # === MEC（モデル誤差抑制補償器）パラメータ ===
    # Cε(s) = mec_gain / (1 + s/mec_omega)
    mec_gain: float = 0.3         # MEC利得 k（大きいほど誤差補償が強い）
    mec_omega: float = 0.05       # MEC低域ろ波のカットオフ周波数 [rad/s]（低めから始める）

    # === 計測ノイズ抑制 ===
    filter_time_constant: float = 2.0  # 出力yのローパスフィルタ時定数 [秒]

    # === 出力制限 ===
    du_max: float = 0.15          # 1ステップあたりの最大変化量（スルーレート制限）

    # === デッドバンド ===
    deadband: float = 2.0         # BPM（目標値付近で制御を緩める）


class RobustController:
    """
    頑健制御コントローラー（MEC: モデル誤差抑制補償器）

    構成:
    1. 基本制御器 K0: PI制御で目標追従
    2. 名目モデル Ĝ: 一次遅れ + むだ時間で人体応答を近似
    3. MEC Cε: モデル誤差 ε = y - ŷ を補償

    動作原理:
    - u = u0 + u_mec
    - u0 = Kp*e + Ki*∫e （基本PI制御）
    - ε = y - ŷ （モデル誤差 = 実出力 - 名目モデル出力）
    - u_mec = -Cε(s)*ε （誤差を低域ろ波付き利得で返す）

    頑健性:
    - 名目モデルと実際の人体の差（個人差、日内変動）をMECが補償
    - 低域ろ波により測定ノイズを増幅しない
    """

    def __init__(self, config: Optional[RobustConfig] = None):
        self.config = config or RobustConfig()

        # 目標心拍数
        self._target_hr: float = 70.0

        # 時間管理
        self._last_time: Optional[float] = None

        # === 基本PI制御器の状態 ===
        self._integral: float = 0.0

        # === 計測フィルタ状態 ===
        self._filtered_hr: float = 70.0

        # === 名目モデルの状態 ===
        # 一次遅れ系: τ * d(ŷ)/dt + ŷ = K * u_delayed
        self._model_output: float = 70.0  # ŷ: 名目モデル出力
        self._input_buffer: List[Tuple[float, float]] = []  # むだ時間用バッファ (time, u)

        # === MEC（モデル誤差抑制補償器）の状態 ===
        # Cε(s) = k / (1 + s/ω) を離散化
        self._mec_filtered_error: float = 0.0  # ローパスフィルタ後のε

        # === 出力履歴 ===
        self._last_output: float = 1.0

    def reset(self) -> None:
        """制御状態をリセット"""
        self._integral = 0.0
        self._filtered_hr = self._target_hr
        self._model_output = self._target_hr
        self._input_buffer.clear()
        self._mec_filtered_error = 0.0
        self._last_output = 1.0
        self._last_time = None
        print("Robust Controller (MEC) reset")

    def set_target_hr(self, target_hr: float) -> None:
        """目標心拍数を設定"""
        self._target_hr = max(40.0, min(180.0, target_hr))

    def get_target_hr(self) -> float:
        return self._target_hr

    def update(self, current_hr: float, min_output: float, max_output: float) -> Tuple[float, dict]:
        """
        頑健制御（MEC）による出力計算

        Args:
            current_hr: 現在の心拍数 (BPM)
            min_output: 出力の最小値
            max_output: 出力の最大値

        Returns:
            Tuple[float, dict]: (抑揚レベル, デバッグ情報)
        """
        current_time = time.time()
        cfg = self.config

        # サンプリング時間の計算
        dt = 1.0
        if self._last_time is not None:
            dt = current_time - self._last_time
            if dt <= 0 or dt > 10.0:
                dt = 1.0

        # === 計測値のローパスフィルタ（ノイズ抑制） ===
        alpha_y = dt / (cfg.filter_time_constant + dt)
        self._filtered_hr += alpha_y * (current_hr - self._filtered_hr)

        # === 名目モデルの更新 ===
        # むだ時間バッファから遅延入力を取得
        self._input_buffer.append((current_time, self._last_output))
        delayed_input = 1.0  # デフォルト（ベースライン）
        cutoff_time = current_time - cfg.model_delay
        while self._input_buffer and self._input_buffer[0][0] < cutoff_time:
            delayed_input = self._input_buffer.pop(0)[1]

        # 一次遅れ系の更新: τ * d(ŷ)/dt + ŷ = K * (u - 1) + baseline
        # 離散化: ŷ[k+1] = ŷ[k] + (dt/τ) * (K*(u-1) + baseline - ŷ[k])
        baseline_hr = self._target_hr  # ベースラインを目標HRとする
        model_target = baseline_hr + cfg.model_gain * (delayed_input - 1.0)
        alpha_m = dt / (cfg.model_tau + dt)
        self._model_output += alpha_m * (model_target - self._model_output)

        # === モデル誤差の計算 ===
        # ε = y - ŷ （実出力 - 名目モデル出力）
        model_error = self._filtered_hr - self._model_output

        # === MEC: モデル誤差抑制補償 ===
        # Cε(s) = k / (1 + s/ω) を離散化
        # y[k+1] = y[k] + (dt*ω) * (k*ε - y[k])
        alpha_mec = dt * cfg.mec_omega
        alpha_mec = min(alpha_mec, 1.0)  # 安定性のため
        self._mec_filtered_error += alpha_mec * (cfg.mec_gain * model_error - self._mec_filtered_error)

        # MEC補償入力: u_mec = -Cε(s)*ε
        u_mec = -self._mec_filtered_error

        # === 基本PI制御器 ===
        error = self._target_hr - self._filtered_hr

        # デッドバンド処理
        if abs(error) < cfg.deadband:
            effective_error = 0.0
        else:
            effective_error = error

        # 比例項
        p_term = cfg.kp * effective_error

        # 積分項（アンチワインドアップ付き）
        if dt > 0:
            self._integral += cfg.ki * effective_error * dt
            self._integral = max(-cfg.integral_max, min(cfg.integral_max, self._integral))

        # 基本制御出力
        u0 = 1.0 + p_term + self._integral

        # === 総合出力 ===
        # u = u0 + u_mec
        raw_output = u0 + u_mec

        # === スルーレート制限 ===
        delta = raw_output - self._last_output
        if abs(delta) > cfg.du_max:
            raw_output = self._last_output + cfg.du_max * (1.0 if delta > 0 else -1.0)

        # === 出力範囲制限 ===
        output = max(min_output, min(max_output, raw_output))

        # 状態更新
        self._last_time = current_time
        self._last_output = output

        debug_info = {
            "control_mode": "Robust(MEC)",
            "target_hr": self._target_hr,
            "current_hr": current_hr,
            "filtered_hr": self._filtered_hr,
            "model_output": self._model_output,
            "model_error": model_error,
            "error": error,
            "effective_error": effective_error,
            "p_term": p_term,
            "integral": self._integral,
            "u0": u0,
            "u_mec": u_mec,
            "raw_output": raw_output,
            "output": output
        }

        return output, debug_info

    def set_kp(self, kp: float) -> None:
        """比例ゲインを設定"""
        self.config.kp = max(0.001, min(0.1, kp))
        print(f"Robust Kp set to {self.config.kp}")

    def set_ki(self, ki: float) -> None:
        """積分ゲインを設定"""
        self.config.ki = max(0.0, min(0.01, ki))
        print(f"Robust Ki set to {self.config.ki}")

    def set_mec_gain(self, gain: float) -> None:
        """MEC利得を設定"""
        self.config.mec_gain = max(0.0, min(1.0, gain))
        print(f"MEC gain set to {self.config.mec_gain}")

    def set_mec_omega(self, omega: float) -> None:
        """MECカットオフ周波数を設定"""
        self.config.mec_omega = max(0.01, min(0.5, omega))
        print(f"MEC omega set to {self.config.mec_omega}")


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
