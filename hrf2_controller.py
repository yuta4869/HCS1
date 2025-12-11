# hrf2_controller.py
"""
HRF2 Controller - PID制御による心拍数追従モード

目標心拍数に向けて抑揚レベルを自動調整し、
被験者の心拍数を目標値に追従させる。

- BPMが目標より低い → 抑揚を上げて興奮させる
- BPMが目標より高い → 抑揚を下げて落ち着かせる
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class HRF2Config:
    """HRF2制御のパラメータ設定"""
    # PIDゲイン
    kp: float = 0.02       # 比例ゲイン
    ki: float = 0.005      # 積分ゲイン
    kd: float = 0.01       # 微分ゲイン

    # 目標心拍数
    target_hr: float = 70.0

    # 抑揚の出力範囲
    min_output: float = 0.3
    max_output: float = 1.8

    # 積分項のアンチワインドアップ
    integral_max: float = 20.0

    # デッドバンド（目標値付近で制御を緩める）
    deadband: float = 3.0  # BPM


class HRF2Controller:
    """
    HRF2 - PID制御による心拍数追従コントローラー

    目標心拍数に対して現在の心拍数を追従させるため、
    抑揚パラメータをPID制御で調整する。
    """

    def __init__(self, config: Optional[HRF2Config] = None):
        self.config = config or HRF2Config()

        # PID制御の内部状態
        self._integral: float = 0.0
        self._last_error: Optional[float] = None
        self._last_time: Optional[float] = None

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
    def target_hr(self) -> float:
        return self.config.target_hr

    @target_hr.setter
    def target_hr(self, value: float) -> None:
        self.config.target_hr = max(40.0, min(180.0, value))
        print(f"HRF2 target HR set to {self.config.target_hr} BPM")

    def reset(self) -> None:
        """制御状態をリセット"""
        self._integral = 0.0
        self._last_error = None
        self._last_time = None
        self._last_output = 1.0
        print("HRF2 Controller reset")

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

    def get_status_text(self, current_hr: float) -> str:
        """現在の状態をテキストで返す"""
        if not self._enabled:
            return "HRF2: 無効"

        error = self.config.target_hr - current_hr
        direction = "↑" if error > 0 else "↓" if error < 0 else "→"

        return (f"HRF2: 目標{self.config.target_hr:.0f}BPM "
                f"現在{current_hr:.0f}BPM {direction} "
                f"抑揚{self._last_output:.2f}")

    def set_pid_gains(self, kp: float, ki: float, kd: float) -> None:
        """PIDゲインを設定"""
        self.config.kp = max(0.0, kp)
        self.config.ki = max(0.0, ki)
        self.config.kd = max(0.0, kd)
        print(f"HRF2 PID gains: Kp={self.config.kp}, Ki={self.config.ki}, Kd={self.config.kd}")

    def set_output_range(self, min_output: float, max_output: float) -> None:
        """出力範囲を設定"""
        self.config.min_output = max(0.0, min(2.0, min_output))
        self.config.max_output = max(0.0, min(2.0, max_output))
        print(f"HRF2 output range: {self.config.min_output} - {self.config.max_output}")
