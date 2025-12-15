# ロバスト制御 (H∞ループ整形) - HCS v6.16.0

## 1. 概要

心拍数フィードバック制御において、被験者ごとの個人差（ゲイン変動、時定数の違い）や測定ノイズに対してロバスト（頑健）な制御を実現するため、H∞ループ整形法に基づく制御器を実装しました。

## 2. 制御対象モデル

### 2.1 心拍数応答のモデル化

抑揚レベル u(t) から心拍数 HR(t) への応答を、一次遅れ＋むだ時間系としてモデル化：

```
         K
P(s) = ――――― × e^(-Ls)
       τs + 1

K  = 1.5   : 定常ゲイン（抑揚変化に対する心拍数変化の比）
τ  = 30.0秒: 時定数（心拍応答の遅れ）
L  = 5.0秒 : むだ時間（刺激から応答開始までの遅延）
```

### 2.2 不確かさのモデル化

実際のプラントは被験者によって異なるため、乗法的不確かさを考慮：

```
P_real(s) = P_nominal(s) × (1 + Δ × W_m(s))

|Δ| ≤ 1

W_m(s) の特性:
- 低周波（0.01 rad/s以下）: 30%の不確かさ
- 高周波（0.1 rad/s以上）: 150%の不確かさ
```

## 3. 混合感度問題

### 3.1 設計仕様

以下の混合感度問題を解いて制御器を設計：

```
    ┌        ┐
    │ Ws × S │
    │        │  < γ
    │ Wt × T │∞
    └        ┘

S(s) = 1/(1 + P(s)K(s))     : 感度関数
T(s) = P(s)K(s)/(1+P(s)K(s)): 相補感度関数
```

### 3.2 重み関数

**感度重み Ws(s)**（目標追従性能を規定）：
```
         s/Ms + ωb
Ws(s) = ―――――――――――
         s + ωb×εs

ωb  = 0.05 rad/s : 帯域幅
Ms  = 2.0        : 感度ピーク上限
εs  = 0.01       : 定常偏差の逆数（1/εs = 100）
```

**相補感度重み Wt(s)**（ロバスト安定性を規定）：
```
         s + ωt/Mt
Wt(s) = ―――――――――――
         εt×s + ωt

ωt  = 0.3 rad/s  : ロールオフ周波数
Mt  = 1.5        : 低周波ゲイン
εt  = 1/1.2      : 高周波ゲイン上限
```

## 4. 制御器実装

### 4.1 制御器構造

PI制御器 + ローパスフィルタをベースとした2次IIRフィルタ形式：

```
                1          1
K(s) = Kp × (1 + ―――) × ―――――――
               Ti×s     Tf×s + 1

Kp = 0.03  : 比例ゲイン
Ti = 10.0秒: 積分時定数
Tf = 2.0秒 : フィルタ時定数
```

### 4.2 離散化（双一次変換）

サンプリング時間 T で離散化：

```
       B0 + B1×z⁻¹ + B2×z⁻²
K(z) = ―――――――――――――――――――――――
       1 + A1×z⁻¹ + A2×z⁻²
```

実装コード（hrf2_controller.py:311-375）：
```python
def _update_controller_coefficients(self) -> None:
    T = self._sample_time
    cfg = self.config

    # クロスオーバー周波数
    omega_c = cfg.ws_bandwidth * 2

    # プラントの位相遅れを考慮
    plant_phase = -math.atan(omega_c * cfg.plant_time_constant) \
                  - omega_c * cfg.plant_dead_time

    # PI + フィルタ係数
    Kp = cfg.loop_gain * cfg.plant_gain
    Ti = 1.0 / (cfg.integral_gain / Kp)
    Tf = cfg.filter_time_constant

    # 離散化
    alpha = T / Ti
    beta = Tf / (Tf + T)

    # 2次IIRフィルタ係数
    self._B0 = Kp * (1 + alpha/2)
    self._B1 = Kp * alpha
    self._B2 = Kp * (alpha/2 - 1) * 0.1
    self._A1 = -(2 * beta - 1)
    self._A2 = beta * 0.5
```

### 4.3 制御アルゴリズム

1ステップの処理（hrf2_controller.py:398-518）：

```
入力: current_hr（現在心拍数）
出力: output（抑揚レベル 0.0〜2.0）

Step 1: ローパスフィルタ
  filtered_hr += α × (current_hr - filtered_hr)
  α = T / (Tf + T)

Step 2: 誤差計算
  error = target_hr - filtered_hr

Step 3: デッドバンド処理
  effective_error = 0 if |error| < 2.0 else error

Step 4: スミス予測器（むだ時間補償）
  predicted_effect = K × (u[k-delay] - 1.0)
  effective_error += predicted_effect × 0.5

Step 5: H∞制御器（2次IIRフィルタ）
  y = B0×e + x1
  x1_new = B1×e + x2 - A1×y
  x2_new = B2×e - A2×y

Step 6: 積分器（定常偏差除去）
  integral += Ki × e × T
  integral = clamp(integral, -15, +15)  # アンチワインドアップ

Step 7: 適応ゲイン調整
  if 誤差が減少していない:
    adapted_gain *= 1.01
  else:
    adapted_gain *= 0.995

Step 8: 出力合成
  raw_output = 1.0 + (adapted_gain/loop_gain) × y + integral

Step 9: スルーレート制限
  Δu = clamp(raw_output - last_output, -0.2, +0.2)
  output = last_output + Δu

Step 10: 出力飽和
  output = clamp(output, 0.0, 2.0)
```

## 5. パラメータ設定

### 5.1 RobustConfig クラス

```python
@dataclass
class RobustConfig:
    # プラントモデル
    plant_gain: float = 1.5           # 定常ゲイン
    plant_time_constant: float = 30.0 # 時定数 [秒]
    plant_dead_time: float = 5.0      # むだ時間 [秒]

    # 不確かさ
    uncertainty_low_freq: float = 0.3   # 低周波不確かさ
    uncertainty_high_freq: float = 1.5  # 高周波不確かさ

    # 感度重み Ws
    ws_bandwidth: float = 0.05          # ωb [rad/s]
    ws_low_freq_gain: float = 100.0     # 1/εs
    ws_peak: float = 2.0                # Ms

    # 相補感度重み Wt
    wt_bandwidth: float = 0.3           # ωt [rad/s]
    wt_high_freq_gain: float = 1.2      # 1/εt
    wt_peak: float = 1.5                # Mt

    # ループ整形
    loop_gain: float = 0.02             # 基本ゲイン
    phase_margin_target: float = 45.0   # 目標位相余裕 [度]
    gain_margin_target: float = 6.0     # 目標ゲイン余裕 [dB]

    # 積分器
    integral_gain: float = 0.003        # 積分ゲイン
    integral_max: float = 15.0          # アンチワインドアップ上限

    # フィルタ
    filter_time_constant: float = 2.0   # LPF時定数 [秒]
    smith_predictor_enabled: bool = True

    # 出力制限
    u_min: float = 0.0
    u_max: float = 2.0
    du_max: float = 0.2                 # スルーレート

    # デッドバンド
    deadband: float = 2.0               # [BPM]

    # 適応機能
    adaptive_enabled: bool = True
    adaptation_rate: float = 0.01
```

### 5.2 チューニング指針

| パラメータ | 効果 | 推奨範囲 |
|-----------|------|---------|
| loop_gain | 応答速度 | 0.01〜0.05 |
| ws_bandwidth | 追従帯域 | 0.03〜0.1 rad/s |
| integral_gain | 定常偏差 | 0.001〜0.01 |
| filter_time_constant | ノイズ除去 | 1.0〜5.0秒 |
| deadband | 不感帯 | 1.0〜3.0 BPM |
| du_max | 応答滑らかさ | 0.1〜0.3 |

## 6. ブロック線図

```
                              ┌─────────────────┐
                              │   Smith 予測器   │
                              │  (むだ時間補償)  │
                              └────────┬────────┘
                                       │
  target_hr ──(+)─→ [デッドバンド] ──→ [H∞制御器] ─┬→ [積分器] ─┐
              │-                        (2次IIR)   │            │
              │                                    │            │
              │    ┌──────────────────────────────┘            │
              │    │                                            │
              │    └──→ [ゲイン調整] ──→ (+) ←─────────────────┘
              │                           │
              │                           ↓
              │                    [スルーレート制限]
              │                           │
              │                           ↓
              │                      [飽和処理]
              │                           │
              │                           ↓ output (抑揚)
              │                    ┌──────┴──────┐
              │                    │             │
              │                    ↓             │
              │              [ プラント ]        │
              │              (心拍応答)          │
              │                    │             │
              │                    ↓             │
              └────────────── [LPフィルタ] ←────┘
                              current_hr
```

## 7. 他の制御モードとの比較

### 7.1 PID制御との違い

| 項目 | PID | Robust |
|------|-----|--------|
| むだ時間補償 | なし | スミス予測器 |
| ノイズ耐性 | 微分項で悪化 | 高周波重みで設計 |
| パラメータ変動 | 性能劣化 | ロバスト性保証 |
| 設計法 | 試行錯誤 | 周波数領域設計 |

### 7.2 適応制御(MRAC)との違い

| 項目 | MRAC | Robust |
|------|------|--------|
| パラメータ同定 | オンライン推定 | 不確かさ範囲で設計 |
| 収束性 | 時間がかかる | 初期から安定 |
| 計算量 | 大きい | 小さい |
| 過渡特性 | 同定誤差に依存 | 安定余裕で保証 |

## 8. 使用例

### 8.1 GUIでの使用

1. HCSアプリを起動
2. 「会話システム」タブ → 「HRF制御」セクション
3. 「有効」チェックボックスをON
4. 制御モード選択で「Robust」を選択
5. 必要に応じてパラメータを調整

### 8.2 プログラムからの使用

```python
from hrf2_controller import HRF2Controller, ControlMode, RobustConfig

# デフォルト設定で作成
controller = HRF2Controller(
    target_hr=70.0,
    control_mode=ControlMode.ROBUST
)

# カスタム設定
config = RobustConfig(
    loop_gain=0.03,          # より積極的な制御
    ws_bandwidth=0.08,       # 広い帯域
    deadband=1.5,            # 狭いデッドバンド
    du_max=0.15              # 滑らかな応答
)
controller._robust_controller.config = config
controller._robust_controller._update_controller_coefficients()

# 制御ループ
while running:
    current_hr = get_heart_rate()
    intonation = controller.compute(current_hr, dt=1.0)
    play_audio_with_intonation(intonation)
```

## 9. 参考文献

1. Paradiso, R., et al. "WEALTHY - A Wearable Healthcare System Based on Knitted Integrated Sensors." IEEE Transactions on Information Technology in Biomedicine, Vol.9, No.3, 2005.

2. Aranda, E., et al. "Robust Heart Rate Control Using Quantitative Feedback Theory (QFT)." European Control Conference (ECC), 2007.

3. Skogestad, S., & Postlethwaite, I. "Multivariable Feedback Control: Analysis and Design." 2nd ed., Wiley, 2005.

4. 川田昌克, 西岡勝博. "MATLABによる制御系設計." 東京電機大学出版局, 2003.
