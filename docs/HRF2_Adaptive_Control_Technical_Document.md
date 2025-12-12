# HRF2適応制御システム - 技術解説

## 1. 概要

HRF2システムは、被験者の心拍数（HR）を目標値に追従させるため、音声の抑揚（intonation）パラメータを自動調整する制御システムです。本実装では、従来のPID制御に加えて、**MRAC（Model Reference Adaptive Control: モデル規範型適応制御）** を選択可能にしました。

## 2. 制御対象システムのモデル

心拍数フィードバックシステムは以下の特性を持ちます：

- **入力 u**: 抑揚レベル（0.0 〜 2.0）
- **出力 y**: 心拍数 HR（BPM）
- **遅延**: 数十秒オーダーの応答遅れ
- **非線形性**: 興奮と鎮静で応答特性が異なる
- **個人差**: パラメータが被験者ごとに大きく異なる

このような不確かさを持つシステムに対して、適応制御は有効なアプローチです。

## 3. モデル規範型適応制御（MRAC）

### 3.1 基本構造

MRACは、実際のシステム出力が**参照モデル**の出力に追従するように、コントローラパラメータを適応的に調整する制御方式です。

```
                    ┌──────────────┐
  目標値 r ────────▶│  参照モデル   │────▶ 参照出力 y_m
                    └──────────────┘
                           │
                           ▼ 追従誤差 e = y_m - y
  目標値 r ────┬──▶┌──────────────┐     ┌──────────────┐
               │   │ コントローラ  │──u─▶│ プラント(人体)│──▶ 出力 y
               │   │   θ(t)      │     └──────────────┘     │
               │   └──────────────┘                          │
               │          ▲                                  │
               │          │ 適応則                           │
               │   ┌──────────────┐                          │
               └──▶│   MIT則      │◀─────────────────────────┘
                   └──────────────┘
```

### 3.2 参照モデル

理想的な心拍数応答として、一次遅れ系を採用：

$$\tau \frac{dy_m}{dt} + y_m = r$$

ここで：
- $y_m$: 参照モデル出力（理想的な心拍数）
- $r$: 目標心拍数
- $\tau$: 時定数（デフォルト30秒）

離散化すると：

$$y_m[k+1] = y_m[k] + \frac{\Delta t}{\tau}(r - y_m[k])$$

時定数τは心拍数の生理的な応答遅れを考慮して設定します。

### 3.3 MIT則（Massachusetts Institute of Technology Rule）

MIT則は、追従誤差の二乗を最小化するようにパラメータを更新する適応則です。

**コスト関数**:
$$J(\theta) = \frac{1}{2}e^2$$

ここで追従誤差は：
$$e = y_m - y$$

**パラメータ更新則**（勾配降下法）:
$$\frac{d\theta}{dt} = -\gamma \frac{\partial J}{\partial \theta} = -\gamma e \frac{\partial e}{\partial \theta}$$

感度項 $\frac{\partial e}{\partial \theta}$ は、出力yのパラメータθに対する感度を表します。本実装では簡略化して：

$$\frac{\partial e}{\partial \theta} \approx -\frac{\partial y}{\partial \theta} \approx -u$$

よって更新則は：
$$\frac{d\theta}{dt} = \gamma \cdot e \cdot u$$

### 3.4 正規化MIT則

数値安定性のため、正規化を導入：

$$\frac{d\theta}{dt} = \frac{\gamma \cdot e \cdot u}{1 + \mu \cdot u^2}$$

ここでμは正規化ゲインです。

## 4. 実装詳細

### 4.1 制御則

制御出力（抑揚レベル）は以下で計算：

$$u = 1.0 + \theta \cdot e_c$$

ここで：
- $u$: 抑揚レベル
- $\theta$: 適応パラメータ（ゲイン）
- $e_c = r - y$: 制御誤差（目標HR - 現在HR）
- ベース値 1.0: 抑揚の中立値

### 4.2 パラメータ更新（離散時間）

```python
# 追従誤差
tracking_error = reference_hr - current_hr

# 正規化MIT則
sensitivity = last_output
norm_factor = 1.0 + normalization_gain * sensitivity**2
delta_theta = gamma * tracking_error * sensitivity / norm_factor

# 忘却係数の適用（ドリフト防止）
theta = forgetting_factor * theta + delta_theta

# パラメータのクランプ
theta = max(theta_min, min(theta_max, theta))
```

### 4.3 安定化メカニズム

| メカニズム | 目的 | パラメータ |
|-----------|------|-----------|
| **パラメータ制限** | 発散防止 | θ_min=0.005, θ_max=0.1 |
| **デッドゾーン** | 小誤差での過剰適応防止 | 2.0 BPM |
| **忘却係数** | パラメータドリフト防止 | 0.995 |
| **正規化** | 数値安定性 | μ=0.01 |

## 5. パラメータ設定ガイド

### 5.1 適応ゲイン γ

$$\gamma \in [0.0001, 0.01]$$

- **小さい値（0.0001）**: 安定だが追従が遅い
- **大きい値（0.01）**: 速い追従だが不安定になりやすい
- **推奨**: 0.001（デフォルト）

### 5.2 参照モデル時定数 τ

$$\tau \in [5, 120] \text{ 秒}$$

- **小さい値（5秒）**: 速い応答を期待
- **大きい値（120秒）**: ゆっくりとした追従
- **推奨**: 30秒（心拍数の生理的応答を考慮）

### 5.3 適応パラメータの初期値と範囲

$$\theta_0 = 0.02, \quad \theta \in [0.005, 0.1]$$

## 6. PID制御との比較

| 特性 | PID制御 | 適応制御（MRAC） |
|------|--------|-----------------|
| **パラメータ** | 固定（Kp, Ki, Kd） | 適応的に変化（θ） |
| **個人差への対応** | 手動調整が必要 | 自動的に適応 |
| **実装複雑度** | 低い | 中程度 |
| **安定性解析** | 容易 | 複雑 |
| **収束性** | 保証される | 条件付き |
| **適用場面** | 特性が既知の場合 | 不確かさがある場合 |

## 7. 動作例

目標HR=80BPMに対して、初期HR=65BPMから追従する場合：

```
Step  HR   Output   θ       状況
────────────────────────────────────
  1   65   1.300   0.0200   誤差大 → 高出力
  2   68   1.270   0.0225   θ増加
  3   72   1.180   0.0225   HR上昇中
  4   76   1.062   0.0155   目標に近づく
  5   78   1.014   0.0072   θ減少
  6   80   1.000   0.0050   目標到達
  7   82   0.990   0.0050   オーバーシュート
  8   80   1.000   0.0050   安定
```

## 8. 数学的背景

### 8.1 安定性（Lyapunov解析）

Lyapunov関数：
$$V = \frac{1}{2}e^2 + \frac{1}{2\gamma}(\theta - \theta^*)^2$$

ここでθ*は理想的なパラメータ。MIT則のもとで：

$$\dot{V} = e\dot{e} + \frac{1}{\gamma}(\theta - \theta^*)\dot{\theta}$$

適切な条件下で $\dot{V} \leq 0$ となり、安定性が保証されます。

### 8.2 収束条件

- 持続的励振条件（Persistent Excitation）
- 制御誤差が十分大きい
- パラメータ範囲内で動作

## 9. 実装ファイル構成

```
HCS_ver4.0/
├── hrf2_controller.py      # 制御アルゴリズム本体
│   ├── ControlMode         # 制御モード列挙型（PID/Adaptive）
│   ├── HRF2Config          # PID制御設定
│   ├── AdaptiveConfig      # 適応制御設定
│   ├── AdaptiveController  # MRAC実装
│   └── HRF2Controller      # 統合コントローラ
├── audio_processing.py     # 音声処理・ProsodySettings
└── gui.py                  # GUI（制御モード選択UI）
```

## 10. 参考文献

1. Åström, K. J., & Wittenmark, B. (2008). *Adaptive Control*. Dover Publications.
2. Ioannou, P. A., & Sun, J. (2012). *Robust Adaptive Control*. Dover Publications.
3. Mitsuhashi et al. (2022). "Heart Rate Variability Control Using a Biofeedback and Wearable System." *MDPI Sensors*.
4. Narendra, K. S., & Annaswamy, A. M. (2005). *Stable Adaptive Systems*. Dover Publications.

---

この適応制御システムにより、被験者ごとの個人差や時間経過に伴う特性変化に対して、自動的にパラメータを調整し、より効果的な心拍数追従が期待できます。
