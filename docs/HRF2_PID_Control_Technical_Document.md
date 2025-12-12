# HRF2 PID制御システム - 技術解説

## 1. 概要

HRF2システムは、被験者の心拍数（HR）を目標値に追従させるため、音声の抑揚（intonation）パラメータを自動調整する制御システムです。本ドキュメントでは、**PID制御（Proportional-Integral-Derivative Control）** の実装について解説します。

## 2. PID制御の基本原理

### 2.1 制御の目的

- **入力**: 目標心拍数 $r$（BPM）
- **出力**: 抑揚レベル $u$（0.0 〜 2.0）
- **フィードバック**: 現在の心拍数 $y$（BPM）
- **目標**: 誤差 $e = r - y$ を最小化

### 2.2 ブロック図

```
                         ┌─────────────────────────────┐
                         │       PID コントローラ       │
                         │                             │
  目標HR r ──(+)──e──▶  │  ┌───┐ ┌───┐ ┌───┐        │     ┌──────────┐
            ↑  │        │  │ P │+│ I │+│ D │──▶ u   │────▶│ プラント │──▶ 現在HR y
            │  │        │  └───┘ └───┘ └───┘        │     │  (人体)  │     │
            │  │        └─────────────────────────────┘     └──────────┘     │
            │  │                                                             │
            │  └─────────────────────────────────────────────────────────────┘
            │                            (-)
            └────────────────────────────────────────────────────────────────┘
                                    フィードバック
```

## 3. PID制御の数式

### 3.1 連続時間系

PID制御の出力は以下の式で表されます：

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$$

ここで：
- $u(t)$: 制御出力（抑揚レベル）
- $e(t) = r - y(t)$: 誤差（目標HR - 現在HR）
- $K_p$: 比例ゲイン（Proportional gain）
- $K_i$: 積分ゲイン（Integral gain）
- $K_d$: 微分ゲイン（Derivative gain）

### 3.2 各項の役割

#### 比例項（P項）
$$P = K_p \cdot e(t)$$

- 現在の誤差に比例した出力
- 誤差が大きいほど、大きな修正を加える
- **効果**: 応答速度を上げる
- **欠点**: 定常偏差が残る

#### 積分項（I項）
$$I = K_i \int_0^t e(\tau) d\tau$$

- 誤差の累積に比例した出力
- 過去の誤差を積み上げて修正
- **効果**: 定常偏差を除去
- **欠点**: オーバーシュート、積分飽和（ワインドアップ）

#### 微分項（D項）
$$D = K_d \frac{de(t)}{dt}$$

- 誤差の変化率に比例した出力
- 誤差の変化を予測して先行的に修正
- **効果**: オーバーシュートを抑制、応答を滑らかに
- **欠点**: ノイズに敏感

### 3.3 離散時間系（実装）

サンプリング周期 $\Delta t$ での離散化：

$$u[k] = u_{base} + K_p \cdot e[k] + K_i \sum_{j=0}^{k} e[j] \cdot \Delta t + K_d \frac{e[k] - e[k-1]}{\Delta t}$$

本実装では $u_{base} = 1.0$（抑揚の中立値）を採用。

## 4. 実装詳細

### 4.1 デッドバンド処理

目標値付近での不要な制御動作を防ぐため、デッドバンドを導入：

$$e_{effective} = \begin{cases} 0 & \text{if } |e| < \delta \\ e & \text{otherwise} \end{cases}$$

ここで $\delta = 3.0$ BPM（デフォルト）

```python
# デッドバンド処理
if abs(error) < self.config.deadband:
    error = 0.0
```

### 4.2 アンチワインドアップ

積分項の飽和（ワインドアップ）を防ぐため、積分値を制限：

$$I_{limited} = \text{clamp}(I, -I_{max}, I_{max})$$

```python
# 積分項の更新
if dt > 0:
    self._integral += error * dt
    # アンチワインドアップ
    self._integral = max(-self.config.integral_max,
                         min(self.config.integral_max, self._integral))
```

### 4.3 出力クランプ

抑揚レベルを有効範囲内に制限：

$$u_{clamped} = \text{clamp}(u, u_{min}, u_{max})$$

```python
# 出力範囲にクランプ
output = max(self.config.min_output,
             min(self.config.max_output, raw_output))
```

### 4.4 完全な実装コード

```python
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

    return output, debug_info
```

## 5. パラメータ設定

### 5.1 デフォルト値

| パラメータ | 記号 | デフォルト値 | 説明 |
|-----------|------|-------------|------|
| 比例ゲイン | $K_p$ | 0.02 | 誤差に対する即座の応答 |
| 積分ゲイン | $K_i$ | 0.005 | 定常偏差の除去 |
| 微分ゲイン | $K_d$ | 0.01 | オーバーシュート抑制 |
| 目標心拍数 | $r$ | 70.0 BPM | 追従目標 |
| 出力最小値 | $u_{min}$ | 0.0 | 抑揚の下限 |
| 出力最大値 | $u_{max}$ | 2.0 | 抑揚の上限 |
| 積分上限 | $I_{max}$ | 20.0 | アンチワインドアップ |
| デッドバンド | $\delta$ | 3.0 BPM | 不感帯幅 |

### 5.2 パラメータ調整の指針

#### Ziegler-Nichols法（参考）

1. $K_i = K_d = 0$ として、$K_p$ を徐々に増加
2. 持続振動が発生する $K_p = K_u$（限界ゲイン）を見つける
3. 振動周期 $T_u$ を測定
4. 以下の式でパラメータを設定：

| 制御タイプ | $K_p$ | $K_i$ | $K_d$ |
|-----------|-------|-------|-------|
| P | $0.5 K_u$ | - | - |
| PI | $0.45 K_u$ | $0.54 K_u / T_u$ | - |
| PID | $0.6 K_u$ | $1.2 K_u / T_u$ | $0.075 K_u T_u$ |

#### 心拍数制御での経験則

心拍数フィードバックシステムは応答が遅く、以下の点に注意：

1. **$K_p$ は小さめ**: 0.01 〜 0.05
   - 大きすぎると不安定

2. **$K_i$ は非常に小さく**: 0.001 〜 0.01
   - 積分飽和を防ぐ

3. **$K_d$ は控えめ**: 0.005 〜 0.02
   - ノイズの影響を軽減

## 6. 動作特性

### 6.1 ステップ応答の例

目標HR=80BPMに対して、初期HR=65BPMからの応答：

```
時刻  現在HR  誤差   P項    I項    D項    出力   状態
─────────────────────────────────────────────────────
 0s    65    +15   +0.30  +0.00  +0.00  1.30   急上昇
 5s    70    +10   +0.20  +0.25  -0.10  1.35   上昇中
10s    75     +5   +0.10  +0.38  -0.10  1.38   収束中
15s    78     +2   +0.04  +0.40  -0.06  1.38   目標付近
20s    80      0   +0.00  +0.40  -0.04  1.36   デッドバンド内
25s    79     +1   +0.02  +0.40  +0.02  1.44   微調整
30s    80      0   +0.00  +0.40  -0.02  1.38   安定
```

### 6.2 各項の寄与グラフ

```
出力
 ↑
2.0├─────────────────────────────
   │
1.5├─────────╱╲─────────────────  ← 合計出力
   │       ╱    ╲
1.3├─────╱────────╲─────────────
   │   ╱            ╲___________
1.0├──╱──────────────────────────  ← ベースライン
   │
0.5├─────────────────────────────
   │
0.0└─────────────────────────────▶ 時間
    0    10    20    30    40   (秒)
```

## 7. 誤差の符号と制御の関係

本システムでは以下の制御方向を採用：

| 状態 | 誤差 $e$ | 制御動作 | 期待効果 |
|------|---------|---------|---------|
| HR < 目標 | $e > 0$ | 抑揚↑ | 興奮させてHR↑ |
| HR = 目標 | $e = 0$ | 維持 | 現状維持 |
| HR > 目標 | $e < 0$ | 抑揚↓ | 落ち着かせてHR↓ |

```python
# 誤差計算
error = target_hr - current_hr  # 正なら抑揚を上げる
```

## 8. 安定性解析

### 8.1 閉ループ伝達関数

プラント（人体の心拍数応答）を一次遅れ系と仮定：

$$G(s) = \frac{K}{1 + T_p s}$$

PIDコントローラ：

$$C(s) = K_p + \frac{K_i}{s} + K_d s$$

閉ループ伝達関数：

$$H(s) = \frac{C(s)G(s)}{1 + C(s)G(s)}$$

### 8.2 安定条件

Routh-Hurwitzの安定判別法により、以下の条件で安定：

1. すべてのゲインが正
2. $K_p$, $K_i$, $K_d$ の適切なバランス
3. 位相余裕 > 45°（推奨）
4. ゲイン余裕 > 6dB（推奨）

### 8.3 実用的な安定性確保

```python
# ゲインの範囲制限
self.config.kp = max(0.0, kp)  # 非負
self.config.ki = max(0.0, ki)  # 非負
self.config.kd = max(0.0, kd)  # 非負

# 出力の範囲制限
output = max(min_output, min(max_output, output))
```

## 9. トラブルシューティング

### 9.1 よくある問題と対策

| 症状 | 原因 | 対策 |
|------|------|------|
| 振動が止まらない | $K_p$ が大きすぎる | $K_p$ を下げる |
| 目標に到達しない | $K_i$ が小さすぎる | $K_i$ を上げる |
| オーバーシュート大 | $K_d$ が小さすぎる | $K_d$ を上げる |
| 応答が遅すぎる | 全体的にゲインが低い | バランスよく上げる |
| 急激な出力変化 | $K_d$ が大きすぎる | $K_d$ を下げる |

### 9.2 デバッグ情報

実装では以下のデバッグ情報を出力：

```python
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
```

## 10. 適応制御との比較

| 項目 | PID制御 | 適応制御（MRAC） |
|------|--------|-----------------|
| パラメータ | 固定（手動調整） | 自動調整 |
| 設計の容易さ | 容易 | 中程度 |
| 安定性保証 | 明確 | 条件付き |
| 個人差対応 | 手動再調整 | 自動適応 |
| 計算負荷 | 低い | やや高い |
| 推奨場面 | 特性既知・安定環境 | 不確かさ大・変動環境 |

## 11. 参考文献

1. Åström, K. J., & Murray, R. M. (2021). *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton University Press.
2. Ogata, K. (2010). *Modern Control Engineering*. 5th Edition, Pearson.
3. Franklin, G. F., Powell, J. D., & Emami-Naeini, A. (2015). *Feedback Control of Dynamic Systems*. 7th Edition, Pearson.
4. Ziegler, J. G., & Nichols, N. B. (1942). "Optimum settings for automatic controllers." *Transactions of the ASME*, 64(11), 759-768.

---

PID制御は古典的ながら効果的な制御手法であり、適切なパラメータ調整により、心拍数フィードバックシステムで良好な追従性能を実現できます。
