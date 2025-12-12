# HRF2 ゲインスケジューリング制御システム - 技術解説

## 1. 概要

ゲインスケジューリング（Gain Scheduling）は、制御誤差の大きさに応じてPIDゲインを動的に切り替える制御手法です。HRF2システムでは、心拍数の目標値からの偏差に基づいて3つのゾーンを設定し、各ゾーンで異なるゲインとゲインタイプを使用します。

## 2. 基本原理

### 2.1 なぜゲインスケジューリングが必要か

固定ゲインPID制御の問題点：
- **大きな誤差時**: ゲインが小さいと応答が遅い
- **小さな誤差時**: ゲインが大きいとオーバーシュートや振動が発生

ゲインスケジューリングの解決策：
- 誤差が大きい → 強い制御（高ゲイン）で素早く目標に近づける
- 誤差が小さい → 穏やかな制御（低ゲイン）で安定させる

### 2.2 ブロック図

```
                              ┌─────────────────────────────────┐
                              │    ゲインスケジューラ            │
                              │                                 │
  目標HR r ──(+)──e──▶       │  |e| ≥ 高閾値 → 高ゲイン(PID)  │
            ↑  │              │  |e| ≥ 中閾値 → 中ゲイン(PI)   │     ┌──────────┐
            │  │              │  |e| < 中閾値 → 低ゲイン(P)    │────▶│ プラント │──▶ 現在HR y
            │  │              │                                 │     │  (人体)  │     │
            │  │              └─────────────────────────────────┘     └──────────┘     │
            │  │                                                                        │
            │  └────────────────────────────────────────────────────────────────────────┘
            │                            (-) フィードバック
            └───────────────────────────────────────────────────────────────────────────┘
```

## 3. ゾーン設計

### 3.1 3ゾーン構成

```
誤差 |e| = |目標HR - 現在HR|

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   |e| ≥ 15 BPM        高ゾーン (High Zone)                 │
    │   ─────────────────────────────────────────────────────────│
    │   7 ≤ |e| < 15 BPM    中ゾーン (Medium Zone)               │
    │   ─────────────────────────────────────────────────────────│
    │   |e| < 7 BPM         低ゾーン (Low Zone)                  │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
         ▲                       ▲
         │                       │
      高閾値(15)              中閾値(7)
```

### 3.2 閾値の意味

| 閾値名 | デフォルト値 | 意味 |
|--------|-------------|------|
| 高閾値 (error_threshold_high) | 15.0 BPM | この値以上の誤差で「高ゾーン」 |
| 中閾値 (error_threshold_medium) | 7.0 BPM | この値以上の誤差で「中ゾーン」 |

**注**: 低閾値は不要。中閾値未満なら自動的に「低ゾーン」となる。

### 3.3 具体例

目標心拍数 = 70 BPM の場合：

| 現在HR | 誤差 |e| | 判定 | ゾーン |
|--------|--------|------|------|
| 50 BPM | 20 | ≥15 | 高 |
| 55 BPM | 15 | ≥15 | 高 |
| 60 BPM | 10 | ≥7, <15 | 中 |
| 65 BPM | 5 | <7 | 低 |
| 70 BPM | 0 | <7 | 低 |
| 75 BPM | 5 | <7 | 低 |
| 80 BPM | 10 | ≥7, <15 | 中 |
| 90 BPM | 20 | ≥15 | 高 |

## 4. ゲインタイプ選択

### 4.1 4種類のゲインタイプ

各ゾーンで使用するPID制御の種類を選択できます：

| タイプ | 使用する項 | 特徴 |
|--------|-----------|------|
| **P** | 比例のみ | シンプル、定常偏差あり |
| **PI** | 比例 + 積分 | 定常偏差を除去 |
| **PD** | 比例 + 微分 | オーバーシュート抑制 |
| **PID** | 全項使用 | 最も高性能、調整が複雑 |

### 4.2 デフォルト設定の根拠

| ゾーン | デフォルトタイプ | 理由 |
|--------|-----------------|------|
| 高 | PID | 大きな誤差を素早く修正、微分で急激な変化を抑制 |
| 中 | PI | 中程度の誤差、積分で偏差を徐々に除去 |
| 低 | P | 目標付近では単純な比例制御で安定維持 |

### 4.3 ゲインタイプによる制御則

ゲインタイプに応じて、ki と kd が有効/無効になります：

```python
# 高ゾーンの場合
if gain_type == PID:
    kp, ki, kd = kp_high, ki_high, kd_high  # 全て有効
elif gain_type == PI:
    kp, ki, kd = kp_high, ki_high, 0.0      # kd無効
elif gain_type == PD:
    kp, ki, kd = kp_high, 0.0, kd_high      # ki無効
elif gain_type == P:
    kp, ki, kd = kp_high, 0.0, 0.0          # kp のみ
```

## 5. 実装詳細

### 5.1 ゾーン判定とゲイン取得

```python
def _get_target_gains(self, abs_error: float) -> Tuple[float, float, float, str, GainType]:
    """誤差の大きさに基づいてターゲットゲインを取得"""

    if abs_error >= self.config.error_threshold_high:
        # 高ゾーン
        gt = self.config.gain_type_high
        kp = self.config.kp_high
        ki = self.config.ki_high if gt in (GainType.PI, GainType.PID) else 0.0
        kd = self.config.kd_high if gt in (GainType.PD, GainType.PID) else 0.0
        return (kp, ki, kd, "high", gt)

    elif abs_error >= self.config.error_threshold_medium:
        # 中ゾーン
        gt = self.config.gain_type_medium
        kp = self.config.kp_medium
        ki = self.config.ki_medium if gt in (GainType.PI, GainType.PID) else 0.0
        kd = self.config.kd_medium if gt in (GainType.PD, GainType.PID) else 0.0
        return (kp, ki, kd, "medium", gt)

    else:
        # 低ゾーン
        gt = self.config.gain_type_low
        kp = self.config.kp_low
        ki = self.config.ki_low if gt in (GainType.PI, GainType.PID) else 0.0
        kd = self.config.kd_low if gt in (GainType.PD, GainType.PID) else 0.0
        return (kp, ki, kd, "low", gt)
```

### 5.2 ゲインスムージング

ゾーン間の切り替え時に出力が急変しないよう、ゲインを徐々に変化させます：

```python
# スムージング係数 (0.0〜1.0)
smoothing_factor = 0.3  # デフォルト

# 現在のゲインを目標ゲインに向けて徐々に更新
self._current_kp += smoothing_factor * (target_kp - self._current_kp)
self._current_ki += smoothing_factor * (target_ki - self._current_ki)
self._current_kd += smoothing_factor * (target_kd - self._current_kd)
```

**スムージング係数の効果:**
- 0.0: 変化なし（実用的でない）
- 0.3: 緩やかな遷移（デフォルト、推奨）
- 1.0: 即座に切り替え（チャタリングの可能性）

### 5.3 PID計算

```python
def compute(self, current_hr: float) -> Tuple[float, dict]:
    # 誤差計算
    error = self.config.target_hr - current_hr
    abs_error = abs(error)

    # ターゲットゲインを取得
    target_kp, target_ki, target_kd, zone, gain_type = self._get_target_gains(abs_error)

    # ゲインをスムージング
    self._current_kp += self.config.smoothing_factor * (target_kp - self._current_kp)
    self._current_ki += self.config.smoothing_factor * (target_ki - self._current_ki)
    self._current_kd += self.config.smoothing_factor * (target_kd - self._current_kd)

    # PID各項の計算
    p_term = self._current_kp * error

    if dt > 0:
        self._integral += error * dt
        self._integral = max(-self.config.integral_max,
                            min(self.config.integral_max, self._integral))
    i_term = self._current_ki * self._integral

    d_term = 0.0
    if self._last_error is not None and dt > 0:
        derivative = (error - self._last_error) / dt
        d_term = self._current_kd * derivative

    # 出力計算
    raw_output = 1.0 + p_term + i_term + d_term
    output = max(self.config.min_output, min(self.config.max_output, raw_output))

    return output, debug_info
```

## 6. パラメータ設定

### 6.1 デフォルト値一覧

| パラメータ | 高ゾーン | 中ゾーン | 低ゾーン |
|-----------|---------|---------|---------|
| Kp | 0.03 | 0.02 | 0.01 |
| Ki | 0.008 | 0.005 | 0.002 |
| Kd | 0.015 | 0.01 | 0.005 |
| ゲインタイプ | PID | PI | P |

| 共通パラメータ | デフォルト値 |
|---------------|-------------|
| 高閾値 | 15.0 BPM |
| 中閾値 | 7.0 BPM |
| スムージング係数 | 0.3 |
| 積分上限 | 20.0 |
| 出力最小値 | 0.0 |
| 出力最大値 | 2.0 |

### 6.2 パラメータ調整の指針

#### 閾値の調整

```
高閾値を下げる → 高ゲインが使われる範囲が広がる → より積極的な制御
中閾値を下げる → 低ゲインが使われる範囲が狭まる → 精密な制御範囲が減少
```

#### ゲインタイプの選択

| 状況 | 推奨設定 |
|------|---------|
| 振動が多い | 高ゾーンを PD に変更 |
| 定常偏差が残る | 低ゾーンを PI に変更 |
| 応答を速くしたい | 全ゾーンを PID に |
| シンプルにしたい | 全ゾーンを P に |

## 7. 動作特性

### 7.1 シナリオ例：目標HR=80BPM、初期HR=55BPM

```
時刻  現在HR  誤差  ゾーン  タイプ  出力   状態
────────────────────────────────────────────────
 0s    55    -25    高     PID    1.75   大きく抑揚上昇
 5s    60    -20    高     PID    1.60   引き続き上昇
10s    65    -15    高     PID    1.45   高ゾーン維持
15s    70    -10    中     PI     1.30   中ゾーンへ遷移
20s    74     -6    低     P      1.12   低ゾーンへ
25s    77     -3    低     P      1.06   目標に接近
30s    79     -1    低     P      1.02   ほぼ安定
35s    80      0    低     P      1.00   目標到達
```

### 7.2 ゾーン遷移の可視化

```
出力(抑揚)
  ↑
2.0├─────────────────────────────────────────────
   │
1.8├────╲
   │     ╲ 高ゾーン(PID)
1.6├──────╲
   │       ╲
1.4├────────╲───────
   │         ╲ 中ゾーン(PI)
1.2├──────────╲─────────────
   │           ╲╲ 低ゾーン(P)
1.0├─────────────╲____________  ← 目標到達
   │
0.8├─────────────────────────────────────────────
   └────────────────────────────────────────────▶ 時間
    0    10    20    30    40    50   (秒)
```

## 8. 他の制御方式との比較

| 項目 | 固定PID | ゲインスケジューリング | 適応制御(MRAC) |
|------|--------|---------------------|---------------|
| ゲイン調整 | 固定 | 誤差に応じて切替 | 自動適応 |
| 設計の容易さ | 容易 | 中程度 | 複雑 |
| 個人差対応 | 手動調整 | 事前設定 | 自動対応 |
| 安定性保証 | 明確 | 中程度 | 条件付き |
| 計算負荷 | 低い | 低い | やや高い |
| 推奨場面 | 特性既知 | 広い動作範囲 | 不確かさ大 |

## 9. GUI操作

### 9.1 設定項目

GainScheduledモード選択時、以下の設定が可能：

```
┌─ GS設定 ─────────────────────────────────────────────────────┐
│ 高閾値: [15.0] 中閾値: [7.0] [適用] zone: high              │
│ | 高: [PID▼] 中: [PI▼] 低: [P▼] type: PID                   │
└─────────────────────────────────────────────────────────────┘
```

- **高閾値/中閾値**: 数値入力で変更
- **高/中/低**: ドロップダウンでゲインタイプ選択
- **zone**: 現在のゾーン表示
- **type**: 現在使用中のゲインタイプ表示

### 9.2 リアルタイム表示

HRF2ステータス行に現在の状態が表示されます：

```
HRF2(GainScheduled): 目標80BPM / 現在65BPM / high(PID)
```

## 10. トラブルシューティング

| 症状 | 考えられる原因 | 対策 |
|------|---------------|------|
| 応答が遅い | 閾値が低すぎる | 高閾値・中閾値を上げる |
| 振動する | 高ゾーンのゲインが大きすぎる | 高ゾーンを PD に変更 |
| 定常偏差が残る | 低ゾーンが P のみ | 低ゾーンを PI に変更 |
| 急激な出力変化 | スムージング不足 | smoothing_factor を小さく |
| ゾーンが頻繁に切り替わる | 閾値が近すぎる | 閾値間の差を広げる |

## 11. 参考文献

1. Åström, K. J., & Hägglund, T. (2006). *Advanced PID Control*. ISA.
2. Shamma, J. S., & Athans, M. (1990). "Analysis of gain scheduled control for nonlinear plants." *IEEE Transactions on Automatic Control*, 35(8), 898-907.
3. Rugh, W. J., & Shamma, J. S. (2000). "Research on gain scheduling." *Automatica*, 36(10), 1401-1425.

---

ゲインスケジューリングは、固定PIDと適応制御の中間に位置する実用的な手法であり、心拍数フィードバックシステムのように広い動作範囲を持つシステムに適しています。
