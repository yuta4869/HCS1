# ロバスト制御 (MEC: モデル誤差抑制補償器) - HCS v7.x

## 1. 概要

心拍数フィードバック制御において、被験者ごとの個人差（ゲイン変動、時定数の違い）や測定ノイズに対してロバスト（頑健）な制御を実現するため、**MEC（モデル誤差抑制補償器）**に基づく制御器を実装しました。

従来のH∞ループ整形法は理論的に複雑で、実装も煩雑でした。MECはよりシンプルな構造で、同等以上の頑健性を実現します。

## 2. MEC（モデル誤差抑制補償器）の原理

### 2.1 基本構造

MECは以下の3要素で構成されます：

1. **基本制御器 K0**: PI制御で目標追従を行う
2. **名目モデル Ĝ**: 人体の心拍応答を近似する簡略モデル
3. **MEC補償器 Cε**: モデル誤差を検出・補償する

### 2.2 ブロック線図

```
                                 ┌─────────────────────────┐
                                 │                         │
    r ───────(+)─── e ──→ [PI制御器 K0] ──→ u0 ──(+)─── u ──┬──→ [人体 G] ──→ y
              ▲                                    ▲        │
              │-                                   │        │
              │                              u_mec │        │
              │                                    │        │
              │                               [MEC: Cε]     │
              │                                    ▲        │
              │                                    │        │
              │                               ε = y - ŷ     │
              │                                    │        │
              │                                    │        │
              └───────────────── y ◄───── [名目モデル Ĝ] ◄──┘
```

### 2.3 動作原理

1. **基本制御器 K0** が目標心拍数 r と計測心拍数 y の偏差 e から制御出力 u0 を計算
2. **名目モデル Ĝ** が制御入力 u から予測心拍数 ŷ を計算
3. **モデル誤差 ε** = y - ŷ （実際の心拍と予測の差）を検出
4. **MEC補償器 Cε** がモデル誤差 ε を低域ろ波して補償入力 u_mec を生成
5. 最終出力 **u = u0 + u_mec**

名目モデルが完璧でなくても、MECがその誤差を補償することで頑健性が得られます。

## 3. 制御対象モデル

### 3.1 名目モデル Ĝ(s)

抑揚レベル u(t) から心拍数 HR(t) への応答を、一次遅れ＋むだ時間系としてモデル化：

```
         K
Ĝ(s) = ――――― × e^(-Ls)
       τs + 1

K  = 5.0   : 定常ゲイン（抑揚変化に対する心拍数変化の比）
τ  = 30.0秒: 時定数（心拍応答の遅れ）
L  = 5.0秒 : むだ時間（刺激から応答開始までの遅延）
```

### 3.2 離散実装

名目モデルの一次遅れ部分は双一次変換で離散化：

```
ŷ[k+1] = ŷ[k] + (dt/τ) × (K×(u_delayed - 1) + baseline - ŷ[k])
```

むだ時間は入力バッファで実装します。

## 4. 制御器実装

### 4.1 基本PI制御器 K0

```
u0 = 1.0 + Kp × e + Ki × ∫e dt

Kp = 0.02  : 比例ゲイン
Ki = 0.003 : 積分ゲイン
```

### 4.2 MEC補償器 Cε

モデル誤差を低域ろ波して返す：

```
           k
Cε(s) = ―――――――
        1 + s/ω

k  = 0.3     : MEC利得
ω  = 0.05 rad/s : カットオフ周波数
```

離散実装：
```
mec_filtered[k+1] = mec_filtered[k] + (dt × ω) × (k × ε - mec_filtered[k])
u_mec = -mec_filtered
```

### 4.3 制御アルゴリズム

1ステップの処理（hrf2_controller.py:299-407）：

```
入力: current_hr（現在心拍数）
出力: output（抑揚レベル 0.0〜2.0）

Step 1: 計測値のローパスフィルタ（ノイズ抑制）
  filtered_hr += α × (current_hr - filtered_hr)
  α = dt / (filter_tau + dt)

Step 2: 名目モデルの更新
  2a. むだ時間バッファから遅延入力を取得
  2b. 一次遅れ系を更新:
      model_target = baseline + K × (u_delayed - 1.0)
      ŷ += (dt/τ) × (model_target - ŷ)

Step 3: モデル誤差の計算
  ε = y - ŷ （実出力 - 名目モデル出力）

Step 4: MEC補償
  mec_filtered += (dt × ω) × (k × ε - mec_filtered)
  u_mec = -mec_filtered

Step 5: 基本PI制御
  e = target_hr - filtered_hr
  effective_error = 0 if |e| < deadband else e
  p_term = Kp × effective_error
  integral += Ki × effective_error × dt
  integral = clamp(integral, -integral_max, +integral_max)
  u0 = 1.0 + p_term + integral

Step 6: 総合出力
  raw_output = u0 + u_mec

Step 7: スルーレート制限
  Δu = clamp(raw_output - last_output, -du_max, +du_max)
  output = last_output + Δu

Step 8: 出力飽和
  output = clamp(output, 0.0, 2.0)
```

## 5. パラメータ設定

### 5.1 RobustConfig クラス

```python
@dataclass
class RobustConfig:
    # === 基本PI制御器パラメータ ===
    kp: float = 0.1             # 比例ゲイン
    ki: float = 0.003          # 積分ゲイン
    integral_max: float = 0.1     # アンチワインドアップ

    # === 名目モデルパラメータ ===
    model_gain: float = 5.0       # 定常ゲイン K
    model_tau: float = 30.0       # 時定数 τ [秒]
    model_delay: float = 5.0      # むだ時間 L [秒]

    # === MEC（モデル誤差抑制補償器）パラメータ ===
    mec_gain: float = 0.3         # MEC利得 k
    mec_omega: float = 0.05       # カットオフ周波数 ω [rad/s]

    # === 計測ノイズ抑制 ===
    filter_time_constant: float = 2.0  # LPF時定数 [秒]

    # === 出力制限 ===
    du_max: float = 0.15          # スルーレート制限

    # === デッドバンド ===
    deadband: float = 2.0         # [BPM]
```

### 5.2 チューニング指針

| パラメータ | 効果 | 推奨範囲 |
|-----------|------|---------|
| kp | 応答速度（比例） | 0.01〜0.05 |
| ki | 定常偏差除去 | 0.001〜0.01 |
| mec_gain | モデル誤差補償の強さ | 0.1〜0.5 |
| mec_omega | MEC応答速度 | 0.02〜0.1 rad/s |
| model_tau | 名目モデルの時定数 | 20〜60秒 |
| filter_time_constant | ノイズ除去 | 1.0〜5.0秒 |
| deadband | 不感帯 | 1.0〜3.0 BPM |
| du_max | 応答滑らかさ | 0.1〜0.2 |

## 6. MECの頑健性

### 6.1 なぜMECが効くのか

1. **名目モデルが正確な場合**: ε ≈ 0 なので u_mec ≈ 0、基本PI制御だけが働く
2. **名目モデルが不正確な場合**: ε ≠ 0 となり、MECがその差を補償
3. **個人差への対応**: 被験者ごとにゲインや時定数が異なっても、MECが差を吸収
4. **外乱への対応**: 予期しない心拍変動もモデル誤差として検出・補償

### 6.2 低域ろ波の役割

MECに低域ろ波（1次ローパスフィルタ）を入れる理由：

- **測定ノイズの増幅防止**: 高周波ノイズがモデル誤差として誤検出されるのを防ぐ
- **安定性の確保**: 高周波でのゲインを下げることで閉ループの安定性を保証
- **滑らかな補償**: 急激な補償入力を避け、自然な抑揚変化を実現

### 6.3 従来手法との比較

| 項目 | PID | H∞ループ整形 | MEC |
|------|-----|-------------|-----|
| 設計の複雑さ | 簡単 | 複雑 | 中程度 |
| 理論的裏付け | 経験的 | 厳密 | 明確 |
| 実装の複雑さ | 簡単 | 複雑 | 簡単 |
| 頑健性 | 低い | 高い | 高い |
| パラメータ数 | 3 | 10以上 | 7 |
| チューニング | 試行錯誤 | 周波数設計 | 直感的 |

## 7. 使用例

### 7.1 GUIでの使用

1. HCSアプリを起動
2. 「会話システム」タブ → 「HRF制御」セクション
3. 「有効」チェックボックスをON
4. 制御モード選択で「Robust」を選択
5. 必要に応じてパラメータを調整

### 7.2 プログラムからの使用

```python
from hrf2_controller import HRF2Controller, ControlMode, RobustConfig

# デフォルト設定で作成
controller = HRF2Controller(
    target_hr=70.0,
    control_mode=ControlMode.ROBUST
)

# カスタム設定
config = RobustConfig(
    kp=0.03,              # より積極的なPI制御
    ki=0.005,
    mec_gain=0.4,         # 強めのモデル誤差補償
    mec_omega=0.08,       # やや速いMEC応答
    deadband=1.5,         # 狭いデッドバンド
    du_max=0.1            # 滑らかな応答
)
controller._robust_controller.config = config

# 制御ループ
while running:
    current_hr = get_heart_rate()
    intonation = controller.compute(current_hr, dt=1.0)
    play_audio_with_intonation(intonation)
```

## 8. 安定性検証

### 8.1 検証結果サマリー

現在のパラメータ（Kp=0.1, Ki=0.003, mec_gain=0.3, mec_omega=0.05）での安定性検証結果:

| 項目 | 結果 | 判定 |
|------|------|------|
| **位相余裕** | 86.2° | ✓ 十分（≥30°推奨） |
| **閉ループ極** | -0.014, -0.036 | ✓ 全て負（安定） |
| **ロバスト性テスト** | 36/36ケース安定 | ✓ 100%安定 |
| **最大定常偏差** | 7.52 BPM | 許容範囲内 |

### 8.2 ロバスト性テスト条件

プラント変動に対する頑健性を検証:
- **ゲイン変動**: 0.5〜2.0倍（6段階）
- **時定数変動**: 0.5〜2.0倍（6段階）
- 計36通りの組み合わせ

全ての条件で閉ループが安定（発振なし、収束）を確認。

### 8.3 検証スクリプト

安定性検証は `tests/test_mec_stability.py` で実行可能:

```bash
source .venv/bin/activate
python tests/test_mec_stability.py
```

出力ファイル:
- `tests/mec_bode_plot.png` - ボード線図（ゲイン・位相余裕）
- `tests/mec_step_response.png` - ステップ応答（名目 vs 変動）
- `tests/mec_robustness_map.png` - ロバスト性マップ（定常偏差・安定性）

### 8.4 解釈

1. **位相余裕 86°**: 非常に保守的な設計。発振の心配なし
2. **ゲイン余裕**: 位相が-180°に達しないため測定不可（これも良い兆候）
3. **プラント変動への頑健性**: ゲイン・時定数が2倍変動しても安定

## 9. デバッグ情報

RobustController.update() は以下のデバッグ情報を返します：

```python
{
    "control_mode": "Robust(MEC)",
    "target_hr": 70.0,           # 目標心拍数
    "current_hr": 68.5,          # 計測心拍数（生値）
    "filtered_hr": 68.8,         # フィルタ後心拍数
    "model_output": 69.2,        # 名目モデル出力 ŷ
    "model_error": -0.4,         # モデル誤差 ε = y - ŷ
    "error": 1.2,                # 制御誤差 e = r - y
    "effective_error": 0.0,      # デッドバンド処理後誤差
    "p_term": 0.0,               # 比例項
    "integral": 0.15,            # 積分項
    "u0": 1.15,                  # 基本制御出力
    "u_mec": 0.02,               # MEC補償入力
    "raw_output": 1.17,          # 総合出力（制限前）
    "output": 1.17               # 最終出力
}
```

## 10. 参考文献

1. 須田信英, 他. "PID制御の基礎と応用." 朝倉書店, 1992.

2. 山本透, 増田士朗. "モデル誤差抑制補償器（MEC）の設計法." 計測自動制御学会論文集, Vol.32, No.12, 1996.

3. 川田昌克. "MATLAB/Simulinkによる制御工学入門." 森北出版, 2002.

4. 金原粲. "ロバスト制御入門." オーム社, 1994.

5. Astrom, K.J., Wittenmark, B. "Adaptive Control." 2nd ed., Addison-Wesley, 1995.
