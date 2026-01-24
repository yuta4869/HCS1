# ローカル変更点メモ (GitHub vs ローカル)

**作成日**: 2026-01-20

## 概要

GitHub (`origin/main`) と比較して、ローカルには以下の変更があります。

---

## 1. gui/status_window.py - Amadeusスタイルへの大幅リデザイン

**変更内容**: シンプルなテキスト表示ウィンドウから、Amadeusスタイルのビジュアルリッチな会話表示ウィンドウへ変更

### 主な変更点:
- **ダークテーマUI**: 背景色 `#0a0f1e`、アクセントカラー（青/赤/緑/黄）
- **円形ビジュアライザー**: 状態に応じたアニメーション付き円形表示
- **状態表示**: idle/listening/speaking/thinkingの4状態
- **アニメーション**: パルスエフェクト、波形表示、発話アニメーション
- **心拍数表示機能追加**: `set_heart_rate(hr)` メソッド
- **ウィンドウサイズ**: 600x280 → 700x500

### 新規メソッド:
- `set_state(state)` - 状態を設定
- `set_heart_rate(hr)` - 心拍数を表示
- `_animate()` - アニメーションループ
- `_draw_visualizer()` - 円形ビジュアライザー描画
- `_draw_status_indicator()` - ステータスインジケーター描画

---

## 2. gui/timeseries_analysis.py - 制御率の許容誤差設定UI追加

**変更内容**: 制御メトリクス計算時の許容誤差(tolerance_bpm)をGUIから変更可能に

### 追加されたUI要素:
- Spinbox: 許容誤差設定 (1.0〜20.0 BPM、刻み0.5、デフォルト5.0)
- ラベル: 「制御率の許容誤差:」「BPM (目標±この値以内を制御成功とする)」

### コード変更:
- `ts_tolerance_bpm_var` (DoubleVar) 追加
- `ts_tolerance_spinbox` (Spinbox) 追加
- `_ts_toggle_metrics_options()`: スピンボックスの有効/無効切り替え追加
- `_ts_calculate_and_log_metrics()`: `ControlMetricsAnalyzer(tolerance_bpm=...)` に変更

---

## 3. 未追跡ファイル (新規作成)

### analyze_data.py
- Excelファイル解析用のユーティリティスクリプト
- 条件別のグループ化と統計計算

### analyze_surveys.py
- アンケートデータ解析用スクリプト

---

## 4. 更新されたデータファイル (コミット不要)

- `PANAS_analysis_v2/` - PANAS解析結果（再解析済み）
- `input.wav`, `output.wav` - 音声ファイル（テスト用）
- `.DS_Store` - macOSメタデータ

---

## 推奨アクション

1. **コミット推奨**: `gui/status_window.py`, `gui/timeseries_analysis.py`
2. **コミット任意**: `analyze_data.py`, `analyze_surveys.py` (ユーティリティスクリプト)
3. **コミット不要**: データファイル、`.DS_Store`
