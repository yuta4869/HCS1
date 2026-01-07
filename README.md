# HCS (Human Conversation System)

心拍数と音声韻律を連携させた対話システム

A conversation system that integrates heart rate monitoring with speech prosody control.

---

## 📥 ダウンロード / Download

### 最新版 (v7.10.0)

**方法1: スタンドアロンアプリ (推奨)**

| OS | ファイル | インストール方法 |
|----|----------|-----------------|
| macOS | `HCS_App_x.x.x_macOS.dmg` | DMG を開き、アプリを Applications にドラッグ |
| Ubuntu | `hcs-app_x.x.x_amd64.deb` | `sudo dpkg -i hcs-app_x.x.x_amd64.deb` |

[Releases ページ](https://github.com/yuta4869/HCS1/releases) からダウンロード

**方法2: Git clone (開発者向け)**
```bash
git clone https://github.com/yuta4869/HCS1.git
cd HCS1
```

**方法3: 特定バージョンをダウンロード**
```bash
# 最新リリース
git clone --branch v7.6.0 --depth 1 https://github.com/yuta4869/HCS1.git

# または ZIP でダウンロード
# https://github.com/yuta4869/HCS1/archive/refs/tags/v7.6.0.zip
```

---

## 🇯🇵 日本語

### 概要

HCS (Human Conversation System) は、心拍数データと音声韻律（リズム、抑揚、強勢）を解析・連携させる対話システムです。Polar 心拍センサーからリアルタイムで心拍数を取得し、AI の音声出力の抑揚を自動調整することで、ユーザーの心拍数を目標値に誘導します。

### 主な機能

- **リアルタイム音声認識**: `faster-whisper` による高速な音声テキスト変換
- **音声活動検出 (VAD)**: 発話の開始・終了を自動検出
- **心拍数モニタリング**: Polar センサー（H10, OH1 等）との BLE 接続
- **心拍数フィードバック制御 (HRF2)**:
  - PID 制御
  - 適応制御 (MRAC)
  - ゲインスケジューリング制御
  - ロバスト制御 (MEC: モデル誤差抑制補償器)
- **LLM 対話**: OpenAI API またはローカル LLM（llama.cpp）対応
- **GUI**: Tkinter ベースの操作画面
- **データログ**: 会話履歴、心拍数、HRV などの研究用ログ出力

### 必要環境

- Python 3.8 以上
- macOS または Ubuntu
- Polar 心拍センサー（オプション）
- OpenAI API キー（オプション）

### セットアップ

**macOS:**
```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

**Ubuntu:**
```bash
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh
```

**手動インストール:**
```bash
# macOS
brew install portaudio

# Ubuntu
sudo apt-get install -y portaudio19-dev libbluetooth-dev python3-venv

# 仮想環境作成
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 環境変数

OpenAI API を使用する場合:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### ローカル LLM セットアップ（オプション）

OpenAI API を使わずにローカルで LLM を動かす場合:

1. **モデルをダウンロード** (GGUF 形式):
```bash
# 例: ELYZA Japanese Llama 3
huggingface-cli download mmnga/Llama-3-ELYZA-JP-8B-gguf Llama-3-ELYZA-JP-8B-q4_k_m.gguf --local-dir ./models
```

2. **GPU アクセラレーション** (オプション):
```bash
# macOS (Metal)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# NVIDIA GPU (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

3. **GUI で設定**:
   - LLM 選択で「ローカル LLM」を選択
   - モデルファイルのパスを指定

### 使い方

```bash
source .venv/bin/activate
python3 main.py
```

#### 基本的な流れ

1. **GUI 起動**: アプリを起動すると4つのタブがあるウィンドウが開きます
2. **センサー接続**: 「センサー類 接続」ボタンで Polar デバイスを接続
3. **被験者番号入力**: セッション情報に被験者IDを入力（ログファイル名に使用）
4. **会話開始**: 「音声対話 開始」ボタンをクリック
5. **対話**: 話しかけると AI が音声で応答します
6. **終了**: 「音声対話 停止」ボタンで終了

#### タブ説明

| タブ | 機能 |
|------|------|
| **会話システム** | メインの対話画面。心拍数モニター、LLM設定、音声設定、プロンプト編集など |
| **リアルタイムモニター** | 心拍数・ECG・HRV(SDNN)のリアルタイムグラフ表示 |
| **ECG/HRV解析** | 保存されたECGデータからLF/HF・RMSSDを計算 |
| **時系列解析** | HR/RMSSD/SDNNの時系列グラフを生成 |
| **アンケート解析(旧)** | 旧フォーマットのアンケートデータ解析（C列に条件番号） |
| **アンケート解析(新)** | 新フォーマットのアンケートデータ解析（ファイル名から条件を自動検出） |

#### 会話システムタブの設定

- **心拍数モニター**: 基準心拍数の設定、Polarセンサー接続
- **セッション情報**: 被験者番号（ログファイル名に使用）
- **LLM設定**: OpenAI API / ローカルLLM の切り替え
- **AIプロンプト**: システムプロンプトの編集
- **音声設定**: VOICEVOXの話者・速度・ピッチの調整
- **HRF制御**: 心拍数フィードバック制御の有効/無効、制御モード選択

#### リアルタイムモニター

1. 「会話システム」タブでセンサーを接続
2. 「リアルタイムモニター」タブに移動
3. 「モニター開始」ボタンをクリック
4. 心拍数、ECG波形、SDNN（心拍変動）がリアルタイム表示されます
5. SDNNは30秒バッファで毎秒計算され、CSVファイルにも保存されます

#### ECG/HRV 解析

保存されたECGデータからHRV指標を計算します。

**入力ファイル命名規則:**
ECGファイルは以下の形式で自動認識されます:
```
h10_ecg_session_No{被験者番号}_{日付}_{時刻}_{条件名}.csv
例: h10_ecg_session_No1_20251225_143000_HRF2_PID.csv
```

**対応する条件名:**
| 条件名 | 説明 |
|--------|------|
| `Fixed` | 固定韻律（制御なし） |
| `HRF` | 心拍数フィードバック（旧方式） |
| `Sin` | 正弦波韻律変調 |
| `HRF2_PID` | HRF2 PID制御 |
| `HRF2_Adaptive` | HRF2 適応制御（MRAC） |
| `HRF2_GS` | HRF2 ゲインスケジューリング |
| `HRF2_Robust` | HRF2 ロバスト制御（MEC） |

**解析パラメータ（GUI設定可能）:**
| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| センサーサンプルレート | 130 Hz | Polar H10のECGサンプリングレート |
| リサンプリング周波数 | 1.0 Hz | RRI等間隔化後のレート |
| 分位点フィルタ（下限） | 3.8%ile | 異常値除去の下限 |
| 分位点フィルタ（上限） | 96.2%ile | 異常値除去の上限 |
| 最小心拍数 | 45 BPM | 生理学的下限 |
| 最大心拍数 | 210 BPM | 生理学的上限 |
| 解析ウィンドウ | 30秒 | スライディングウィンドウ幅 |
| 解析開始オフセット | 30秒 | データ開始からの除外時間 |
| 解析終了オフセット | 5分30秒 | 解析終了時刻 |

**使い方:**
1. 「ECG/HRV解析」タブに移動
2. 「入力フォルダを選択」でECGファイルが入ったフォルダを選択
   - 出力フォルダが未設定の場合、入力フォルダと同じ場所が自動設定されます
3. 必要に応じて「出力フォルダを選択」で出力先を変更
4. 解析する条件にチェック（デフォルトで全条件選択）
5. 必要に応じて詳細パラメータを調整
6. 「HRV解析 実行」をクリック

**出力される解析結果:**
| ファイル名 | 内容 |
|-----------|------|
| `{被験者ID}_{条件}_result.xlsx` | 時系列LF/HF・RMSSD・SDNN（30秒スライディングウィンドウ） |
| `{被験者ID}_{条件}_resultLFHF5min.xlsx` | 全体平均LF/HF |
| `{被験者ID}_Combined_HRV_Analysis.xlsx` | 全条件統合結果 |
| `LFHF_Boxplot.png` | LF/HF箱ひげ図 |
| `RMSSD_Boxplot.png` | RMSSD箱ひげ図 |
| `SDNN_Boxplot.png` | SDNN箱ひげ図 |

**複数被験者の統合解析:**
1. 各被験者の解析結果フォルダを用意（例: `No1/`, `No2/`）
2. 「被験者データフォルダを選択」で親フォルダを選択
3. 「複数被験者の統合グラフ作成」をクリック
4. 条件ごとの比較グラフが生成されます

#### 時系列解析

HRセッションファイルからHR/RMSSD/SDNNの時系列グラフを生成します。

**入力ファイル:**
- `{被験者ID}_{条件}_result.xlsx` - ECG/HRV解析の出力ファイル
- `verity_hr_session_*.csv` - Verity Sense心拍数ログ
- `h10_hr_session_*.csv` - H10心拍数ログ
- `conversation_log_*.csv` - 会話ログ（発言時間帯表示用）

**グラフオプション（GUI設定）:**
| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| 基準心拍数 | 0 (非表示) | 緑の点線で表示 |
| 目標心拍数 | 0 (非表示) | 青の一点鎖線で表示 |
| 基準心拍数ラインを表示 | ON | 基準HRライン表示 |
| 目標心拍数ラインを表示 | ON | 目標HRライン表示（値が0より大きい場合のみ） |
| 全条件統合グラフも生成 | ON | Combined グラフを生成 |
| 発言時間帯を色付け表示 | ON | 会話ログから発言区間を背景色で表示 |

**目標心拍数ラインについて:**
- 「目標心拍数 (BPM)」に値を入力すると、グラフに目標ラインが表示されます
- 0の場合は非表示になります
- HRF2制御で使用した目標心拍数を入力してください

**出力グラフ:**
| ファイル名 | 内容 |
|-----------|------|
| `{被験者ID}_{条件}_RMSSD.png` | 条件別RMSSDグラフ |
| `{被験者ID}_{条件}_SDNN.png` | 条件別SDNNグラフ |
| `{被験者ID}_{条件}_H10_HR.png` | H10心拍数グラフ |
| `{被験者ID}_{条件}_Verity_HR.png` | Verity心拍数グラフ |
| `{被験者ID}_Combined_RMSSD.png` | 全条件統合RMSSDグラフ |
| `{被験者ID}_Combined_SDNN.png` | 全条件統合SDNNグラフ |

#### アンケート解析

アンケートデータの統計解析とグラフ作成を行います。**旧フォーマット**と**新フォーマット**の2種類のタブがあります。

##### アンケート解析(旧) - 旧フォーマット

C列に条件番号が含まれる形式のアンケートファイル用です。

**入力ファイル形式（Excel .xlsx）:**

```
┌─────┬─────────┬───────┬───────┬───────┬───────┬─────────────┐
│  A  │    B    │   C   │   D   │   E   │  ...  │     Z〜     │
├─────┼─────────┼───────┼───────┼───────┼───────┼─────────────┤
│ No  │ 被験者ID │ 条件  │ Q1    │ Q2    │  ...  │ PANAS項目   │
├─────┼─────────┼───────┼───────┼───────┼───────┼─────────────┤
│  1  │    1    │   1   │   4   │   3   │  ...  │     5       │
│  2  │    1    │   4   │   5   │   4   │  ...  │     3       │
│  3  │    2    │   1   │   3   │   4   │  ...  │     4       │
└─────┴─────────┴───────┴───────┴───────┴───────┴─────────────┘
```

| 列 | 内容 | 備考 |
|----|------|------|
| A列 | 行番号 | 任意 |
| B列 | 被験者ID | 数値または文字列 |
| C列 | 条件番号 | 1-7の番号または条件名 |
| D列〜 | アンケート回答 | リッカート尺度等 |

**列範囲の指定:**
- **独自アンケート**: デフォルト D列〜Y列
- **PANAS**: デフォルト Z列〜AO列

**使い方:**
1. 「アンケート解析(旧)」タブに移動
2. 「Excelファイルを選択」でファイルを選択
3. 解析する条件にチェック
4. 列範囲を確認（必要に応じて変更）
5. 「箱ひげ図を作成」または「PANAS解析を実行」をクリック

##### アンケート解析(新) - 新フォーマット

ファイル名から条件を自動検出する形式です。条件ごとに別ファイルとして保存されたアンケート用です。

**ファイル命名規則:**
```
実験後アンケート{条件番号}（回答）.xlsx
例: 実験後アンケート1（回答）.xlsx → Fixed条件
    実験後アンケート4（回答）.xlsx → HRF2_PID条件
```

**条件番号マッピング:**
| 番号 | 条件名 |
|------|--------|
| 0 | Test（テスト用、解析対象外） |
| 1 | Fixed |
| 2 | HRF |
| 3 | Sin |
| 4 | HRF2_PID |
| 5 | HRF2_Adaptive |
| 6 | HRF2_GS |
| 7 | HRF2_Robust |

**入力ファイル形式（Excel .xlsx）:**

```
┌─────┬─────────┬───────┬───────┬───────┬───────┬─────────────┐
│  A  │    B    │   C   │   D   │   E   │  ...  │     U〜     │
├─────┼─────────┼───────┼───────┼───────┼───────┼─────────────┤
│時刻 │ 被験者ID │ Q1    │ Q2    │ Q3    │  ...  │ PANAS項目   │
├─────┼─────────┼───────┼───────┼───────┼───────┼─────────────┤
│     │    1    │   4   │   3   │   5   │  ...  │     5       │
│     │    2    │   5   │   4   │   4   │  ...  │     3       │
└─────┴─────────┴───────┴───────┴───────┴───────┴─────────────┘
```

| 列 | 内容 | 備考 |
|----|------|------|
| A列 | タイムスタンプ | 任意 |
| B列 | 被験者ID | 数値または文字列 |
| C〜T列 | 設問（18問） | デフォルト範囲 |
| U〜AK列 | PANAS（16項目） | デフォルト範囲 |

**使い方:**
1. 「アンケート解析(新)」タブに移動
2. 「フォルダを選択」でアンケートファイルが入ったフォルダを選択
   - ファイル名から条件が自動検出され、検出結果が表示されます
   - 出力フォルダが未設定の場合、入力フォルダと同じ場所が自動設定されます
3. 必要に応じて「出力フォルダを選択」で出力先を変更
4. 解析する条件にチェック
5. 列範囲を確認（必要に応じて変更）
6. 「箱ひげ図を作成」または「PANAS解析を実行」をクリック

**出力（独自アンケート）:**
```
{出力フォルダ}/
├── question_boxplots_grid_v2.png   # 全設問グリッド表示
└── question_boxplots_v2/           # 個別グラフフォルダ
    ├── 設問名1_boxplot.png
    ├── 設問名2_boxplot.png
    └── ...
```

**出力（PANAS）:**
```
{出力フォルダ}/PANAS_analysis_v2/
├── PANAS_boxplot_v2.png    # PA/NA箱ひげ図（並列表示）
├── PANAS_barplot_v2.png    # PA/NA棒グラフ（平均±SD）
└── PANAS_results_v2.xlsx   # 統計結果
    ├── 条件別統計シート    # n, 平均, SD, 中央値
    ├── 内的一貫性シート    # α係数, ω係数
    └── 個人データシート    # Subject, Condition, PA_Score, NA_Score
```

##### PANAS項目

日本語版PANAS（Positive and Negative Affect Schedule）の解析機能です。

**PANAS項目（16項目）:**
- **PA（ポジティブ情動）8項目**: 活気のある、誇らしい、強気な、きっぱりとした、気合いの入った、わくわくした、機敏な、熱狂した
- **NA（ネガティブ情動）8項目**: びくびくした、おびえた、うろたえた、心配した、ぴりぴりした、苦悩した、恥じた、いらだった

**得点範囲**: 各項目1〜6点（6件法）、PA/NA合計 8〜48点

**信頼性係数:**
- クロンバックのα係数
- McDonald's ω係数（簡易版）

#### ログファイル

会話中のデータは `logs/` フォルダに自動保存されます:

| ファイル名 | 内容 |
|-----------|------|
| `conversation_log_{subject_id}_{timestamp}.txt` | 会話本文（全セッション通し） |
| `conversation_log_{subject_id}_{session_timestamp}_{mode}.csv` | 会話履歴CSV |
| `verity_hr_prosody_{subject_id}_{session_timestamp}_{mode}.csv` | Verity HR + 韻律レベル |
| `verity_hr_session_{subject_id}_{session_timestamp}_{mode}.csv` | Verity Sense心拍数 |
| `h10_ecg_session_{subject_id}_{session_timestamp}_{mode}.csv` | H10 ECG生データ（130Hz） |
| `h10_hr_session_{subject_id}_{session_timestamp}_{mode}.csv` | H10 心拍数 |
| `h10_sdnn_session_{subject_id}_{session_timestamp}_{mode}.csv` | H10 SDNN時系列 |
| `heartrate_after_tts_{subject_id}_{session_timestamp}_{mode}.csv` | TTS後心拍数 |
| `heartrate_at_recording_start_{subject_id}_{session_timestamp}_{mode}.csv` | 録音開始時心拍数 |
| `interaction_events_{subject_id}_{session_timestamp}_{mode}.csv` | 対話イベントログ |

- `{subject_id}`: 被験者番号（例: No1）
- `{session_timestamp}`: セッション開始時刻（例: 20250728_160807）
- `{mode}`: 条件名（Sin/HRF/Fixed）

#### ECG/HRV解析 出力ファイル

| ファイル名 | 内容 |
|-----------|------|
| `{subject_id}_{condition}_result.xlsx` | 時系列LF/HF・RMSSD・SDNN（30秒スライディングウィンドウ） |
| `{subject_id}_{condition}_resultLFHF5min.xlsx` | 全体平均LF/HF |
| `{subject_id}_Combined_HRV_Analysis.xlsx` | 全条件統合結果 |
| `LFHF_Boxplot.png` | LF/HF箱ひげ図（複数被験者統合時） |
| `RMSSD_Boxplot.png` | RMSSD箱ひげ図 |
| `SDNN_Boxplot.png` | SDNN箱ひげ図 |

#### 時系列解析 出力ファイル

| ファイル名 | 内容 |
|-----------|------|
| `{subject_id}_{condition}_RMSSD.png` | 条件別RMSSDグラフ |
| `{subject_id}_{condition}_SDNN.png` | 条件別SDNNグラフ |
| `{subject_id}_{condition}_{H10\|Verity}_HR.png` | 条件別HRグラフ（デバイス別） |
| `{subject_id}_Combined_RMSSD.png` | 全条件統合RMSSDグラフ |
| `{subject_id}_Combined_SDNN.png` | 全条件統合SDNNグラフ |
| `{subject_id}_Combined_HR.png` | 全条件統合HRグラフ |

---

## 🇬🇧 English

### Overview

HCS (Human Conversation System) is a dialogue system that analyzes and integrates heart rate data with speech prosody (rhythm, intonation, stress). It retrieves real-time heart rate from Polar sensors and automatically adjusts the AI's voice prosody to guide the user's heart rate toward a target value.

### Key Features

- **Real-time Transcription**: Fast speech-to-text using `faster-whisper`
- **Voice Activity Detection (VAD)**: Automatic speech start/end detection
- **Heart Rate Monitoring**: BLE connection with Polar sensors (H10, OH1, etc.)
- **Heart Rate Feedback Control (HRF2)**:
  - PID Control
  - Adaptive Control (MRAC)
  - Gain Scheduling Control
  - Robust Control (MEC: Model Error Compensation)
- **LLM Conversation**: OpenAI API or local LLM (llama.cpp) support
- **GUI**: Tkinter-based interface
- **Data Logging**: Conversation history, heart rate, HRV logging for research

### Requirements

- Python 3.8+
- macOS or Ubuntu
- Polar heart rate sensor (optional)
- OpenAI API key (optional)

### Setup

**macOS:**
```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

**Ubuntu:**
```bash
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh
```

**Manual Installation:**
```bash
# macOS
brew install portaudio

# Ubuntu
sudo apt-get install -y portaudio19-dev libbluetooth-dev python3-venv

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

For OpenAI API:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Local LLM Setup (Optional)

To run LLM locally without OpenAI API:

1. **Download a model** (GGUF format):
```bash
# Example: ELYZA Japanese Llama 3
huggingface-cli download mmnga/Llama-3-ELYZA-JP-8B-gguf Llama-3-ELYZA-JP-8B-q4_k_m.gguf --local-dir ./models
```

2. **GPU Acceleration** (optional):
```bash
# macOS (Metal)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# NVIDIA GPU (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

3. **Configure in GUI**:
   - Select "Local LLM" in LLM selection
   - Specify the model file path

### Usage

```bash
source .venv/bin/activate
python3 main.py
```

#### Basic Flow

1. **Launch GUI**: Opens a window with 4 tabs
2. **Connect Sensor**: Click "Connect Sensors" to connect Polar devices
3. **Enter Subject ID**: Input subject ID in session info (used for log filenames)
4. **Start Conversation**: Click "Start Voice Dialogue" button
5. **Dialogue**: Speak and AI will respond with voice
6. **End**: Click "Stop Voice Dialogue" button

#### Tab Overview

| Tab | Function |
|-----|----------|
| **Conversation System** | Main dialogue screen with HR monitor, LLM settings, voice settings, prompt editor |
| **Realtime Monitor** | Real-time graphs of HR, ECG, and HRV (SDNN) |
| **ECG/HRV Analysis** | Calculate LF/HF and RMSSD from saved ECG data |
| **Timeseries Analysis** | Generate HR/RMSSD/SDNN time series graphs |
| **Questionnaire Analysis (Old)** | Questionnaire analysis for old format (condition number in column C) |
| **Questionnaire Analysis (New)** | Questionnaire analysis for new format (condition auto-detected from filename) |

#### Conversation System Settings

- **Heart Rate Monitor**: Reference HR setting, Polar sensor connection
- **Session Info**: Subject ID (used in log filenames)
- **LLM Settings**: Switch between OpenAI API / Local LLM
- **AI Prompt**: Edit system prompt
- **Voice Settings**: VOICEVOX speaker, speed, pitch adjustment
- **HRF Control**: Enable/disable heart rate feedback control, select control mode

#### Realtime Monitor

1. Connect sensors in "Conversation System" tab
2. Switch to "Realtime Monitor" tab
3. Click "Start Monitor" button
4. Heart rate, ECG waveform, and SDNN (heart rate variability) are displayed in real-time
5. SDNN is calculated every second from a 30-second buffer and saved to CSV

#### ECG/HRV Analysis

Calculate HRV indices from saved ECG data.

**Input File Naming Convention:**
ECG files are automatically recognized in the following format:
```
h10_ecg_session_No{SubjectNumber}_{Date}_{Time}_{ConditionName}.csv
Example: h10_ecg_session_No1_20251225_143000_HRF2_PID.csv
```

**Supported Condition Names:**
| Condition | Description |
|-----------|-------------|
| `Fixed` | Fixed prosody (no control) |
| `HRF` | Heart rate feedback (legacy) |
| `Sin` | Sinusoidal prosody modulation |
| `HRF2_PID` | HRF2 PID control |
| `HRF2_Adaptive` | HRF2 Adaptive control (MRAC) |
| `HRF2_GS` | HRF2 Gain scheduling |
| `HRF2_Robust` | HRF2 Robust control (MEC) |

**Analysis Parameters (GUI configurable):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| Sensor sample rate | 130 Hz | Polar H10 ECG sampling rate |
| Resampling frequency | 1.0 Hz | Rate after RRI interpolation |
| Quantile filter (low) | 3.8%ile | Lower outlier threshold |
| Quantile filter (high) | 96.2%ile | Upper outlier threshold |
| Min heart rate | 45 BPM | Physiological lower bound |
| Max heart rate | 210 BPM | Physiological upper bound |
| Analysis window | 30 sec | Sliding window width |
| Analysis start offset | 30 sec | Exclusion from data start |
| Analysis end offset | 5m 30s | Analysis end time |

**Usage:**
1. Go to "ECG/HRV Analysis" tab
2. Click "Select Input Folder" to choose folder containing ECG files
   - Output folder is automatically set to the same location if not specified
3. Optionally click "Select Output Folder" to change output location
4. Check conditions to analyze (all selected by default)
5. Adjust detailed parameters if needed
6. Click "Run HRV Analysis"

**Output Files:**
| Filename | Content |
|----------|---------|
| `{SubjectID}_{Condition}_result.xlsx` | Time series LF/HF, RMSSD, SDNN (30-sec sliding window) |
| `{SubjectID}_{Condition}_resultLFHF5min.xlsx` | Overall average LF/HF |
| `{SubjectID}_Combined_HRV_Analysis.xlsx` | Combined results for all conditions |
| `LFHF_Boxplot.png` | LF/HF box plot |
| `RMSSD_Boxplot.png` | RMSSD box plot |
| `SDNN_Boxplot.png` | SDNN box plot |

**Multi-Subject Integration:**
1. Prepare analysis result folders for each subject (e.g., `No1/`, `No2/`)
2. Click "Select Subject Data Folder" to choose the parent folder
3. Click "Create Multi-Subject Integrated Graphs"
4. Comparison graphs by condition are generated

#### Timeseries Analysis

Generate HR/RMSSD/SDNN time series graphs from HR session files.

**Input Files:**
- `{SubjectID}_{Condition}_result.xlsx` - ECG/HRV analysis output
- `verity_hr_session_*.csv` - Verity Sense heart rate log
- `h10_hr_session_*.csv` - H10 heart rate log
- `conversation_log_*.csv` - Conversation log (for speech interval display)

**Graph Options (GUI settings):**
| Option | Default | Description |
|--------|---------|-------------|
| Reference HR | 0 (hidden) | Shown as green dashed line |
| Target HR | 0 (hidden) | Shown as blue dash-dot line |
| Show reference HR line | ON | Display reference HR line |
| Show target HR line | ON | Display target HR line (only if value > 0) |
| Generate combined graphs | ON | Generate Combined graphs |
| Show speech intervals | ON | Highlight speech intervals from conversation log |

**About Target HR Line:**
- Enter a value in "Target HR (BPM)" to display the target line on graphs
- Set to 0 to hide
- Enter the target HR used in HRF2 control

**Output Graphs:**
| Filename | Content |
|----------|---------|
| `{SubjectID}_{Condition}_RMSSD.png` | RMSSD graph by condition |
| `{SubjectID}_{Condition}_SDNN.png` | SDNN graph by condition |
| `{SubjectID}_{Condition}_H10_HR.png` | H10 heart rate graph |
| `{SubjectID}_{Condition}_Verity_HR.png` | Verity heart rate graph |
| `{SubjectID}_Combined_RMSSD.png` | Combined RMSSD graph (all conditions) |
| `{SubjectID}_Combined_SDNN.png` | Combined SDNN graph (all conditions) |

#### Questionnaire Analysis

Statistical analysis and graph creation for questionnaire data. Two tabs are available: **Old Format** and **New Format**.

##### Questionnaire Analysis (Old) - Old Format

For questionnaire files with condition number in column C.

**Excel File Format:**
```
    |   A    |    B    |    C     |   D  |   E  |  ...  |   Z~  |
----+--------+---------+----------+------+------+-------+-------+
  1 | (any)  | Subject | Condition|  Q1  |  Q2  |  ...  | PANAS |
----+--------+---------+----------+------+------+-------+-------+
  2 |        |   No1   |    1     |  5   |  4   |  ...  |   3   |
  3 |        |   No1   |    4     |  4   |  5   |  ...  |   4   |
  4 |        |   No2   |    1     |  3   |  4   |  ...  |   5   |
```

- **Column B**: Subject ID
- **Column C**: Condition number (1-7) or name
- **Columns D~**: Questionnaire items (default D-Y)
- **PANAS columns**: Default Z-AO

**Usage:**
1. Go to "Questionnaire Analysis (Old)" tab
2. Click "Select Excel File" to choose file
3. Check desired conditions
4. Verify column range (adjust if needed)
5. Click "Create Box Plots" or "Run PANAS Analysis"

##### Questionnaire Analysis (New) - New Format

For questionnaire files where condition is auto-detected from filename. Each condition is saved as a separate file.

**File Naming Convention:**
```
実験後アンケート{ConditionNumber}（回答）.xlsx
Example: 実験後アンケート1（回答）.xlsx → Fixed condition
         実験後アンケート4（回答）.xlsx → HRF2_PID condition
```

**Condition Number Mapping:**
| Number | Condition Name |
|--------|---------------|
| 0 | Test (excluded from analysis) |
| 1 | Fixed |
| 2 | HRF |
| 3 | Sin |
| 4 | HRF2_PID |
| 5 | HRF2_Adaptive |
| 6 | HRF2_GS |
| 7 | HRF2_Robust |

**Excel File Format:**
```
    |   A    |    B    |   C  |   D  |   E  |  ...  |   U~  |
----+--------+---------+------+------+------+-------+-------+
  1 | Time   | Subject |  Q1  |  Q2  |  Q3  |  ...  | PANAS |
----+--------+---------+------+------+------+-------+-------+
  2 |        |    1    |  4   |  3   |  5   |  ...  |   5   |
  3 |        |    2    |  5   |  4   |  4   |  ...  |   3   |
```

- **Column A**: Timestamp (optional)
- **Column B**: Subject ID
- **Columns C-T**: Questions (18 items, default)
- **Columns U-AK**: PANAS (16 items, default)

**Usage:**
1. Go to "Questionnaire Analysis (New)" tab
2. Click "Select Folder" to choose folder containing questionnaire files
   - Conditions are auto-detected from filenames and displayed
   - Output folder is automatically set to the same location if not specified
3. Optionally click "Select Output Folder" to change output location
4. Check desired conditions
5. Verify column range (adjust if needed)
6. Click "Create Box Plots" or "Run PANAS Analysis"

**Output (Custom Questionnaire):**
```
{OutputFolder}/
├── question_boxplots_grid_v2.png   # Summary grid
└── question_boxplots_v2/           # Individual plots folder
    ├── Question1_boxplot.png
    ├── Question2_boxplot.png
    └── ...
```

**Output (PANAS):**
```
{OutputFolder}/PANAS_analysis_v2/
├── PANAS_boxplot_v2.png    # PA/NA box plots
├── PANAS_barplot_v2.png    # PA/NA bar plots (mean±SD)
└── PANAS_results_v2.xlsx   # Statistics results
    ├── Condition Statistics  # n, mean, SD, median
    ├── Internal Consistency  # α, ω coefficients
    └── Individual Data       # Subject, Condition, PA_Score, NA_Score
```

##### PANAS Items

Japanese PANAS (Positive and Negative Affect Schedule) analysis.

**PANAS Items (16 items):**
- **PA (Positive Affect) 8 items**: 活気のある, 誇らしい, 強気な, きっぱりとした, 気合いの入った, わくわくした, 機敏な, 熱狂した
- **NA (Negative Affect) 8 items**: びくびくした, おびえた, うろたえた, 心配した, ぴりぴりした, 苦悩した, 恥じた, いらだった

**Score Range**: 1-6 per item, PA/NA total 8-48

**Reliability Coefficients:**
- **Cronbach's α**: Internal consistency
- **McDonald's ω**: Factor-based reliability (simplified)

#### Log Files

Data during conversation is auto-saved to `logs/` folder:

| Filename | Content |
|----------|---------|
| `conversation_log_{subject_id}_{timestamp}.txt` | Conversation text (all sessions) |
| `conversation_log_{subject_id}_{session_timestamp}_{mode}.csv` | Conversation history CSV |
| `verity_hr_prosody_{subject_id}_{session_timestamp}_{mode}.csv` | Verity HR + prosody level |
| `verity_hr_session_{subject_id}_{session_timestamp}_{mode}.csv` | Verity Sense heart rate |
| `h10_ecg_session_{subject_id}_{session_timestamp}_{mode}.csv` | H10 raw ECG (130Hz) |
| `h10_hr_session_{subject_id}_{session_timestamp}_{mode}.csv` | H10 heart rate |
| `h10_sdnn_session_{subject_id}_{session_timestamp}_{mode}.csv` | H10 SDNN time series |
| `heartrate_after_tts_{subject_id}_{session_timestamp}_{mode}.csv` | HR after TTS |
| `heartrate_at_recording_start_{subject_id}_{session_timestamp}_{mode}.csv` | HR at recording start |
| `interaction_events_{subject_id}_{session_timestamp}_{mode}.csv` | Interaction event log |

- `{subject_id}`: Subject ID (e.g., No1)
- `{session_timestamp}`: Session start time (e.g., 20250728_160807)
- `{mode}`: Condition name (Sin/HRF/Fixed)

#### ECG/HRV Analysis Output Files

| Filename | Content |
|----------|---------|
| `{subject_id}_{condition}_result.xlsx` | Time series LF/HF, RMSSD, SDNN (30-sec sliding window) |
| `{subject_id}_{condition}_resultLFHF5min.xlsx` | Overall average LF/HF |
| `{subject_id}_Combined_HRV_Analysis.xlsx` | Combined results for all conditions |
| `LFHF_Boxplot.png` | LF/HF box plot (multi-subject) |
| `RMSSD_Boxplot.png` | RMSSD box plot |
| `SDNN_Boxplot.png` | SDNN box plot |

#### Timeseries Analysis Output Files

| Filename | Content |
|----------|---------|
| `{subject_id}_{condition}_RMSSD.png` | RMSSD graph by condition |
| `{subject_id}_{condition}_SDNN.png` | SDNN graph by condition |
| `{subject_id}_{condition}_{H10\|Verity}_HR.png` | HR graph by condition (per device) |
| `{subject_id}_Combined_RMSSD.png` | Combined RMSSD graph (all conditions) |
| `{subject_id}_Combined_SDNN.png` | Combined SDNN graph (all conditions) |
| `{subject_id}_Combined_HR.png` | Combined HR graph (all conditions) |

---

## 📁 Project Structure

```
HCS1/
├── main.py                 # Entry point
├── gui/                    # GUI package (Tkinter)
│   ├── __init__.py
│   ├── application.py      # Main Application class
│   ├── status_window.py    # Status display window
│   ├── realtime_monitor.py # Realtime HR/ECG/SDNN monitor
│   ├── ecg_analysis.py     # ECG/HRV analysis
│   ├── questionnaire_analysis.py    # Questionnaire analysis (old format)
│   ├── questionnaire_analysis_v2.py # Questionnaire analysis (new format)
│   ├── panas_analysis.py   # PANAS analysis
│   └── timeseries_analysis.py # Timeseries analysis
├── audio_processing.py     # Audio processing & TTS
├── hrf2_controller.py      # Heart rate feedback controller
├── polar_monitor.py        # Polar sensor BLE connection
├── conversation_manager.py # LLM conversation management
├── config.py               # Configuration
├── requirements.txt        # Python dependencies
├── setup_mac.sh            # macOS setup script
├── setup_ubuntu.sh         # Ubuntu setup script
├── build_mac.sh            # macOS build script (DMG)
├── build_ubuntu.sh         # Ubuntu build script (DEB)
├── HCS_App.spec            # PyInstaller spec file
└── docs/                   # Technical documentation
```

## 🔨 ビルド / Build

スタンドアロンアプリをビルドする場合:

**macOS:**
```bash
chmod +x build_mac.sh
./build_mac.sh
# DMGを作成するか聞かれます → 出力: dist/HCS_App_x.x.x_macOS.dmg
```

**Ubuntu:**
```bash
chmod +x build_ubuntu.sh
./build_ubuntu.sh
# DEBを作成するか聞かれます → 出力: dist/hcs-app_x.x.x_amd64.deb
```

## 📄 License

MIT License

## 📝 Version History

- **v7.11.0** - Advanced analysis features and bug fixes:
  - **高度解析タブ追加**: 非線形HRV解析（Sample Entropy, DFA, Poincaré）、統計検定（ANOVA, Friedman）、相関分析
  - **制御メトリクス機能**: 時系列解析でRMSE, MAE, 制御率, 収束率, 立ち上がり時間, 整定時間, オーバーシュートを計算・CSV出力
  - **ECG解析Time列修正**: 解析区間のオフセットがTime列に反映されるように修正。例：60〜360秒で解析→Time列は60〜360
  - **時系列解析の時間同期**: ECG解析と同じ解析区間を指定すれば、_result.xlsx・HRセッション・発言ログの全データで時間軸が一致。全て「記録開始からの経過秒数」を基準に計算
  - **ECG→時系列解析の時間設定連携**: ECG解析で時間区間を指定すると、時系列解析タブに自動反映。「ECG設定から取得」ボタンで手動取得も可能
  - **解析区間デフォルト値変更**: 60〜360秒に変更（旧：30〜330秒）
  - **箱ひげ図改善**: 被験者ごと＋全体統合の両方を生成、統合データ（AllSubjects_HRV_Long.xlsx, AllSubjects_HRV_Wide.xlsx）も出力
  - **ファイル検索修正**: `{subject_id}_Combined_HRV_Analysis.xlsx` 形式に対応
  - **起動エラー修正**: main.pyの `import queue9` タイポを `import queue` に修正
- **v7.10.0** - ECG data validation at conversation start (warns if H10 ECG not recording)
- **v7.9.0** - Battery level display for Polar sensors (Verity Sense & H10)
- **v7.8.0** - H10 heart rate fallback when Verity Sense connection lost, connection status display (接続断/H10代替/HR断/ECG断)
- **v7.6.0** - Robust control rewritten using MEC (Model Error Compensation), stability verification added
- **v7.5.0** - Realtime monitor always shows target HR line with legend
- **v7.4.1** - ECG analysis partial condition support, matplotlib thread-safe backend fix
- **v7.4.0** - HRF2 settings persistence (gains, target HR), audio playback noise fix (macOS afplay), target HR realtime update
- **v7.3.0** - Output file naming with subject_id and mode, timeseries analysis tab
- **v7.2.0** - Timeseries analysis feature for individual subject HR/RMSSD/SDNN graphs
- **v7.1.0** - Target HR line display in realtime monitor when HRF2 enabled
- **v6.16.0** - Robust Control (H∞ loop shaping) added to HRF2 control modes
- **v6.13.1** - Scrollable conversation tab for small windows
- **v6.13.0** - SDNN CSV logging from ECG realtime monitor
- **v6.12.0** - ECG-based SDNN calculation with 130Hz R-peak detection
- **v6.11.0** - Real ECG data display from H10
- **v6.10.0** - Changed HRV display to SDNN, 130Hz sampling
- **v6.9.0** - Realtime monitor module separation
- **v6.8.0** - DMG/DEB package generation added to build scripts
- **v6.7.1** - Fix PyInstaller multiprocessing issue on macOS
- **v6.7.0** - VOICEVOX auto-start support (Mac/Ubuntu)
- **v6.6.0** - Cross-platform build system (PyInstaller)
- **v6.5.0** - GUI modular package structure
- **v6.3.2** - Gain type selection (P/PI/PD/PID) for GainSchedule control
- **v6.3.0** - GainSchedule control mode added
- **v6.2.0** - Adaptive control (MRAC) added
- **v6.0.0** - HRF2 PID control system added
- **v5.2.0** - Local LLM support added
