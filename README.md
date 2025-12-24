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
| **アンケート解析** | アンケートデータの統計解析・グラフ作成 |

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

**ファイル命名規則:**
ECGファイルは以下の形式で命名してください:
```
{被験者ID}_{条件名}.csv
例: S01_No.csv, S01_MRAC.csv, S02_PID.csv
```

**対応する条件名:**
- `No` - 制御なし
- `PID` - PID制御
- `MRAC` - 適応制御
- `GainSchedule` - ゲインスケジューリング制御

**使い方:**
1. 「ECG/HRV解析」タブに移動
2. 「入力フォルダを選択」でECGファイルが入ったフォルダを選択
3. 「出力フォルダを選択」で結果の保存先を選択
4. 「バッチ解析 実行」をクリック

**出力される解析結果:**
- `{条件名}_result.xlsx` - 時系列のLF/HF・RMSSD（30秒スライディングウィンドウ）
- `{条件名}_resultLFHF5min.xlsx` - 全体平均LF/HF
- `Combined_HRV_Analysis.xlsx` - 全条件の統合結果
- `LFHF_Boxplot.png` - LF/HFの箱ひげ図
- `RMSSD_Boxplot.png` - RMSSDの箱ひげ図

**複数被験者の統合解析:**
1. 各被験者の解析結果フォルダを用意（例: `S01/`, `S02/`）
2. 「被験者データフォルダを選択」で親フォルダを選択
3. 「複数被験者の統合グラフ作成」をクリック
4. 条件ごとの比較グラフが生成されます

#### アンケート解析

アンケートデータの統計解析とグラフ作成を行います。

**ファイル形式:**
Excelファイル（.xlsx）に以下の形式でデータを入力:
- 1行目: ヘッダー（質問項目名）
- 条件名を含む列名（例: `No_Q1`, `PID_Q1`, `MRAC_Q1`）

**使い方:**
1. 「アンケート解析」タブに移動
2. 「アンケートファイルを選択」でExcelファイルを選択
3. 「出力フォルダを選択」で保存先を選択
4. 「解析 & グラフ作成」をクリック

**出力されるグラフ:**
- 条件ごとの比較棒グラフ
- エラーバー付きの統計グラフ

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
| **Questionnaire Analysis** | Statistical analysis and graph generation for questionnaire data |

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

**File Naming Convention:**
ECG files should be named as follows:
```
{SubjectID}_{ConditionName}.csv
Example: S01_No.csv, S01_MRAC.csv, S02_PID.csv
```

**Supported Condition Names:**
- `No` - No control
- `PID` - PID control
- `MRAC` - Adaptive control
- `GainSchedule` - Gain scheduling control

**Usage:**
1. Go to "ECG/HRV Analysis" tab
2. Click "Select Input Folder" to choose folder containing ECG files
3. Click "Select Output Folder" to choose where results will be saved
4. Click "Run Batch Analysis"

**Output Files:**
- `{ConditionName}_result.xlsx` - Time series LF/HF & RMSSD (30-sec sliding window)
- `{ConditionName}_resultLFHF5min.xlsx` - Overall average LF/HF
- `Combined_HRV_Analysis.xlsx` - Combined results for all conditions
- `LFHF_Boxplot.png` - LF/HF box plot
- `RMSSD_Boxplot.png` - RMSSD box plot

**Multi-Subject Integration:**
1. Prepare analysis result folders for each subject (e.g., `S01/`, `S02/`)
2. Click "Select Subject Data Folder" to choose the parent folder
3. Click "Create Multi-Subject Integrated Graphs"
4. Comparison graphs by condition are generated

#### Questionnaire Analysis

Statistical analysis and graph creation for questionnaire data.

**File Format:**
Excel file (.xlsx) with the following format:
- Row 1: Header (question item names)
- Column names including condition names (e.g., `No_Q1`, `PID_Q1`, `MRAC_Q1`)

**Usage:**
1. Go to "Questionnaire Analysis" tab
2. Click "Select Questionnaire File" to choose Excel file
3. Click "Select Output Folder" to choose save location
4. Click "Analyze & Create Graphs"

**Output Graphs:**
- Comparison bar graphs by condition
- Statistical graphs with error bars

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
│   └── questionnaire_analysis.py
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
