# HCS (Human Conversation System)

心拍数と音声韻律を連携させた対話システム

A conversation system that integrates heart rate monitoring with speech prosody control.

---

## 📥 ダウンロード / Download

### 最新版 (v6.8.0)

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
git clone --branch v6.8.0 --depth 1 https://github.com/yuta4869/HCS1.git

# または ZIP でダウンロード
# https://github.com/yuta4869/HCS1/archive/refs/tags/v6.8.0.zip
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

1. GUI が起動します
2. Polar センサーを接続（任意）
3. 「会話開始」ボタンをクリック
4. 話しかけると AI が応答します

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

1. GUI will launch
2. Connect Polar sensor (optional)
3. Click "Start Conversation" button
4. Speak and AI will respond

---

## 📁 Project Structure

```
HCS1/
├── main.py                 # Entry point
├── gui/                    # GUI package (Tkinter)
│   ├── __init__.py
│   ├── application.py      # Main Application class
│   ├── status_window.py    # Status display window
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
