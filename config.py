# config.py

import os
import sys # resource_path のために追加

# --- Helper function for PyInstaller ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Development mode or when not bundled by PyInstaller
        try:
            # If __file__ is defined (e.g. when run as a script)
            base_path = os.path.abspath(os.path.dirname(__file__))
        except NameError:
            # If __file__ is not defined (e.g. in an interactive interpreter or frozen environment without __file__)
            base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- GPU Detection ---
def _detect_gpu_backend():
    """利用可能なGPUバックエンドを検出する

    Returns:
        tuple: (backend_name, gpu_layers)
            backend_name: "cuda", "mps", "cpu"
            gpu_layers: -1 (全レイヤーをGPUに), 0 (CPUのみ)
    """
    try:
        import torch

        # CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU検出: CUDA ({device_name})")
            return "cuda", -1

        # MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("GPU検出: MPS (Apple Silicon)")
            return "mps", -1

        # CPU fallback
        print("GPU検出: なし (CPUを使用)")
        return "cpu", 0

    except ImportError:
        print("警告: PyTorchがインストールされていません (CPUを使用)")
        return "cpu", 0

# GPU設定を検出
GPU_BACKEND, _DEFAULT_GPU_LAYERS = _detect_gpu_backend()

# --- General Configuration ---
# Whisper設定: CUDA/CPUに応じて自動選択
# 注意: faster-whisperはCUDAのみGPU対応、MPSは非対応
def _detect_whisper_settings():
    """GPU利用可能かどうかで最適なWhisper設定を返す"""
    try:
        import torch
        if torch.cuda.is_available():
            # CUDA: より大きなモデルと高精度な計算
            return "medium", "float16"
        else:
            # CPU/MPS: faster-whisperはMPS非対応なのでCPU設定
            # int8はCPUで高速、float32は互換性が高い
            return "base", "int8"
    except ImportError:
        # torchがない場合はCPU設定
        return "base", "int8"

WHISPER_MODEL_NAME, WHISPER_COMPUTE_TYPE = _detect_whisper_settings()
WHISPER_TRANSCRIBE_BEAM_SIZE = 5       # For faster-whisper. Smaller values (e.g., 1 or 3) can be faster but less accurate.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# --- Audio Processing Settings ---
AUDIO_BUFFER_SECONDS = 30  # seconds
TRANSCRIPTION_INTERVAL = 1.0 # seconds
SAVE_UTTERANCE_WAV = True  # For research, save each detected utterance to a WAV file
UTTERANCE_WAV_DIR = os.path.join(LOG_DIR, "utterances")

# --- Audio Device Settings ---
# マイクデバイスの設定（環境に応じて変更）
# デバイス名の一部を指定すると自動的にマッチするデバイスを検索
# 複数のキーワードをリストで指定可能（優先順位順）
# 利用可能なデバイスは以下のコマンドで確認:
#   python -c "import sounddevice as sd; print(sd.query_devices())"

# 会話システム用マイク（音声認識に使用）
# 例: ["インカム", "Headset", "USB Audio"] - インカムやヘッドセット優先
# ヘッドセット/マイク専用デバイスを優先（ウェブカメラより先にマッチさせる）
# HSOH20などのヘッドセットは "HSOH" や "Headset" を含むことが多い
# "USB Audio" はウェブカメラにもマッチするため後ろに
CONVERSATION_MIC_KEYWORDS = ["HSOH", "Headset", "インカム", "マイク", "USB Audio"]

# 映像録画用マイク（ビデオ録音に使用）
# 例: ["Webcam", "USB Camera", "C920"] - USBカメラのマイク優先
VIDEO_MIC_KEYWORDS = ["Webcam", "USB Camera", "C920", "C922", "Camera"]

# フォールバック: キーワードにマッチしない場合はデフォルトデバイスを使用
# True = デフォルトデバイスを使用, False = エラーとして扱う
USE_DEFAULT_MIC_AS_FALLBACK = True

# --- Audio Stream Settings (Linux/Ubuntu向け安定性設定) ---
# "input overflow"エラーが頻発する場合はこれらの値を調整
# AUDIO_CHUNK_SIZE: バッファサイズ（大きいほど安定だがレイテンシー増加）
# AUDIO_LATENCY: "low", "high", または秒数（例: 0.2）
#   - macOS: "low"で問題なし
#   - Linux/Ubuntu (PulseAudio): "high"推奨
import platform
if platform.system() == "Linux":
    AUDIO_CHUNK_SIZE = 4096  # Linux: 大きめのバッファ
    AUDIO_LATENCY = "high"   # Linux: 高レイテンシーで安定性確保
else:
    AUDIO_CHUNK_SIZE = 2048  # macOS/Windows: 標準サイズ
    AUDIO_LATENCY = "high"   # デフォルトは高レイテンシー

# ビデオ録画時の音声キャプチャ設定
# 録画用マイクは会話用とは別スレッドで動くため、より大きなバッファと低サンプリングレートで安定させる
VIDEO_AUDIO_SAMPLE_RATE = 16000  # Hz （必要に応じて 8000〜48000 の間で調整）
VIDEO_AUDIO_CHANNELS = 1
VIDEO_AUDIO_CHUNK_SIZE = 8192  # 4096 では不安定な環境向けに大きめをデフォルト化
VIDEO_AUDIO_LATENCY = "high"

INPUT_WAV_FILE = "input.wav"           # Filename for recorded audio
OUTPUT_WAV_FILE = "output.wav"         # Filename for synthesized audio
CONFIG_FILE = resource_path("config_heartrate_prosody.json")

# --- Log File Templates ---
# モード名: Sin (正弦波), HRF (心拍フィードバック), Fixed (抑揚固定)
CONVERSATION_LOG_FILE_TEMPLATE = os.path.join(LOG_DIR, "conversation_log_{subject_id}_{timestamp}.txt")
CONVERSATION_CSV_LOG_FILE_TEMPLATE = os.path.join(LOG_DIR, "conversation_log_{subject_id}_{session_timestamp}_{mode}.csv")
HR_PROSODY_CSV_TEMPLATE = os.path.join(LOG_DIR, "verity_hr_prosody_{subject_id}_{session_timestamp}_{mode}.csv")

HEARTRATE_AFTER_TTS_CSV_TEMPLATE = os.path.join(LOG_DIR, "heartrate_after_tts_{subject_id}_{session_timestamp}_{mode}.csv")
HEARTRATE_AT_RECORDING_START_CSV_TEMPLATE = os.path.join(LOG_DIR, "heartrate_at_recording_start_{subject_id}_{session_timestamp}_{mode}.csv")

VERITY_HR_SESSION_CSV_TEMPLATE = os.path.join(LOG_DIR, "verity_hr_session_{subject_id}_{session_timestamp}_{mode}.csv")
H10_ECG_SESSION_CSV_TEMPLATE = os.path.join(LOG_DIR, "h10_ecg_session_{subject_id}_{session_timestamp}_{mode}.csv")
H10_HR_SESSION_CSV_TEMPLATE = os.path.join(LOG_DIR, "h10_hr_session_{subject_id}_{session_timestamp}_{mode}.csv")
H10_SDNN_SESSION_CSV_TEMPLATE = os.path.join(LOG_DIR, "h10_sdnn_session_{subject_id}_{session_timestamp}_{mode}.csv")
INTERACTION_EVENT_LOG_FILE_TEMPLATE = os.path.join(LOG_DIR, "interaction_events_{subject_id}_{session_timestamp}_{mode}.csv")

# --- API and Service URLs ---
VOICEVOX_URL = "http://localhost:50021"

# --- VOICEVOX Auto-Start Settings ---
# True: main.py起動時にVOICEVOXも自動起動
VOICEVOX_AUTO_START = True

# macOS: VOICEVOXアプリケーションのパス
# 通常は /Applications/VOICEVOX.app
VOICEVOX_MAC_APP_PATH = "/Applications/VOICEVOX.app"

# Ubuntu: Dockerイメージとコンテナ設定
VOICEVOX_DOCKER_IMAGE = "voicevox/voicevox_engine:cpu-ubuntu20.04-latest"
VOICEVOX_DOCKER_CONTAINER_NAME = "voicevox_engine"
# GPU版を使用する場合: "voicevox/voicevox_engine:nvidia-ubuntu20.04-latest"
# VOICEVOX_DOCKER_IMAGE = "voicevox/voicevox_engine:nvidia-ubuntu20.04-latest"

# --- LLM API Settings ---
# ローカルLLM使用時は USE_LOCAL_LLM = True に設定
# デフォルトは True (ローカルLLM を使用)
USE_LOCAL_LLM = True

# ローカルLLM設定 (llama.cpp, Ollama, LM Studio など)
LOCAL_LLM_BASE_URL = "http://localhost:8080/v1"  # llama.cpp server
LOCAL_LLM_MODEL = "local-model"  # ローカルモデル名（サーバーによっては無視される）
LOCAL_LLM_API_KEY = "not-needed"  # ローカルLLMでは通常不要

# ローカルLLMサーバー自動起動設定
LOCAL_LLM_AUTO_START = True  # True: main.py起動時にLLMサーバーも自動起動
# モデルファイルのパス (GGUFフォーマット)
# 例: ./models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf
LOCAL_LLM_MODEL_PATH = "./models/model.gguf"  # 使用するモデルファイル
LOCAL_LLM_HOST = "127.0.0.1"
LOCAL_LLM_PORT = 8080
LOCAL_LLM_CONTEXT_SIZE = 4096  # コンテキストサイズ
# GPUレイヤー数: 自動検出された値を使用
# -1: 全レイヤーをGPUに載せる (Metal/CUDA検出時)
#  0: CPUのみ (GPU未検出時)
LOCAL_LLM_GPU_LAYERS = _DEFAULT_GPU_LAYERS

# OpenAI API設定 (USE_LOCAL_LLM = False の場合に使用)
OPENAI_MODEL = "gpt-4o-mini"

# 共通設定
LLM_MAX_TOKENS = 200
LLM_TEMPERATURE = 0.7
LLM_TIMEOUT = 30

# 後方互換性のため
OPENAI_MAX_TOKENS = LLM_MAX_TOKENS
OPENAI_TEMPERATURE = LLM_TEMPERATURE
OPENAI_TIMEOUT = LLM_TIMEOUT

# --- Polar Device Settings ---
POLAR_VERITY_SENSE_NAME = "Polar Sense"
POLAR_H10_NAME = "Polar H10"

# Characteristic UUIDs
HR_CHARACTERISTIC_UUID = '00002a37-0000-1000-8000-00805f9b34fb' # Heart Rate Measurement

# Polar H10 PMD (Polar Measurement Data) Service and Characteristics
PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8" # For commands
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"    # For data (ECG, ACC, etc.)

# Command to start ECG stream on Polar H10
ECG_WRITE = bytearray([
    0x02, # Request type: Start measurement
    0x00, # Measurement type: ECG
    0x00, # Setting type: Sample rate
    0x01, # Setting array length: 1
    0x82, # Sample rate: 130 Hz (0x8200 in little endian -> 130)
    0x00,
    0x01, # Setting type: Resolution
    0x01, # Setting array length: 1
    0x0E, # Resolution: 14-bit (0x0E00 -> 14)
    0x00
])

# --- Baseline Measurement Settings ---
DEFAULT_BASELINE_MEASUREMENT_DURATION = 30  # seconds
MIN_SAMPLES_FOR_MEDIAN = 5  # Minimum HR samples needed to calculate median
# v4.0 以降のGUIとの互換用（旧名→新名の橋渡し）
BASELINE_MEASUREMENT_DURATION = DEFAULT_BASELINE_MEASUREMENT_DURATION

# --- Logger Names ---
LOGGER_HR_PROSODY = "hr_prosody_verity"
LOGGER_HR_AFTER_TTS = "hr_after_tts"
LOGGER_HR_AT_RECORDING_START = "hr_at_recording_start"
LOGGER_VERITY_HR_SESSION = "verity_hr_session"
LOGGER_H10_ECG_SESSION = "h10_ecg_session"
LOGGER_H10_HR_SESSION = "h10_hr_session"
LOGGER_H10_SDNN_SESSION = "h10_sdnn_session"
LOGGER_CONVERSATION_CSV = "conversation_csv"
LOGGER_INTERACTION_EVENTS = "interaction_events"
