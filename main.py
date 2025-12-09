# main.py
import os
import sys

# --- Bootstrap to ensure the correct virtual environment is used ---
# This is a workaround to help users who are not activating the venv.
# It makes the script "magically" work even if called with the wrong python.
try:
    # The absolute path to the project's root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    # The expected path to the virtual environment's python executable
    venv_python_path = os.path.join(project_root, ".venv", "bin", "python")

    # The path of the python executable that is currently running this script
    current_python_path = sys.executable

    # If the current python is not the one from 
    # our venv, and the venv exists
    if current_python_path != venv_python_path and os.path.exists(venv_python_path):
        print("---")
        print("WARNING: Incorrect Python interpreter detected.")
        print(f"  Running with: {current_python_path}")
        print(f"  Expected:     {venv_python_path}")
        print("Attempting to re-launch with the correct interpreter from the virtual environment...")
        print("---")
        # Replace the current process with the correct one
        os.execv(venv_python_path, [venv_python_path, __file__] + sys.argv[1:])
except Exception as e:
    # If anything goes wrong, print a warning and continue,
    # allowing the original error (e.g., ModuleNotFoundError) to occur naturally.
    print(f"WARNING: An error occurred in the bootstrap section: {e}")
# --- End of Bootstrap ---

import queue
import tkinter as tk # For messagebox, though it's often part of Application
from tkinter import messagebox
import torch # For checking CUDA availability for faster-whisper
from faster_whisper import WhisperModel
import openai

# Import modules from the project
import config
from logger_utils import initialize_log_directory # log_queue is created here
# LoggingThread is initialized within Application
from polar_monitor import HeartRateMonitor, H10Monitor
from conversation_manager import ConversationManager
from audio_processing import ProsodySettings, VoicevoxManager, SpeakerSettings, AudioProcessor
from audio_device_utils import get_conversation_mic_device, print_device_info
from gui import Application # The main Tkinter application class

def main():
    """Main function to initialize and run the application."""
    # Create a hidden root window to handle messageboxes before the main app UI is built
    root = tk.Tk()
    root.withdraw()

    print("アプリケーションの初期化を開始します...")

    # 1. Initialize Log Directory
    try:
        initialize_log_directory(config.LOG_DIR)
    except Exception as e_log_dir:
        messagebox.showerror("ログディレクトリ初期化エラー",
                             f"ログディレクトリの作成に失敗しました: {config.LOG_DIR}\nエラー: {e_log_dir}", parent=root)
        sys.exit(1)

    # 2. Check for OpenAI API Key (only when using OpenAI backend)
    if config.LLM_BACKEND == "openai" and not os.getenv("OPENAI_API_KEY"):
        warning_msg = ("環境変数 'OPENAI_API_KEY' が設定されていません。\n"
                       "AI応答生成機能は利用できません。")
        print(f"WARNING: {warning_msg.replace(chr(10), ' ')}")
        messagebox.showwarning("OpenAI APIキー警告", warning_msg, parent=root)
    elif config.LLM_BACKEND == "local":
        print(f"LLMバックエンド: ローカルLLM (モデル: {config.LOCAL_LLM_MODEL_PATH})")

    # 3. Load faster-whisper Model
    faster_whisper_model_instance: WhisperModel
    print("faster-whisperモデルのロードを開始します...")
    try:
        # Determine device and compute type for faster-whisper
        model_name = config.WHISPER_MODEL_NAME
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        env_device_override = os.getenv("WHISPER_DEVICE")
        if env_device_override and env_device_override.lower() in ["cuda", "cpu"]:
            device = env_device_override.lower()
            print(f"  環境変数によりWhisperデバイスを '{device}' にオーバーライドしました。")

        compute_type = config.WHISPER_COMPUTE_TYPE
        # CPUでint8_float16は使えないのでint8に変更
        if device == "cpu" and compute_type == "int8_float16":
            print(f"  警告: CPUでは 'int8_float16' を使用できません。'int8' に変更します。")
            compute_type = "int8"
        
        print(f"  モデル '{model_name}' をロード中 (デバイス: {device}, 計算タイプ: {compute_type})...")
        faster_whisper_model_instance = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("faster-whisperモデルのロード成功。")
    except Exception as e_whisper:
        messagebox.showerror("モデルロードエラー",
                             f"faster-whisperモデルのロードに失敗しました:\n{e_whisper}", parent=root)
        sys.exit(1)

    # 4. Check VOICEVOX Server
    print("VOICEVOXサーバーの接続を確認中...")
    if not VoicevoxManager.check_server():
        messagebox.showwarning("VOICEVOX警告",
                               "VOICEVOXサーバーに接続できません。\n"
                               "音声合成機能は利用できません。", parent=root)

    # 5. Initialize Core Components
    app_instance = None
    try:
        print("コアコンポーネントを初期化中...")
        log_q = queue.Queue()
        prosody_settings = ProsodySettings()

        speakers_list_data = VoicevoxManager.get_speakers()
        if not speakers_list_data:
            print("警告: VOICEVOXから話者リストを取得できませんでした。または話者がいません。")
            speakers_list_data = [{'name': '話者取得失敗/なし', 'id': 0}]
        speaker_settings = SpeakerSettings(speakers_list_data)

        hr_monitor = HeartRateMonitor(log_queue_ref=log_q)
        h10_monitor = H10Monitor(log_queue_ref=log_q)

        # 会話システム用マイクの自動検出
        print("\n--- オーディオデバイス設定 ---")
        conversation_mic_device = get_conversation_mic_device()

        audio_processor = AudioProcessor(
            prosody_settings=prosody_settings,
            speaker_settings=speaker_settings,
            hr_monitor=hr_monitor,
            h10_monitor=h10_monitor,
            log_queue_ref=log_q,
            faster_whisper_model_instance=faster_whisper_model_instance,
            input_device_index=conversation_mic_device
        )
        print("コアコンポーネントの初期化成功。")

        # 6. Start the Application UI
        print("アプリケーションUIを起動中...")
        # The main app is now a Toplevel window, which will be managed by the hidden root
        app_instance = Application(
            master=root,
            prosody_settings=prosody_settings,
            speaker_settings=speaker_settings,
            audio_processor=audio_processor,
            hr_monitor=hr_monitor,
            h10_monitor=h10_monitor,
            log_queue_ref=log_q
        )
        
        # The mainloop is called on the root window
        root.mainloop()

    except SystemExit:
        print("アプリケーションが sys.exit() により終了しました。")
    except Exception as e_main:
        import traceback
        traceback.print_exc()
        messagebox.showerror("実行時エラー", f"アプリケーションの実行中に致命的なエラーが発生しました:\n{e_main}", parent=root)
        sys.exit(1)

    print("メインプロセス終了。")

if __name__ == "__main__":
    # Setup for potential multiprocessing issues on Windows/macOS if any part uses it
    # (though this app seems primarily thread-based)
    if sys.platform.startswith('win'):
        # import multiprocessing
        # multiprocessing.freeze_support() # If multiprocessing were used
        pass # No specific freeze_support needed for this threading model

    main()
