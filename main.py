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

    # If the current python is not the one from our venv, and the venv exists
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
from gui import Application # The main Tkinter application class

def main():
    """Main function to initialize and run the application."""
    print("アプリケーションの初期化を開始します...")

    # 1. Initialize Log Directory and Queue
    # LOG_DIR is accessed from the imported config module
    try:
        initialize_log_directory(config.LOG_DIR)
    except Exception as e_log_dir:
        messagebox.showerror("ログディレクトリ初期化エラー",
                             f"ログディレクトリの作成に失敗しました: {config.LOG_DIR}\nエラー: {e_log_dir}")
        print(f"ログディレクトリ '{config.LOG_DIR}' の作成または確認に失敗しました: {e_log_dir}")
        # Continue without directory if it's just a print, or sys.exit(1) if critical
        # For now, we'll let it print and continue, as some logs might still work in current dir.

    log_q = queue.Queue() # Central log queue

    # 2. Load faster-whisper Model
    faster_whisper_model_instance: WhisperModel
    print("faster-whisperモデルのロードを開始します...")
    try:
        # Determine device and compute type for faster-whisper
        model_name = config.WHISPER_MODEL_NAME # Read model name from config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Allow overriding device via environment variable (useful for testing)
        env_device_override = os.getenv("WHISPER_DEVICE")
        if env_device_override and env_device_override.lower() in ["cuda", "cpu"]:
            device = env_device_override.lower()
            print(f"  環境変数によりWhisperデバイスを '{device}' にオーバーライドしました。")

        # Read compute_type from config. This gives user full control.
        # For CPU, "int8" is often a good choice for performance.
        # For CUDA, "int8_float16" or "float16" are good choices.
        compute_type = config.WHISPER_COMPUTE_TYPE
        
        print(f"  モデル '{model_name}' をロード中 (デバイス: {device}, 計算タイプ: {compute_type})...")
        faster_whisper_model_instance = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("faster-whisperモデルのロード成功。")
    except Exception as e_whisper:
        messagebox.showerror("モデルロードエラー",
                             f"faster-whisperモデルのロードに失敗しました:\n{e_whisper}\n\n"
                             "環境を確認してください (例: 'pip install faster-whisper torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX')。")
        sys.exit(1)

    # 3. Check VOICEVOX Server
    print("VOICEVOXサーバーの接続を確認中...")
    if not VoicevoxManager.check_server():
        # This is a warning, app can still run without TTS
        messagebox.showwarning("VOICEVOX警告",
                               "VOICEVOXサーバーに接続できません。\n"
                               "音声合成機能は利用できません。")

    # 4. Set OpenAI API Key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        # This is a warning, app can run but AI responses will fail
        messagebox.showwarning("OpenAI APIキー警告",
                               "環境変数 'OPENAI_API_KEY' が設定されていません。\n"
                               "AIアシスタントの応答生成機能は利用できません。")

    # 5. Initialize Core Components
    app_instance = None # To allow app.on_close() in finally block
    try:
        print("コアコンポーネントを初期化中...")
        prosody_settings = ProsodySettings()

        # Get speakers for SpeakerSettings
        # This might try to connect to Voicevox, handle potential unavailability
        speakers_list_data = VoicevoxManager.get_speakers()
        if not speakers_list_data:
            print("警告: VOICEVOXから話者リストを取得できませんでした。または話者がいません。")
            # Provide a dummy speaker entry if list is empty to prevent errors in SpeakerSettings
            speakers_list_data = [{'name': '話者取得失敗/なし', 'id': 0}]
        speaker_settings = SpeakerSettings(speakers_list_data)

        hr_monitor = HeartRateMonitor(log_queue_ref=log_q)
        h10_monitor = H10Monitor(log_queue_ref=log_q)

        # Pass the loaded whisper model to AudioProcessor
        audio_processor = AudioProcessor(
            prosody_settings=prosody_settings,
            speaker_settings=speaker_settings,
            hr_monitor=hr_monitor,
            h10_monitor=h10_monitor,
            log_queue_ref=log_q,
            faster_whisper_model_instance=faster_whisper_model_instance
        )
        print("コアコンポーネントの初期化成功。")

        # 6. Start the Application UI
        print("アプリケーションUIを起動中...")
        # Pass the central log_q to Application, which will then pass it to LoggingThread
        app_instance = Application(
            prosody_settings=prosody_settings,
            speaker_settings=speaker_settings,
            audio_processor=audio_processor,
            hr_monitor=hr_monitor,
            h10_monitor=h10_monitor,
            log_queue_ref=log_q # Pass the queue here
        )
        app_instance.mainloop()

    except SystemExit: # Handle sys.exit() calls gracefully
        print("アプリケーションが sys.exit() により終了しました。")
    except tk.TclError as e_tk:
        # Handles cases like display not found, etc.
        print(f"Tkinterエラーが発生しました: {e_tk}")
        messagebox.showerror("UIエラー", f"GUIの初期化または実行中にエラーが発生しました:\n{e_tk}\n"
                                     "グラフィカル環境が利用可能か確認してください。")
    except Exception as e_main:
        # Catch-all for other unexpected errors during component init or app launch
        print(f"致命的なエラー: アプリケーション実行中に予期せぬエラーが発生しました: {e_main}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("実行時エラー", f"アプリケーションの実行中に致命的なエラーが発生しました:\n{e_main}")
        if app_instance and hasattr(app_instance, '_closing') and not app_instance._closing:
            try:
                app_instance.on_close() # Attempt graceful shutdown of the app
            except Exception as e_on_close_fatal:
                print(f"on_close中の追加エラー (無視): {e_on_close_fatal}")
        sys.exit(1) # Ensure exit after fatal error

    print("メインプロセス終了。")

if __name__ == "__main__":
    # Setup for potential multiprocessing issues on Windows/macOS if any part uses it
    # (though this app seems primarily thread-based)
    if sys.platform.startswith('win'):
        # import multiprocessing
        # multiprocessing.freeze_support() # If multiprocessing were used
        pass # No specific freeze_support needed for this threading model

    main()