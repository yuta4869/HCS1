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
import subprocess
import time
import tkinter as tk # For messagebox, though it's often part of Application
from tkinter import messagebox
import torch # For checking CUDA availability for faster-whisper
from faster_whisper import WhisperModel
import openai
import requests
import atexit
import signal

# Import modules from the project
import config

# グローバル変数: LLMサーバープロセス
_llm_server_process = None

def print_gpu_status():
    """GPU使用状況を表示する"""
    print("")
    print("=" * 50)
    print("          GPU / アクセラレータ状況")
    print("=" * 50)

    # GPU Backend
    gpu_backend = config.GPU_BACKEND
    if gpu_backend == "cuda":
        try:
            device_name = torch.cuda.get_device_name(0)
            print(f"  GPU Backend:    CUDA ({device_name})")
        except:
            print(f"  GPU Backend:    CUDA")
    elif gpu_backend == "mps":
        print(f"  GPU Backend:    MPS (Apple Silicon)")
    else:
        print(f"  GPU Backend:    CPU (GPUなし)")

    # LLM GPU Layers
    gpu_layers = config.LOCAL_LLM_GPU_LAYERS
    if gpu_layers == -1:
        print(f"  LLM GPU Layers: -1 (全レイヤーをGPUに)")
    elif gpu_layers == 0:
        print(f"  LLM GPU Layers: 0 (CPUのみ)")
    else:
        print(f"  LLM GPU Layers: {gpu_layers}")

    # Whisper
    whisper_info = f"{config.WHISPER_MODEL_NAME} / {config.WHISPER_COMPUTE_TYPE}"
    if config.WHISPER_COMPUTE_TYPE == "float16":
        print(f"  Whisper:        {whisper_info} (GPU)")
    else:
        print(f"  Whisper:        {whisper_info} (CPU)")

    # LLM Mode
    if config.USE_LOCAL_LLM:
        print(f"  LLM Mode:       ローカルLLM (llama.cpp)")
    else:
        print(f"  LLM Mode:       OpenAI API ({config.OPENAI_MODEL})")

    print("=" * 50)
    print("")

def start_local_llm_server():
    """ローカルLLMサーバー(llama-cpp-python)を起動する"""
    global _llm_server_process

    if not config.USE_LOCAL_LLM or not config.LOCAL_LLM_AUTO_START:
        return True

    # サーバーが既に起動しているかチェック (/v1/models エンドポイントを使用)
    try:
        response = requests.get(f"http://{config.LOCAL_LLM_HOST}:{config.LOCAL_LLM_PORT}/v1/models", timeout=2)
        if response.status_code == 200:
            print("ローカルLLMサーバーは既に起動しています。")
            return True
    except requests.exceptions.RequestException:
        pass  # サーバーが起動していない場合

    # モデルファイルの存在確認
    if not os.path.exists(config.LOCAL_LLM_MODEL_PATH):
        print(f"エラー: LLMモデルファイルが見つかりません: {config.LOCAL_LLM_MODEL_PATH}")
        return False

    # llama-cpp-pythonのサーバーを起動
    # python -m llama_cpp.server --model <path> --host <host> --port <port>
    cmd = [
        sys.executable,  # 現在のPython実行ファイル
        "-m", "llama_cpp.server",
        "--model", config.LOCAL_LLM_MODEL_PATH,
        "--host", config.LOCAL_LLM_HOST,
        "--port", str(config.LOCAL_LLM_PORT),
        "--n_ctx", str(config.LOCAL_LLM_CONTEXT_SIZE),
        "--n_gpu_layers", str(config.LOCAL_LLM_GPU_LAYERS),
    ]

    print(f"ローカルLLMサーバーを起動中...")
    print(f"  モデル: {config.LOCAL_LLM_MODEL_PATH}")
    print(f"  ホスト: {config.LOCAL_LLM_HOST}:{config.LOCAL_LLM_PORT}")

    try:
        # サブプロセスとしてサーバーを起動（バックグラウンド）
        # 注: stdout/stderrをDEVNULLにしないとバッファリングでサーバーがブロックする
        _llm_server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if sys.platform != 'win32' else None
        )

        # サーバーが起動するまで待機 (/v1/models エンドポイントを使用)
        print("  サーバーの起動を待機中...")
        max_wait = 30  # 通常は数秒で起動
        for i in range(max_wait):
            try:
                response = requests.get(f"http://{config.LOCAL_LLM_HOST}:{config.LOCAL_LLM_PORT}/v1/models", timeout=2)
                if response.status_code == 200:
                    print(f"ローカルLLMサーバーが起動しました (http://{config.LOCAL_LLM_HOST}:{config.LOCAL_LLM_PORT})")
                    return True
            except requests.exceptions.RequestException:
                pass

            # プロセスが終了していないかチェック
            if _llm_server_process.poll() is not None:
                print(f"エラー: LLMサーバーが予期せず終了しました (exit code: {_llm_server_process.returncode})")
                return False

            time.sleep(1)
            if (i + 1) % 10 == 0:
                print(f"  まだ待機中... ({i + 1}秒)")

        print("エラー: LLMサーバーの起動がタイムアウトしました")
        stop_local_llm_server()
        return False

    except Exception as e:
        print(f"エラー: LLMサーバーの起動に失敗しました: {e}")
        return False

def stop_local_llm_server():
    """ローカルLLMサーバーを停止する"""
    global _llm_server_process

    if _llm_server_process is not None:
        print("ローカルLLMサーバーを停止中...")
        try:
            if sys.platform != 'win32':
                # プロセスグループ全体を終了
                os.killpg(os.getpgid(_llm_server_process.pid), signal.SIGTERM)
            else:
                _llm_server_process.terminate()

            _llm_server_process.wait(timeout=5)
            print("ローカルLLMサーバーを停止しました。")
        except Exception as e:
            print(f"LLMサーバー停止中にエラー: {e}")
            try:
                _llm_server_process.kill()
            except:
                pass
        finally:
            _llm_server_process = None

# アプリケーション終了時にLLMサーバーも停止
atexit.register(stop_local_llm_server)

# グローバル変数: VOICEVOXプロセス/コンテナ
_voicevox_process = None
_voicevox_docker_started = False

def start_voicevox():
    """VOICEVOXを自動起動する（Mac: .app, Ubuntu: Docker）"""
    global _voicevox_process, _voicevox_docker_started

    if not config.VOICEVOX_AUTO_START:
        return True

    # 既にサーバーが起動しているかチェック
    try:
        response = requests.get(f"{config.VOICEVOX_URL}/version", timeout=2)
        if response.status_code == 200:
            print("VOICEVOXサーバーは既に起動しています。")
            return True
    except requests.exceptions.RequestException:
        pass  # サーバーが起動していない場合

    # macOSの場合: .appを起動
    if sys.platform == 'darwin':
        return _start_voicevox_mac()
    # Linuxの場合: Dockerコンテナを起動
    elif sys.platform.startswith('linux'):
        return _start_voicevox_docker()
    else:
        print(f"警告: {sys.platform} でのVOICEVOX自動起動は未対応です。")
        return False

def _start_voicevox_mac():
    """macOS: VOICEVOX.appを起動"""
    global _voicevox_process

    app_path = config.VOICEVOX_MAC_APP_PATH

    if not os.path.exists(app_path):
        print(f"警告: VOICEVOXアプリが見つかりません: {app_path}")
        print("  手動でVOICEVOXを起動してください。")
        return False

    print(f"VOICEVOXを起動中... ({app_path})")

    try:
        # open コマンドでアプリを起動（バックグラウンド）
        _voicevox_process = subprocess.Popen(
            ['open', '-a', app_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # サーバーが起動するまで待機
        print("  VOICEVOXサーバーの起動を待機中...")
        max_wait = 30  # 最大30秒待機
        for i in range(max_wait):
            try:
                response = requests.get(f"{config.VOICEVOX_URL}/version", timeout=2)
                if response.status_code == 200:
                    version = response.json() if response.text else "unknown"
                    print(f"VOICEVOXサーバーが起動しました (version: {version})")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(1)
            if (i + 1) % 10 == 0:
                print(f"  まだ待機中... ({i + 1}秒)")

        print("警告: VOICEVOXサーバーの起動がタイムアウトしました。")
        return False

    except Exception as e:
        print(f"エラー: VOICEVOXの起動に失敗しました: {e}")
        return False

def _start_voicevox_docker():
    """Ubuntu: Docker コンテナを起動"""
    global _voicevox_docker_started

    # Dockerがインストールされているか確認
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("警告: Dockerがインストールされていません。")
            return False
    except FileNotFoundError:
        print("警告: Dockerがインストールされていません。")
        return False

    container_name = config.VOICEVOX_DOCKER_CONTAINER_NAME
    image_name = config.VOICEVOX_DOCKER_IMAGE

    # 既存のコンテナがあるか確認
    result = subprocess.run(
        ['docker', 'ps', '-a', '--filter', f'name={container_name}', '--format', '{{.Names}}'],
        capture_output=True, text=True
    )

    if container_name in result.stdout:
        # コンテナが存在する場合、起動
        print(f"既存のVOICEVOXコンテナを起動中... ({container_name})")
        subprocess.run(['docker', 'start', container_name], capture_output=True)
    else:
        # 新規コンテナを作成・起動
        print(f"VOICEVOXコンテナを作成・起動中... ({image_name})")

        # GPU使用可否を確認
        gpu_available = False
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True)
            gpu_available = result.returncode == 0
        except FileNotFoundError:
            pass

        cmd = [
            'docker', 'run', '-d',
            '--name', container_name,
            '-p', '50021:50021',
        ]

        # GPU使用可能で、nvidia imageの場合はGPUを有効化
        if gpu_available and 'nvidia' in image_name:
            cmd.extend(['--gpus', 'all'])

        cmd.append(image_name)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"警告: VOICEVOXコンテナの起動に失敗しました: {result.stderr}")
            return False

    _voicevox_docker_started = True

    # サーバーが起動するまで待機
    print("  VOICEVOXサーバーの起動を待機中...")
    max_wait = 60  # Dockerは起動に時間がかかる
    for i in range(max_wait):
        try:
            response = requests.get(f"{config.VOICEVOX_URL}/version", timeout=2)
            if response.status_code == 200:
                version = response.json() if response.text else "unknown"
                print(f"VOICEVOXサーバーが起動しました (version: {version})")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"  まだ待機中... ({i + 1}秒)")

    print("警告: VOICEVOXサーバーの起動がタイムアウトしました。")
    return False

def stop_voicevox():
    """VOICEVOXを停止する"""
    global _voicevox_docker_started

    # Dockerコンテナの場合のみ停止（Macの.appは手動管理）
    if _voicevox_docker_started and sys.platform.startswith('linux'):
        container_name = config.VOICEVOX_DOCKER_CONTAINER_NAME
        print(f"VOICEVOXコンテナを停止中... ({container_name})")
        try:
            subprocess.run(['docker', 'stop', container_name], capture_output=True, timeout=10)
            print("VOICEVOXコンテナを停止しました。")
        except Exception as e:
            print(f"警告: VOICEVOXコンテナの停止に失敗しました: {e}")
        _voicevox_docker_started = False

# アプリケーション終了時にVOICEVOXも停止（Dockerの場合のみ）
atexit.register(stop_voicevox)

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

    # 0. Display GPU Status
    print_gpu_status()

    # 1. Initialize Log Directory
    try:
        initialize_log_directory(config.LOG_DIR)
    except Exception as e_log_dir:
        messagebox.showerror("ログディレクトリ初期化エラー",
                             f"ログディレクトリの作成に失敗しました: {config.LOG_DIR}\nエラー: {e_log_dir}", parent=root)
        sys.exit(1)

    # 2. Start Local LLM Server or Check OpenAI API Key
    if config.USE_LOCAL_LLM:
        print("ローカルLLMモードが有効です。")
        if config.LOCAL_LLM_AUTO_START:
            if not start_local_llm_server():
                messagebox.showerror("LLMサーバーエラー",
                                   "ローカルLLMサーバーの起動に失敗しました。\n"
                                   f"モデルパス: {config.LOCAL_LLM_MODEL_PATH}", parent=root)
                sys.exit(1)
    else:
        # OpenAI APIキーのチェック
        if not os.getenv("OPENAI_API_KEY"):
            warning_msg = ("環境変数 'OPENAI_API_KEY' が設定されていません。\n"
                           "AI応答生成機能は利用できません。")
            print(f"WARNING: {warning_msg.replace(chr(10), ' ')}")
            messagebox.showwarning("OpenAI APIキー警告", warning_msg, parent=root)

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

    # 4. Start/Check VOICEVOX Server
    print("VOICEVOXサーバーを確認/起動中...")
    if not start_voicevox():
        # 自動起動に失敗した場合、手動確認
        if not VoicevoxManager.check_server():
            messagebox.showwarning("VOICEVOX警告",
                                   "VOICEVOXサーバーに接続できません。\n"
                                   "音声合成機能は利用できません。\n\n"
                                   "Mac: VOICEVOXアプリを手動で起動してください。\n"
                                   "Ubuntu: docker start voicevox_engine", parent=root)

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
