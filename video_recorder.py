# video_recorder.py
"""
動画録画モジュール（音声付き）
会話セッション中のビデオ＋音声録画を管理する
"""

import cv2
import threading
import queue
import os
import time
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
from typing import Optional, List

import config
from audio_device_utils import get_video_mic_device
from logger_utils import format_subject_id_for_filename


class VideoRecorder:
    """
    会話セッション中のビデオ＋音声録画を管理するクラス

    使用例:
        recorder = VideoRecorder()
        if recorder.start_recording(session_timestamp="20231215_143022"):
            # 会話処理...
            recorder.stop_recording()
    """

    # 録画ファイルの保存先テンプレート
    # モード名: Sin (正弦波), HRF (心拍フィードバック), Fixed (抑揚固定)
    VIDEO_FILE_TEMPLATE = os.path.join(config.LOG_DIR, "video_session_{subject_id}_{session_timestamp}_{mode}.mp4")
    TEMP_VIDEO_TEMPLATE = os.path.join(config.LOG_DIR, "temp_video_{subject_id}_{session_timestamp}_{mode}.mp4")
    TEMP_AUDIO_TEMPLATE = os.path.join(config.LOG_DIR, "temp_audio_{subject_id}_{session_timestamp}_{mode}.wav")

    def __init__(self,
                 camera_index: int = 0,
                 fps: float = 30.0,
                 resolution: Optional[tuple] = None,
                 codec: str = "mp4v",
                 audio_sample_rate: int = 48000,
                 audio_channels: int = 1,
                 record_audio: bool = True,
                 audio_device_index: Optional[int] = None,
                 auto_detect_audio_device: bool = True):
        """
        VideoRecorderを初期化する

        Args:
            camera_index: 使用するカメラのインデックス (0=デフォルト/内蔵, 1,2...=外部USB等)
            fps: 録画のフレームレート
            resolution: 解像度 (width, height) のタプル。Noneの場合はカメラのデフォルト
            codec: 動画コーデック (デフォルト: mp4v)
            audio_sample_rate: 音声サンプリングレート (Hz、デフォルト: 48000)
            audio_channels: 音声チャンネル数 (1=モノラル、デフォルト: 1)
            record_audio: 音声も録音するかどうか (デフォルト: True)
            audio_device_index: 音声録音に使用するデバイスのインデックス
                               (None=自動検出、または明示的に指定)
            auto_detect_audio_device: True=config.pyのキーワードで自動検出、False=audio_device_indexを直接使用
        """
        self.camera_index = camera_index
        self.fps = fps
        self.resolution = resolution
        self.codec = codec
        self.audio_sample_rate = audio_sample_rate
        self.audio_channels = audio_channels
        self.record_audio = record_audio

        # 音声デバイスの決定
        if auto_detect_audio_device and audio_device_index is None:
            # config.pyのキーワードに基づいて自動検出
            self.audio_device_index = get_video_mic_device()
        else:
            self.audio_device_index = audio_device_index

        self._capture: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording_thread: Optional[threading.Thread] = None
        self._audio_recording_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_recording = False
        self._current_filepath: Optional[str] = None
        self._temp_video_path: Optional[str] = None
        self._temp_audio_path: Optional[str] = None
        self._frame_count = 0
        self._start_time: Optional[datetime] = None
        self._audio_data = []

    @property
    def is_recording(self) -> bool:
        """録画中かどうかを返す"""
        return self._is_recording

    @property
    def current_filepath(self) -> Optional[str]:
        """現在の録画ファイルパスを返す"""
        return self._current_filepath

    @property
    def frame_count(self) -> int:
        """録画したフレーム数を返す"""
        return self._frame_count

    def _detect_available_camera(self) -> int:
        """
        利用可能なカメラを検出する

        Returns:
            利用可能なカメラのインデックス (-1 = カメラなし)
        """
        # まず指定されたインデックスを試す
        cap = cv2.VideoCapture(self.camera_index)
        if cap.isOpened():
            cap.release()
            print(f"[VideoRecorder] カメラ {self.camera_index} を使用します")
            return self.camera_index

        # 指定インデックスが使えない場合、0から順に検索
        for i in range(5):  # 最大5台まで検索
            if i == self.camera_index:
                continue
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                print(f"[VideoRecorder] カメラ {i} を代替として使用します")
                return i

        return -1

    def _get_output_filepath(self, session_timestamp: str, mode: str, subject_id: Optional[str] = None) -> str:
        """
        録画ファイルの出力パスを生成する

        Args:
            session_timestamp: セッションタイムスタンプ
            mode: モード名 (Sin/HRF/Fixed)

        Returns:
            出力ファイルパス
        """
        subject_component = format_subject_id_for_filename(subject_id)
        return self.VIDEO_FILE_TEMPLATE.format(
            session_timestamp=session_timestamp,
            mode=mode,
            subject_id=subject_component
        )

    def _audio_callback(self, indata, frames, time_info, status):
        """音声録音コールバック"""
        if status:
            print(f"[VideoRecorder] 音声録音警告: {status}")
        self._audio_data.append(indata.copy())

    def _recording_loop(self) -> None:
        """映像録画スレッドのメインループ"""
        print(f"[VideoRecorder] 映像録画ループ開始")

        while not self._stop_event.is_set():
            if self._capture is None or not self._capture.isOpened():
                print("[VideoRecorder] エラー: カメラが開かれていません")
                break

            ret, frame = self._capture.read()
            if not ret:
                print("[VideoRecorder] 警告: フレーム取得に失敗しました")
                time.sleep(0.01)
                continue

            if self._writer is not None:
                self._writer.write(frame)
                self._frame_count += 1

            # CPU負荷軽減のための小休止
            time.sleep(1.0 / self.fps * 0.5)

        print(f"[VideoRecorder] 映像録画ループ終了 (総フレーム数: {self._frame_count})")

    def _merge_audio_video(self, video_path: str, audio_path: Optional[str], output_path: str) -> bool:
        """
        ffmpegを使用して映像と音声を結合/再エンコードする
        （H.264 + AACに変換して一般的なプレイヤーで確実に再生可能にする）

        Args:
            video_path: 映像ファイルパス
            audio_path: 音声ファイルパス (Noneの場合は映像のみ)
            output_path: 出力ファイルパス

        Returns:
            True: 成功, False: 失敗
        """
        try:
            print(f"[VideoRecorder] FFmpegで最終ファイルを生成中... (audio={'あり' if audio_path else 'なし'})")

            cmd = [
                'ffmpeg',
                '-loglevel', 'error',
                '-y',                 # 既存ファイルは問答無用で上書き
                '-i', video_path,
            ]

            if audio_path:
                cmd += ['-i', audio_path]

            # libx264で再エンコードし、互換性の高いyuv420pに揃える
            cmd += [
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
            ]

            if audio_path:
                cmd += ['-c:a', 'aac', '-b:a', '128k']
            else:
                cmd += ['-an']  # 音声トラックなし

            cmd += [
                '-movflags', '+faststart',  # moovを冒頭に配置して再生互換性を向上
                output_path
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120  # 再エンコードが入るので余裕を持たせる
            )

            if result.returncode == 0:
                print(f"[VideoRecorder] FFmpeg処理成功: {output_path}")
                return True
            else:
                print(f"[VideoRecorder] FFmpegエラー: {result.stderr.decode()}")
                return False

        except subprocess.TimeoutExpired:
            print(f"[VideoRecorder] FFmpeg処理がタイムアウトしました")
            return False
        except Exception as e:
            print(f"[VideoRecorder] FFmpeg実行中に例外発生: {e}")
            return False

    def start_recording(self, session_timestamp: str, mode: str = "Fixed", subject_id: Optional[str] = None) -> bool:
        """
        録画を開始する（映像＋音声）

        Args:
            session_timestamp: セッションタイムスタンプ (ファイル名に使用)
            mode: モード名 (Sin/HRF/Fixed)

        Returns:
            True: 録画開始成功, False: 録画開始失敗
        """
        if self._is_recording:
            print("[VideoRecorder] 警告: 既に録画中です")
            return False

        # logsディレクトリの確認/作成
        os.makedirs(config.LOG_DIR, exist_ok=True)

        # 利用可能なカメラを検出
        available_camera = self._detect_available_camera()
        if available_camera < 0:
            print("[VideoRecorder] エラー: 利用可能なカメラが見つかりません")
            return False

        # カメラを開く
        self._capture = cv2.VideoCapture(available_camera)
        if not self._capture.isOpened():
            print(f"[VideoRecorder] エラー: カメラ {available_camera} を開けませんでした")
            return False

        # 解像度設定（指定がある場合）
        if self.resolution:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # 実際の解像度を取得
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[VideoRecorder] カメラ解像度: {actual_width}x{actual_height}")

        # 出力ファイルパスを設定
        subject_component = format_subject_id_for_filename(subject_id)
        self._current_filepath = self._get_output_filepath(session_timestamp, mode, subject_id)
        self._temp_video_path = self.TEMP_VIDEO_TEMPLATE.format(
            session_timestamp=session_timestamp,
            mode=mode,
            subject_id=subject_component
        )
        self._temp_audio_path = self.TEMP_AUDIO_TEMPLATE.format(
            session_timestamp=session_timestamp,
            mode=mode,
            subject_id=subject_component
        )

        # VideoWriterを初期化（一時ファイルに保存）
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            self._temp_video_path,
            fourcc,
            self.fps,
            (actual_width, actual_height)
        )

        if not self._writer.isOpened():
            print(f"[VideoRecorder] エラー: VideoWriterを初期化できませんでした")
            self._capture.release()
            self._capture = None
            return False

        # 音声録音の準備
        self._audio_data = []

        # 録画開始
        self._stop_event.clear()
        self._frame_count = 0
        self._start_time = datetime.now()
        self._is_recording = True

        # 映像録画スレッド開始
        self._recording_thread = threading.Thread(
            target=self._recording_loop,
            daemon=True,
            name="VideoRecorderThread"
        )
        self._recording_thread.start()

        # 音声録音開始（有効な場合のみ）
        if self.record_audio:
            try:
                # デバイスインデックスが指定されている場合はそのデバイスを使用
                device_info = ""
                if self.audio_device_index is not None:
                    try:
                        dev = sd.query_devices(self.audio_device_index)
                        device_info = f", デバイス: {dev['name']}"
                    except Exception:
                        device_info = f", デバイスID: {self.audio_device_index}"

                self._audio_stream = sd.InputStream(
                    samplerate=self.audio_sample_rate,
                    channels=self.audio_channels,
                    device=self.audio_device_index,  # None=デフォルト、または指定デバイス
                    callback=self._audio_callback
                )
                self._audio_stream.start()
                print(f"[VideoRecorder] 音声録音開始 ({self.audio_sample_rate}Hz, {self.audio_channels}ch{device_info})")
            except Exception as e:
                print(f"[VideoRecorder] 警告: 音声録音の開始に失敗しました: {e}")
                print(f"[VideoRecorder] 映像のみで録画を続行します")
                self.record_audio = False  # 音声録音を無効化
        else:
            print(f"[VideoRecorder] 映像のみ録画モード")

        print(f"[VideoRecorder] 録画開始: {self._current_filepath}")
        return True

    def stop_recording(self) -> Optional[str]:
        """
        録画を停止する（映像＋音声を結合して保存）

        Returns:
            録画したファイルのパス (録画していなかった場合はNone)
        """
        if not self._is_recording:
            print("[VideoRecorder] 録画は実行されていません")
            return None

        print("[VideoRecorder] 録画停止リクエスト受信...")

        # 停止シグナルを送信
        self._stop_event.set()

        # 音声録音を停止
        audio_recorded = False
        try:
            if hasattr(self, '_audio_stream') and self._audio_stream:
                self._audio_stream.stop()
                self._audio_stream.close()

                # 音声データを保存
                if len(self._audio_data) > 0:
                    audio_array = np.concatenate(self._audio_data, axis=0)
                    sf.write(self._temp_audio_path, audio_array, self.audio_sample_rate)
                    audio_recorded = True
                    print(f"[VideoRecorder] 音声データ保存完了: {len(audio_array)} サンプル")
        except Exception as e:
            print(f"[VideoRecorder] 音声データ保存エラー: {e}")

        # 映像録画スレッド終了を待機
        if self._recording_thread and self._recording_thread.is_alive():
            print("[VideoRecorder] 映像録画スレッドの終了を待機中...")
            self._recording_thread.join(timeout=5.0)
            if self._recording_thread.is_alive():
                print("[VideoRecorder] 警告: 映像録画スレッドが時間内に終了しませんでした")

        # リソースを解放
        if self._writer:
            self._writer.release()
            self._writer = None

        if self._capture:
            self._capture.release()
            self._capture = None

        self._is_recording = False
        self._recording_thread = None

        duration = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0

        # 映像/音声ファイルをH.264ベースで最終化
        if os.path.exists(self._temp_video_path):
            audio_path = None
            if audio_recorded and self._temp_audio_path and os.path.exists(self._temp_audio_path):
                audio_path = self._temp_audio_path
            elif audio_recorded:
                print("[VideoRecorder] 警告: 音声有効のはずですが一時音声ファイルが見つかりません。映像のみで保存します。")

            if self._merge_audio_video(self._temp_video_path, audio_path, self._current_filepath):
                try:
                    os.remove(self._temp_video_path)
                    if audio_path:
                        os.remove(audio_path)
                    print(f"[VideoRecorder] 一時ファイルを削除しました")
                except Exception as e:
                    print(f"[VideoRecorder] 一時ファイル削除エラー: {e}")
            else:
                print(f"[VideoRecorder] 警告: FFmpeg処理に失敗したため、一時映像をそのまま保存します (互換性低)")
                try:
                    os.rename(self._temp_video_path, self._current_filepath)
                    if audio_path and os.path.exists(audio_path):
                        os.remove(audio_path)
                except Exception as e:
                    print(f"[VideoRecorder] ファイル保存エラー: {e}")

        filepath = self._current_filepath

        print(f"[VideoRecorder] 録画停止完了")
        print(f"[VideoRecorder] 保存先: {filepath}")
        print(f"[VideoRecorder] 録画時間: {duration:.1f}秒, フレーム数: {self._frame_count}")

        self._current_filepath = None
        self._temp_video_path = None
        self._temp_audio_path = None
        self._start_time = None
        self._audio_data = []

        return filepath

    def get_recording_duration(self) -> float:
        """
        現在の録画時間を秒数で返す

        Returns:
            録画時間（秒）、録画中でない場合は0
        """
        if self._start_time and self._is_recording:
            return (datetime.now() - self._start_time).total_seconds()
        return 0.0

    @staticmethod
    def list_available_cameras(max_cameras: int = 5) -> list:
        """
        利用可能なカメラのリストを返す（デバッグ用）

        Args:
            max_cameras: 検索する最大カメラ数

        Returns:
            利用可能なカメラインデックスのリスト
        """
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append({
                    'index': i,
                    'resolution': f"{width}x{height}"
                })
                cap.release()
        return available


# モジュールテスト用
if __name__ == "__main__":
    print("=== VideoRecorder テスト（音声付き） ===")

    # 利用可能なカメラを表示
    cameras = VideoRecorder.list_available_cameras()
    if cameras:
        print(f"利用可能なカメラ: {cameras}")
    else:
        print("利用可能なカメラが見つかりませんでした")
        exit(1)

    # 5秒間のテスト録画
    recorder = VideoRecorder()
    test_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if recorder.start_recording(test_timestamp):
        print("5秒間録画中（音声＋映像）...")
        time.sleep(5)
        filepath = recorder.stop_recording()
        print(f"テスト録画完了: {filepath}")
    else:
        print("録画開始に失敗しました")
