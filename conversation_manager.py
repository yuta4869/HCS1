# conversation_manager.py

import datetime
import logging
import time
import queue
from typing import Optional, List, Dict, Any
import os
import soundfile as sf
import numpy as np

import config
from logger_utils import get_timestamped_log_path, format_subject_id_for_filename


class ConversationManager:
    """
    会話の履歴管理と、会話内容の記録（本文と表形式）を担当する簡易管理クラス。
    """

    def __init__(
        self,
        audio_processor: Any,
        hr_monitor: Any,
        h10_monitor: Any,
        app_ref: Any,
        log_queue_ref: Optional[queue.Queue] = None,
        max_history: int = 10,
    ) -> None:
        self.audio_processor = audio_processor
        self.hr_monitor = hr_monitor
        self.h10_monitor = h10_monitor
        self.app = app_ref
        self.max_history: int = max_history
        self.conversation_history: List[Dict[str, Any]] = []
        self.log_queue: Optional[queue.Queue] = log_queue_ref
        self.log_filepath: Optional[str] = None
        self.csv_log_filepath: Optional[str] = None
        self.current_session_timestamp_for_csv: Optional[str] = None
        self.turn_counter: int = 0
        self._stop_conversation: bool = False
        self.subject_id: Optional[str] = None

    def set_subject_id(self, subject_id: Optional[str]) -> None:
        """被験者番号を更新し、ファイル名に反映できるようにする。"""
        self.subject_id = subject_id.strip() if subject_id else None

    def set_system_prompt(self, prompt: str) -> None:
        """system役の最初の文を履歴の先頭に入れる（既存のsystemは消す）。"""
        # 既存の system を取り除く
        self.conversation_history = [m for m in self.conversation_history if m.get("role") != "system"]
        # 先頭に追加
        self.conversation_history.insert(0, {"role": "system", "content": prompt})

    def update_system_prompt(self, new_prompt: str):
        """システムプロンプトを更新する。"""
        if not new_prompt:
            print("Warning: Attempted to set an empty system prompt.")
            return
        self.set_system_prompt(new_prompt) # Use existing method
        print(f"System prompt updated: {new_prompt[:100]}...")
        if self.log_filepath:
            try:
                with open(self.log_filepath, 'a', encoding='utf-8') as f:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    f.write(f"[{timestamp}] System Prompt Updated: {new_prompt}\n\n")
            except Exception as e:
                print(f"Failed to log system prompt update: {e}")

    def add_message(
        self,
        role: str,
        content: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        wav_filename: Optional[str] = None,
    ) -> None:
        """履歴に一件追加し、必要であればログにも書き出す。"""
        if not content:
            return

        msg: Dict[str, Any] = {"role": role, "content": content}
        if start_time:
            msg["start_time"] = start_time
        if end_time:
            msg["end_time"] = end_time
        if wav_filename:
            msg["wav_filename"] = wav_filename

        if role == "system":
            self.set_system_prompt(content)
        else:
            self.conversation_history.append(msg)
            non_system = [m for m in self.conversation_history if m.get("role") != "system"]
            if len(non_system) > self.max_history:
                # Keep only the last `max_history` user/assistant messages
                self.conversation_history = [m for m in self.conversation_history if m.get("role") == "system"] + non_system[-self.max_history:]

        self._log_message(role, content, start_time, end_time, wav_filename)

    def get_messages(self) -> List[Dict[str, str]]:
        """大規模言語モデルに渡す形（role/contentだけ）で履歴を返す。"""
        # Ensure a system prompt exists before returning messages for LLM
        if not any(m["role"] == "system" for m in self.conversation_history):
            print("Warning: No system prompt in conversation history. Adding a default one.")
            self.set_system_prompt("You are a helpful assistant.") # Add a default if missing
        return [{"role": m["role"], "content": m["content"]} for m in self.conversation_history]
    
    def clear_history(self) -> None:
        """Clears user/assistant messages, retaining system prompts."""
        system_messages = [m for m in self.conversation_history if m["role"] == "system"]
        self.conversation_history = system_messages # Retain only system messages
        self.turn_counter = 0 # Reset turn counter
        # When history is cleared, logs specific to the session should also be reset
        self.initialize_conversation_log() # Reinitialize text log
        self.close_conversation_csv_log() # Close current CSV log

    def initialize_conversation_log(self) -> None:
        """Initializes the text-based conversation log file."""
        # Use existing _ensure_text_log_open for consistency
        self._ensure_text_log_open()

    def initialize_conversation_csv_log(self,
                                        session_timestamp: str,
                                        mode: str = "Fixed",
                                        subject_id: Optional[str] = None) -> None:
        """
        Initializes the CSV conversation log for the current session via the log_queue.

        Args:
            session_timestamp: セッションタイムスタンプ
            mode: モード名 (Sin/HRF/Fixed)
        """
        if not self.log_queue:
            print("Warning: Log queue not provided to ConversationManager. CSV logging disabled.")
            return

        # Create utterance directory if it doesn't exist
        if config.SAVE_UTTERANCE_WAV:
            os.makedirs(config.UTTERANCE_WAV_DIR, exist_ok=True)

        if subject_id:
            self.set_subject_id(subject_id)
        self.current_session_timestamp_for_csv = session_timestamp
        self.current_session_mode = mode  # モードを保存
        csv_path = get_timestamped_log_path(
            config.CONVERSATION_CSV_LOG_FILE_TEMPLATE,
            session_timestamp=self.current_session_timestamp_for_csv,
            mode=mode,
            subject_id=self.subject_id
        )
        self.csv_log_filepath = csv_path

        # ヘッダーにモード列を追加
        header = ["timestamp", "role", "content", "start_time", "end_time", "wav_filename", "mode"]
        try:
            # Ensure directory exists before telling logger thread to add handler
            os.makedirs(os.path.dirname(self.csv_log_filepath), exist_ok=True)
            self.log_queue.put(("add_handler", config.LOGGER_CONVERSATION_CSV, self.csv_log_filepath, header))
            print(f"Conversation CSV log ready: {self.csv_log_filepath}")
        except Exception as e:
            print(f"Failed to initialize conversation CSV log: {e}")
            self.csv_log_filepath = None
            self.current_session_timestamp_for_csv = None

    def close_conversation_csv_log(self) -> None:
        """Closes the current CSV conversation log via the log_queue."""
        if self.csv_log_filepath and self.log_queue:
            print(f"Requesting close of conversation CSV log: {self.csv_log_filepath}")
            self.log_queue.put(("remove_handler", config.LOGGER_CONVERSATION_CSV))
            self.csv_log_filepath = None
            self.current_session_timestamp_for_csv = None
            
    def _ensure_text_log_open(self) -> None:
        """本文ログ用の文章ファイルを開く。"""
        if self.log_filepath:
            return
        template = config.CONVERSATION_LOG_FILE_TEMPLATE
        # session_timestampとmodeがあれば使用、なければフォールバック
        session_ts = getattr(self, 'current_session_timestamp_for_csv', None)
        mode = getattr(self, 'current_session_mode', None) or "Unknown"
        self.log_filepath = get_timestamped_log_path(
            template,
            subject_id=self.subject_id,
            session_timestamp=session_ts,
            mode=mode
        )
        try:
            os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                f.write(f"=== Conversation log started at {timestamp} ===\n\n")
            print(f"Conversation text log: {self.log_filepath}")
        except Exception as e:
            print(f"Failed to open conversation text log: {e}")
            self.log_filepath = None

    def _log_message(
        self,
        role: str,
        content: str,
        start_time: Optional[datetime.datetime],
        end_time: Optional[datetime.datetime],
        wav_filename: Optional[str] = None,
    ) -> None:
        """一件の会話を文章ファイルおよびCSVに記録する。"""
        timestamp = datetime.datetime.now()
        self._ensure_text_log_open()
        if self.log_filepath:
            try:
                with open(self.log_filepath, "a", encoding="utf-8") as f:
                    ts = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
                    f.write(f"[{ts}] {role.upper()}: {content}\n")
                    if wav_filename:
                        f.write(f"    WAV: {wav_filename}\n")
                    f.write("\n")
            except Exception as e:
                print(f"Failed to write to conversation text log: {e}")

        if self.log_queue:
            # self._ensure_csv_log_open() # Call this in initialize_conversation_csv_log
            if self.csv_log_filepath:
                # モードを取得（保存されていない場合はFixedをデフォルト）
                mode = getattr(self, 'current_session_mode', 'Fixed')
                payload = [
                    timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    role,
                    content,
                    start_time.strftime("%Y-%m-%d %H:%M:%S.%f") if start_time else "",
                    end_time.strftime("%Y-%m-%d %H:%M:%S.%f") if end_time else "",
                    wav_filename or "",
                    mode,  # モード列を追加
                ]
                record = logging.LogRecord(
                    name=config.LOGGER_CONVERSATION_CSV, level=logging.INFO, pathname="",
                    lineno=0, msg="", args=(payload,), exc_info=None)
                record.payload = payload
                self.log_queue.put(record)

    def close(self) -> None:
        """会話ログ用のCSVハンドラを閉じるようロギングすれっどに依頼する。"""
        self.close_conversation_csv_log() # Use the new method

    def generate_reply(self, user_text: str, system_prompt: str, wav_filename: Optional[str] = None) -> str:
        """
        利用中の大規模言語モデルとの結合は環境ごとに異なるはずなので、
        ここでは「単純な固定応答＋エコー」にしておく。
        """
        if system_prompt:
            self.add_message("system", system_prompt)

        self.add_message("user", user_text, wav_filename=wav_filename)
        reply = f"あなたは「{user_text}」と言いました。今は心拍数連動の音声と動作確認用の簡易応答です。"
        self.add_message("assistant", reply)
        return reply

    def conversation_loop(self) -> None:
        """
        VADによって確定されたテキストをキューから取得し、応答生成→音声出力、をくり返す会話ループ。
        """
        self._stop_conversation = False
        if self.app:
            self.app.set_status("Ready to talk. Please speak.", "green")

        while not self._stop_conversation:
            try:
                user_text, audio_data = self.audio_processor.final_text_queue.get(timeout=1.0)
                
                if not user_text:
                    continue

                self.turn_counter += 1
                wav_filename = None
                if config.SAVE_UTTERANCE_WAV and isinstance(audio_data, np.ndarray):
                    session_id = self.current_session_timestamp_for_csv or "session"
                    subject_component = format_subject_id_for_filename(self.subject_id)
                    filename = f"{subject_component}_{session_id}_turn_{self.turn_counter:03d}.wav"
                    wav_filename = os.path.join(config.UTTERANCE_WAV_DIR, filename)
                    try:
                        sf.write(wav_filename, audio_data, self.audio_processor.sample_rate)
                        print(f"Saved utterance to {wav_filename}")
                    except Exception as e:
                        print(f"Failed to save utterance WAV file: {e}")
                        wav_filename = None
                
                if self.app:
                    self.app.append_log(f"[User] {user_text}")

                sys_prompt = self.app.system_prompt.get("1.0", "end").strip() if self.app and hasattr(self.app, "system_prompt") else ""
                reply_text = self.generate_reply(user_text, sys_prompt, wav_filename=wav_filename)

                if self.app:
                    self.app.append_log(f"[Assistant] {reply_text}")
                    self.app.set_status("Generating and playing audio response...", "green")

                self.audio_processor.text_to_speech(reply_text)
                
                if self.app:
                    self.app.set_status("Ready to talk. Please speak.", "green")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in conversation loop: {e}")
                if self.app:
                    self.app.set_status(f"Error in conversation loop: {e}", "red")
                break

        if self.app:
            self.app.set_status("Conversation loop stopped.", "blue")

    def stop_conversation(self) -> None:
        self._stop_conversation = True
