# conversation_manager.py

import datetime
import logging # For creating LogRecord instances
import os # For os.path.join if get_timestamped_log_path is not used for text logs
import queue # For type hinting log_queue_ref
from typing import Optional, List, Dict, Any

# Assuming config.py and logger_utils.py are in the same directory or accessible in PYTHONPATH
import config
from logger_utils import get_timestamped_log_path # If used for text log path generation as well

class ConversationManager:
    """Manages conversation history and logging for text and CSV formats."""
    def __init__(self, max_history: int = 10, log_queue_ref: Optional[queue.Queue] = None):
        self.max_history: int = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.log_filepath: Optional[str] = None # For text log
        self.csv_log_filepath: Optional[str] = None # For CSV log
        self.log_queue: Optional[queue.Queue] = log_queue_ref
        self.current_session_timestamp_for_csv: Optional[str] = None

        self.initialize_conversation_log() # Initialize text log on creation

    def initialize_conversation_log(self):
        """Initializes the text-based conversation log file."""
        try:
            # Using get_timestamped_log_path for consistency, assuming LOG_DIR is part of the template path in config
            self.log_filepath = get_timestamped_log_path(config.CONVERSATION_LOG_FILE_TEMPLATE)
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
            with open(self.log_filepath, 'w', encoding='utf-8') as f:
                f.write("=== Conversation Log ===\n")
                f.write(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            print(f"Conversation text log initialized: {self.log_filepath}")
        except Exception as e:
            print(f"Failed to initialize conversation text log file: {e}")
            self.log_filepath = None

    def initialize_conversation_csv_log(self, session_timestamp: str) -> None:
        """Initializes the CSV conversation log for the current session via the log_queue."""
        if not self.log_queue:
            print("Warning: Log queue not provided to ConversationManager. CSV logging disabled.")
            return
        try:
            self.current_session_timestamp_for_csv = session_timestamp
            self.csv_log_filepath = get_timestamped_log_path(
                config.CONVERSATION_CSV_LOG_FILE_TEMPLATE,
                session_timestamp=self.current_session_timestamp_for_csv
            )
            # ヘッダーに Start Time と End Time を追加
            header = ["Timestamp", "Role", "Content", "Start Time", "End Time"]
            # Ensure directory exists before telling logger thread to add handler
            os.makedirs(os.path.dirname(self.csv_log_filepath), exist_ok=True)
            self.log_queue.put(("add_handler", config.LOGGER_CONVERSATION_CSV, self.csv_log_filepath, header))
            print(f"Conversation CSV log ready: {self.csv_log_filepath}")
        except Exception as e:
            print(f"Failed to initialize conversation CSV log: {e}")
            self.csv_log_filepath = None

    def close_conversation_csv_log(self) -> None:
        """Closes the current CSV conversation log via the log_queue."""
        if self.csv_log_filepath and self.log_queue:
            print(f"Requesting close of conversation CSV log: {self.csv_log_filepath}")
            self.log_queue.put(("remove_handler", config.LOGGER_CONVERSATION_CSV))
            self.csv_log_filepath = None
            self.current_session_timestamp_for_csv = None

    def add_message(self, role: str, content: str, start_time: Optional[datetime.datetime] = None, end_time: Optional[datetime.datetime] = None) -> None:
        """Adds a message to the conversation history and logs it."""
        if not content:
            return

        message = {"role": role, "content": content}
        self.conversation_history.append(message)

        # Prune history: keep system messages + last max_history user/assistant messages
        system_messages = [m for m in self.conversation_history if m["role"] == "system"]
        user_assistant_messages = [m for m in self.conversation_history if m["role"] != "system"]
        if len(user_assistant_messages) > self.max_history * 2: # *2 because user and assistant alternate
            user_assistant_messages = user_assistant_messages[-(self.max_history * 2):]
        self.conversation_history = system_messages + user_assistant_messages

        current_timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        # Log to text file (if not a system message)
        if role != "system" and self.log_filepath:
            try:
                with open(self.log_filepath, 'a', encoding='utf-8') as f:
                    prefix = "User" if role == "user" else "Assistant"
                    f.write(f"[{current_timestamp_str}] {prefix}: {content}\n")
                    if role == "assistant":
                        f.write("\n") # Add a newline after assistant messages for readability
            except Exception as e:
                print(f"Failed to write to conversation text log: {e}")

        # Log to CSV via queue (if not a system message and CSV log is active)
        if role != "system" and self.csv_log_filepath and self.log_queue:
            try:
                # 開始時刻と終了時刻を文字列に変換
                start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S.%f') if start_time else ""
                end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S.%f') if end_time else ""

                # ペイロードに開始・終了時刻を追加
                payload = [current_timestamp_str, role.capitalize(), content, start_time_str, end_time_str]
                record = logging.LogRecord(
                    name=config.LOGGER_CONVERSATION_CSV, level=logging.INFO,
                    pathname="", lineno=0, msg="", args=(payload,),
                    exc_info=None, func=""
                )
                record.payload = payload # type: ignore
                self.log_queue.put(record)
            except Exception as e:
                print(f"Failed to queue conversation CSV log: {e}")

    def get_messages(self) -> List[Dict[str, str]]:
        """Returns the current conversation history, ensuring a system prompt exists."""
        if not any(m["role"] == "system" for m in self.conversation_history):
            print("Warning: No system prompt in conversation history. Adding a default one.")
            # This will add a default prompt and log it if add_message handles system prompts for logging.
            # For simplicity, we assume update_system_prompt is called initially.
            # Or, add a simple default here:
            self.conversation_history.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        return list(self.conversation_history) # Return a copy

    def clear_history(self) -> None:
        """Clears user/assistant messages, retaining system prompts, and reinitializes logs."""
        system_messages = [m for m in self.conversation_history if m["role"] == "system"]
        self.conversation_history = system_messages # Retain only system messages

        # Reinitialize the text log for a new "session" within the same manager instance
        self.initialize_conversation_log()

        # Close the current CSV log; a new one will be created if a new conversation starts
        self.close_conversation_csv_log()
        print("Conversation history cleared (system prompt retained). Text log reinitialized, CSV log closed.")

    def update_system_prompt(self, new_prompt: str):
        """Updates or adds the system prompt."""
        if not new_prompt:
            print("Warning: Attempted to set an empty system prompt.")
            return

        # Remove existing system prompts
        self.conversation_history = [m for m in self.conversation_history if m["role"] != "system"]
        # Add the new system prompt at the beginning
        self.conversation_history.insert(0, {"role": "system", "content": new_prompt})
        print(f"System prompt updated: {new_prompt[:100]}...")
        # Optionally, log this change to the text log if desired
        if self.log_filepath:
            try:
                with open(self.log_filepath, 'a', encoding='utf-8') as f:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    f.write(f"[{timestamp}] System Prompt Updated: {new_prompt}\n\n")
            except Exception as e:
                print(f"Failed to log system prompt update: {e}")
