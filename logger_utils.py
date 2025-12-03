# logger_utils.py

import logging
import logging.handlers
import csv
import datetime
import os
import queue # queue.Queue は main.py などでインスタンス化され、LoggingThread に渡される
import threading
from typing import Optional, List, Any

# config.py から LOG_DIR をインポートすることを想定
# 例: from . import config (同じディレクトリにある場合)
# または、initialize_log_directory が LOG_DIR を引数で受け取るように変更も可能

def get_timestamped_log_path(template: str, session_timestamp: Optional[str] = None, mode: str = "Fixed") -> str:
    """
    Generates a log file path with the current or session timestamp and mode.

    Args:
        template: ファイルパステンプレート（{timestamp}, {session_timestamp}, {mode}を含む）
        session_timestamp: セッションタイムスタンプ（Noneの場合は現在時刻）
        mode: モード名 (Sin/HRF/Fixed)

    Returns:
        フォーマット済みのファイルパス
    """
    if session_timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        timestamp = session_timestamp
    return template.format(timestamp=timestamp, session_timestamp=timestamp, mode=mode)

def initialize_log_directory(log_dir_path: str):
    """Creates the log directory if it doesn't exist."""
    os.makedirs(log_dir_path, exist_ok=True)
    print(f"Log directory checked/created: {log_dir_path}")

# --- Log Record Factory ---
def record_factory(*args, **kwargs):
    """
    Custom log record factory that moves args to a 'payload' attribute
    if args is a list or dict, to prevent formatting attempts by the logger.
    """
    record = logging.LogRecord(*args, **kwargs)
    # Check if args is a list/tuple/dict and not empty, and if msg is empty (or a placeholder)
    # This indicates that args is likely the intended payload for CsvFileHandler
    if isinstance(record.args, (dict, list, tuple)) and len(record.args) > 0:
        record.payload = record.args[0] # Assuming the first element of args is the payload
        record.args = () # Clear args to prevent default formatting
    else:
        record.payload = None
    return record

# --- CSV File Handler for Logging ---
class CsvFileHandler(logging.FileHandler):
    """Custom logging handler to write specific data to CSV files."""
    def __init__(self, filename, mode='a', encoding='utf-8-sig', delay=False, header: Optional[List[str]] = None):
        super().__init__(filename, mode, encoding, delay)
        self.header = header
        self.writer = None
        # Ensure stream is opened before checking/writing header if not delayed
        if not delay:
            self._check_and_write_header()

    def _check_and_write_header(self):
        """Writes the header if the file is new or empty."""
        if self.stream is None: # Ensure stream is open
            self.stream = self._open()
        if self.stream and self.header:
            self.stream.seek(0, os.SEEK_END) # Go to the end of the file
            if self.stream.tell() == 0:      # Check if the file is empty
                self.stream.seek(0)          # Go back to the beginning to write the header
                try:
                    csv_writer = csv.writer(self.stream)
                    csv_writer.writerow(self.header)
                    self.stream.flush()
                    print(f"Log file header written: {self.baseFilename}")
                except Exception as e:
                    print(f"Error writing CSV header ({self.baseFilename}): {e}")

    def emit(self, record: logging.LogRecord):
        """Emit a record. Writes the payload to the CSV."""
        if self.stream is None: # Ensure stream is open, especially if 'delay' was true
             self.stream = self._open()

        if self.writer is None and self.stream: # Initialize writer if not already done
             self.writer = csv.writer(self.stream)
             self._check_and_write_header() # Check header again, useful if file was created with delay

        if hasattr(record, 'payload') and record.payload is not None and self.writer:
            try:
                if isinstance(record.payload, (list, tuple)):
                    self.writer.writerow(record.payload)
                    self.flush() # Ensure data is written to disk
                else:
                    # This case should ideally not happen if record_factory and log calls are correct
                    print(f"Warning: Invalid payload type ({type(record.payload)}) in CsvFileHandler, cannot write to CSV: {record.payload}")
            except Exception as e:
                # Avoid printing common errors when the application is closing
                if 'I/O operation on closed file' not in str(e).lower() and \
                   'bad file descriptor' not in str(e).lower():
                    print(f"CSV log writing error ({self.baseFilename}): {e}")
        # If there's no payload, it's a normal log message not intended for this CSV handler,
        # or an incorrectly formatted one. We choose to silently ignore it or log a warning.
        # else:
        #     if not hasattr(record, 'payload') or record.payload is None:
        #         print(f"Debug: Record has no payload, not writing to CSV: {record.msg}")


# --- Logging Thread ---
class LoggingThread(threading.Thread):
    """
    A dedicated thread for handling logging operations asynchronously.
    Receives log records or commands via a queue.
    """
    def __init__(self, log_queue_ref: queue.Queue):
        super().__init__(daemon=True, name="LoggingThread")
        self.log_queue = log_queue_ref
        self.handlers: dict[str, CsvFileHandler] = {} # Logger name to Handler mapping
        self._stop_event = threading.Event()
        print("Logging thread initialized")

    def add_handler(self, logger_name: str, handler: CsvFileHandler):
        """Add a handler for a specific logger name (not logger instance)."""
        # This method is now primarily for direct use if needed,
        # but "add_handler" command via queue is preferred for thread safety.
        print(f"Adding logging handler: {logger_name} -> {handler.baseFilename}")
        self.handlers[logger_name] = handler
        # logging.getLogger(logger_name) # Not strictly necessary here as we manage handlers directly

    def remove_handler(self, logger_name: str):
        """Remove and close a handler by logger name."""
        if logger_name in self.handlers:
            handler = self.handlers.pop(logger_name)
            print(f"Removing and closing logging handler: {logger_name} -> {handler.baseFilename}")
            try:
                handler.close()
            except Exception as e:
                print(f"Error closing handler ({logger_name}): {e}")
        else:
             print(f"Warning: Handler to remove not found: {logger_name}")

    def stop(self):
        """Signal the thread to stop and close handlers."""
        print("Logging thread stop request received")
        self._stop_event.set()
        self.log_queue.put(None) # Put a sentinel value to unblock the queue.get()

    def run(self):
        """Main loop for the logging thread."""
        print("Logging thread started")
        # Configure basicConfig for other parts of the application if needed,
        # but this thread manages its CsvFileHandlers explicitly.
        # logging.basicConfig(level=logging.NOTSET) # Potentially configure elsewhere (e.g. main.py)

        original_factory = logging.getLogRecordFactory()
        logging.setLogRecordFactory(record_factory) # Set our custom factory

        while not self._stop_event.is_set():
            try:
                record_or_command = self.log_queue.get(timeout=0.5) # Timeout to check stop_event
                if record_or_command is None: # Sentinel for stopping
                    print("Logging thread: Stop signal (None) received from queue")
                    break

                if isinstance(record_or_command, tuple) and isinstance(record_or_command[0], str):
                    command = record_or_command[0]
                    if command == "add_handler":
                        # expects: ("add_handler", logger_name, filepath, header_list)
                        _, logger_name, filepath, header = record_or_command
                        # Ensure the directory exists before creating the handler
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        handler = CsvFileHandler(filepath, header=header)
                        self.handlers[logger_name] = handler
                        print(f"Logging handler dynamically added: {logger_name} -> {filepath}")
                    elif command == "remove_handler":
                        # expects: ("remove_handler", logger_name)
                        _, logger_name = record_or_command
                        self.remove_handler(logger_name)
                    continue # Processed command, go to next queue item

                # If it's a LogRecord
                if isinstance(record_or_command, logging.LogRecord):
                    record = record_or_command
                    logger_name = record.name
                    if logger_name in self.handlers:
                        handler = self.handlers[logger_name]
                        # The CsvFileHandler itself will check if it should handle this record (e.g. based on payload)
                        # No need for handler.filter(record) if CsvFileHandler.emit does the job
                        handler.handle(record) # Let the handler process it
                    # else:
                        # Optionally log if a record is received for an unknown logger_name
                        # print(f"LoggingThread: No handler for logger_name '{logger_name}'")
                else:
                    print(f"LoggingThread: Received unknown item in queue: {type(record_or_command)}")

            except queue.Empty:
                # Timeout occurred, loop back to check stop_event
                continue
            except Exception as e:
                print(f"Logging thread error: {e}")
                import traceback
                traceback.print_exc()

        print("Logging thread: Cleaning up remaining handlers...")
        remaining_handlers = list(self.handlers.keys()) # Iterate over a copy of keys
        for logger_name in remaining_handlers:
             self.remove_handler(logger_name)

        logging.setLogRecordFactory(original_factory) # Restore original factory
        print("Logging thread finished")