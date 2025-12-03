# polar_monitor.py

import asyncio
import datetime
import logging
import queue
import statistics
import threading
from typing import Optional, List, Any, Dict

from bleak import BleakClient, BleakScanner

import config
from logger_utils import get_timestamped_log_path


class HeartRateMonitor:
    """Monitors heart rate from Verity Sense and manages data logging via queue."""
    def __init__(self, log_queue_ref: queue.Queue):
        self.current_hr: int = 0
        self.reference_hr: int = 70 # Default, can be updated
        self.last_timestamp: Optional[datetime.datetime] = None
        self.is_connected: bool = False
        self.client: Optional[BleakClient] = None
        self.hr_lock = threading.Lock()
        self.device: Optional[Any] = None # Bleak device object
        self.current_prosody_level: float = 1.0
        self.log_queue: queue.Queue = log_queue_ref
        self.hr_prosody_filepath: str = ""
        self.verity_hr_session_filepath: str = ""
        self.baseline_hr_samples: List[int] = []
        self.is_measuring_baseline: bool = False

    async def start_monitoring_async(self) -> bool:
        """Asynchronous method to connect and start monitoring."""
        try:
            self.is_connected = await self.connect_to_device()
        except Exception as e:
            print(f"Error starting Verity Sense monitoring: {e}")
            self.is_connected = False
        return self.is_connected

    async def stop_monitoring_async(self):
        """Asynchronous method to disconnect and stop monitoring."""
        try:
            if self.is_connected and self.client:
                await self.disconnect()
        except Exception as e:
            print(f"Error stopping Verity Sense monitoring: {e}")
        finally:
            self.is_connected = False


    def initialize_hr_prosody_csv(self) -> None:
        """Stores the intended path for the HR/Prosody log and signals logger thread."""
        try:
            self.hr_prosody_filepath = get_timestamped_log_path(config.HR_PROSODY_CSV_TEMPLATE)
            header = ["Timestamp", "Heart Rate (BPM)", "Prosody Level Applied", "Reference HR"]
            self.log_queue.put(("add_handler", config.LOGGER_HR_PROSODY, self.hr_prosody_filepath, header))
            print(f"HR and Prosody log (Verity) ready: {self.hr_prosody_filepath}")
        except Exception as e:
            print(f"Failed to set HR and Prosody log path (Verity): {e}")
            self.hr_prosody_filepath = ""

    def initialize_verity_hr_session_csv(self, filepath: str) -> None:
        """Stores the intended path for the Verity HR session log and signals logger thread."""
        try:
            self.verity_hr_session_filepath = filepath
            header = ["Timestamp", "Heart Rate (BPM)"]
            self.log_queue.put(("add_handler", config.LOGGER_VERITY_HR_SESSION, self.verity_hr_session_filepath, header))
            print(f"Conversation HR log (Verity) ready: {self.verity_hr_session_filepath}")
        except Exception as e:
            print(f"Failed to set conversation HR log path (Verity): {e}")
            self.verity_hr_session_filepath = ""

    def close_verity_hr_session_csv(self) -> None:
        """Signals the logger thread to remove the Verity HR session handler."""
        if self.verity_hr_session_filepath:
             print(f"Requesting close of conversation HR log (Verity): {self.verity_hr_session_filepath}")
             self.log_queue.put(("remove_handler", config.LOGGER_VERITY_HR_SESSION))
             self.verity_hr_session_filepath = ""

    def set_reference_hr(self, value: int) -> None:
        """Sets the reference heart rate."""
        with self.hr_lock:
            self.reference_hr = value
            print(f"Reference HR set to: {value}")
        self.log_hr_with_prosody()

    def get_current_hr(self) -> int:
        """Gets the current heart rate."""
        with self.hr_lock:
            return self.current_hr

    def get_reference_hr(self) -> int:
        """Gets the reference heart rate."""
        with self.hr_lock:
            return self.reference_hr

    def update_prosody_level(self, level: float) -> None:
        """Updates the currently applied prosody level and logs it."""
        with self.hr_lock:
            self.current_prosody_level = level
        self.log_hr_with_prosody()

    def log_hr_with_prosody(self) -> None:
        """Puts the current HR and the *applied* prosody level into the log queue."""
        if not self.hr_prosody_filepath: return
        try:
            with self.hr_lock:
                current_hr = self.current_hr
                prosody_level = self.current_prosody_level
                ref_hr = self.reference_hr
                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            payload = [timestamp_str, current_hr, f"{prosody_level:.2f}", ref_hr]
            record = logging.LogRecord(
                name=config.LOGGER_HR_PROSODY, level=logging.INFO, pathname="", lineno=0,
                msg="", args=(payload,), exc_info=None, func=""
            )
            record.payload = payload # type: ignore
            self.log_queue.put(record)
        except Exception as e:
            print(f"Unexpected error while queuing HR/Prosody log (Verity): {e}")

    def log_verity_hr_to_session(self, timestamp_dt: datetime.datetime, hr_value: int) -> None:
        """Puts the Verity Sense HR into the log queue for session logging."""
        if not self.verity_hr_session_filepath: return
        try:
            timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            payload = [timestamp_str, hr_value]
            record = logging.LogRecord(
                name=config.LOGGER_VERITY_HR_SESSION, level=logging.INFO, pathname="", lineno=0,
                msg="", args=(payload,), exc_info=None, func=""
            )
            record.payload = payload # type: ignore
            self.log_queue.put(record)
        except Exception as e:
            print(f"Unexpected error while queuing conversation HR (Verity): {e}")

    def hr_notification_handler(self, sender: int, data: bytearray):
        """Callback for receiving heart rate data from Verity Sense."""
        try:
            if not data: return

            flags = data[0]
            hr_format_bit = (flags >> 0) & 1 # 0 for UINT8, 1 for UINT16
            byte_index = 1
            hr_value = 0

            if hr_format_bit == 1: # UINT16 format
                if len(data) >= 3:
                    hr_value = int.from_bytes(data[byte_index:byte_index+2], byteorder='little')
                else: return # Not enough data
            else: # UINT8 format
                if len(data) >= 2:
                    hr_value = data[byte_index]
                else: return # Not enough data

            timestamp = datetime.datetime.now()

            with self.hr_lock:
                self.current_hr = hr_value
                self.last_timestamp = timestamp
                if self.is_measuring_baseline:
                    self.baseline_hr_samples.append(hr_value)

            self.log_hr_with_prosody() # Log with current prosody
            self.log_verity_hr_to_session(timestamp, hr_value) # Log to session CSV

        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                print(f"Runtime error during HR processing (Verity): {e}")
        except Exception as e:
            print(f"Unexpected error during HR processing (Verity): {e}")

    async def connect_to_device(self) -> bool:
        """Connects to the Verity Sense heart rate sensor."""
        try:
            print("Searching for Polar Verity Sense...")
            devices = await BleakScanner.discover(timeout=15.0, return_adv=False)
            self.device = next((d for d in devices if d.name and config.POLAR_VERITY_SENSE_NAME.lower() in d.name.lower()), None)

            if not self.device:
                print(f"{config.POLAR_VERITY_SENSE_NAME} not found.")
                return False

            print(f"Attempting to connect to Verity Sense: {self.device.address}")
            self.client = BleakClient(self.device.address, timeout=20.0) # Use address string
            await self.client.connect()

            if self.client.is_connected:
                print(f"Connected to {config.POLAR_VERITY_SENSE_NAME}: {self.device.address}")
                self.initialize_hr_prosody_csv() # Initialize log for this connection session
                await self.client.start_notify(config.HR_CHARACTERISTIC_UUID, self.hr_notification_handler)
                print("Started receiving heart rate data (Verity)")
                self.is_connected = True
                return True
            else:
                print(f"Failed to connect to {config.POLAR_VERITY_SENSE_NAME}.")
                self.client = None
                return False
        except asyncio.TimeoutError:
            print("Verity Sense search or connection timed out.")
            self.client = None
            return False
        except Exception as e:
            print(f"Error during Verity Sense connection: {e}")
            if self.client and self.client.is_connected:
                try: await self.client.disconnect()
                except Exception: pass
            self.client = None
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnects from the Verity Sense sensor."""
        print("Starting Verity Sense disconnection...")
        self.is_measuring_baseline = False # Stop baseline measurement on disconnect

        client = self.client # Local variable to avoid race condition if self.client is set to None elsewhere
        if client and client.is_connected:
            try:
                await client.stop_notify(config.HR_CHARACTERISTIC_UUID)
            except Exception as e:
                print(f"Error stopping HR notifications (Verity): {e}")
            try:
                await client.disconnect()
                print(f"Disconnected from {config.POLAR_VERITY_SENSE_NAME}")
            except Exception as e:
                print(f"Error disconnecting Verity Sense: {e}")
        else:
            print("Verity Sense is not connected.")

        self.is_connected = False
        self.client = None
        # Close general HR/Prosody log
        if self.hr_prosody_filepath:
             print(f"Requesting close of HR/Prosody log (Verity): {self.hr_prosody_filepath}")
             self.log_queue.put(("remove_handler", config.LOGGER_HR_PROSODY))
             self.hr_prosody_filepath = ""
        print("Verity Sense disconnection process complete.")


    def start_baseline_measurement(self, duration_seconds: int) -> bool:
        """Starts collecting HR data for baseline calculation."""
        if not self.is_connected:
            print("Error: Cannot start baseline measurement, Verity Sense not connected.")
            return False
        if self.is_measuring_baseline:
            print("Warning: Baseline measurement already in progress.")
            return False

        print(f"Starting baseline HR measurement for {duration_seconds} seconds...")
        with self.hr_lock:
            self.baseline_hr_samples = [] # Clear previous samples
            self.is_measuring_baseline = True
        return True

    def stop_baseline_measurement(self) -> Optional[int]:
        """Stops baseline measurement and calculates the median HR."""
        if not self.is_measuring_baseline:
            print("Warning: Baseline measurement was not running.")
            return None

        print("Stopping baseline HR measurement...")
        with self.hr_lock:
            self.is_measuring_baseline = False
            samples = list(self.baseline_hr_samples) # Make a copy

        if len(samples) < config.MIN_SAMPLES_FOR_MEDIAN:
            print(f"Insufficient data for median calculation (collected {len(samples)}, need {config.MIN_SAMPLES_FOR_MEDIAN}).")
            return None

        try:
            median_hr = int(statistics.median(samples))
            print(f"Baseline measurement complete. Samples: {len(samples)}, Median HR: {median_hr}")
            return median_hr
        except statistics.StatisticsError as e:
            print(f"Error calculating median HR: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during median calculation: {e}")
            return None


class H10Monitor:
    """Monitors ECG and HR from Polar H10 and manages data logging via queue."""
    def __init__(self, log_queue_ref: queue.Queue):
        self.is_connected: bool = False
        self.client: Optional[BleakClient] = None
        self.device: Optional[Any] = None # Bleak device object
        self.ecg_lock = threading.Lock() # For synchronizing access to ECG data if needed elsewhere
        self.hr_lock = threading.Lock()
        self.current_h10_hr: int = 0
        self.log_queue: queue.Queue = log_queue_ref
        self.h10_ecg_session_filepath: str = ""
        self.h10_hr_session_filepath: str = ""

    async def start_monitoring_async(self):
        """Asynchronous method to connect and start monitoring."""
        try:
            self.is_connected = await self.connect_to_device()
        except Exception as e:
            print(f"Error starting H10 monitoring: {e}")
            self.is_connected = False
        return self.is_connected

    async def stop_monitoring_async(self):
        """Asynchronous method to disconnect and stop monitoring."""
        try:
            if self.is_connected and self.client:
                await self.disconnect()
        except Exception as e:
            print(f"Error stopping H10 monitoring: {e}")
        finally:
            self.is_connected = False

    def get_current_rr(self) -> int:
        """Gets the current RR interval (which is essentially HR for H10)."""
        with self.hr_lock:
            # H10 provides HR, not RR interval directly in this context.
            # Assuming 'RR interval (ms)' label in GUI should show HR in BPM.
            # If actual RR intervals are needed, a different characteristic or calculation would be required.
            return self.current_h10_hr


    def initialize_h10_ecg_session_csv(self, filepath: str):
        """Stores the intended path for the H10 ECG session log and signals logger thread."""
        try:
            self.h10_ecg_session_filepath = filepath
            header = ["Timestamp", "Device Timestamp (ns)", "ECG Value (uV)"]
            self.log_queue.put(("add_handler", config.LOGGER_H10_ECG_SESSION, self.h10_ecg_session_filepath, header))
            print(f"Conversation ECG log (H10) ready: {self.h10_ecg_session_filepath}")
        except Exception as e:
            print(f"Failed to set conversation ECG log path (H10): {e}")
            self.h10_ecg_session_filepath = ""

    def initialize_h10_hr_session_csv(self, filepath: str) -> None:
        """Stores the intended path for the H10 HR session log and signals logger thread."""
        try:
            self.h10_hr_session_filepath = filepath
            header = ["Timestamp", "Heart Rate (BPM)"]
            self.log_queue.put(("add_handler", config.LOGGER_H10_HR_SESSION, self.h10_hr_session_filepath, header))
            print(f"Conversation HR log (H10) ready: {self.h10_hr_session_filepath}")
        except Exception as e:
            print(f"Failed to set conversation HR log path (H10): {e}")
            self.h10_hr_session_filepath = ""

    def close_h10_ecg_session_csv(self) -> None:
        """Signals the logger thread to remove the H10 ECG session handler."""
        if self.h10_ecg_session_filepath:
            print(f"Requesting close of conversation ECG log (H10): {self.h10_ecg_session_filepath}")
            self.log_queue.put(("remove_handler", config.LOGGER_H10_ECG_SESSION))
            self.h10_ecg_session_filepath = ""

    def close_h10_hr_session_csv(self) -> None:
        """Signals the logger thread to remove the H10 HR session handler."""
        if self.h10_hr_session_filepath:
            print(f"Requesting close of conversation HR log (H10): {self.h10_hr_session_filepath}")
            self.log_queue.put(("remove_handler", config.LOGGER_H10_HR_SESSION))
            self.h10_hr_session_filepath = ""

    def ecg_data_handler(self, sender: int, data: bytearray):
        """Callback for receiving ECG data from H10 PMD characteristic."""
        try:
            if not data: return
            if not self.h10_ecg_session_filepath: return # Don't process if no log file is set

            # Type 0 for ECG data
            if data[0] == 0x00: # ECG data
                device_timestamp_ns = self.convert_to_unsigned_long(data, 1, 8)
                current_time = datetime.datetime.now()
                timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
                step = 3  # 3 bytes per sample
                samples_raw = data[10:] # Skip type and timestamp
                offset = 0
                payloads_to_log = []

                # No need for ecg_lock here as this handler processes data serially for this device
                while offset + step <= len(samples_raw):
                    # ECG value is in microvolts (uV)
                    raw_value = int.from_bytes(samples_raw[offset:offset+step], byteorder='little', signed=True)
                    ecg_value_uv = float(raw_value) # Original code had / 1.0, which is redundant
                    payloads_to_log.append([timestamp_str, device_timestamp_ns, f"{ecg_value_uv:.2f}"])
                    offset += step

                for payload_item in payloads_to_log:
                    record = logging.LogRecord(
                        name=config.LOGGER_H10_ECG_SESSION, level=logging.INFO, pathname="", lineno=0,
                        msg="", args=(payload_item,), exc_info=None, func=""
                    )
                    record.payload = payload_item # type: ignore
                    self.log_queue.put(record)

        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                print(f"Runtime error during ECG processing (H10): {e}")
        except Exception as e:
            print(f"Unexpected error during ECG processing (H10): {e}")

    def h10_hr_notification_handler(self, sender: int, data: bytearray):
        """Callback for receiving heart rate data from H10 HR characteristic."""
        try:
            if not data: return

            flags = data[0]
            hr_format_bit = (flags >> 0) & 1 # 0 for UINT8, 1 for UINT16
            byte_index = 1
            hr_value = 0

            if hr_format_bit == 1: # UINT16 format
                if len(data) >= 3:
                    hr_value = int.from_bytes(data[byte_index:byte_index+2], byteorder='little')
                else: return # Not enough data
            else: # UINT8 format
                if len(data) >= 2:
                    hr_value = data[byte_index]
                else: return # Not enough data

            timestamp = datetime.datetime.now()
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

            with self.hr_lock:
                self.current_h10_hr = hr_value

            # ログファイルが設定されている場合のみ記録
            if self.h10_hr_session_filepath:
                payload = [timestamp_str, hr_value]
                record = logging.LogRecord(
                    name=config.LOGGER_H10_HR_SESSION, level=logging.INFO, pathname="", lineno=0,
                    msg="", args=(payload,), exc_info=None, func=""
                )
                record.payload = payload # type: ignore
                self.log_queue.put(record)

        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                print(f"Runtime error during HR processing (H10): {e}")
        except Exception as e:
            print(f"Unexpected error during HR processing (H10): {e}")

    def convert_to_unsigned_long(self, data: bytearray, offset: int, length: int) -> int:
        """Helper function to convert bytes to unsigned long."""
        return int.from_bytes(data[offset : offset + length], byteorder="little", signed=False)

    async def connect_to_device(self) -> bool:
        """Connects to the Polar H10 sensor."""
        try:
            print("Searching for Polar H10...")
            devices = await BleakScanner.discover(timeout=15.0, return_adv=False)
            self.device = next((d for d in devices if d.name and config.POLAR_H10_NAME.lower() in d.name.lower()), None)

            if not self.device:
                print(f"{config.POLAR_H10_NAME} not found.")
                return False

            print(f"Attempting to connect to H10: {self.device.address}")
            self.client = BleakClient(self.device.address, timeout=20.0) # Use address string
            await self.client.connect()

            if self.client.is_connected:
                print(f"Connected to {config.POLAR_H10_NAME}: {self.device.address}")

                # Enable ECG stream
                print("Sending ECG stream start command...")
                await self.client.write_gatt_char(config.PMD_CONTROL, config.ECG_WRITE, response=True)
                print("ECG stream start command sent.")
                await self.client.start_notify(config.PMD_DATA, self.ecg_data_handler)
                print("Started receiving ECG data (H10)")

                # Enable HR stream
                print("Starting HR stream...")
                await self.client.start_notify(config.HR_CHARACTERISTIC_UUID, self.h10_hr_notification_handler)
                print("Started receiving heart rate data (H10)")

                self.is_connected = True
                return True
            else:
                print(f"Failed to connect to {config.POLAR_H10_NAME}.")
                self.client = None
                return False
        except asyncio.TimeoutError:
            print("H10 search or connection timed out.")
            self.client = None
            return False
        except Exception as e:
            print(f"Error during H10 device connection: {e}")
            if self.client and self.client.is_connected:
                try: await self.client.disconnect()
                except Exception: pass
            self.client = None
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnects from the H10 sensor."""
        print("Starting H10 disconnection...")

        client = self.client # Local variable
        if client and client.is_connected:
            try:
                await client.stop_notify(config.PMD_DATA)
                print("Stopped ECG data notifications (H10).")
            except Exception as e:
                print(f"Error stopping ECG data notifications (H10): {e}")
            try:
                await client.stop_notify(config.HR_CHARACTERISTIC_UUID)
                print("Stopped heart rate data notifications (H10).")
            except Exception as e:
                print(f"Error stopping heart rate data notifications (H10): {e}")
            try:
                await client.disconnect()
                print(f"Disconnected from {config.POLAR_H10_NAME}")
            except Exception as e:
                print(f"Error disconnecting H10 device: {e}")
        else:
            print("H10 is not connected.")

        self.is_connected = False
        self.client = None
        self.current_h10_hr = 0
        print("H10 disconnection process complete.")