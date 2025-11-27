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

# --- General Configuration ---
CONFIG_FILE = resource_path("config_heartrate_prosody.json")
LOG_DIR = os.path.join(os.path.expanduser("~"), "/Users/user/Research/HCS_ver4.0", "Logs")

# --- Log File Templates ---
CONVERSATION_LOG_FILE_TEMPLATE = os.path.join(LOG_DIR, "conversation_log_{timestamp}.txt")
CONVERSATION_CSV_LOG_FILE_TEMPLATE = os.path.join(LOG_DIR, "conversation_log_{session_timestamp}.csv")
HR_PROSODY_CSV_TEMPLATE = os.path.join(LOG_DIR, "verity_hr_prosody_{timestamp}.csv")

# HEARTRATE_AFTER_TTS_CSV = os.path.join(LOG_DIR, "heartrate_after_tts.csv") # 変更前
HEARTRATE_AFTER_TTS_CSV_TEMPLATE = os.path.join(LOG_DIR, "heartrate_after_tts_{session_timestamp}.csv") # 変更後

# HEARTRATE_AT_RECORDING_START_CSV = os.path.join(LOG_DIR, "heartrate_at_recording_start.csv") # 変更前
HEARTRATE_AT_RECORDING_START_CSV_TEMPLATE = os.path.join(LOG_DIR, "heartrate_at_recording_start_{session_timestamp}.csv") # 変更後

VERITY_HR_SESSION_CSV_TEMPLATE = os.path.join(LOG_DIR, "verity_hr_session_{session_timestamp}.csv")
H10_ECG_SESSION_CSV_TEMPLATE = os.path.join(LOG_DIR, "h10_ecg_session_{session_timestamp}.csv")
H10_HR_SESSION_CSV_TEMPLATE = os.path.join(LOG_DIR, "h10_hr_session_{session_timestamp}.csv")
INTERACTION_EVENT_LOG_FILE_TEMPLATE = os.path.join(LOG_DIR, "interaction_events_{session_timestamp}.csv")

# --- API and Service URLs ---
VOICEVOX_URL = "http://localhost:50021"

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
LOGGER_CONVERSATION_CSV = "conversation_csv"
LOGGER_INTERACTION_EVENTS = "interaction_events"