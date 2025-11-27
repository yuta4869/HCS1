# HCS_ver4.0: Human Computer System for Heart Rate and Prosody Analysis

## Project Overview

This project, `HCS_ver4.0`, appears to be a Human Computer System focusing on the analysis of heart rate data and prosody (speech rhythm, intonation, stress) to understand human states or interactions. It likely integrates various modules for audio processing, physiological monitoring, and conversational management, potentially with a graphical user interface.

## Features (Inferred)

*   **Heart Rate Monitoring:** Integration with physiological sensors (e.g., Polar devices via `polar_monitor.py` or simulated data).
*   **Audio Processing:** Analysis of speech prosody, potentially through `audio_processing.py`.
*   **Conversation Management:** Logic for managing interactions, possibly handled by `conversation_manager.py`.
*   **Graphical User Interface (GUI):** A user-friendly interface built with `gui.py`.
*   **Logging:** Utilities for logging various data and events (`logger_utils.py`).
*   **Configuration:** Flexible configuration options via `config.py` and `config_heartrate_prosody.json`.

## Setup and Installation

This project requires Python 3.x. It is highly recommended to use a Python virtual environment to manage dependencies.

1.  **Navigate to the project directory:**
    ```bash
    cd /Users/user/Research/HCS_ver4.0
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install required dependencies:**
    The project uses several external libraries. Based on observed dependencies, these include:
    ```bash
    pip install matplotlib torch bleak opencv-python sounddevice numpy
    ```
    *(Note: If you encounter issues with `numpy` during installation, it might be due to a conflict with a system-installed version. Using a virtual environment typically resolves this. If problems persist, consider uninstalling a system `numpy` if feasible, or ensure your `pip` command points to the virtual environment's `pip`.)*

## Usage

After setting up the environment and installing dependencies:

1.  **Run the main application:**
    ```bash
    python main.py
    ```
    *(The specific functionality will depend on the implementation within `main.py` and other modules.)*

## Configuration

*   **`config.py`:** Contains general configuration settings for the application.
*   **`config_heartrate_prosody.json`:** Likely holds specific parameters related to heart rate and prosody analysis. You may need to adjust these files to suit your specific use case or hardware setup.

## Project Structure (Inferred)

*   `main.py`: The main entry point of the application.
*   `gui.py`: Handles the graphical user interface.
*   `audio_processing.py`: Contains logic for processing audio input.
*   `polar_monitor.py`: Potentially interfaces with Polar heart rate sensors.
*   `conversation_manager.py`: Manages the flow and logic of human-computer conversations.
*   `config.py`, `config_heartrate_prosody.json`: Configuration files.
*   `logger_utils.py`: Utilities for logging.
*   `.gitignore`: Specifies files and directories to be ignored by Git (e.g., `venv/`, `__pycache__/`, `build/`, `dist/`).

## Contributing

(If this were an open-source project, instructions on how to contribute would go here.)

## License

(Specify the project's license here.)
