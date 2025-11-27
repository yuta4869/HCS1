# HCS_ver4.0: Human Computer System for Heart Rate and Prosody Analysis

## Project Overview

This project, `HCS_ver4.0`, appears to be a Human Computer System focusing on the analysis of heart rate data and prosody (speech rhythm, intonation, stress) to understand human states or interactions. It likely integrates various modules for audio processing, physiological monitoring, and conversational management, potentially with a graphical user interface.

## Version

v1.0.1

## Features (Inferred)

*   **Heart Rate Monitoring:** Integration with physiological sensors (e.g., Polar devices via `polar_monitor.py`).
*   **Audio Processing:** Analysis of speech prosody, potentially through `audio_processing.py`.
*   **Conversation Management:** Logic for managing interactions, possibly handled by `conversation_manager.py`.
*   **Graphical User Interface (GUI):** A user-friendly interface built with `gui.py`.
*   **Logging:** Utilities for logging various data and events (`logger_utils.py`).
*   **Configuration:** Flexible configuration options via `config.py` and `config_heartrate_prosody.json`.

## Setup and Installation

This project requires Python 3.x and platform-specific system libraries. Automated setup scripts are provided for convenience.

### Automated Setup (Recommended)

Choose the script for your operating system. These scripts will install necessary system dependencies, create a Python virtual environment (`venv`), and install all required Python packages.

**For macOS:**
1. Make the script executable:
   ```bash
   chmod +x setup_mac.sh
   ```
2. Run the script:
   ```bash
   ./setup_mac.sh
   ```

**For Ubuntu:**
1. Make the script executable:
   ```bash
   chmod +x setup_ubuntu.sh
   ```
2. Run the script:
   ```bash
   ./setup_ubuntu.sh
   ```

After the setup is complete, follow the activation instructions printed by the script to start using the application.

---

### Manual Installation

If you prefer to set up the environment manually, follow these steps.

1.  **Install System Dependencies:**
    *   **On macOS:** You need [Homebrew](https://brew.sh/).
        ```bash
        brew install portaudio
        ```
    *   **On Ubuntu:**
        ```bash
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev libbluetooth-dev python3-venv
        ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Navigate to the project directory
    cd /path/to/HCS_ver4.0

    # Create a virtual environment
    python3 -m venv venv

    # Activate it
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    With the virtual environment activated, install all required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

After setting up the environment and activating it (`source venv/bin/activate`):

1.  **Run the main application:**
    ```bash
    python main.py
    ```
    *(The specific functionality will depend on the implementation within `main.py` and other modules.)*

2.  **Deactivate the environment** when you are finished:
    ```bash
    deactivate
    ```

## Configuration

*   **`config.py`:** Contains general configuration settings for the application.
*   **`config_heartrate_prosody.json`:** Likely holds specific parameters related to heart rate and prosody analysis. You may need to adjust these files to suit your specific use case or hardware setup.

## Project Structure

*   `main.py`: The main entry point of the application.
*   `gui.py`: Handles the graphical user interface.
*   `audio_processing.py`: Contains logic for processing audio input.
*   ... and other project files.

## Contributing

(If this were an open-source project, instructions on how to contribute would go here.)

## License

(Specify the project's license here.)