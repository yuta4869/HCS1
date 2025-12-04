# HCS_ver4.0: Human Combersation System for Heart Rate and Prosody Analysis

## Project Overview

This project, `HCS_ver4.0`, is a Human-Computer System focusing on the analysis of heart rate data and prosody (speech rhythm, intonation, stress) to understand human states or interactions. It integrates modules for real-time audio processing, physiological monitoring, and conversational management with a graphical user interface.

## Version

v3.5.1

## Features

*   **Real-time Transcription:** Continuous audio streaming from the microphone with real-time speech-to-text using `faster-whisper`.
*   **Voice Activity Detection (VAD):** Automatically detects the start and end of speech to create complete utterances for processing.
*   **Heart Rate Monitoring:** Integration with Polar physiological sensors (via `polar_monitor.py`).
*   **Heart-rate feedback (HFB):** Modulates AI voice prosody based on user heart rate.
*   **Conversation Management:** Manages conversation history and generates replies (currently placeholder, intended for LLM integration).
*   **Graphical User Interface (GUI):** A user-friendly interface built with Tkinter (`gui.py`).
*   **Data Logging:** Comprehensive logging for conversations, heart rate, and other events for research purposes.
*   **Configuration:** Flexible configuration options via `config.py` and `config_heartrate_prosody.json`.

## Setup and Installation

This project requires Python 3.8+ and platform-specific system libraries. Automated setup scripts are provided for convenience.

### Automated Setup (Recommended)

Choose the script for your operating system. These scripts will install necessary system dependencies, create a Python virtual environment (`.venv`), and install all required Python packages.

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
    It is crucial to name the virtual environment `.venv` for the application's bootstrap script to work correctly.
    ```bash
    # Navigate to the project directory
    cd /path/to/HCS_ver4.0/HCS1

    # Create a virtual environment named .venv
    python3 -m venv .venv

    # Activate it
    source .venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    With the virtual environment activated, install all required packages from `requirements.txt`.
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

### Environment Variable

This application uses the OpenAI API for generating replies. You must set an environment variable with your API key.

```bash
export OPENAI_API_KEY="your_api_key_here"
```
You can add this line to your shell's startup file (e.g., `~/.bashrc` or `~/.zshrc`) to set it automatically.

## Usage

After setting up the environment and activating it (`source .venv/bin/activate`):

1.  **Run the main application:**
    ```bash
    python3 main.py
    ```
    The application GUI will launch. The system will be in a stand-by state, continuously listening to the microphone.

2.  **Start a conversation:**
    Click the "Start Conversation" button. When you speak, the system will detect your utterance, transcribe it, and generate a response.

3.  **Deactivate the environment** when you are finished:
    ```bash
    deactivate
    ```

### Subject Number Tagging

The GUI now contains a **セッション情報** section where you must enter the participant (被験者) number before starting a conversation.  
Only half-width alphanumeric characters, hyphen (`-`), and underscore (`_`) are allowed; other characters are ignored.  
The entered number is embedded into the filenames of conversation logs, physiological CSVs, and recorded videos so that every artifact can be traced back to a specific participant.

## Configuration

*   **`config.py`:** Contains general configuration for the application, including Whisper model settings, compute types, and logging options.
*   **`config_heartrate_prosody.json`:** Holds specific parameters related to heart rate and prosody analysis.
