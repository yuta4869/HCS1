#!/bin/bash
# This script correctly sets up the environment and runs the application.

# Get the directory where the script is located to ensure we run from the right place
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Changing directory to: $SCRIPT_DIR"

VENV_PATH="./.venv/bin/activate"

if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "Please run the setup_mac.sh or setup_ubuntu.sh script first."
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH"

# Run the main python script
echo "Starting HCS_ver4.0 application..."
python main.py

echo "Application finished. Deactivating virtual environment."
deactivate
