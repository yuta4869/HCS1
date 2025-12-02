#!/bin/bash
# macOS setup script for HCS_ver4.0

echo "Starting setup for macOS..."

# --- Step 1: Check for Homebrew ---
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install it first by following the instructions at https://brew.sh/"
    exit 1
fi

echo "Homebrew found. Updating..."
brew update

# --- Step 2: Install System Dependencies ---
echo "Installing system dependencies (portaudio)..."
brew install portaudio

# --- Step 3: Create Python Virtual Environment ---
echo "Creating Python virtual environment in '.venv'..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install it first."
    exit 1
fi
python3 -m .venv .venv

# --- Step 4: Install Python Packages ---
echo "Installing Python packages from requirements.txt..."
# Ensure we use the pip from the created .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

# --- Final Step: Instructions ---
echo ""
echo "----------------------------------------"
echo "Setup complete!"
echo ""
echo "To run the application, follow these steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run the main script:"
echo "   python3 main.py"
echo ""
echo "3. When you are finished, deactivate the environment:"
echo "   deactivate"
echo "----------------------------------------"