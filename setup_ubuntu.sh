#!/bin/bash
# Ubuntu setup script for HCS_ver4.0

echo "Starting setup for Ubuntu..."

# --- Step 1: Update apt ---
echo "Updating apt package list..."
sudo apt-get update

# --- Step 2: Install System Dependencies ---
echo "Installing system dependencies (portaudio, bluetooth, python3-venv)..."
sudo apt-get install -y portaudio19-dev libbluetooth-dev python3-venv

# --- Step 3: Create Python Virtual Environment ---
echo "Creating Python virtual environment in 'venv'..."
python3 -m venv venv

# --- Step 4: Install Python Packages ---
echo "Installing Python packages from requirements.txt..."
# Ensure we use the pip from the created venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# --- Final Step: Instructions ---
echo ""
echo "----------------------------------------"
echo "Setup complete!"
echo ""
echo "To run the application, follow these steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the main script:"
echo "   python3 main.py"
echo ""
echo "3. When you are finished, deactivate the environment:"
echo "   deactivate"
echo "----------------------------------------"
