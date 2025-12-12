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
echo "Creating Python virtual environment in '.venv'..."
python3 -m venv .venv

# --- Step 4: Install Python Packages ---
echo "Installing Python packages from requirements.txt..."
# Ensure we use the pip from the created .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

# --- Step 5: Check for NVIDIA GPU and install CUDA version of llama-cpp-python ---
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo ""
    read -p "Do you want to install llama-cpp-python with CUDA support? (y/N): " install_cuda
    if [[ "$install_cuda" =~ ^[Yy]$ ]]; then
        echo "Installing llama-cpp-python with CUDA support..."
        CMAKE_ARGS="-DGGML_CUDA=on" ./.venv/bin/pip install llama-cpp-python --force-reinstall --no-cache-dir
        echo "CUDA version of llama-cpp-python installed!"
    else
        echo "Skipping CUDA installation. Using CPU version."
    fi
else
    echo "No NVIDIA GPU detected. Using CPU version of llama-cpp-python."
fi

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
echo ""
echo "--- Optional: Local LLM Setup ---"
echo "If you want to use a local LLM instead of OpenAI API:"
echo "1. Download a GGUF model:"
echo "   huggingface-cli download mmnga/Llama-3-ELYZA-JP-8B-gguf Llama-3-ELYZA-JP-8B-q4_k_m.gguf --local-dir ./models"
echo "2. Edit config.py:"
echo "   - Set USE_LOCAL_LLM = True"
echo "   - Set LOCAL_LLM_MODEL_PATH to your model file"
echo "----------------------------------------"
