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
python3 -m venv .venv

# --- Step 4: Install Python Packages ---
echo "Installing Python packages from requirements.txt..."
# Ensure we use the pip from the created .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

# --- Step 5: Check for Apple Silicon and install Metal version of llama-cpp-python ---
if [[ $(uname -m) == "arm64" ]]; then
    echo ""
    echo "Apple Silicon (M1/M2/M3) detected!"
    echo ""
    read -p "Do you want to install llama-cpp-python with Metal (GPU) support? (y/N): " install_metal
    if [[ "$install_metal" =~ ^[Yy]$ ]]; then
        echo "Installing llama-cpp-python with Metal support..."
        CMAKE_ARGS="-DGGML_METAL=on" ./.venv/bin/pip install llama-cpp-python --force-reinstall --no-cache-dir
        echo "Metal version of llama-cpp-python installed!"
    else
        echo "Skipping Metal installation. Using CPU version."
    fi
else
    echo "Intel Mac detected. Using CPU version of llama-cpp-python."
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
