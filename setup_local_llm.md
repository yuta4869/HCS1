# ローカルLLM セットアップガイド

このガイドでは、HCS ver4.0 で Youri 7B Chat (rinna) をローカルで実行するための設定方法を説明します。

## 概要

- **モデル**: [Youri 7B Chat](https://huggingface.co/rinna/youri-7b-chat) (rinna)
- **フォーマット**: GGUF (llama.cpp 用に量子化)
- **ライブラリ**: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## 1. llama-cpp-python のインストール

### Mac (Apple Silicon / Metal)

Metal (GPU) アクセラレーションを有効にしてインストール:

```bash
# 仮想環境を有効化
cd /path/to/HCS_ver4.0
source .venv/bin/activate

# Metal サポート付きでインストール
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Mac (Intel)

Intel Mac の場合は CPU のみで実行:

```bash
source .venv/bin/activate
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Ubuntu (CUDA / NVIDIA GPU)

CUDA を使用する場合:

```bash
# 仮想環境を有効化
source .venv/bin/activate

# CUDA サポート付きでインストール (CUDA 12.x の場合)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# CUDA 11.x の場合は以下を使用
# CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Ubuntu (CPU のみ)

GPU がない場合:

```bash
source .venv/bin/activate
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## 2. モデルのダウンロード

### ダウンロード先

GGUF フォーマットのモデルは以下から入手できます:
- [mmnga/rinna-youri-7b-chat-gguf](https://huggingface.co/mmnga/rinna-youri-7b-chat-gguf)

### 推奨モデル

| 量子化 | ファイルサイズ | 品質 | 速度 | 推奨環境 |
|--------|---------------|------|------|----------|
| q4_K_M | ~4.4 GB | 良好 | 速い | **推奨** (バランス型) |
| q5_K_M | ~5.1 GB | より良い | 中程度 | メモリに余裕がある場合 |
| q8_0 | ~7.7 GB | 最高 | 遅い | 高品質が必要な場合 |
| q3_K_M | ~3.5 GB | 低め | 最速 | メモリが限られる場合 |

### ダウンロード手順

```bash
# models ディレクトリを作成
cd /path/to/HCS_ver4.0
mkdir -p models
cd models

# Hugging Face CLI でダウンロード (推奨)
pip install huggingface_hub
huggingface-cli download mmnga/rinna-youri-7b-chat-gguf rinna-youri-7b-chat-q4_K_M.gguf --local-dir .

# または wget で直接ダウンロード
# wget https://huggingface.co/mmnga/rinna-youri-7b-chat-gguf/resolve/main/rinna-youri-7b-chat-q4_K_M.gguf
```

## 3. config.py の設定

`config.py` で以下の設定を確認・調整してください:

```python
# LLMバックエンドを "local" に設定
LLM_BACKEND = "local"

# モデルファイルのパス
LOCAL_LLM_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rinna-youri-7b-chat-q4_K_M.gguf")

# GPU レイヤー数
# -1 = 全てのレイヤーをGPUに載せる (推奨)
#  0 = CPU のみ
# 20 = 一部のレイヤーのみGPU
LOCAL_LLM_N_GPU_LAYERS = -1
```

### 環境別の推奨設定

#### Mac (Apple Silicon M1/M2/M3)
```python
LOCAL_LLM_N_GPU_LAYERS = -1  # Metal で全レイヤー処理
LOCAL_LLM_N_CTX = 2048
```

#### Ubuntu (NVIDIA GPU, 8GB+ VRAM)
```python
LOCAL_LLM_N_GPU_LAYERS = -1  # CUDA で全レイヤー処理
LOCAL_LLM_N_CTX = 2048
```

#### CPU のみ
```python
LOCAL_LLM_N_GPU_LAYERS = 0   # CPU のみ
LOCAL_LLM_N_CTX = 1024       # メモリ節約のためコンテキスト長を短く
LOCAL_LLM_MAX_TOKENS = 100   # 生成トークン数を制限
```

## 4. 動作確認

```bash
cd /path/to/HCS_ver4.0
source .venv/bin/activate
python main.py
```

起動時に以下のようなメッセージが表示されれば成功です:

```
LLMバックエンド: ローカルLLM (llama-cpp-python)
ローカルLLMモデルをロード中: /path/to/models/rinna-youri-7b-chat-q4_K_M.gguf
  - コンテキスト長: 2048
  - GPUレイヤー数: -1
ローカルLLMモデルのロード完了
```

## 5. トラブルシューティング

### モデルファイルが見つからない

```
FileNotFoundError: モデルファイルが見つかりません: /path/to/models/rinna-youri-7b-chat-q4_K_M.gguf
```

→ `models/` ディレクトリにモデルファイルをダウンロードしてください。

### llama-cpp-python がインストールできない

```
error: command 'cmake' failed
```

→ CMake をインストールしてください:
- Mac: `brew install cmake`
- Ubuntu: `sudo apt install cmake`

### Metal が認識されない (Mac)

```bash
# インストール時のログを確認
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python -v

# Metal Framework が存在するか確認
ls /System/Library/Frameworks/Metal.framework
```

### CUDA が認識されない (Ubuntu)

```bash
# CUDA が正しくインストールされているか確認
nvcc --version
nvidia-smi

# CUDA パスを設定
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### メモリ不足

```
RuntimeError: CUDA out of memory
```

→ `config.py` で以下を調整:
```python
LOCAL_LLM_N_GPU_LAYERS = 20  # GPU に載せるレイヤー数を減らす
LOCAL_LLM_N_CTX = 1024       # コンテキスト長を短くする
```

## 6. OpenAI API に戻す方法

ローカルLLMではなく OpenAI API を使用したい場合は、`config.py` で以下を変更:

```python
LLM_BACKEND = "openai"  # "local" から "openai" に変更
```

環境変数 `OPENAI_API_KEY` を設定することを忘れないでください。

## 参考リンク

- [rinna/youri-7b-chat](https://huggingface.co/rinna/youri-7b-chat) - 元モデル
- [mmnga/rinna-youri-7b-chat-gguf](https://huggingface.co/mmnga/rinna-youri-7b-chat-gguf) - GGUF 変換版
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python バインディング
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 推論エンジン
