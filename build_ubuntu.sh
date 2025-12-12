#!/bin/bash
# build_ubuntu.sh - Ubuntu用ビルドスクリプト
# HCS (Heart Rate Conversation System) アプリケーションをビルドします

set -e  # エラー時に停止

echo "========================================"
echo "  HCS App Builder for Ubuntu"
echo "========================================"
echo ""

# --- 環境確認 ---
echo "1. 環境を確認中..."

# Python確認
if ! command -v python3 &> /dev/null; then
    echo "エラー: Python 3が見つかりません。"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "   Python: $PYTHON_VERSION"

# 仮想環境確認
if [ ! -d ".venv" ]; then
    echo "エラー: 仮想環境 (.venv) が見つかりません。"
    echo "先に setup_ubuntu.sh を実行してください。"
    exit 1
fi

# --- 追加の依存関係をインストール ---
echo ""
echo "2. システム依存関係を確認中..."

# binutils (strip コマンド用)
if ! command -v strip &> /dev/null; then
    echo "   binutils をインストール中..."
    sudo apt-get install -y binutils
fi

# --- 仮想環境をアクティベート ---
echo ""
echo "3. 仮想環境をアクティベート..."
source .venv/bin/activate

# --- PyInstallerのインストール確認 ---
echo ""
echo "4. PyInstallerを確認中..."
if ! pip show pyinstaller > /dev/null 2>&1; then
    echo "   PyInstallerをインストール中..."
    pip install pyinstaller
else
    echo "   PyInstaller: インストール済み"
fi

# --- ビルドディレクトリのクリーンアップ ---
echo ""
echo "5. 前回のビルドをクリーンアップ..."
rm -rf build dist

# --- ビルド実行 ---
echo ""
echo "6. アプリケーションをビルド中..."
echo "   (これには数分かかる場合があります)"
echo ""

pyinstaller HCS_App.spec --noconfirm

# --- ビルド結果確認 ---
echo ""
echo "========================================"
if [ -f "dist/HCS_App" ]; then
    echo "  ビルド成功!"
    echo "========================================"
    echo ""
    echo "出力先: dist/HCS_App"
    echo ""
    echo "実行方法:"
    echo "  ./dist/HCS_App"
    echo ""
    echo "注意事項:"
    echo "  - VOICEVOXが起動している必要があります"
    echo "  - ローカルLLM使用時はモデルファイルが必要です"
    echo "    (デフォルト: ./models/model.gguf)"
    echo "  - Polarセンサーを使用するにはBluetoothが必要です"
    echo "    (sudo権限が必要な場合があります)"
    echo ""

    # ファイルサイズを表示
    FILE_SIZE=$(du -sh dist/HCS_App | cut -f1)
    echo "ファイルサイズ: $FILE_SIZE"

    # 実行可能にする
    chmod +x dist/HCS_App
    echo ""
    echo "実行権限を付与しました。"
else
    echo "  ビルド失敗"
    echo "========================================"
    echo ""
    echo "エラーログを確認してください。"
    exit 1
fi

# --- オプション: デスクトップエントリ作成 ---
echo ""
read -p "デスクトップショートカットを作成しますか? (y/N): " create_desktop
if [[ "$create_desktop" =~ ^[Yy]$ ]]; then
    DESKTOP_FILE="$HOME/.local/share/applications/hcs-app.desktop"
    APP_PATH="$(pwd)/dist/HCS_App"

    mkdir -p "$HOME/.local/share/applications"

    cat > "$DESKTOP_FILE" << EOL
[Desktop Entry]
Name=HCS App
Comment=Heart Rate Conversation System
Exec=$APP_PATH
Terminal=true
Type=Application
Categories=Science;Education;
EOL

    echo "デスクトップエントリを作成しました: $DESKTOP_FILE"
fi
