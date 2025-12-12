#!/bin/bash
# build_mac.sh - macOS用ビルドスクリプト
# HCS (Heart Rate Conversation System) アプリケーションをビルドします

set -e  # エラー時に停止

echo "========================================"
echo "  HCS App Builder for macOS"
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
    echo "先に setup_mac.sh を実行してください。"
    exit 1
fi

# --- 仮想環境をアクティベート ---
echo ""
echo "2. 仮想環境をアクティベート..."
source .venv/bin/activate

# --- PyInstallerのインストール確認 ---
echo ""
echo "3. PyInstallerを確認中..."
if ! pip show pyinstaller > /dev/null 2>&1; then
    echo "   PyInstallerをインストール中..."
    pip install pyinstaller
else
    echo "   PyInstaller: インストール済み"
fi

# --- ビルドディレクトリのクリーンアップ ---
echo ""
echo "4. 前回のビルドをクリーンアップ..."
rm -rf build dist

# --- ビルド実行 ---
echo ""
echo "5. アプリケーションをビルド中..."
echo "   (これには数分かかる場合があります)"
echo ""

pyinstaller HCS_App.spec --noconfirm

# --- ビルド結果確認 ---
echo ""
echo "========================================"
if [ -d "dist/HCS_App.app" ]; then
    echo "  ビルド成功!"
    echo "========================================"
    echo ""
    echo "出力先: dist/HCS_App.app"
    echo ""
    echo "実行方法:"
    echo "  1. dist/HCS_App.app をダブルクリック"
    echo "  または"
    echo "  2. open dist/HCS_App.app"
    echo ""
    echo "注意事項:"
    echo "  - VOICEVOXが起動している必要があります"
    echo "  - ローカルLLM使用時はモデルファイルが必要です"
    echo "    (デフォルト: ./models/model.gguf)"
    echo "  - Polarセンサーを使用するにはBluetoothが必要です"
    echo ""

    # アプリサイズを表示
    APP_SIZE=$(du -sh dist/HCS_App.app | cut -f1)
    echo "アプリサイズ: $APP_SIZE"
elif [ -f "dist/HCS_App" ]; then
    echo "  ビルド成功! (実行ファイル)"
    echo "========================================"
    echo ""
    echo "出力先: dist/HCS_App"
    echo ""
    echo "実行方法: ./dist/HCS_App"

    # ファイルサイズを表示
    FILE_SIZE=$(du -sh dist/HCS_App | cut -f1)
    echo "ファイルサイズ: $FILE_SIZE"
else
    echo "  ビルド失敗"
    echo "========================================"
    echo ""
    echo "エラーログを確認してください。"
    exit 1
fi
