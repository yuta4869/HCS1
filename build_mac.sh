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

    # --- DMG作成 ---
    echo ""
    read -p "DMGファイルを作成しますか? (y/N): " create_dmg
    if [[ "$create_dmg" =~ ^[Yy]$ ]]; then
        echo ""
        echo "6. DMGファイルを作成中..."

        # create-dmg がインストールされているか確認
        if ! command -v create-dmg &> /dev/null; then
            echo "   create-dmg をインストール中..."
            brew install create-dmg
        fi

        # バージョン取得（HCS_App.spec から）
        VERSION=$(grep -o "CFBundleShortVersionString.*" HCS_App.spec | grep -o "'[0-9.]*'" | tr -d "'")
        if [ -z "$VERSION" ]; then
            VERSION="1.0.0"
        fi

        DMG_NAME="HCS_App_${VERSION}_macOS.dmg"

        # 既存のDMGを削除
        rm -f "dist/$DMG_NAME"

        # DMG作成
        create-dmg \
            --volname "HCS App $VERSION" \
            --volicon "dist/HCS_App.app/Contents/Resources/icon.icns" 2>/dev/null || true

        # create-dmg でDMG作成（シンプル版）
        create-dmg \
            --volname "HCS App" \
            --window-size 600 400 \
            --icon "HCS_App.app" 150 185 \
            --app-drop-link 450 185 \
            --no-internet-enable \
            "dist/$DMG_NAME" \
            "dist/HCS_App.app" 2>/dev/null || {
                # create-dmg が失敗した場合、hdiutil で作成
                echo "   create-dmg が失敗したため、hdiutil で作成します..."
                hdiutil create -volname "HCS App" -srcfolder "dist/HCS_App.app" -ov -format UDZO "dist/$DMG_NAME"
            }

        if [ -f "dist/$DMG_NAME" ]; then
            DMG_SIZE=$(du -sh "dist/$DMG_NAME" | cut -f1)
            echo ""
            echo "   DMG作成完了: dist/$DMG_NAME"
            echo "   DMGサイズ: $DMG_SIZE"
            echo ""
            echo "   GitHub Releases へのアップロード:"
            echo "   gh release create v$VERSION dist/$DMG_NAME --title \"v$VERSION\""
        else
            echo "   DMG作成に失敗しました"
        fi
    fi

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
