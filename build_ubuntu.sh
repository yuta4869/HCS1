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

# --- オプション: DEBパッケージ作成 ---
echo ""
read -p "DEBパッケージを作成しますか? (y/N): " create_deb
if [[ "$create_deb" =~ ^[Yy]$ ]]; then
    echo ""
    echo "7. DEBパッケージを作成中..."

    # バージョン取得（HCS_App.spec から）
    VERSION=$(grep -o "CFBundleShortVersionString.*" HCS_App.spec | grep -o "'[0-9.]*'" | tr -d "'")
    if [ -z "$VERSION" ]; then
        VERSION="1.0.0"
    fi

    # アーキテクチャ検出
    ARCH=$(dpkg --print-architecture 2>/dev/null || echo "amd64")

    DEB_NAME="hcs-app_${VERSION}_${ARCH}"
    DEB_DIR="dist/$DEB_NAME"

    # DEBパッケージ用ディレクトリ構造を作成
    rm -rf "$DEB_DIR"
    mkdir -p "$DEB_DIR/DEBIAN"
    mkdir -p "$DEB_DIR/usr/local/bin"
    mkdir -p "$DEB_DIR/usr/share/applications"
    mkdir -p "$DEB_DIR/usr/share/doc/hcs-app"

    # 実行ファイルをコピー
    if [ -d "dist/HCS_App" ]; then
        # onedirモードの場合
        mkdir -p "$DEB_DIR/opt/hcs-app"
        cp -r dist/HCS_App/* "$DEB_DIR/opt/hcs-app/"
        # シンボリックリンクを作成
        ln -sf /opt/hcs-app/HCS_App "$DEB_DIR/usr/local/bin/hcs-app"
    else
        # onefileモードの場合
        cp dist/HCS_App "$DEB_DIR/usr/local/bin/hcs-app"
    fi
    chmod +x "$DEB_DIR/usr/local/bin/hcs-app" 2>/dev/null || true

    # control ファイル作成
    cat > "$DEB_DIR/DEBIAN/control" << EOF
Package: hcs-app
Version: $VERSION
Section: science
Priority: optional
Architecture: $ARCH
Depends: libc6, libgtk-3-0, libasound2, libpulse0
Maintainer: HCS Developer <hcs@example.com>
Description: Heart Rate Conversation System
 A heart rate feedback conversation system with voice recognition,
 text-to-speech, and Polar sensor integration.
 .
 Features:
  - Voice recognition (Whisper)
  - Text-to-speech (VOICEVOX)
  - Heart rate monitoring (Polar Verity Sense, H10)
  - LLM integration (Local or OpenAI)
EOF

    # postinst スクリプト（インストール後の処理）
    cat > "$DEB_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
echo "HCS App がインストールされました。"
echo "実行: hcs-app"
echo ""
echo "注意: VOICEVOXが必要です。Dockerで起動してください:"
echo "  docker run -d --name voicevox -p 50021:50021 voicevox/voicevox_engine:cpu-ubuntu20.04-latest"
EOF
    chmod +x "$DEB_DIR/DEBIAN/postinst"

    # デスクトップエントリ
    cat > "$DEB_DIR/usr/share/applications/hcs-app.desktop" << EOF
[Desktop Entry]
Name=HCS App
Comment=Heart Rate Conversation System
Exec=hcs-app
Terminal=true
Type=Application
Categories=Science;Education;Medical;
Keywords=heart;rate;conversation;voice;
EOF

    # copyright ファイル
    cat > "$DEB_DIR/usr/share/doc/hcs-app/copyright" << EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: hcs-app
Source: https://github.com/yuta4869/HCS1

Files: *
Copyright: $(date +%Y) HCS Developers
License: MIT
EOF

    # DEBパッケージをビルド
    dpkg-deb --build "$DEB_DIR" "dist/${DEB_NAME}.deb"

    if [ -f "dist/${DEB_NAME}.deb" ]; then
        DEB_SIZE=$(du -sh "dist/${DEB_NAME}.deb" | cut -f1)
        echo ""
        echo "   DEB作成完了: dist/${DEB_NAME}.deb"
        echo "   DEBサイズ: $DEB_SIZE"
        echo ""
        echo "   インストール方法:"
        echo "   sudo dpkg -i dist/${DEB_NAME}.deb"
        echo ""
        echo "   GitHub Releases へのアップロード:"
        echo "   gh release create v$VERSION dist/${DEB_NAME}.deb --title \"v$VERSION\""

        # 一時ディレクトリを削除
        rm -rf "$DEB_DIR"
    else
        echo "   DEB作成に失敗しました"
    fi
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
