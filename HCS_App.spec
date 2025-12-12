# -*- mode: python ; coding: utf-8 -*-
# HCS_App.spec - PyInstaller spec file for Heart Rate Conversation System
# Build: pyinstaller HCS_App.spec

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 必要なデータファイル
datas = [
    ('config_heartrate_prosody.json', '.'),
]

# hidden imports - 動的にインポートされるモジュール
hiddenimports = [
    # GUI関連
    'gui',
    'gui.application',
    'gui.status_window',
    'gui.ecg_analysis',
    'gui.questionnaire_analysis',
    # 音声処理
    'sounddevice',
    'soundfile',
    # Whisper
    'faster_whisper',
    'ctranslate2',
    # LLM
    'llama_cpp',
    'llama_cpp.server',
    # Bluetooth
    'bleak',
    # 科学計算
    'scipy',
    'scipy.signal',
    'scipy.interpolate',
    'numpy',
    'pandas',
    # グラフ
    'matplotlib',
    'matplotlib.backends.backend_tkagg',
    'seaborn',
    'japanize_matplotlib',
    # OpenAI
    'openai',
    # その他
    'requests',
    'openpyxl',
    'cv2',
    'torch',
]

# 追加の隠しインポートを収集
hiddenimports += collect_submodules('bleak')
hiddenimports += collect_submodules('faster_whisper')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter.test',
        # 'unittest',  # torch が内部で unittest を必要とするため除外しない
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# onedirモード（macOSの.appバンドルとの互換性）
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HCS_App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # デバッグ用にコンソール表示
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HCS_App',
)

# macOS用 .app バンドル
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='HCS_App.app',
        icon=None,
        bundle_identifier='com.hcs.heartrate-conversation',
        info_plist={
            'NSMicrophoneUsageDescription': 'This app requires microphone access for voice recognition.',
            'NSBluetoothAlwaysUsageDescription': 'This app requires Bluetooth access for heart rate monitoring.',
            'NSBluetoothPeripheralUsageDescription': 'This app requires Bluetooth access for heart rate monitoring.',
            'CFBundleShortVersionString': '6.5.0',
            'NSSupportsSuddenTermination': False,  # 突然の終了を許可しない
            'NSSupportsAutomaticTermination': False,  # 自動終了を許可しない
        },
    )
