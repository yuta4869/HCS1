# audio_device_utils.py
"""
オーディオデバイス検索ユーティリティ
環境に依存しないデバイス名ベースのマイク検索機能を提供
"""

import sounddevice as sd
from typing import Optional, List, Dict, Any


def list_input_devices() -> List[Dict[str, Any]]:
    """
    利用可能な入力デバイス（マイク）の一覧を取得する

    Returns:
        入力デバイス情報のリスト
    """
    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append({
                'index': i,
                'name': dev['name'],
                'channels': dev['max_input_channels'],
                'sample_rate': dev['default_samplerate'],
                'is_default': i == sd.default.device[0]
            })

    return input_devices


def find_device_by_keywords(
    keywords: List[str],
    use_default_as_fallback: bool = True,
    device_type: str = "input"
) -> Optional[int]:
    """
    キーワードリストに基づいてオーディオデバイスを検索する

    Args:
        keywords: 検索キーワードのリスト（優先順位順）
        use_default_as_fallback: マッチしない場合にデフォルトデバイスを使用するか
        device_type: "input" (マイク) または "output" (スピーカー)

    Returns:
        デバイスインデックス（見つからない場合はNone）
    """
    devices = sd.query_devices()

    # 入力/出力デバイスをフィルタリング
    if device_type == "input":
        valid_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        default_device = sd.default.device[0]
    else:
        valid_devices = [(i, d) for i, d in enumerate(devices) if d['max_output_channels'] > 0]
        default_device = sd.default.device[1]

    # 会話用マイク検索時にウェブカメラを除外するキーワード
    webcam_exclude_keywords = ["webcam", "camera", "c270", "c920", "c922", "hd webcam"]

    # キーワードの優先順位順に検索
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for idx, dev in valid_devices:
            dev_name_lower = dev['name'].lower()
            if keyword_lower in dev_name_lower:
                # ウェブカメラを除外（"USB Audio"などの汎用キーワードでマッチした場合）
                is_webcam = any(wc in dev_name_lower for wc in webcam_exclude_keywords)
                if is_webcam and keyword_lower in ["usb audio", "usb", "audio"]:
                    print(f"[AudioDevice] キーワード '{keyword}' にマッチしたがウェブカメラのためスキップ: [{idx}] {dev['name']}")
                    continue
                print(f"[AudioDevice] キーワード '{keyword}' にマッチ: [{idx}] {dev['name']}")
                return idx

    # マッチしない場合
    if use_default_as_fallback:
        if default_device is not None:
            dev_info = devices[default_device]
            print(f"[AudioDevice] キーワードにマッチするデバイスなし。デフォルトを使用: [{default_device}] {dev_info['name']}")
            return default_device
        else:
            print("[AudioDevice] 警告: デフォルトデバイスが設定されていません")
            return None
    else:
        print(f"[AudioDevice] 警告: キーワード {keywords} にマッチするデバイスが見つかりません")
        return None


def get_conversation_mic_device(
    keywords: Optional[List[str]] = None,
    use_default_as_fallback: bool = True
) -> Optional[int]:
    """
    会話システム用のマイクデバイスを取得する

    Args:
        keywords: 検索キーワード（Noneの場合はconfig.pyから取得）
        use_default_as_fallback: フォールバック設定

    Returns:
        デバイスインデックス
    """
    if keywords is None:
        try:
            import config
            keywords = getattr(config, 'CONVERSATION_MIC_KEYWORDS', [])
            use_default_as_fallback = getattr(config, 'USE_DEFAULT_MIC_AS_FALLBACK', True)
        except ImportError:
            keywords = []

    if not keywords:
        # キーワード未設定の場合はデフォルトを使用
        return sd.default.device[0]

    return find_device_by_keywords(keywords, use_default_as_fallback, "input")


def get_video_mic_device(
    keywords: Optional[List[str]] = None,
    use_default_as_fallback: bool = True
) -> Optional[int]:
    """
    映像録画用のマイクデバイスを取得する

    Args:
        keywords: 検索キーワード（Noneの場合はconfig.pyから取得）
        use_default_as_fallback: フォールバック設定

    Returns:
        デバイスインデックス
    """
    if keywords is None:
        try:
            import config
            keywords = getattr(config, 'VIDEO_MIC_KEYWORDS', [])
            use_default_as_fallback = getattr(config, 'USE_DEFAULT_MIC_AS_FALLBACK', True)
        except ImportError:
            keywords = []

    if not keywords:
        # キーワード未設定の場合はデフォルトを使用
        return sd.default.device[0]

    return find_device_by_keywords(keywords, use_default_as_fallback, "input")


def print_device_info():
    """デバイス情報を表示する（デバッグ/設定確認用）"""
    print("=" * 60)
    print("利用可能なオーディオ入力デバイス（マイク）")
    print("=" * 60)

    devices = list_input_devices()
    for dev in devices:
        default_mark = " [デフォルト]" if dev['is_default'] else ""
        print(f"  [{dev['index']:2d}] {dev['name']}{default_mark}")
        print(f"       チャンネル: {dev['channels']}, サンプルレート: {dev['sample_rate']}Hz")

    print("")
    print("=" * 60)
    print("現在の設定による検索結果")
    print("=" * 60)

    try:
        import config

        print(f"\n会話システム用マイク検索キーワード: {config.CONVERSATION_MIC_KEYWORDS}")
        conv_device = get_conversation_mic_device()
        if conv_device is not None:
            dev_info = sd.query_devices(conv_device)
            print(f"  → 選択: [{conv_device}] {dev_info['name']}")

        print(f"\n映像録画用マイク検索キーワード: {config.VIDEO_MIC_KEYWORDS}")
        video_device = get_video_mic_device()
        if video_device is not None:
            dev_info = sd.query_devices(video_device)
            print(f"  → 選択: [{video_device}] {dev_info['name']}")

        print("")
        if conv_device == video_device:
            print("⚠ 警告: 会話システムと映像録画が同じマイクを使用します")
            print("   競合を避けるには、config.pyで異なるキーワードを設定してください")
        else:
            print("✓ 会話システムと映像録画が異なるマイクを使用します（競合なし）")

    except ImportError:
        print("config.pyが見つかりません")

    print("")


def get_secondary_mic_device(primary_device_index: Optional[int] = None) -> Optional[int]:
    """
    会話システム用マイク以外の2つ目のマイクを取得する
    映像録画用の音声に使用

    Args:
        primary_device_index: 会話システムが使用するマイクのインデックス

    Returns:
        2つ目のマイクのインデックス（存在しない場合はNone）
    """
    devices = sd.query_devices()
    input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

    if len(input_devices) < 2:
        print("[AudioDevice] 入力デバイスが1つしかないため、映像録音用の音声は無効")
        return None

    # プライマリデバイスが指定されていない場合は取得
    if primary_device_index is None:
        primary_device_index = get_conversation_mic_device()

    # プライマリ以外のデバイスを探す
    # まずVIDEO_MIC_KEYWORDSで検索
    try:
        import config
        video_keywords = getattr(config, 'VIDEO_MIC_KEYWORDS', [])
    except ImportError:
        video_keywords = []

    # キーワードに基づいて検索（プライマリ以外）
    for keyword in video_keywords:
        keyword_lower = keyword.lower()
        for idx, dev in input_devices:
            if idx != primary_device_index and keyword_lower in dev['name'].lower():
                print(f"[AudioDevice] 映像録音用マイク: [{idx}] {dev['name']}")
                return idx

    # キーワードにマッチしない場合、プライマリ以外の最初のデバイスを使用
    for idx, dev in input_devices:
        if idx != primary_device_index:
            print(f"[AudioDevice] 映像録音用マイク（2番目のデバイス）: [{idx}] {dev['name']}")
            return idx

    return None


# モジュールテスト用
if __name__ == "__main__":
    print_device_info()

    print("\n" + "=" * 60)
    print("セカンダリマイク検索テスト")
    print("=" * 60)
    primary = get_conversation_mic_device()
    secondary = get_secondary_mic_device(primary)
    if secondary is not None:
        print(f"✓ 2つ目のマイクが利用可能 → 映像に音声を付けられます")
    else:
        print("✗ 2つ目のマイクなし → 映像は音声なしになります")
