# gui/ecg_analysis.py
"""ECG/HRV解析関連のヘルパー関数"""

import os
import re
from itertools import accumulate

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as mpatches
import seaborn as sns
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

# 解析用の定数
_ECG_CONDITIONS = [
    "Fixed",
    "HRF",
    "Sin",
    "HRF2_PID",
    "HRF2_Adaptive",
    "HRF2_GS",
    "HRF2_Robust",
]
_FILENAME_COND_PATTERN = "|".join(_ECG_CONDITIONS)
# ECGファイルのみを対象とする（h10_ecg_sessionで始まるファイル）
FILENAME_PATTERN = re.compile(
    rf"h10_ecg_session_No(?P<subject>\d+)_\d{{8}}_\d{{6}}_(?P<condition>{_FILENAME_COND_PATTERN})\.csv$",
    re.IGNORECASE,
)
ANALYS_CONDITION_MAP = {
    "sin": "Sin",
    "fixed": "Fixed",
    "hrf": "HRF",
    "hrf2_pid": "HRF2_PID",
    "hrf2_adaptive": "HRF2_Adaptive",
    "hrf2_gs": "HRF2_GS",
    "hrf2_gainscheduled": "HRF2_GS",
    "hrf2_robust": "HRF2_Robust",
}
ANALYS_CONDITION_ORDER = _ECG_CONDITIONS

DEFAULT_CONDITION_LABELS = {
    'Fixed': '固定会話',
    'Sin': '正弦波',
    'HRF': '調整会話',
    'HRF2_PID': 'HRF2 (PID)',
    'HRF2_Adaptive': 'HRF2 (Adaptive)',
    'HRF2_GS': 'HRF2 (GS)',
    'HRF2_Robust': 'HRF2 (Robust)',
}

CONDITION_COLORS = {
    'Fixed': 'lightcoral',
    'Sin': 'lightblue',
    'HRF': 'lightyellow',
    'HRF2_PID': '#ffcc99',
    'HRF2_Adaptive': '#b2d8b2',
    'HRF2_GS': '#c9c3ff',
    'HRF2_Robust': '#f4b6c2',
}


def bandpass_filter(signal_data, fs):
    """バンドパスフィルタ (0.5Hz - 50Hz)"""
    from scipy.signal import butter, filtfilt
    # 5次バターワースフィルタには最低 3 * (5 * 2 + 1) = 33 サンプル必要
    min_samples = 34
    if len(signal_data) < min_samples:
        raise ValueError(
            f"データが短すぎます: {len(signal_data)}サンプル（最低{min_samples}サンプル≒{min_samples/fs:.2f}秒必要）"
        )
    lowcut = 0.5
    highcut = 50
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal_data)


def ecg_to_rri(file_path, fs=130, analysis_start_offset=None, analysis_end_offset=None):
    """ECGデータからRRIを算出する"""
    from scipy.signal import find_peaks
    try:
        data = pd.read_csv(file_path, delimiter=',', encoding="utf-8", skiprows=1, header=None, usecols=[0, 2], names=['timestamp', 'ecg'])
    except ValueError:
        print(f"エラー: {os.path.basename(file_path)} の読み込みに失敗しました。")
        return np.array([]), None, None

    # データが空の場合（ヘッダーのみ）のチェック
    if data.empty or len(data) == 0:
        print(f"エラー: {os.path.basename(file_path)} にデータがありません（H10接続断の可能性）。")
        return np.array([]), None, None

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    start_time_real = data['timestamp'].iloc[0]
    end_time_real = data['timestamp'].iloc[-1]
    duration_seconds = (end_time_real - start_time_real).total_seconds()

    custom_window = analysis_start_offset is not None or analysis_end_offset is not None

    if custom_window:
        start_offset = max(0.0, analysis_start_offset or 0.0)
        analysis_start_time = start_time_real + pd.Timedelta(seconds=start_offset)
        if analysis_end_offset is not None:
            if analysis_end_offset <= start_offset:
                print("  -> エラー: 解析終了時刻が開始時刻以下です。")
                return np.array([]), None, None
            analysis_end_time = start_time_real + pd.Timedelta(seconds=analysis_end_offset)
        else:
            analysis_end_time = end_time_real

        if analysis_start_time >= end_time_real:
            print("  -> エラー: 指定された解析開始時刻がデータ範囲外です。")
            return np.array([]), None, None

        if analysis_end_time > end_time_real:
            analysis_end_time = end_time_real

        if analysis_end_time <= analysis_start_time:
            print("  -> エラー: 指定された解析区間が無効です。")
            return np.array([]), None, None

        data_filtered = data[(data['timestamp'] >= analysis_start_time) & (data['timestamp'] <= analysis_end_time)]
        actual_end_offset = (analysis_end_time - start_time_real).total_seconds()
        print(f"  -> 指定区間で解析: {start_offset:.1f}秒 〜 {actual_end_offset:.1f}秒")
    else:
        if duration_seconds < 60:
            print(f"  -> 短いデータを検出 ({duration_seconds:.1f}秒): 全期間を解析します")
            analysis_start_time = start_time_real
            analysis_end_time = end_time_real
            data_filtered = data
        else:
            print(f"  -> 長いデータを検出 ({duration_seconds:.1f}秒): 30秒後から5分間を解析します")
            analysis_start_time = start_time_real + pd.Timedelta(seconds=30)
            analysis_end_time = start_time_real + pd.Timedelta(minutes=5, seconds=30)
            data_filtered = data[(data['timestamp'] >= analysis_start_time) & (data['timestamp'] <= analysis_end_time)]

    if data_filtered.empty:
        print("  -> エラー: 解析対象期間のデータが空です。")
        return np.array([]), None, None

    ecg_data = data_filtered['ecg'].values
    filtered_ecg = bandpass_filter(ecg_data, fs)
    diff_ecg = np.diff(filtered_ecg)
    squared_ecg = diff_ecg ** 2
    window_size = int(0.150 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_size) / window_size, mode='same')
    height_threshold = np.mean(integrated_ecg) * 0.4
    distance = fs * 0.3
    peaks, _ = find_peaks(integrated_ecg, distance=distance, height=height_threshold)
    rri_data = np.diff(peaks) * 1000 / fs

    return rri_data, analysis_start_time, analysis_end_time


def calculate_hrv_indices(
    file_path,
    label,
    fs=130,
    analysis_start_offset=None,
    analysis_end_offset=None,
    resampling_freq=1.0,
    quantile_low=0.038,
    quantile_high=0.962,
    min_hr=45.0,
    max_hr=210.0,
    analysis_window_seconds=30.0,
):
    """指定されたファイルを解析し、時系列データと全体LF/HF値を返す"""
    from scipy import interpolate
    import scipy.signal

    print(f"--- 解析開始: {label} ({os.path.basename(file_path)}) ---")
    rri_data, start_time, end_time = ecg_to_rri(
        file_path,
        fs,
        analysis_start_offset=analysis_start_offset,
        analysis_end_offset=analysis_end_offset
    )

    if len(rri_data) == 0:
        return None, None

    if resampling_freq <= 0:
        resampling_freq = 1.0
    quantile_low = max(0.0, min(quantile_low, 1.0))
    quantile_high = max(0.0, min(quantile_high, 1.0))
    if quantile_high <= quantile_low:
        quantile_low, quantile_high = 0.038, 0.962
    if min_hr <= 0:
        min_hr = 45.0
    if max_hr <= min_hr:
        max_hr = min_hr + 1.0
    if analysis_window_seconds <= 0:
        analysis_window_seconds = 30.0

    time_data = list(accumulate(rri_data / 1000))
    if not time_data:
        return None, None

    duration_total = time_data[-1]
    time_axis = np.arange(0, duration_total, 1 / resampling_freq)

    if len(time_data) < 4:
        print("  -> データ点が少なすぎるためスキップします。")
        return None, None

    spline_func = interpolate.interp1d(time_data, rri_data, fill_value="extrapolate", kind='cubic')
    rri = spline_func(time_axis)

    min_samples_for_quantile = int(max(resampling_freq * 10, 10))
    if len(rri) > min_samples_for_quantile:
        low_threshold = np.quantile(rri, quantile_low)
        high_threshold = np.quantile(rri, quantile_high)
        rri[(rri < low_threshold) | (rri > high_threshold)] = np.nan

    df_temp = pd.DataFrame(data=rri, index=time_axis, columns=["rri"])
    df_temp.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
    rri = df_temp["rri"].values

    rri[(rri > 60000 / min_hr) | (rri < 60000 / max_hr)] = np.nan
    df_temp = pd.DataFrame(data=rri, index=time_axis, columns=["rri"])
    df_temp.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
    rri = df_temp["rri"].values

    if len(rri) > 1:
        prerri = np.roll(rri, 1)
        prerri[0] = rri[0]
        safe_prerri = np.where(prerri == 0, 1, prerri)
        change_ratio = rri / safe_prerri
        rri[(change_ratio < 0.7) | (change_ratio > 1.3)] = np.nan
        df_temp = pd.DataFrame(data=rri, index=time_axis, columns=["rri"])
        df_temp.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
        rri = df_temp["rri"].values

    N_total = len(rri)
    dt_total = 1 / resampling_freq
    window_total = scipy.signal.windows.hann(N_total)
    F_total = np.fft.fft(rri * window_total)
    freq_total = np.fft.fftfreq(N_total, d=dt_total)
    Amp_total = np.abs(F_total / (N_total / 2))

    lf_mask_total = (freq_total >= 0.04) & (freq_total < 0.15)
    hf_mask_total = (freq_total >= 0.15) & (freq_total < 0.4)
    LF_total = np.sum(Amp_total[lf_mask_total])
    HF_total = np.sum(Amp_total[hf_mask_total])
    overall_lf_hf_val = LF_total / HF_total if HF_total != 0 else 0
    print(f"  -> 全体LF/HF値: {overall_lf_hf_val:.4f}")

    target_window_samples = max(int(round(analysis_window_seconds * resampling_freq)), 1)
    min_window_samples = max(int(round(10 * resampling_freq)), 1)

    if len(rri) >= target_window_samples:
        analysis_window = target_window_samples
    elif len(rri) >= min_window_samples:
        analysis_window = len(rri)
        actual_seconds = analysis_window / resampling_freq
        print(f"  -> データが短いため、ウィンドウサイズを {actual_seconds:.1f}秒 に短縮して解析します。")
    else:
        print("  -> データが短すぎて解析できません（約10秒未満）。")
        return None, None

    LF_HF_sliding = []
    RMSSD_sliding = []
    SDNN_sliding = []
    time_points = []

    i = 0
    while i <= (len(rri) - analysis_window):
        rri_window = rri[i: analysis_window + i]

        N = len(rri_window)
        dt = 1 / resampling_freq
        window = scipy.signal.windows.hann(N)
        F = np.fft.fft(rri_window * window)
        freq = np.fft.fftfreq(N, d=dt)
        Amp = np.abs(F / (N / 2))

        lf_mask = (freq >= 0.04) & (freq < 0.15)
        hf_mask = (freq >= 0.15) & (freq < 0.4)
        LF = np.sum(Amp[lf_mask])
        HF = np.sum(Amp[hf_mask])
        lf_hf_val = LF / HF if HF != 0 else 0
        LF_HF_sliding.append(lf_hf_val)

        if len(rri_window) > 1:
            # RMSSD: Root Mean Square of Successive Differences
            diff_rri = np.diff(rri_window)
            mssd = np.mean(np.square(diff_rri))
            rmssd_val = np.sqrt(mssd)
            RMSSD_sliding.append(rmssd_val)

            # SDNN: Standard Deviation of NN intervals
            sdnn_val = np.std(rri_window)
            SDNN_sliding.append(sdnn_val)
        else:
            RMSSD_sliding.append(np.nan)
            SDNN_sliding.append(np.nan)

        time_points.append(i)
        i += 1

    sliding_result_df = pd.DataFrame({
        'Time': time_points,
        'LF/HF': LF_HF_sliding,
        'RMSSD': RMSSD_sliding,
        'SDNN': SDNN_sliding
    })

    return sliding_result_df, overall_lf_hf_val


def run_batch_analysis(
    files_map,
    output_dir,
    analysis_start_offset=None,
    analysis_end_offset=None,
    sensor_sample_rate=130.0,
    resampling_freq=1.0,
    quantile_low=0.038,
    quantile_high=0.962,
    min_hr=45.0,
    max_hr=210.0,
    analysis_window_seconds=30.0,
    subject_id=None,
):
    """バッチ解析を実行し、結果をExcelファイルに保存する

    Args:
        subject_id: 被験者ID（出力ファイル名に含める。Noneの場合はフォルダ名から推測）
    """
    os.makedirs(output_dir, exist_ok=True)
    combined_df = None
    print("=== バッチ解析を開始します ===")

    # subject_idが指定されていない場合、出力フォルダ名から推測
    if subject_id is None:
        folder_name = os.path.basename(output_dir.rstrip('/\\'))
        # No1, No2 などのパターンをチェック
        match = re.match(r'(No\d+)', folder_name, re.IGNORECASE)
        if match:
            subject_id = match.group(1)
        else:
            subject_id = folder_name

    for label, filename in files_map.items():
        file_path = filename
        if not os.path.exists(file_path):
            print(f"警告: ファイルが見つかりません -> {file_path}")
            continue

        sliding_df, overall_lfhf = calculate_hrv_indices(
            file_path,
            label,
            fs=sensor_sample_rate,
            analysis_start_offset=analysis_start_offset,
            analysis_end_offset=analysis_end_offset,
            resampling_freq=resampling_freq,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            min_hr=min_hr,
            max_hr=max_hr,
            analysis_window_seconds=analysis_window_seconds,
        )

        if sliding_df is not None and not sliding_df.empty:
            # 被験者番号を含むファイル名: {subject_id}_{condition}_result.xlsx
            sliding_output_path = os.path.join(output_dir, f"{subject_id}_{label}_result.xlsx")
            sliding_df.to_excel(sliding_output_path, index=False)
            print(f"  -> 時系列結果を保存: {os.path.basename(sliding_output_path)}")

            overall_output_path = os.path.join(output_dir, f"{subject_id}_{label}_resultLFHF5min.xlsx")
            overall_df_file = pd.DataFrame({'File Name': [filename], 'LF/HF (Overall)': [overall_lfhf]})
            overall_df_file.to_excel(overall_output_path, index=False)
            print(f"  -> 全体平均結果を保存: {os.path.basename(overall_output_path)}")

            df_renamed = sliding_df.copy()
            df_renamed.columns = ['Time', f'{label}_LF/HF', f'{label}_RMSSD', f'{label}_SDNN']

            if combined_df is None:
                combined_df = df_renamed
            else:
                combined_df = pd.merge(combined_df, df_renamed, on='Time', how='outer')
        else:
            print(f"  -> {label} の解析結果が得られませんでした。")

    if combined_df is not None:
        combined_df.sort_values('Time', inplace=True)
        cols = ['Time']
        for label in files_map.keys():
            if f'{label}_LF/HF' in combined_df.columns:
                cols.append(f'{label}_LF/HF')
                cols.append(f'{label}_RMSSD')
                cols.append(f'{label}_SDNN')

        combined_df = combined_df[cols]
        combined_output_path = os.path.join(output_dir, f"{subject_id}_Combined_HRV_Analysis.xlsx")
        combined_df.to_excel(combined_output_path, index=False)
        print(f"\n=== 全データの結合ファイルを保存しました ===")
        print(f"保存先: {combined_output_path}")
    else:
        print("\n有効な解析結果が1つもありませんでした。")

    print("\n処理完了。")


def generate_box_plots(
    input_file_path,
    output_dir,
    condition_labels=None,
    condition_order=None,
):
    """箱ひげ図を生成する"""
    condition_labels = condition_labels or DEFAULT_CONDITION_LABELS
    condition_order = condition_order or ANALYS_CONDITION_ORDER
    colors = CONDITION_COLORS

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {input_file_path}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_excel(input_file_path)
    print("データの読み込みに成功しました。")

    saved_files = []

    for metric_suffix, title, output_filename in [
        ('_LF/HF', 'LF/HFの比較（時系列分布）', 'LFHF_Boxplot.png'),
        ('_RMSSD', 'RMSSDの比較（時系列分布）', 'RMSSD_Boxplot.png'),
        ('_SDNN', 'SDNNの比較（時系列分布）', 'SDNN_Boxplot.png')
    ]:
        print(f"--- {title} のグラフを作成中 ---")
        plot_data = pd.DataFrame()
        found_cols = False

        for eng_key, display_label in condition_labels.items():
            col_name = f"{eng_key}{metric_suffix}"
            if col_name in df.columns:
                label = display_label or eng_key
                plot_data[label] = df[col_name]
                found_cols = True

        if not found_cols:
            print(f"エラー: {metric_suffix} に関するデータが見つかりませんでした。")
            continue

        df_melted = plot_data.melt(var_name='Condition', value_name='Value')
        df_melted = df_melted.dropna()

        # Aggバックエンドを直接使用（スレッドセーフ）
        fig = Figure(figsize=(10, 7))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        available_labels = set(df_melted['Condition'])
        order_labels = []
        for cond in condition_order:
            label = condition_labels.get(cond)
            if label and label in available_labels:
                order_labels.append(label)
        if not order_labels:
            order_labels = list(df_melted['Condition'].unique())

        palette = {}
        for cond in condition_order:
            label = condition_labels.get(cond)
            if label and label in order_labels:
                palette[label] = colors.get(cond, 'lightgray')
        for label in order_labels:
            if label not in palette:
                palette[label] = 'lightgray'
        sns.boxplot(x='Condition', y='Value', data=df_melted, palette=palette, ax=ax, showfliers=False, width=0.5, order=order_labels)

        legend_patches = [mpatches.Patch(color=palette[label], label=label) for label in order_labels]
        ax.legend(handles=legend_patches, title="条件", loc='upper right')
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(metric_suffix.replace('_', ''), fontsize=14)
        ax.set_xlabel("条件", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        save_path = os.path.join(output_dir, output_filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        print(f"保存完了: {save_path}")
        saved_files.append(save_path)

    print("\nすべてのグラフ作成が完了しました。")
    return saved_files


# 後方互換性のためのエイリアス（analys_プレフィックス付き）
analys_bandpass_filter = bandpass_filter
analys_ecg_to_rri = ecg_to_rri
analys_calculate_hrv_indices = calculate_hrv_indices
analys_run_batch_analysis = run_batch_analysis
analys_generate_box_plots = generate_box_plots
