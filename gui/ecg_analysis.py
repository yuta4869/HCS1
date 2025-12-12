# gui/ecg_analysis.py
"""ECG/HRV解析関連のヘルパー関数"""

import os
import re
from itertools import accumulate

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

# 解析用の定数
FILENAME_PATTERN = re.compile(r".*_No(?P<subject>\d+)_\d{8}_\d{6}_(?P<condition>Sin|Fixed|HRF)\.csv$", re.IGNORECASE)
ANALYS_CONDITION_MAP = {"sin": "Sin", "fixed": "Fixed", "hrf": "HRF"}
ANALYS_CONDITION_ORDER = ["Sin", "Fixed", "HRF"]


def bandpass_filter(signal_data, fs):
    """バンドパスフィルタ (0.5Hz - 50Hz)"""
    from scipy.signal import butter, filtfilt
    lowcut = 0.5
    highcut = 50
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal_data)


def ecg_to_rri(file_path, fs=130):
    """ECGデータからRRIを算出する"""
    from scipy.signal import find_peaks
    try:
        data = pd.read_csv(file_path, delimiter=',', encoding="utf-8", skiprows=1, header=None, usecols=[0, 2], names=['timestamp', 'ecg'])
    except ValueError:
        print(f"エラー: {os.path.basename(file_path)} の読み込みに失敗しました。")
        return np.array([]), None, None

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    start_time_real = data['timestamp'].iloc[0]
    end_time_real = data['timestamp'].iloc[-1]
    duration_seconds = (end_time_real - start_time_real).total_seconds()

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


def calculate_hrv_indices(file_path, label, fs=130):
    """指定されたファイルを解析し、時系列データと全体LF/HF値を返す"""
    from scipy import interpolate
    import scipy.signal

    print(f"--- 解析開始: {label} ({os.path.basename(file_path)}) ---")
    rri_data, start_time, end_time = ecg_to_rri(file_path, fs)

    if len(rri_data) == 0:
        return None, None

    time_data = list(accumulate(rri_data / 1000))
    if not time_data:
        return None, None

    resampling_freq = 1
    duration_total = int(time_data[-1])
    time_axis = np.arange(0, duration_total, 1 / resampling_freq)

    if len(time_data) < 4:
        print("  -> データ点が少なすぎるためスキップします。")
        return None, None

    spline_func = interpolate.interp1d(time_data, rri_data, fill_value="extrapolate", kind='cubic')
    rri = spline_func(time_axis)

    if len(rri) > 10:
        low_threshold = np.quantile(rri, 0.038)
        high_threshold = np.quantile(rri, 0.962)
        rri[(rri < low_threshold) | (rri > high_threshold)] = np.nan

    df_temp = pd.DataFrame(data=rri, index=time_axis, columns=["rri"])
    df_temp.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
    rri = df_temp["rri"].values

    MinHR, MaxHR = 45, 210
    rri[(rri > 60000 / MinHR) | (rri < 60000 / MaxHR)] = np.nan
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

    if len(rri) >= 30:
        analysis_window = 30
    elif len(rri) >= 10:
        analysis_window = 10
        print(f"  -> データが短いため、ウィンドウサイズを {analysis_window}秒 に短縮して解析します。")
    else:
        print("  -> データが短すぎて解析できません（10秒未満）。")
        return None, None

    LF_HF_sliding = []
    RMSSD_sliding = []
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
            diff_rri = np.diff(rri_window)
            mssd = np.mean(np.square(diff_rri))
            rmssd_val = np.sqrt(mssd)
            RMSSD_sliding.append(rmssd_val)
        else:
            RMSSD_sliding.append(np.nan)

        time_points.append(i)
        i += 1

    sliding_result_df = pd.DataFrame({
        'Time': time_points,
        'LF/HF': LF_HF_sliding,
        'RMSSD': RMSSD_sliding
    })

    return sliding_result_df, overall_lf_hf_val


def run_batch_analysis(files_map, output_dir):
    """バッチ解析を実行し、結果をExcelファイルに保存する"""
    os.makedirs(output_dir, exist_ok=True)
    combined_df = None
    print("=== バッチ解析を開始します ===")

    for label, filename in files_map.items():
        file_path = filename
        if not os.path.exists(file_path):
            print(f"警告: ファイルが見つかりません -> {file_path}")
            continue

        sliding_df, overall_lfhf = calculate_hrv_indices(file_path, label)

        if sliding_df is not None and not sliding_df.empty:
            sliding_output_path = os.path.join(output_dir, f"{label}_result.xlsx")
            sliding_df.to_excel(sliding_output_path, index=False)
            print(f"  -> 時系列結果を保存: {os.path.basename(sliding_output_path)}")

            overall_output_path = os.path.join(output_dir, f"{label}_resultLFHF5min.xlsx")
            overall_df_file = pd.DataFrame({'File Name': [filename], 'LF/HF (Overall)': [overall_lfhf]})
            overall_df_file.to_excel(overall_output_path, index=False)
            print(f"  -> 全体平均結果を保存: {os.path.basename(overall_output_path)}")

            df_renamed = sliding_df.copy()
            df_renamed.columns = ['Time', f'{label}_LF/HF', f'{label}_RMSSD']

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

        combined_df = combined_df[cols]
        combined_output_path = os.path.join(output_dir, "Combined_HRV_Analysis.xlsx")
        combined_df.to_excel(combined_output_path, index=False)
        print(f"\n=== 全データの結合ファイルを保存しました ===")
        print(f"保存先: {combined_output_path}")
    else:
        print("\n有効な解析結果が1つもありませんでした。")

    print("\n処理完了。")


def generate_box_plots(input_file_path, output_dir):
    """箱ひげ図を生成する"""
    conditions = {'Fixed': '固定会話', 'HRF': '調整会話', 'Sin': '正弦波'}
    colors = {'固定会話': 'lightcoral', '調整会話': 'lightyellow', '正弦波': 'lightblue'}

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {input_file_path}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_excel(input_file_path)
    print("データの読み込みに成功しました。")

    saved_files = []

    for metric_suffix, title, output_filename in [
        ('_LF/HF', 'LF/HFの比較（時系列分布）', 'LFHF_Boxplot.png'),
        ('_RMSSD', 'RMSSDの比較（時系列分布）', 'RMSSD_Boxplot.png')
    ]:
        print(f"--- {title} のグラフを作成中 ---")
        plot_data = pd.DataFrame()
        found_cols = False

        for eng_key, jp_label in conditions.items():
            col_name = f"{eng_key}{metric_suffix}"
            if col_name in df.columns:
                plot_data[jp_label] = df[col_name]
                found_cols = True

        if not found_cols:
            print(f"エラー: {metric_suffix} に関するデータが見つかりませんでした。")
            continue

        df_melted = plot_data.melt(var_name='Condition', value_name='Value')
        df_melted = df_melted.dropna()

        fig, ax = plt.subplots(figsize=(10, 7))
        order_labels = [conditions[key] for key in ANALYS_CONDITION_ORDER if conditions[key] in set(df_melted['Condition'])]
        if not order_labels:
            order_labels = list(df_melted['Condition'].unique())

        palette = {label: colors.get(label, 'lightgray') for label in order_labels}
        sns.boxplot(x='Condition', y='Value', data=df_melted, palette=palette, ax=ax, showfliers=False, width=0.5, order=order_labels)

        legend_patches = [mpatches.Patch(color=palette[label], label=label) for label in order_labels]
        ax.legend(handles=legend_patches, title="条件", loc='upper right')
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(metric_suffix.replace('_', ''), fontsize=14)
        ax.set_xlabel("条件", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        save_path = os.path.join(output_dir, output_filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"保存完了: {save_path}")
        plt.close()
        saved_files.append(save_path)

    print("\nすべてのグラフ作成が完了しました。")
    return saved_files


# 後方互換性のためのエイリアス（analys_プレフィックス付き）
analys_bandpass_filter = bandpass_filter
analys_ecg_to_rri = ecg_to_rri
analys_calculate_hrv_indices = calculate_hrv_indices
analys_run_batch_analysis = run_batch_analysis
analys_generate_box_plots = generate_box_plots
