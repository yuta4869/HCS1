"""
MEC（モデル誤差抑制補償器）の安定性検証

検証項目:
1. 名目システムの閉ループ安定性（極配置）
2. ロバスト安定性（ゲイン/位相余裕）
3. 名目モデルと実システムのミスマッチに対する頑健性
4. 時間応答シミュレーション
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from dataclasses import dataclass
from typing import Tuple, List
import os
import sys

# 日本語フォント設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'sans-serif']


@dataclass
class MECParams:
    """MECパラメータ（現在の実装値）"""
    # PI制御器
    kp: float = 0.1
    ki: float = 0.003

    # 名目モデル
    model_gain: float = 5.0      # K
    model_tau: float = 30.0      # τ [秒]
    model_delay: float = 5.0     # L [秒]

    # MEC
    mec_gain: float = 0.3        # k
    mec_omega: float = 0.05      # ω [rad/s]

    # フィルタ
    filter_tau: float = 2.0      # 計測LPF時定数


def transfer_function_eval(num_coeffs: List[float], den_coeffs: List[float],
                           s: complex) -> complex:
    """伝達関数 G(s) = num(s)/den(s) を評価"""
    num = sum(c * (s ** i) for i, c in enumerate(reversed(num_coeffs)))
    den = sum(c * (s ** i) for i, c in enumerate(reversed(den_coeffs)))
    return num / den if abs(den) > 1e-15 else complex(1e10, 0)


def get_plant_tf(K: float, tau: float, L: float, s: complex) -> complex:
    """
    プラント伝達関数: G(s) = K * exp(-Ls) / (τs + 1)
    """
    return K * np.exp(-L * s) / (tau * s + 1)


def get_pi_controller_tf(kp: float, ki: float, s: complex) -> complex:
    """
    PI制御器: K0(s) = kp + ki/s = (kp*s + ki) / s
    """
    if abs(s) < 1e-15:
        return complex(1e10, 0)
    return kp + ki / s


def get_mec_tf(k: float, omega: float, s: complex) -> complex:
    """
    MEC補償器: Cε(s) = k / (1 + s/ω) = k*ω / (s + ω)
    """
    return k * omega / (s + omega)


def get_measurement_filter_tf(tau: float, s: complex) -> complex:
    """
    計測フィルタ: F(s) = 1 / (τs + 1)
    """
    return 1.0 / (tau * s + 1)


def analyze_loop_gain(params: MECParams, omega_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    開ループ伝達関数 L(s) の周波数応答を計算

    閉ループ構造:
    L(s) = K0(s) * G(s) * F(s) + Cε(s) * (G(s) - Ĝ(s)) * F(s)

    名目システム（G = Ĝ）では:
    L_nominal(s) = K0(s) * G(s) * F(s)
    """
    gains = []
    phases = []

    for omega in omega_range:
        s = 1j * omega

        # 各伝達関数
        K0 = get_pi_controller_tf(params.kp, params.ki, s)
        G = get_plant_tf(params.model_gain, params.model_tau, params.model_delay, s)
        F = get_measurement_filter_tf(params.filter_tau, s)

        # 名目開ループ伝達関数
        L = K0 * G * F

        gains.append(20 * np.log10(abs(L) + 1e-15))
        phases.append(np.angle(L, deg=True))

    return omega_range, np.array(gains), np.array(phases)


def find_margins(omega_range: np.ndarray, gains: np.ndarray, phases: np.ndarray) -> dict:
    """
    ゲイン余裕と位相余裕を計算
    """
    # 位相交差周波数（位相 = -180度）を探す
    phase_crossover_idx = None
    for i in range(len(phases) - 1):
        if phases[i] > -180 and phases[i+1] <= -180:
            phase_crossover_idx = i
            break

    # ゲイン交差周波数（ゲイン = 0dB）を探す
    gain_crossover_idx = None
    for i in range(len(gains) - 1):
        if gains[i] > 0 and gains[i+1] <= 0:
            gain_crossover_idx = i
            break

    result = {
        'gain_margin_db': None,
        'phase_margin_deg': None,
        'gain_crossover_freq': None,
        'phase_crossover_freq': None
    }

    if phase_crossover_idx is not None:
        result['gain_margin_db'] = -gains[phase_crossover_idx]
        result['phase_crossover_freq'] = omega_range[phase_crossover_idx]

    if gain_crossover_idx is not None:
        result['phase_margin_deg'] = 180 + phases[gain_crossover_idx]
        result['gain_crossover_freq'] = omega_range[gain_crossover_idx]

    return result


def analyze_closed_loop_poles(params: MECParams) -> List[complex]:
    """
    閉ループ極を近似計算（むだ時間をPade近似）

    簡略化: 2次Pade近似で離散的な極を求める
    """
    # 2次Pade近似: exp(-Ls) ≈ (1 - Ls/2 + (Ls)^2/12) / (1 + Ls/2 + (Ls)^2/12)
    L = params.model_delay
    tau = params.model_tau
    K = params.model_gain
    kp = params.kp
    ki = params.ki

    # 簡略化のため、むだ時間なしの場合の極を計算
    # 閉ループ: 1 + K0(s) * G(s) = 0
    # s*(τs+1) + (kp*s + ki)*K = 0
    # τs^2 + (1 + kp*K)*s + ki*K = 0

    a = tau
    b = 1 + kp * K
    c = ki * K

    discriminant = b**2 - 4*a*c

    if discriminant >= 0:
        s1 = (-b + np.sqrt(discriminant)) / (2*a)
        s2 = (-b - np.sqrt(discriminant)) / (2*a)
        return [complex(s1, 0), complex(s2, 0)]
    else:
        real_part = -b / (2*a)
        imag_part = np.sqrt(-discriminant) / (2*a)
        return [complex(real_part, imag_part), complex(real_part, -imag_part)]


def simulate_step_response(params: MECParams,
                           plant_gain_ratio: float = 1.0,
                           plant_tau_ratio: float = 1.0,
                           duration: float = 300.0,
                           dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ステップ応答シミュレーション

    Args:
        params: MECパラメータ
        plant_gain_ratio: 実プラントゲイン / 名目ゲイン
        plant_tau_ratio: 実プラント時定数 / 名目時定数
        duration: シミュレーション時間 [秒]
        dt: サンプリング時間 [秒]

    Returns:
        time, hr, control_output
    """
    # 実プラントパラメータ
    real_gain = params.model_gain * plant_gain_ratio
    real_tau = params.model_tau * plant_tau_ratio
    real_delay = params.model_delay

    # 状態変数
    baseline_hr = 70.0
    target_hr = 80.0  # ステップ入力: 70 → 80

    # プラント状態
    plant_hr = baseline_hr

    # 名目モデル状態
    model_hr = baseline_hr

    # 計測フィルタ状態
    filtered_hr = baseline_hr

    # PI制御器状態
    integral = 0.0

    # MEC状態
    mec_filtered = 0.0

    # 入力バッファ（むだ時間用）
    delay_steps = int(real_delay / dt)
    input_buffer = [1.0] * (delay_steps + 1)
    model_input_buffer = [1.0] * (int(params.model_delay / dt) + 1)

    # 出力
    last_output = 1.0

    # 記録
    times = []
    hrs = []
    outputs = []

    for step in range(int(duration / dt)):
        t = step * dt
        times.append(t)

        # === 計測フィルタ ===
        alpha_f = dt / (params.filter_tau + dt)
        filtered_hr += alpha_f * (plant_hr - filtered_hr)

        # === 名目モデル更新 ===
        model_delay_steps = int(params.model_delay / dt)
        if len(model_input_buffer) > model_delay_steps:
            delayed_model_input = model_input_buffer[-model_delay_steps-1]
        else:
            delayed_model_input = 1.0

        model_target = baseline_hr + params.model_gain * (delayed_model_input - 1.0)
        alpha_m = dt / (params.model_tau + dt)
        model_hr += alpha_m * (model_target - model_hr)

        # === モデル誤差 ===
        model_error = filtered_hr - model_hr

        # === MEC ===
        alpha_mec = dt * params.mec_omega
        alpha_mec = min(alpha_mec, 1.0)
        mec_filtered += alpha_mec * (params.mec_gain * model_error - mec_filtered)
        u_mec = -mec_filtered

        # === PI制御 ===
        error = target_hr - filtered_hr
        p_term = params.kp * error
        integral += params.ki * error * dt
        integral = max(-0.1, min(0.1, integral))
        u0 = 1.0 + p_term + integral

        # === 総合出力 ===
        output = u0 + u_mec
        output = max(0.0, min(2.0, output))

        outputs.append(output)

        # === 入力バッファ更新 ===
        input_buffer.append(output)
        model_input_buffer.append(output)

        # === 実プラント更新（むだ時間付き） ===
        if len(input_buffer) > delay_steps:
            delayed_input = input_buffer[-delay_steps-1]
        else:
            delayed_input = 1.0

        plant_target = baseline_hr + real_gain * (delayed_input - 1.0)
        alpha_p = dt / (real_tau + dt)
        plant_hr += alpha_p * (plant_target - plant_hr)

        hrs.append(plant_hr)

    return np.array(times), np.array(hrs), np.array(outputs)


def evaluate_robustness(params: MECParams) -> dict:
    """
    様々なプラント変動に対する頑健性を評価
    """
    results = []

    # ゲイン変動: 0.5〜2.0倍
    # 時定数変動: 0.5〜2.0倍
    gain_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    tau_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    for g_ratio in gain_ratios:
        for t_ratio in tau_ratios:
            times, hrs, outputs = simulate_step_response(
                params,
                plant_gain_ratio=g_ratio,
                plant_tau_ratio=t_ratio,
                duration=300.0
            )

            # 評価指標
            target = 80.0
            steady_state_hr = np.mean(hrs[-30:])  # 最後30秒の平均
            steady_state_error = abs(target - steady_state_hr)
            overshoot = (np.max(hrs) - target) / (target - 70.0) * 100 if np.max(hrs) > target else 0

            # 安定性判定（発振していないか）
            hr_std_last = np.std(hrs[-60:])  # 最後60秒の標準偏差
            is_stable = hr_std_last < 2.0  # 2BPM以下なら安定

            results.append({
                'gain_ratio': g_ratio,
                'tau_ratio': t_ratio,
                'steady_state_error': steady_state_error,
                'overshoot': overshoot,
                'is_stable': is_stable,
                'final_hr': steady_state_hr
            })

    return results


def create_stability_report(params: MECParams, output_dir: str):
    """
    安定性検証レポートを生成
    """
    print("=" * 60)
    print("MEC（モデル誤差抑制補償器）安定性検証レポート")
    print("=" * 60)

    # 1. パラメータ表示
    print("\n【1. 現在のパラメータ】")
    print(f"  PI制御器: Kp = {params.kp}, Ki = {params.ki}")
    print(f"  名目モデル: K = {params.model_gain}, τ = {params.model_tau}s, L = {params.model_delay}s")
    print(f"  MEC: k = {params.mec_gain}, ω = {params.mec_omega} rad/s")
    print(f"  計測フィルタ: τ = {params.filter_tau}s")

    # 2. 開ループ解析
    print("\n【2. 開ループ解析（ボード線図）】")
    omega_range = np.logspace(-4, 1, 500)
    omega, gains, phases = analyze_loop_gain(params, omega_range)
    margins = find_margins(omega, gains, phases)

    print(f"  ゲイン余裕: {margins['gain_margin_db']:.1f} dB" if margins['gain_margin_db'] else "  ゲイン余裕: 測定不可（位相が-180度に達しない）")
    print(f"  位相余裕: {margins['phase_margin_deg']:.1f} 度" if margins['phase_margin_deg'] else "  位相余裕: 測定不可（ゲインが0dBに達しない）")

    if margins['gain_crossover_freq']:
        print(f"  ゲイン交差周波数: {margins['gain_crossover_freq']:.4f} rad/s")
    if margins['phase_crossover_freq']:
        print(f"  位相交差周波数: {margins['phase_crossover_freq']:.4f} rad/s")

    # 安定性判定
    is_stable_by_margins = True
    if margins['gain_margin_db'] and margins['gain_margin_db'] < 6:
        print("  ⚠️ 警告: ゲイン余裕が6dB未満")
        is_stable_by_margins = False
    if margins['phase_margin_deg'] and margins['phase_margin_deg'] < 30:
        print("  ⚠️ 警告: 位相余裕が30度未満")
        is_stable_by_margins = False

    if is_stable_by_margins:
        print("  ✓ 名目システムは十分な安定余裕を持つ")

    # 3. 閉ループ極
    print("\n【3. 閉ループ極（むだ時間なし近似）】")
    poles = analyze_closed_loop_poles(params)
    for i, pole in enumerate(poles):
        print(f"  極{i+1}: {pole.real:.4f} + {pole.imag:.4f}j")
        if pole.real >= 0:
            print(f"    ⚠️ 不安定極！")
        else:
            print(f"    ✓ 安定（実部 < 0）")

    # 4. ロバスト性テスト
    print("\n【4. ロバスト性テスト（プラント変動に対する頑健性）】")
    robustness_results = evaluate_robustness(params)

    stable_count = sum(1 for r in robustness_results if r['is_stable'])
    total_count = len(robustness_results)
    print(f"  テストケース: {total_count}件")
    print(f"  安定: {stable_count}件 ({100*stable_count/total_count:.0f}%)")

    # 不安定なケースを表示
    unstable_cases = [r for r in robustness_results if not r['is_stable']]
    if unstable_cases:
        print("  不安定なケース:")
        for case in unstable_cases:
            print(f"    - ゲイン比={case['gain_ratio']}, 時定数比={case['tau_ratio']}")
    else:
        print("  ✓ すべてのテストケースで安定")

    # 5. グラフ生成
    print("\n【5. グラフ生成】")

    # ボード線図
    fig1 = Figure(figsize=(12, 8))
    canvas1 = FigureCanvasAgg(fig1)

    ax1 = fig1.add_subplot(211)
    ax1.semilogx(omega, gains, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('ゲイン [dB]')
    ax1.set_title('開ループ伝達関数 ボード線図')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_xlim([omega[0], omega[-1]])

    if margins['gain_crossover_freq']:
        ax1.axvline(x=margins['gain_crossover_freq'], color='r', linestyle=':', alpha=0.7)
        ax1.annotate(f'ωc={margins["gain_crossover_freq"]:.3f}',
                    (margins['gain_crossover_freq'], 0), fontsize=9)

    ax2 = fig1.add_subplot(212)
    ax2.semilogx(omega, phases, 'b-', linewidth=2)
    ax2.axhline(y=-180, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('周波数 [rad/s]')
    ax2.set_ylabel('位相 [度]')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_xlim([omega[0], omega[-1]])
    ax2.set_ylim([-270, 0])

    if margins['phase_margin_deg'] and margins['gain_crossover_freq']:
        ax2.annotate(f'PM={margins["phase_margin_deg"]:.1f}°',
                    (margins['gain_crossover_freq'], -180 + margins['phase_margin_deg']/2), fontsize=9)

    fig1.tight_layout()
    bode_path = os.path.join(output_dir, 'mec_bode_plot.png')
    fig1.savefig(bode_path, dpi=150)
    print(f"  ボード線図: {bode_path}")

    # ステップ応答（名目 vs 変動）
    fig2 = Figure(figsize=(12, 10))
    canvas2 = FigureCanvasAgg(fig2)

    ax3 = fig2.add_subplot(211)

    # 名目システム
    t_nom, hr_nom, u_nom = simulate_step_response(params, 1.0, 1.0)
    ax3.plot(t_nom, hr_nom, 'b-', linewidth=2, label='名目 (G=Ĝ)')

    # ゲイン2倍
    t_g2, hr_g2, u_g2 = simulate_step_response(params, 2.0, 1.0)
    ax3.plot(t_g2, hr_g2, 'r--', linewidth=1.5, label='ゲイン2倍')

    # ゲイン0.5倍
    t_g05, hr_g05, u_g05 = simulate_step_response(params, 0.5, 1.0)
    ax3.plot(t_g05, hr_g05, 'g--', linewidth=1.5, label='ゲイン0.5倍')

    # 時定数2倍
    t_t2, hr_t2, u_t2 = simulate_step_response(params, 1.0, 2.0)
    ax3.plot(t_t2, hr_t2, 'm:', linewidth=1.5, label='時定数2倍')

    ax3.axhline(y=80, color='k', linestyle='--', alpha=0.5, label='目標')
    ax3.set_ylabel('心拍数 [BPM]')
    ax3.set_title('ステップ応答（目標: 70→80 BPM）')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 300])
    ax3.set_ylim([68, 85])

    ax4 = fig2.add_subplot(212)
    ax4.plot(t_nom, u_nom, 'b-', linewidth=2, label='名目')
    ax4.plot(t_g2, u_g2, 'r--', linewidth=1.5, label='ゲイン2倍')
    ax4.plot(t_g05, u_g05, 'g--', linewidth=1.5, label='ゲイン0.5倍')
    ax4.plot(t_t2, u_t2, 'm:', linewidth=1.5, label='時定数2倍')
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('時間 [秒]')
    ax4.set_ylabel('制御出力（抑揚）')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 300])
    ax4.set_ylim([0.8, 1.6])

    fig2.tight_layout()
    step_path = os.path.join(output_dir, 'mec_step_response.png')
    fig2.savefig(step_path, dpi=150)
    print(f"  ステップ応答: {step_path}")

    # ロバスト性マップ
    fig3 = Figure(figsize=(10, 8))
    canvas3 = FigureCanvasAgg(fig3)

    ax5 = fig3.add_subplot(111)

    gain_ratios = sorted(set(r['gain_ratio'] for r in robustness_results))
    tau_ratios = sorted(set(r['tau_ratio'] for r in robustness_results))

    stability_map = np.zeros((len(tau_ratios), len(gain_ratios)))
    error_map = np.zeros((len(tau_ratios), len(gain_ratios)))

    for r in robustness_results:
        gi = gain_ratios.index(r['gain_ratio'])
        ti = tau_ratios.index(r['tau_ratio'])
        stability_map[ti, gi] = 1 if r['is_stable'] else 0
        error_map[ti, gi] = r['steady_state_error']

    im = ax5.imshow(error_map, cmap='RdYlGn_r', aspect='auto',
                    extent=[min(gain_ratios)-0.125, max(gain_ratios)+0.125,
                           max(tau_ratios)+0.125, min(tau_ratios)-0.125])

    # 不安定領域をマーク
    for r in robustness_results:
        if not r['is_stable']:
            gi = gain_ratios.index(r['gain_ratio'])
            ti = tau_ratios.index(r['tau_ratio'])
            ax5.plot(r['gain_ratio'], r['tau_ratio'], 'kx', markersize=15, markeredgewidth=3)

    ax5.set_xlabel('プラントゲイン比 (実/名目)')
    ax5.set_ylabel('プラント時定数比 (実/名目)')
    ax5.set_title('ロバスト性マップ（色: 定常偏差 [BPM], ×: 不安定）')
    ax5.set_xticks(gain_ratios)
    ax5.set_yticks(tau_ratios)

    cbar = fig3.colorbar(im, ax=ax5)
    cbar.set_label('定常偏差 [BPM]')

    # 名目点をマーク
    ax5.plot(1.0, 1.0, 'b*', markersize=20, label='名目')
    ax5.legend()

    fig3.tight_layout()
    robust_path = os.path.join(output_dir, 'mec_robustness_map.png')
    fig3.savefig(robust_path, dpi=150)
    print(f"  ロバスト性マップ: {robust_path}")

    # 6. 総合評価
    print("\n【6. 総合評価】")
    all_stable = all(r['is_stable'] for r in robustness_results)
    max_error = max(r['steady_state_error'] for r in robustness_results)

    if all_stable:
        print("  ✓ すべてのテスト条件で安定")
    else:
        print("  ⚠️ 一部の条件で不安定")

    print(f"  最大定常偏差: {max_error:.2f} BPM")

    if margins['gain_margin_db'] and margins['gain_margin_db'] >= 6:
        print("  ✓ ゲイン余裕 ≥ 6dB")

    if margins['phase_margin_deg'] and margins['phase_margin_deg'] >= 30:
        print("  ✓ 位相余裕 ≥ 30度")

    print("\n" + "=" * 60)

    return {
        'margins': margins,
        'poles': poles,
        'robustness': robustness_results,
        'all_stable': all_stable
    }


if __name__ == '__main__':
    # 現在の実装パラメータ
    params = MECParams(
        kp=0.1,
        ki=0.003,
        model_gain=5.0,
        model_tau=30.0,
        model_delay=5.0,
        mec_gain=0.3,
        mec_omega=0.05,
        filter_tau=2.0
    )

    # 出力ディレクトリ
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # レポート生成
    results = create_stability_report(params, output_dir)
