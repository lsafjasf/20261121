"""实测项目：交叉频带纯音、快速声级跃迁、分频重组误差、分段电平。"""

from __future__ import annotations

import math

from .chain import ProcessingChain
from .signals import sine, level_step
from .meters import steady_level, rms_db, block_meters


def recombination_error(samples, chain: ProcessingChain) -> float:
    """仅分频+重组（不压缩）的相对 RMS 误差 dB，越小说明分频越透明。"""
    chain.reset()
    bands = chain.crossover.split_block(samples)
    recon = chain.crossover.recombine(bands)
    err2, sig2 = 0.0, 0.0
    for a, b in zip(samples, recon):
        err2 += (a - b) ** 2
        sig2 += a * a
    return 10.0 * math.log10(err2 / (sig2 + 1e-14) + 1e-14)


def tone_crossover_measure(band_params, fs, crossovers, bypass=False,
                           level_dbfs=-20.0, dur=1.0):
    """交叉频带附近纯音：记录各带输入/输出稳态电平与合成输出。"""
    rows = []
    test_freqs = sorted({
        round(fc * m) for fc in crossovers for m in (0.5, 1.0, 2.0)
    } | set(int(fc) for fc in crossovers))
    for fc_tone in test_freqs:
        sig = sine(fc_tone, dur, fs, level_dbfs)
        chain = ProcessingChain(band_params, fs, crossovers, bypass=bypass)
        out, band_in, band_out = chain.process_block(sig)
        in_lvls = [steady_level(b, fs, 0.3, 0.9) for b in band_in]
        out_lvls = [steady_level(b, fs, 0.3, 0.9) for b in band_out]
        rows.append({
            "tone_hz": fc_tone,
            "input_dbfs": round(steady_level(sig, fs, 0.3, 0.9), 1),
            "band_input_dbfs": [round(v, 1) for v in in_lvls],
            "band_output_dbfs": [round(v, 1) for v in out_lvls],
            "sum_output_dbfs": round(steady_level(out, fs, 0.3, 0.9), 1),
        })
    return rows


def _time_to_fraction(series, fs, t0_s, v0, v_inf, fraction):
    """从 t0 起，序列到达 (v_inf-v0)*fraction 变化量所需时间（ms）。"""
    target = v0 + (v_inf - v0) * fraction
    i0 = int(t0_s * fs)
    rising = v_inf > v0
    for i in range(i0, len(series)):
        v = series[i]
        if (rising and v >= target) or (not rising and v <= target):
            return (i - i0) / fs * 1000.0
    return None


def level_transition_measure(band_params, fs, band_index=1, freq_hz=1000.0):
    """1 kHz 纯音在 -30/-12 dBFS 间阶跃，测包络攻击与恢复时间。"""
    schedule = [(0.0, -30.0), (0.5, -12.0), (1.0, -30.0), (1.5, -8.0)]
    sig = level_step(freq_hz, schedule, fs)
    chain = ProcessingChain(band_params, fs, bypass=False)
    chain.reset()
    comp = chain.bands[band_index]
    env_db, gain_db = [], []
    for x in sig:
        chain.process_sample(x)
        env_db.append(comp.last_env_db)
        gain_db.append(comp.last_gain_db)

    def median(seq):
        s = sorted(seq)
        return s[len(s) // 2]

    def steady(seq, t_end):
        a = int((t_end - 0.12) * fs)
        b = int((t_end - 0.02) * fs)
        return median(seq[a:b])

    env_lo = steady(env_db, 0.5)
    env_hi = steady(env_db, 1.0)
    gr_lo = steady(gain_db, 0.5)
    gr_hi = steady(gain_db, 1.0)
    attack_ms = _time_to_fraction(env_db, fs, 0.5, env_lo, env_hi, 0.63)
    release_ms = _time_to_fraction(env_db, fs, 1.0, env_hi, env_lo, 0.37)
    return {
        "schedule": schedule,
        "env_db_steady_low": round(env_lo, 1),
        "env_db_steady_high": round(env_hi, 1),
        "gain_db_steady_low": round(gr_lo, 1),
        "gain_db_steady_high": round(gr_hi, 1),
        "attack_ms_measured": None if attack_ms is None else round(attack_ms, 2),
        "release_ms_measured": None if release_ms is None else round(release_ms, 2),
        "attack_ms_config": comp.params["attack_ms"],
        "release_ms_config": comp.params["release_ms"],
        "block_timeline": block_meters(sig, fs, 50.0),
        "gain_timeline": block_meters(
            [10.0 ** (g / 20.0) for g in gain_db], fs, 50.0),
    }


def demo_band_levels(band_params, fs, crossovers, demo_sig, bypass):
    """演示节目全程每段输入/输出 RMS/峰值，用于旁路对比。"""
    chain = ProcessingChain(band_params, fs, crossovers, bypass=bypass)
    out, band_in, band_out = chain.process_block(demo_sig)
    summary = []
    for i in range(4):
        summary.append({
            "band": i,
            "input_rms_dbfs": round(rms_db(band_in[i]), 1),
            "output_rms_dbfs": round(rms_db(band_out[i]), 1),
            "input_peak_dbfs": round(
                20 * math.log10(max((abs(x) for x in band_in[i]), default=0)
                                + 1e-14), 1),
            "output_peak_dbfs": round(
                20 * math.log10(max((abs(x) for x in band_out[i]), default=0)
                                + 1e-14), 1),
        })
    return out, summary, block_meters(out, fs, 100.0)

