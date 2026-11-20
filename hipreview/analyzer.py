"""主声轨与助听轨的对齐、重复与语言标记核查。

三类问题判定（均输出卷内相对时刻）：

DELAY          助听轨相对主声轨存在超阈值固定时延（换片接线/偏移）。
DUP_TRACK      助听轨直接复制了主混音中心声道：全局高度相关且
               助听轨无语义差异（助听轨本应是独立对白增强轨）。
DUP_SEGMENT    卷内出现重复声轨片段（同一段测试词序列在两处重复）。
LANG_MISMATCH  助听轨 DTMF 语言代号与参考片段指定语言不符。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .synth import (
    SAMPLE_RATE, WORD_COUNT, LANG_CODES,
    _DTMF_LOW, _DTMF_HIGH, _KEYS,
)

DELAY_TOLERANCE_MS = 60.0       # 小于该时延视为同步正常
DUP_CORR_THRESHOLD = 0.90       # 全局相关系数高于此值判定整轨重复
DELAY_CORR_MIN = 0.60           # 互相关峰值低于此值时不报告时延（对齐不可信）
SEG_CORR_THRESHOLD = 0.88       # 片段相关阈值
DUP_RESIDUAL_THRESHOLD = 0.05   # 归一化残差低于此值才算逐样本复制
SEG_WINDOW_SEC = 4.0            # 片段检测窗（秒）
SEG_HOP_SEC = 2.0


@dataclass
class Issue:
    reel_id: str
    category: str
    start_sec: float
    end_sec: float
    detail: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReelResult:
    reel_id: str
    duration_sec: float
    delay_ms: float
    best_corr: float
    detected_lang: str | None
    expected_lang: str | None
    issues: list[Issue] = field(default_factory=list)
    waveform_main: dict | None = None
    waveform_hi: dict | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["issues"] = [i.to_dict() if isinstance(i, Issue) else i
                          for i in self.issues]
        return data


def _norm(sig: np.ndarray) -> np.ndarray:
    std = sig.std()
    return (sig - sig.mean()) / (std + 1e-12)


def xcorr_delay(reference: np.ndarray, target: np.ndarray,
                max_lag_ms: float = 1500.0,
                sr: int = SAMPLE_RATE) -> tuple[int, float]:
    """互相关估计 target 相对 reference 的滞后采样与峰值相关系数。

    返回 lag>0 表示 target 晚于 reference（助听轨延后）。
    用 FFT 全相关再截取 ±max_lag 窗口，纯 numpy 实现。
    """
    n = len(reference) + len(target) - 1
    fft_n = 1 << int(np.ceil(np.log2(n)))
    ref_n = _norm(reference)
    tgt_n = _norm(target)
    corr = np.fft.irfft(np.fft.rfft(ref_n, fft_n)
                        * np.conj(np.fft.rfft(tgt_n, fft_n)), fft_n)
    corr = np.concatenate((corr[-(len(target) - 1):], corr[:len(reference)]))
    # corr[k] 对应 reference 与 target 平移 k；换算为 target 滞后量
    lags = np.arange(-(len(target) - 1), len(reference))
    target_lag = -lags
    max_lag = int(max_lag_ms / 1000.0 * sr)
    mask = np.abs(target_lag) <= max_lag
    peak_idx = np.argmax(np.abs(corr[mask]))
    lag = int(target_lag[mask][peak_idx])
    denom = np.sqrt(np.sum(ref_n ** 2) * np.sum(tgt_n ** 2))
    peak_corr = float(np.abs(corr[mask][peak_idx]) / (denom + 1e-12))
    return lag, peak_corr


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """等长信号皮尔逊相关系数。"""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = np.sqrt(np.sum(a0 ** 2) * np.sum(b0 ** 2))
    return float(np.sum(a0 * b0) / (denom + 1e-12))


def rms(sig: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sig * sig)))


def normalized_residual(a: np.ndarray, b: np.ndarray) -> float:
    """归一化残差能量：逐样本复制趋近 0，独立制作的同义轨明显更大。"""
    if len(a) != len(b) or len(a) < 2:
        return 1.0
    scale = np.dot(a, b) / (np.dot(b, b) + 1e-12)
    residual = a - scale * b
    return float(np.sqrt(np.mean(residual ** 2)) / (rms(a) + 1e-12))


def _goertzel_energy(sig: np.ndarray, freq: float,
                     sr: int = SAMPLE_RATE) -> float:
    """单频能量（Goertzel 思路的直接 DFT 幅度）。"""
    n = len(sig)
    if n == 0:
        return 0.0
    k = np.arange(n)
    basis = np.exp(-2j * np.pi * freq * k / sr)
    return float(np.abs(np.sum(sig * basis)))


def detect_dtmf_keys(sig: np.ndarray, sr: int = SAMPLE_RATE,
                     frame_ms: float = 25.0,
                     hop_ms: float = 10.0) -> list[tuple[float, str]]:
    """滑窗检测 DTMF 键，返回 (起始时刻, 键) 的去重序列。

    状态机：连续 2 帧同键确认发声，中间出现静音/失配帧即结束，
    确保相邻两个键即使间隔很短也不会被合并。
    """
    frame = int(frame_ms / 1000.0 * sr)
    hop = int(hop_ms / 1000.0 * sr)
    keys: list[tuple[float, str]] = []
    current_key = None
    hold = 0
    start_sample = 0
    for start in range(0, max(1, len(sig) - frame), hop):
        frame_sig = sig[start:start + frame]
        key = _classify_dtmf_frame(frame_sig, sr)
        if key is None:
            if current_key is not None and hold >= 2:
                keys.append((start_sample / sr, current_key))
            current_key, hold = None, 0
            continue
        if key == current_key:
            hold += 1
        else:
            if current_key is not None and hold >= 2:
                keys.append((start_sample / sr, current_key))
            current_key, hold, start_sample = key, 1, start
    if current_key is not None and hold >= 2:
        keys.append((start_sample / sr, current_key))
    return keys


def _classify_dtmf_frame(frame_sig: np.ndarray, sr: int) -> str | None:
    """判断单帧是否为合格 DTMF 音，返回键名或 None。"""
    if rms(frame_sig) < 0.03:
        return None
    low_e = [_goertzel_energy(frame_sig, f, sr) for f in _DTMF_LOW]
    high_e = [_goertzel_energy(frame_sig, f, sr) for f in _DTMF_HIGH]
    row, col = int(np.argmax(low_e)), int(np.argmax(high_e))
    # 目标频率能量需显著领先，避免把啁啾扫频误判为 DTMF
    if low_e[row] < 4.0 * sorted(low_e)[-2]:
        return None
    if high_e[col] < 4.0 * sorted(high_e)[-2]:
        return None
    return next((k for k, (r, c) in _KEYS.items() if r == row and c == col), None)


def detect_language(track: np.ndarray, sr: int = SAMPLE_RATE,
                    probe_seconds: int = 6) -> tuple[str | None, list[str]]:
    """统计卷首若干秒的 DTMF 键序列，按多数投票还原语言代号。"""
    head = track[:probe_seconds * sr]
    keys = [k for _, k in detect_dtmf_keys(head, sr)]
    votes: dict[str, int] = {}
    for i in range(0, len(keys) - 1, 2):
        pair = keys[i] + keys[i + 1]
        for lang, code in LANG_CODES.items():
            if code == pair:
                votes[lang] = votes.get(lang, 0) + 1
    if not votes:
        return None, keys
    return max(votes, key=votes.get), keys


def find_duplicate_segments(track: np.ndarray, sr: int = SAMPLE_RATE
                            ) -> list[tuple[float, float, float]]:
    """检测卷内自重复片段，返回 (首次起点, 重复起点, 时长)。

    要求同一对偏移在相邻滑窗上连续命中，避免周期性测试词
    （如每 10 秒词号循环）造成的偶然高相关。
    """
    win = int(SEG_WINDOW_SEC * sr)
    hop = int(SEG_HOP_SEC * sr)
    if len(track) < 2 * win:
        return []
    windows: list[tuple[int, np.ndarray]] = []
    starts = list(range(0, max(1, len(track) - win + 1), hop))
    if not starts or starts[-1] != len(track) - win:
        starts.append(len(track) - win)  # 末尾补一窗，覆盖整段尾部
    for start in starts:
        seg = track[start:start + win]
        if rms(seg) > 0.01:
            windows.append((start, _norm(seg)))
    pair_corr: dict[tuple[int, int], float] = {}
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            corr = pearson(windows[i][1], windows[j][1])
            if corr >= SEG_CORR_THRESHOLD:
                pair_corr[(i, j)] = corr

    confirmed: set[tuple[int, int]] = set()
    for (i, j), strong in pair_corr.items():
        # 强命中直接成立；略低的命中需相邻窗佐证，排除周期词偶然相似
        neighbor = pair_corr.get((i + 1, j + 1), 0.0)
        if strong >= 0.95 or neighbor >= SEG_CORR_THRESHOLD - 0.1:
            confirmed.add((i, j))

    hits: list[tuple[float, float, float]] = []
    claimed: set[int] = set()
    for i, j in sorted(confirmed, key=lambda p: windows[p[1]][0]):
        if j in claimed:
            continue
        claimed.add(j)
        hits.append((windows[i][0] / sr, windows[j][0] / sr,
                     SEG_WINDOW_SEC))
    return hits


def analyze_reel(reel_id: str, main: np.ndarray, hi: np.ndarray,
                 expected_lang: str | None,
                 sr: int = SAMPLE_RATE,
                 waveform_bins: int = 800) -> ReelResult:
    """分析单卷的主声轨与助听轨，生成问题列表与波形证据。"""
    from .audioio import waveform_peaks

    n = min(len(main), len(hi))
    main, hi = main[:n], hi[:n]
    duration = n / sr
    lag, corr = xcorr_delay(main, hi, sr=sr)
    delay_ms = lag / sr * 1000.0
    detected_lang, key_seq = detect_language(hi, sr)
    residual = normalized_residual(hi, main)

    result = ReelResult(
        reel_id=reel_id,
        duration_sec=round(duration, 3),
        delay_ms=round(delay_ms, 2),
        best_corr=round(corr, 4),
        detected_lang=detected_lang,
        expected_lang=expected_lang,
        waveform_main=waveform_peaks(main, waveform_bins),
        waveform_hi=waveform_peaks(hi, waveform_bins),
    )

    if abs(delay_ms) > DELAY_TOLERANCE_MS and corr >= DELAY_CORR_MIN:
        result.issues.append(Issue(
            reel_id=reel_id, category="DELAY",
            start_sec=0.0, end_sec=round(min(2.0, duration), 3),
            detail=f"助听轨相对主声轨延后 {delay_ms:.0f} ms"
                   f"（阈值 ±{DELAY_TOLERANCE_MS:.0f} ms，对齐相关 "
                   f"{corr:.2f} ≥ {DELAY_CORR_MIN}）",
            evidence={"delay_ms": round(delay_ms, 2),
                      "corr": round(corr, 4)},
        ))

    if corr >= DUP_CORR_THRESHOLD and residual < DUP_RESIDUAL_THRESHOLD:
        result.issues.append(Issue(
            reel_id=reel_id, category="DUP_TRACK",
            start_sec=0.0, end_sec=round(duration, 3),
            detail=f"助听轨与主混音中心全程相关系数 {corr:.3f}"
                   f" 且归一化残差 {residual:.4f}（<{DUP_RESIDUAL_THRESHOLD}），"
                   "判定为直接复制主混音而非独立增强轨",
            evidence={"corr": round(corr, 4),
                      "normalized_residual": round(residual, 4),
                      "rms_main": round(rms(main), 4),
                      "rms_hi": round(rms(hi), 4)},
        ))

    for first, again, seg_len in find_duplicate_segments(hi, sr):
        result.issues.append(Issue(
            reel_id=reel_id, category="DUP_SEGMENT",
            start_sec=round(again, 3),
            end_sec=round(again + seg_len, 3),
            detail=f"该片段与 {first:.1f}s 处重复（{seg_len:.0f}s 窗相关"
                   f" ≥{SEG_CORR_THRESHOLD}）",
            evidence={"first_occurrence_sec": round(first, 3),
                      "window_sec": SEG_WINDOW_SEC},
        ))

    if expected_lang is not None and detected_lang != expected_lang:
        result.issues.append(Issue(
            reel_id=reel_id, category="LANG_MISMATCH",
            start_sec=0.0,
            end_sec=round(min(6.0, duration), 3),
            detail=f"助听轨语言标记 {detected_lang or '未识别'} "
                   f"与参考片段指定 {expected_lang} 不符",
            evidence={"detected": detected_lang,
                      "expected": expected_lang,
                      "dtmf_keys": key_seq[:12]},
        ))

    return result


def analyze_playlist(reels: list[dict], sr: int = SAMPLE_RATE) -> list[ReelResult]:
    """批量分析播放清单。

    reels 每项：{reel_id, main(ndarray), hi(ndarray), expected_lang}
    """
    results: list[ReelResult] = []
    for item in reels:
        results.append(analyze_reel(
            item["reel_id"], item["main"], item["hi"],
            item.get("expected_lang"), sr=sr,
        ))
    return results
