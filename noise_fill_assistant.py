#!/usr/bin/env python3
"""Noise-floor sampling assistant for broadcast editing.

Given a dialogue WAV and an edit segment, find the silent window whose
spectrum best matches the ambience at the edit boundaries, reject clearly
unfit candidates (transients, narrowband tones, sound-field switches) and
render a fill preview with short raised-cosine fades.

Pure standard library; no third-party dependencies.
"""

import argparse
import array
import cmath
import json
import math
import os
import struct
import sys
import wave


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------

def read_wav(path):
    """Return (samples as list of float in [-1, 1), sample_rate)."""
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if width != 2:
        raise ValueError("only 16-bit PCM WAV is supported, got %d bytes" % width)
    data = array.array("h")
    data.frombytes(raw)
    if sys.byteorder == "big":
        data.byteswap()
    if channels > 1:
        mono = [0.0] * frames
        for i in range(frames):
            acc = 0.0
            for c in range(channels):
                acc += data[i * channels + c]
            mono[i] = acc / (channels * 32768.0)
        return mono, rate
    scale = 1.0 / 32768.0
    return [s * scale for s in data], rate


def write_wav(path, samples, rate):
    """Write float samples in [-1, 1] to a 16-bit PCM WAV."""
    clipped = [max(-1.0, min(1.0, s)) for s in samples]
    data = array.array("h", (int(round(s * 32767.0)) for s in clipped))
    if sys.byteorder == "big":
        data.byteswap()
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(data.tobytes())


# ---------------------------------------------------------------------------
# DSP primitives
# ---------------------------------------------------------------------------

def fft(values):
    """Iterative radix-2 Cooley-Tukey FFT on a list of complex numbers."""
    n = len(values)
    if n == 1:
        return values[:]
    if n & (n - 1):
        raise ValueError("fft length must be a power of two")
    out = values[:]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            out[i], out[j] = out[j], out[i]
    size = 2
    while size <= n:
        half = size >> 1
        step = -2.0j * math.pi / size
        table = [cmath.exp(step * k) for k in range(half)]
        for start in range(0, n, size):
            for k in range(half):
                even = out[start + k]
                odd = out[start + k + half] * table[k]
                out[start + k] = even + odd
                out[start + k + half] = even - odd
        size <<= 1
    return out


def hann_window(n):
    if n == 1:
        return [1.0]
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)) for i in range(n)]


def db(value, floor=1e-12):
    return 10.0 * math.log10(max(value, floor))


def rms(samples):
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


# ---------------------------------------------------------------------------
# Band spectrum analysis
# ---------------------------------------------------------------------------

FFT_SIZE = 1024
BAND_COUNT = 28
BAND_MIN_HZ = 55.0
BAND_MAX_HZ = 7200.0


def band_edges(rate, bands=BAND_COUNT, fmin=BAND_MIN_HZ, fmax=BAND_MAX_HZ):
    """Log-spaced band edges; returns (edges, centers)."""
    ratio = (fmax / fmin) ** (1.0 / bands)
    edges = [fmin * ratio ** i for i in range(bands + 1)]
    centers = [math.sqrt(edges[i] * edges[i + 1]) for i in range(bands)]
    return edges, centers


def band_spectrum_db(samples, rate, fft_size=FFT_SIZE, window=None):
    """Average band energies (dB) over non-overlapping windowed frames."""
    if len(samples) < fft_size:
        samples = list(samples) + [0.0] * (fft_size - len(samples))
    if window is None:
        window = hann_window(fft_size)
    edges, _ = band_edges(rate)
    bin_hz = rate / fft_size
    nbins = fft_size // 2
    band_bins = [[] for _ in edges[:-1]]
    for b in range(nbins):
        freq = b * bin_hz
        for bi in range(len(edges) - 1):
            if edges[bi] <= freq < edges[bi + 1]:
                band_bins[bi].append(b)
                break
    win_energy = sum(w * w for w in window)
    band_energy = [0.0] * (len(edges) - 1)
    frames = 0
    start = 0
    while start + fft_size <= len(samples):
        frame = [complex(samples[start + i] * window[i], 0.0) for i in range(fft_size)]
        spec = fft(frame)
        for bi, bins in enumerate(band_bins):
            if not bins:
                continue
            e = 0.0
            for b in bins:
                e += spec[b].real * spec[b].real + spec[b].imag * spec[b].imag
            band_energy[bi] += e / (len(bins) * win_energy)
        frames += 1
        start += fft_size
    if frames == 0:
        raise ValueError("segment shorter than one analysis frame")
    result = [db(e / frames) for e in band_energy]
    # Bands with no FFT bins (lowest bands at small fft sizes) inherit the
    # nearest lower band so they do not distort distances.
    for bi in range(len(result)):
        if not band_bins[bi]:
            result[bi] = result[bi - 1] if bi > 0 else 0.0
    return result


def spectral_distance_db(a_db, b_db):
    """Mean absolute per-band difference in dB."""
    return sum(abs(x - y) for x, y in zip(a_db, b_db)) / len(a_db)


def shape_distance_db(a_db, b_db):
    """Band distance after removing the mean level offset."""
    off = (sum(a_db) - sum(b_db)) / len(a_db)
    return sum(abs((x - off) - y) for x, y in zip(a_db, b_db)) / len(a_db)


def band_tilt_db(bands_db, rate, split_hz=1500.0, low_ref_hz=500.0):
    """Energy tilt: mean of bands above split_hz minus mean below low_ref_hz."""
    _, centers = band_edges(rate)
    hi = [v for v, c in zip(bands_db, centers) if c >= split_hz]
    lo = [v for v, c in zip(bands_db, centers) if c <= low_ref_hz]
    return (sum(hi) / len(hi)) - (sum(lo) / len(lo))


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

def rms_envelope(samples, rate, frame_s=0.025, hop_s=0.010):
    frame = max(1, int(frame_s * rate))
    hop = max(1, int(hop_s * rate))
    env = []
    pos = 0
    while pos + frame <= len(samples):
        seg = samples[pos:pos + frame]
        env.append((pos / rate, rms(seg)))
        pos += hop
    return env, hop / rate


def silent_segments(samples, rate, margin_db=8.0, local_s=2.0):
    """Return (silent_segments, non_silent_count, floor_range, threshold_range).

    The noise floor is tracked locally (low percentile of the RMS envelope
    inside a sliding window) so steady ambience in louder sound fields still
    counts as silence; frames quieter than local floor + margin are silent.
    """
    env, hop = rms_envelope(samples, rate)
    levels = [db(v * v) for _, v in env]
    half = max(1, int(local_s / hop / 2))
    floors = []
    for i in range(len(levels)):
        lo = max(0, i - half)
        hi = min(len(levels), i + half + 1)
        window = sorted(levels[lo:hi])
        floors.append(window[int(0.10 * (len(window) - 1))])
    segments = []
    non_silent = 0
    start = None
    for (t, _), level, floor in zip(env, levels, floors):
        if level <= floor + margin_db:
            if start is None:
                start = t
        else:
            non_silent += 1
            if start is not None:
                segments.append((start, t))
                start = None
    if start is not None:
        segments.append((start, env[-1][0] + hop))
    floor_range = [round(min(floors), 3), round(max(floors), 3)]
    thr_range = [round(min(floors) + margin_db, 3),
                 round(max(floors) + margin_db, 3)]
    return segments, non_silent, floor_range, thr_range


# ---------------------------------------------------------------------------
# Candidate evaluation and filtering
# ---------------------------------------------------------------------------

TRANSIENT_RATIO_MAX = 3.2
DERIVATIVE_RATIO_MAX = 12.0
TONAL_PROMINENCE_MAX_DB = 8.0
# Low-frequency hum (mains/fan) is part of the ambience and is handled by
# band-spectrum matching; the tonal filter targets whistles/whines above it.
TONAL_MIN_HZ = 500.0
LEVEL_STEP_MAX_DB = 3.0


def _short_rms_stats(seg, rate, frame_s=0.005):
    frame = max(1, int(frame_s * rate))
    values = []
    pos = 0
    while pos + frame <= len(seg):
        values.append(rms(seg[pos:pos + frame]))
        pos += frame
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    return max(values), max(median, 1e-9)


def transient_ratio(seg, rate):
    peak, median = _short_rms_stats(seg, rate)
    return peak / median


def derivative_ratio(seg):
    if len(seg) < 3:
        return 0.0
    diffs = sorted(abs(seg[i + 1] - seg[i]) for i in range(len(seg) - 1))
    median = max(diffs[(len(diffs) - 1) // 2], 1e-12)
    return diffs[-1] / median


def tonal_prominence(seg, rate, fft_size=FFT_SIZE, window=None):
    """Largest bin prominence (dB) over a local median; returns (db, hz)."""
    if window is None:
        window = hann_window(fft_size)
    nbins = fft_size // 2
    acc = [0.0] * nbins
    frames = 0
    pos = 0
    while pos + fft_size <= len(seg):
        frame = [complex(seg[pos + i] * window[i], 0.0) for i in range(fft_size)]
        spec = fft(frame)
        for b in range(nbins):
            acc[b] += spec[b].real * spec[b].real + spec[b].imag * spec[b].imag
        frames += 1
        pos += fft_size
    if frames == 0:
        return 0.0, 0.0
    power = [v / frames for v in acc]
    bin_hz = rate / fft_size
    min_bin = max(4, int(TONAL_MIN_HZ / bin_hz))
    best_db, best_hz = 0.0, 0.0
    half_width = 12
    for b in range(min_bin, nbins - half_width - 1):
        neighbors = [power[b + k] for k in range(-half_width, half_width + 1)
                     if k != 0 and b + k >= 0]
        neighbors.sort()
        med = max(neighbors[len(neighbors) // 2], 1e-18)
        prom = db(power[b] / med)
        if prom > best_db:
            best_db, best_hz = prom, b * bin_hz
    return best_db, best_hz


def level_step_db(seg):
    half = len(seg) // 2
    if half == 0:
        return 0.0
    return abs(db(rms(seg[:half]) ** 2) - db(rms(seg[half:]) ** 2))


def stationarity_db(seg, rate, parts=4):
    """Mean band spread across sub-window spectra (lower is steadier)."""
    chunk = len(seg) // parts
    if chunk < FFT_SIZE:
        return 0.0
    spectra = [band_spectrum_db(seg[i * chunk:(i + 1) * chunk], rate)
               for i in range(parts)]
    spread = 0.0
    count = 0
    for b in range(len(spectra[0])):
        col = [s[b] for s in spectra]
        mean = sum(col) / len(col)
        spread += sum(abs(v - mean) for v in col) / len(col)
        count += 1
    return spread / max(count, 1)


def evaluate_candidate(samples, rate, start_s, duration_s, target_db,
                       pre_db, post_db, target_level_db):
    """Compute metrics and filter verdicts for one candidate window."""
    i0 = int(round(start_s * rate))
    i1 = int(round((start_s + duration_s) * rate))
    seg = samples[i0:i1]
    cand_db = band_spectrum_db(seg, rate)
    level = db(rms(seg) ** 2)
    spec = spectral_distance_db(cand_db, target_db)
    shape = shape_distance_db(cand_db, target_db)
    metrics = {
        "start": round(start_s, 3),
        "end": round(start_s + duration_s, 3),
        "spectral_distance_db": spec,
        "shape_distance_db": shape,
        "level_delta_db": abs(level - target_level_db),
        "pre_distance_db": spectral_distance_db(cand_db, pre_db),
        "post_distance_db": spectral_distance_db(cand_db, post_db),
        "stationarity_db": stationarity_db(seg, rate),
        "level_step_db": level_step_db(seg),
        "level_db": level,
        "transient_ratio": transient_ratio(seg, rate),
        "derivative_ratio": derivative_ratio(seg),
    }
    prom, hz = tonal_prominence(seg, rate)
    metrics["tonal_prominence_db"] = prom
    metrics["tonal_frequency_hz"] = hz
    return metrics


def rejection_reasons(metrics):
    reasons = []
    if metrics["transient_ratio"] > TRANSIENT_RATIO_MAX:
        reasons.append("transient")
    if metrics["derivative_ratio"] > DERIVATIVE_RATIO_MAX:
        reasons.append("transient")
    if metrics["tonal_prominence_db"] > TONAL_PROMINENCE_MAX_DB:
        reasons.append("narrowband_tone")
    if metrics["level_step_db"] > LEVEL_STEP_MAX_DB:
        reasons.append("soundfield_switch")
    return sorted(set(reasons))


def score_candidate(metrics):
    """Lower is better: spectral match dominates, steadiness breaks ties."""
    return (metrics["shape_distance_db"]
            + 0.5 * metrics["level_delta_db"]
            + 0.1 * metrics["stationarity_db"])


def find_fill(samples, rate, edit_start, edit_end, context_s=0.35,
              step_s=0.05):
    """Search silent windows and rank them against the edit boundary."""
    duration = edit_end - edit_start
    pre = samples[int((edit_start - context_s) * rate):int(edit_start * rate)]
    post = samples[int(edit_end * rate):int((edit_end + context_s) * rate)]
    pre_db = band_spectrum_db(pre, rate)
    post_db = band_spectrum_db(post, rate)
    target_db = [(a + b) / 2.0 for a, b in zip(pre_db, post_db)]
    pre_level = db(rms(pre) ** 2)
    post_level = db(rms(post) ** 2)
    target_level = (pre_level + post_level) / 2.0

    segments, non_silent, floor_range, thr_range = silent_segments(samples, rate)
    accepted, rejected = [], []
    evaluated = 0
    total = len(samples) / rate
    t = 0.0
    while t + duration <= total:
        t = round(t, 6)
        inside = any(a - 1e-9 <= t and t + duration <= b + 1e-9
                     for a, b in segments)
        # Never let the donor overlap the edit segment itself.
        overlaps_edit = t < edit_end and t + duration > edit_start
        if inside and not overlaps_edit:
            evaluated += 1
            m = evaluate_candidate(samples, rate, t, duration, target_db,
                                   pre_db, post_db, target_level)
            m["score_db"] = score_candidate(m)
            reasons = rejection_reasons(m)
            m["rejection_reasons"] = reasons
            m["accepted"] = not reasons
            (accepted if m["accepted"] else rejected).append(m)
        t += step_s
    accepted.sort(key=lambda m: m["score_db"])
    rejected.sort(key=lambda m: m["score_db"])
    analysis = {
        "bands": BAND_COUNT,
        "band_centers_hz": [round(c, 3) for c in band_edges(rate)[1]],
        "boundary_band_db": {
            "target": [round(v, 3) for v in target_db],
            "pre": [round(v, 3) for v in pre_db],
            "post": [round(v, 3) for v in post_db],
        },
        "context_seconds": context_s,
        "silence_floor_range_db": floor_range,
        "silence_threshold_range_db": thr_range,
        "candidate_count": evaluated,
        "non_silent_frames_skipped": non_silent,
    }
    filters = {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejected_transient": sum("transient" in m["rejection_reasons"]
                                  for m in rejected),
        "rejected_narrowband_tone": sum("narrowband_tone" in m["rejection_reasons"]
                                        for m in rejected),
        "rejected_soundfield_switch": sum("soundfield_switch" in m["rejection_reasons"]
                                          for m in rejected),
        "thresholds": {
            "transient_ratio": TRANSIENT_RATIO_MAX,
            "derivative_ratio": DERIVATIVE_RATIO_MAX,
            "tonal_prominence_db": TONAL_PROMINENCE_MAX_DB,
            "level_step_db": LEVEL_STEP_MAX_DB,
        },
    }
    return {
        "duration": duration,
        "accepted": accepted,
        "rejected": rejected,
        "analysis": analysis,
        "filters": filters,
        "target_level_db": target_level,
    }


# ---------------------------------------------------------------------------
# Preview rendering and splice verification
# ---------------------------------------------------------------------------

def raised_cosine_fades(n):
    """Return (fade_in, fade_out) gain ramps of length n."""
    if n <= 0:
        return [], []
    fin = [0.5 - 0.5 * math.cos(math.pi * i / n) for i in range(n)]
    fout = [0.5 + 0.5 * math.cos(math.pi * i / n) for i in range(n)]
    return fin, fout


def make_isolated_fill(samples, rate, donor_start, duration, fade_s):
    """The donor segment itself, edges shaped with short raised-cosine fades."""
    i0 = int(round(donor_start * rate))
    i1 = int(round((donor_start + duration) * rate))
    fill = samples[i0:i1][:]
    nf = min(int(round(fade_s * rate)), len(fill) // 2)
    fin, fout = raised_cosine_fades(nf)
    for i in range(nf):
        fill[i] *= fin[i]
        fill[len(fill) - nf + i] *= fout[i]
    return fill


def build_preview(samples, rate, edit_start, edit_end, donor_start, fade_s=0.012):
    """Replace the edit segment with donor audio, crossfading into the kept
    audio outside the edit so no removed speech leaks into the joins."""
    edit_len = int(round((edit_end - edit_start) * rate))
    nf = int(round(fade_s * rate))
    e0 = int(round(edit_start * rate))
    d0 = int(round(donor_start * rate)) - nf
    donor = samples[d0:d0 + edit_len + 2 * nf]
    if len(donor) < edit_len + 2 * nf:
        raise ValueError("donor window too close to end of file for fades")
    out = samples[:]
    fin, fout = raised_cosine_fades(nf)
    # Head join: kept audio before the edit crossfades into the donor.
    for i in range(nf):
        pos = e0 - nf + i
        out[pos] = samples[pos] * fout[i] + donor[i] * fin[i]
    # Body of the fill.
    out[e0:e0 + edit_len] = donor[nf:nf + edit_len]
    # Tail join: donor crossfades back into the kept audio after the edit.
    for i in range(nf):
        pos = e0 + edit_len + i
        out[pos] = donor[nf + edit_len + i] * fout[i] + samples[pos] * fin[i]
    return out


def splice_jump_db(preview, rate, splice_s, window_s=0.06):
    """Spectral distance between short windows on either side of a splice."""
    n = int(round(window_s * rate))
    c = int(round(splice_s * rate))
    before = preview[c - n:c]
    after = preview[c:c + n]
    return spectral_distance_db(band_spectrum_db(before, rate),
                                band_spectrum_db(after, rate))


def measure_splices(preview, rate, edit_start, edit_end):
    start_db = splice_jump_db(preview, rate, edit_start)
    end_db = splice_jump_db(preview, rate, edit_end)
    return {
        "start_db": round(start_db, 3),
        "end_db": round(end_db, 3),
        "mean_db": round((start_db + end_db) / 2.0, 3),
    }


# ---------------------------------------------------------------------------
# Sound-field labels (demo scenario)
# ---------------------------------------------------------------------------

def field_at(modes, t):
    for name, (a, b) in modes.items():
        if a <= t < b:
            return name
    return None


def round_metrics(m, nd=3):
    out = {}
    for k, v in m.items():
        out[k] = round(v, nd) if isinstance(v, float) else v
    return out


def run_analysis(wav_path, edit_start, edit_end, outdir, scenario=None,
                 fade_s=0.012):
    samples, rate = read_wav(wav_path)
    result = find_fill(samples, rate, edit_start, edit_end)
    if not result["accepted"]:
        raise SystemExit("no acceptable silent window found")
    best = result["accepted"][0]
    duration = result["duration"]

    os.makedirs(outdir, exist_ok=True)
    preview_path = os.path.join(outdir, "recommended_preview.wav")
    fill_path = os.path.join(outdir, "recommended_isolated_fill.wav")
    preview = build_preview(samples, rate, edit_start, edit_end,
                            best["start"], fade_s)
    write_wav(preview_path, preview, rate)
    write_wav(fill_path,
              make_isolated_fill(samples, rate, best["start"], duration, fade_s),
              rate)
    splice = measure_splices(preview, rate, edit_start, edit_end)
    explanation = result_explanation(result)

    report = {
        "schema_version": 1,
        "input": {
            "path": os.path.abspath(wav_path),
            "sample_rate": rate,
            "channels": 1,
            "bit_depth": 16,
            "encoding": "pcm",
            "duration_seconds": round(len(samples) / rate, 3),
        },
        "edit": {
            "start_seconds": edit_start,
            "end_seconds": edit_end,
            "duration_seconds": round(duration, 3),
            "fade_seconds": fade_s,
        },
        "analysis": result["analysis"],
        "recommendation": dict(round_metrics(best),
                               donor_start_seconds=best["start"],
                               donor_end_seconds=round(best["start"] + duration, 3),
                               donor_duration_seconds=round(duration, 3),
                               target_level_db=round(result["target_level_db"], 3),
                               explanation=explanation),
        "filters": result["filters"],
        "candidates": {
            "accepted": [round_metrics(m) for m in result["accepted"][:12]],
            "rejected": [round_metrics(m) for m in result["rejected"][:12]],
        },
        "preview": {
            "joined_path": os.path.abspath(preview_path),
            "fill_path": os.path.abspath(fill_path),
            "sample_rate": rate,
            "channels": 1,
            "fade": "raised_cosine",
        },
        "splice_check": splice,
    }
    demo = None
    if scenario is not None:
        demo = run_scenario_extras(samples, rate, edit_start, edit_end, result,
                                   best, scenario, outdir, fade_s, report)
    report_path = os.path.join(outdir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    if demo is not None:
        demo_path = os.path.join(outdir, "demo_report.json")
        with open(demo_path, "w", encoding="utf-8") as fh:
            json.dump(demo, fh, ensure_ascii=False, indent=2)
    return report, demo


def run_scenario_extras(samples, rate, edit_start, edit_end, result, best,
                        scenario, outdir, fade_s, report):
    """Field attribution, other-field comparison and markdown explanation."""
    modes = scenario["modes"]
    duration = result["duration"]
    edit_field = field_at(modes, (edit_start + edit_end) / 2.0)
    best_field = field_at(modes, best["start"] + duration / 2.0)

    other = None
    for m in result["accepted"]:
        if m is best:
            continue
        f = field_at(modes, m["start"] + duration / 2.0)
        if f is not None and f != edit_field:
            other = m
            other_field = f
            break

    target_db = report["analysis"]["boundary_band_db"]["target"]
    boundary_tilt = band_tilt_db(target_db, rate)
    rec_db = band_spectrum_db(
        samples[int(best["start"] * rate):int((best["start"] + duration) * rate)],
        rate)
    rec_tilt = band_tilt_db(rec_db, rate)

    other_splice = None
    other_tilt = None
    other_preview_path = None
    if other is not None:
        other_preview_path = os.path.join(outdir, "other_field_preview.wav")
        other_preview = build_preview(samples, rate, edit_start, edit_end,
                                      other["start"], fade_s)
        write_wav(other_preview_path, other_preview, rate)
        other_splice = measure_splices(other_preview, rate, edit_start, edit_end)
        other_db = band_spectrum_db(
            samples[int(other["start"] * rate):int((other["start"] + duration) * rate)],
            rate)
        other_tilt = band_tilt_db(other_db, rate)

    splice = report["splice_check"]
    reduction_db = None
    reduction_pct = None
    if other_splice is not None:
        reduction_db = round(other_splice["mean_db"] - splice["mean_db"], 3)
        reduction_pct = round(100.0 * reduction_db / other_splice["mean_db"], 2)

    demo = {
        "scenario": scenario,
        "recommendation": {
            "donor_start_seconds": best["start"],
            "donor_end_seconds": round(best["start"] + duration, 3),
            "field": best_field,
            "spectral_distance_db": round(best["spectral_distance_db"], 3),
            "level_delta_db": round(best["level_delta_db"], 3),
            "band_tilt_db": round(rec_tilt, 3),
            "explanation": report["recommendation"]["explanation"],
        },
        "why_same_field": {
            "edit_field": edit_field,
            "boundary_band_tilt_db": round(boundary_tilt, 3),
            "recommended_band_tilt_db": round(rec_tilt, 3),
            "other_field": other_field if other else None,
            "other_field_window_seconds": [other["start"], other["end"]] if other else None,
            "other_field_band_tilt_db": round(other_tilt, 3) if other else None,
            "other_field_spectral_distance_db": round(other["spectral_distance_db"], 3) if other else None,
        },
        "filter_validation": {
            "transient_rejections": result["filters"]["rejected_transient"],
            "narrowband_rejections": result["filters"]["rejected_narrowband_tone"],
            "transient_example": first_with(result["rejected"], "transient"),
            "narrowband_example": first_with(result["rejected"], "narrowband_tone"),
        },
        "splice_validation": {
            "recommended": splice,
            "other_field": other_splice,
            "reduction_db": reduction_db,
            "reduction_percent": reduction_pct,
        },
        "outputs": {
            "dialogue": os.path.abspath(report["input"]["path"]),
            "recommended_preview": report["preview"]["joined_path"],
            "recommended_fill": report["preview"]["fill_path"],
            "other_field_preview": os.path.abspath(other_preview_path) if other_preview_path else None,
        },
    }
    md = render_explanation_md(scenario, edit_start, edit_end, edit_field,
                               best, best_field, boundary_tilt, rec_tilt,
                               other, other_field if other else None,
                               other_tilt, result, splice, other_splice,
                               reduction_db, reduction_pct)
    with open(os.path.join(outdir, "explanation.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    return demo


def first_with(rejected, reason):
    for m in rejected:
        if reason in m["rejection_reasons"]:
            return round_metrics(m)
    return None


def render_explanation_md(scenario, edit_start, edit_end, edit_field, best,
                          best_field, boundary_tilt, rec_tilt, other,
                          other_field, other_tilt, result, splice,
                          other_splice, reduction_db, reduction_pct):
    modes = scenario.get("modes", {})
    mode_desc = "、".join("%s %.0f–%.0f s" % (k, v[0], v[1])
                         for k, v in modes.items())
    lines = []
    lines.append("# 空调切换声场中的底噪取样验证")
    lines.append("")
    lines.append("合成对话长 %.0f s，声场依次为 %s；待填补剪辑段为 %.1f–%.1f s，"
                 "因此边界处于 `%s` 声场。"
                 % (scenario.get("duration_seconds", 0), mode_desc,
                    edit_start, edit_end, edit_field))
    lines.append("")
    lines.append("## 推荐窗口")
    lines.append("- 推荐 `%.3f–%.3f s`，落在 `%s` 档声场。"
                 % (best["start"], best["end"], best_field))
    lines.append("- 边界平均频带距离 `%.2f dB`，声压差 `%.2f dB`。"
                 % (best["spectral_distance_db"], best["level_delta_db"]))
    lines.append("- 1.5 kHz 以上相对 500 Hz 以下的高频倾斜：边界 `%.2f dB`，"
                 "推荐窗口 `%.2f dB`。" % (boundary_tilt, rec_tilt))
    if other is not None:
        lines.append("- 其他声场最佳候选为 `%s` 档 `%.2f–%.2f s`，频带距离 `%.2f dB`，"
                     "高于推荐窗口；其高频倾斜为 `%.2f dB`，与边界不一致。"
                     % (other_field, other["start"], other["end"],
                        other["spectral_distance_db"], other_tilt))
    lines.append("")
    lines.append("## 不合格候选")
    f = result["filters"]
    tr = first_with(result["rejected"], "transient")
    nb = first_with(result["rejected"], "narrowband_tone")
    lines.append("- 瞬态过滤拒绝 `%d` 个窗口；示例瞬态比 `%.2f`。"
                 % (f["rejected_transient"], tr["transient_ratio"] if tr else 0.0))
    lines.append("- 窄带鸣声过滤拒绝 `%d` 个窗口；示例在 `%.0f Hz` 处突出 `%.2f dB`。"
                 % (f["rejected_narrowband_tone"],
                    nb["tonal_frequency_hz"] if nb else 0.0,
                    nb["tonal_prominence_db"] if nb else 0.0))
    lines.append("")
    lines.append("## 拼接谱跳变")
    lines.append("- 推荐窗口：`%.2f dB`（起点 %.2f dB，终点 %.2f dB）。"
                 % (splice["mean_db"], splice["start_db"], splice["end_db"]))
    if other_splice is not None:
        lines.append("- 其他声场窗口：`%.2f dB`（起点 %.2f dB，终点 %.2f dB）。"
                     % (other_splice["mean_db"], other_splice["start_db"],
                        other_splice["end_db"]))
        lines.append("- 跳变降低 `%.2f dB`，即 `%.1f%%`。"
                     % (reduction_db, reduction_pct))
    lines.append("")
    lines.append("## 审计摘要")
    for sentence in result_explanation(result):
        lines.append("- " + sentence)
    lines.append("")
    return "\n".join(lines)


def result_explanation(result):
    best = result["accepted"][0]
    duration = result["duration"]
    return [
        "供体 %.2f–%.2fs 与边界平均谱相差 %.2f dB。"
        % (best["start"], best["start"] + duration, best["spectral_distance_db"]),
        "频带形状差 %.2f dB，声压差 %.2f dB；前/后边界独立距离分别为 %.2f/%.2f dB。"
        % (best["shape_distance_db"], best["level_delta_db"],
           best["pre_distance_db"], best["post_distance_db"]),
        "共评估 %d 个静音窗口，瞬态剔除 %d 个，窄带鸣声剔除 %d 个，声场阶跃剔除 %d 个。"
        % (result["analysis"]["candidate_count"],
           result["filters"]["rejected_transient"],
           result["filters"]["rejected_narrowband_tone"],
           result["filters"]["rejected_soundfield_switch"]),
    ]


# ---------------------------------------------------------------------------
# Synthetic sound-field generator (demo / self-test)
# ---------------------------------------------------------------------------

def _colored_noise(rng, n, tilt_per_octave_db, hum_hz=0.0, hum_amp=0.0,
                   rate=16000):
    """Gaussian noise shaped by a per-octave tilt, plus optional mains hum."""
    spec = [0.0j] * FFT_SIZE
    out = [0.0] * n
    phase = 0.0
    pos = 0
    while pos < n:
        for b in range(1, FFT_SIZE // 2):
            freq = b * rate / FFT_SIZE
            gain = 10.0 ** (tilt_per_octave_db
                            * math.log2(max(freq, 20.0) / 500.0) / 20.0)
            spec[b] = complex(rng.gauss(0, 1), rng.gauss(0, 1)) * gain
        for b in range(1, FFT_SIZE // 2):
            spec[FFT_SIZE - b] = spec[b].conjugate()
        frame = fft(spec)
        for i in range(FFT_SIZE):
            if pos + i < n:
                out[pos + i] = frame[(FFT_SIZE - i) % FFT_SIZE].real / FFT_SIZE
        pos += FFT_SIZE
    if hum_hz and hum_amp:
        for i in range(n):
            out[i] += hum_amp * math.sin(2.0 * math.pi * hum_hz * i / rate)
    peak = max(abs(v) for v in out) or 1.0
    return [v / peak for v in out]


def synthesize_dialogue(scenario, seed=20261121):
    """Render the AC-switching demo: three noise fields, speech-like bursts,
    a click and a narrowband tone as distractors."""
    import random
    rng = random.Random(seed)
    rate = scenario["sample_rate"]
    total = int(scenario["duration_seconds"] * rate)
    # Field presets: (level_dbfs, tilt per octave dB, hum hz, hum amp)
    fields = {
        "low": (-38.0, -4.0, 0.0, 0.0),
        "high": (-37.0, -1.5, 125.0, 0.004),
        "off": (-47.0, -6.0, 0.0, 0.0),
    }
    samples = [0.0] * total
    for name, (a, b) in scenario["modes"].items():
        level, tilt, hum_hz, hum_amp = fields[name]
        n0, n1 = int(a * rate), int(b * rate)
        noise = _colored_noise(rng, n1 - n0, tilt, hum_hz, hum_amp, rate)
        amp = 10.0 ** (level / 20.0)
        for i in range(n1 - n0):
            samples[n0 + i] = noise[i] * amp
    # Speech-like bursts: band-limited noise syllables with silent gaps.
    t = 0.4
    while t < scenario["duration_seconds"] - 1.0:
        dur = rng.uniform(0.25, 0.9)
        gap = rng.uniform(0.5, 2.2)
        i0 = int(t * rate)
        n = int(dur * rate)
        burst = _colored_noise(rng, n, -8.0, rate=rate)
        amp = 10.0 ** (-16.0 / 20.0)
        for i in range(n):
            if i0 + i < total:
                env = math.sin(math.pi * i / n) ** 2
                samples[i0 + i] += burst[i] * amp * env
        t += dur + gap
    dis = scenario.get("distractors", {})
    if "click_seconds" in dis:
        c = int(dis["click_seconds"] * rate)
        for i in range(-8, 9):
            if 0 <= c + i < total:
                samples[c + i] += (0.35 if i % 2 == 0 else -0.3) * math.exp(-abs(i) / 2.0)
    if "narrowband_tone_seconds" in dis:
        a, b = dis["narrowband_tone_seconds"]
        freq = dis.get("narrowband_tone_hz", 3150.0)
        for i in range(int(a * rate), int(b * rate)):
            if 0 <= i < total:
                samples[i] += 0.01 * math.sin(2.0 * math.pi * freq * i / rate)
    return samples, rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Noise-floor sampling assistant for broadcast editing.")
    sub = parser.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="find a fill window and render previews")
    pa.add_argument("--input", required=True, help="dialogue WAV (16-bit PCM)")
    pa.add_argument("--edit-start", type=float, required=True)
    pa.add_argument("--edit-end", type=float, required=True)
    pa.add_argument("--fade", type=float, default=0.012,
                    help="raised-cosine fade seconds at each splice")
    pa.add_argument("--scenario", help="optional scenario JSON for field labels")
    pa.add_argument("--outdir", required=True)

    ps = sub.add_parser("synthesize", help="render the AC-switching demo WAV")
    ps.add_argument("--scenario", required=True)
    ps.add_argument("--output", required=True)
    ps.add_argument("--seed", type=int, default=20261121)

    args = parser.parse_args(argv)
    if args.command == "synthesize":
        with open(args.scenario, encoding="utf-8") as fh:
            scenario = json.load(fh)
        samples, rate = synthesize_dialogue(scenario, args.seed)
        write_wav(args.output, samples, rate)
        print("wrote %s (%.1f s, %d Hz)" % (args.output, len(samples) / rate, rate))
        return 0

    scenario = None
    if args.scenario:
        with open(args.scenario, encoding="utf-8") as fh:
            scenario = json.load(fh)
    report, demo = run_analysis(args.input, args.edit_start, args.edit_end,
                                args.outdir, scenario, args.fade)
    rec = report["recommendation"]
    print("recommended %.3f–%.3f s (score %.3f dB, distance %.3f dB)"
          % (rec["donor_start_seconds"], rec["donor_end_seconds"],
             rec["score_db"], rec["spectral_distance_db"]))
    print("splice jump %.2f dB (start %.2f, end %.2f)"
          % (report["splice_check"]["mean_db"],
             report["splice_check"]["start_db"],
             report["splice_check"]["end_db"]))
    if demo is not None:
        sv = demo["splice_validation"]
        print("other-field splice %.2f dB -> reduction %.2f dB (%.1f%%)"
              % (sv["other_field"]["mean_db"], sv["reduction_db"],
                 sv["reduction_percent"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

