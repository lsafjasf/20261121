"""End-to-end batch runner for codec delay and pre-onset leakage."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .codecs import (
    CODEC_CONFIGS,
    CodecConfig,
    decode_lossy,
    encode_lossy,
    normalize_reference,
    verify_codecs,
)
from .dsp import DelayEstimate, align_signal, estimate_delay
from .fixtures import build_fixtures
from .metrics import (
    MeasurementSummary,
    MetricSettings,
    evaluate_acceptance,
    measure_onsets,
)
from .wavio import AudioClip, read_mono_wav, write_mono_wav, write_stereo_ab


@dataclass
class CodecResult:
    config: CodecConfig
    delay: DelayEstimate
    metrics: MeasurementSummary
    acceptance: dict[str, object]
    reference: AudioClip
    aligned: AudioClip
    encoded_path: str
    aligned_path: str
    ab_path: str
    unaligned_path: str
    encoded_bytes: int


@dataclass
class CaseResult:
    case_id: str
    title: str
    source_file: str
    description: str
    expected: str
    onsets_s: list[float]
    reference: AudioClip
    reference_path: str
    codecs: list[CodecResult]
    acceptance: dict[str, object]


@dataclass
class LabResult:
    output_dir: Path
    sample_rate: int
    settings: MetricSettings
    cases: list[CaseResult]
    generated_at: str
    acceptance: dict[str, object]


def load_manifest(fixtures_dir: str | Path) -> dict[str, object]:
    path = Path(fixtures_dir) / "manifest.json"
    if not path.exists():
        build_fixtures(fixtures_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _case_acceptance(
    codec_results: list[CodecResult], expected: str
) -> dict[str, object]:
    passed = all(bool(item.acceptance["passed"]) for item in codec_results)
    failed = [
        {
            "codec": item.config.key,
            "message": item.acceptance["message"],
            "unexpected_onset_indices": item.acceptance[
                "unexpected_onset_indices"
            ],
        }
        for item in codec_results
        if not item.acceptance["passed"]
    ]
    return {
        "passed": passed,
        "expected": expected,
        "failed_codecs": failed,
    }


def _run_codec(
    reference: AudioClip,
    reference_path: Path,
    case_dir: Path,
    output_dir: Path,
    onsets_s: list[float],
    expected: str,
    config: CodecConfig,
    settings: MetricSettings,
) -> CodecResult:
    codec_dir = case_dir / config.key
    codec_dir.mkdir(parents=True, exist_ok=True)
    encoded_path = codec_dir / f"encoded{config.extension}"
    unaligned_path = codec_dir / "decoded_unaligned.wav"
    aligned_path = codec_dir / "aligned.wav"
    ab_path = codec_dir / "ab.wav"
    encode_lossy(reference_path, encoded_path, config)
    decode_lossy(encoded_path, unaligned_path, reference.sample_rate)
    unaligned = read_mono_wav(unaligned_path)
    if unaligned.sample_rate != reference.sample_rate:
        raise RuntimeError("decoded sample rate does not match the reference")
    delay = estimate_delay(reference.samples, unaligned.samples, reference.sample_rate)
    aligned_samples = align_signal(
        unaligned.samples, len(reference.samples), delay.delay_samples
    )
    aligned = AudioClip(aligned_samples, reference.sample_rate)
    write_mono_wav(aligned_path, aligned.samples, aligned.sample_rate)
    write_stereo_ab(
        ab_path, reference.samples, aligned.samples, reference.sample_rate
    )
    metrics = measure_onsets(
        reference.samples,
        aligned.samples,
        reference.sample_rate,
        onsets_s,
        settings,
    )
    acceptance = evaluate_acceptance(metrics, expected, delay.confidence, settings)
    return CodecResult(
        config=config,
        delay=delay,
        metrics=metrics,
        acceptance=acceptance,
        reference=reference,
        aligned=aligned,
        encoded_path=_relative(encoded_path, output_dir),
        aligned_path=_relative(aligned_path, output_dir),
        ab_path=_relative(ab_path, output_dir),
        unaligned_path=_relative(unaligned_path, output_dir),
        encoded_bytes=encoded_path.stat().st_size,
    )


def run_lab(
    fixtures_dir: str | Path,
    output_dir: str | Path,
    codecs: tuple[CodecConfig, ...] = CODEC_CONFIGS,
    metric_settings: MetricSettings | None = None,
) -> LabResult:
    fixtures_dir = Path(fixtures_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(fixtures_dir)
    verify_codecs(codecs)
    sample_rate = int(manifest.get("sample_rate", 48_000))
    settings = metric_settings or MetricSettings(
        window_ms=float(manifest.get("window_ms", 20.0)),
        guard_ms=float(manifest.get("guard_ms", 2.0)),
    )
    case_results: list[CaseResult] = []
    raw_cases = manifest.get("cases", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("manifest must contain a non-empty cases list")
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("each manifest case must be an object")
        case_id = str(raw_case["id"])
        if not case_id or "/" in case_id or "\\" in case_id or case_id in {".", ".."}:
            raise ValueError(f"invalid case id: {case_id!r}")
        source_file = Path(str(raw_case["file"]))
        if not source_file.is_absolute():
            source_file = fixtures_dir / source_file
        if not source_file.exists():
            raise FileNotFoundError(source_file)
        onsets = [float(value) for value in raw_case.get("onsets_s", [])]
        if not onsets:
            raise ValueError(f"case {case_id} has no onsets")
        expected = str(raw_case.get("expect", "source-prelude"))
        if expected not in {"codec-added", "source-prelude"}:
            raise ValueError(f"case {case_id} has unknown expectation: {expected!r}")
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        reference_path = case_dir / "reference.wav"
        normalize_reference(source_file, reference_path, sample_rate)
        reference = read_mono_wav(reference_path)
        invalid_onsets = [
            onset for onset in onsets if onset <= 0.0 or onset >= reference.duration_s
        ]
        if invalid_onsets:
            raise ValueError(
                f"case {case_id} has onsets outside its duration: {invalid_onsets}"
            )
        codec_results = [
            _run_codec(
                reference,
                reference_path,
                case_dir,
                output_dir,
                onsets,
                expected,
                config,
                settings,
            )
            for config in codecs
        ]
        case_results.append(
            CaseResult(
                case_id=case_id,
                title=str(raw_case.get("title", case_id)),
                source_file=str(raw_case["file"]),
                description=str(raw_case.get("description", "")),
                expected=expected,
                onsets_s=onsets,
                reference=reference,
                reference_path=_relative(reference_path, output_dir),
                codecs=codec_results,
                acceptance=_case_acceptance(codec_results, expected),
            )
        )
        for item in codec_results:
            print(
                f"{case_id:18s} {item.config.key:7s} "
                f"delay={item.delay.delay_ms:+.2f}ms "
                f"worst_added={item.metrics.worst_added_dbfs} "
                f"{'PASS' if item.acceptance['passed'] else 'CHECK'}"
            )
    acceptance = {
        "passed": all(case.acceptance["passed"] for case in case_results),
        "passed_cases": sum(case.acceptance["passed"] for case in case_results),
        "total_cases": len(case_results),
    }
    result = LabResult(
        output_dir=output_dir,
        sample_rate=sample_rate,
        settings=settings,
        cases=case_results,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        acceptance=acceptance,
    )
    _write_summary_json(result)
    _write_metrics_csv(result)
    from .report import build_report

    build_report(result)
    return result


def _onset_dict(onset) -> dict[str, object]:
    return asdict(onset)


def _codec_dict(codec: CodecResult, reference: AudioClip) -> dict[str, object]:
    duration_s = reference.duration_s
    actual_kbps = codec.encoded_bytes * 8.0 / max(duration_s, 1e-9) / 1000.0
    return {
        "key": codec.config.key,
        "label": codec.config.label,
        "encoder": codec.config.encoder,
        "target": codec.config.target,
        "args": list(codec.config.args),
        "encoded_bytes": codec.encoded_bytes,
        "actual_kbps": round(actual_kbps, 2),
        "delay_samples": codec.delay.delay_samples,
        "delay_ms": round(codec.delay.delay_ms, 4),
        "delay_confidence": round(codec.delay.confidence, 6),
        "delay_method": codec.delay.method,
        "worst_added_dbfs": codec.metrics.worst_added_dbfs,
        "worst_added_onset_index": codec.metrics.worst_added_onset_index,
        "counts": {
            "codec_added": codec.metrics.codec_added_count,
            "source_prelude": codec.metrics.source_prelude_count,
            "mixed": codec.metrics.mixed_count,
            "no_added_energy": codec.metrics.no_added_energy_count,
        },
        "acceptance": codec.acceptance,
        "files": {
            "encoded": codec.encoded_path,
            "aligned": codec.aligned_path,
            "ab": codec.ab_path,
            "unaligned": codec.unaligned_path,
        },
        "onsets": [_onset_dict(onset) for onset in codec.metrics.onsets],
    }


def _write_summary_json(result: LabResult) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": result.generated_at,
        "sample_rate": result.sample_rate,
        "settings": asdict(result.settings),
        "acceptance": result.acceptance,
        "cases": [
            {
                "id": case.case_id,
                "title": case.title,
                "source_file": case.source_file,
                "description": case.description,
                "expected": case.expected,
                "onsets_s": case.onsets_s,
                "reference_file": case.reference_path,
                "acceptance": case.acceptance,
                "codecs": [
                    _codec_dict(codec, case.reference) for codec in case.codecs
                ],
            }
            for case in result.cases
        ],
    }
    (result.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_metrics_csv(result: LabResult) -> None:
    path = result.output_dir / "metrics.csv"
    fields = [
        "case",
        "codec",
        "onset_index",
        "onset_s",
        "window_start_s",
        "window_stop_s",
        "reference_pre_dbfs",
        "decoded_pre_dbfs",
        "codec_added_dbfs",
        "source_dominance_db",
        "level_change_db",
        "reference_pre_to_attack_db",
        "decoded_pre_to_attack_db",
        "classification",
        "delay_samples",
        "delay_ms",
        "delay_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in result.cases:
            for codec in case.codecs:
                for onset in codec.metrics.onsets:
                    writer.writerow(
                        {
                            "case": case.case_id,
                            "codec": codec.config.key,
                            "onset_index": onset.index,
                            "onset_s": f"{onset.onset_s:.6f}",
                            "window_start_s": f"{onset.window_start_s:.6f}",
                            "window_stop_s": f"{onset.window_stop_s:.6f}",
                            "reference_pre_dbfs": f"{onset.reference_pre_dbfs:.3f}",
                            "decoded_pre_dbfs": f"{onset.decoded_pre_dbfs:.3f}",
                            "codec_added_dbfs": (
                                ""
                                if onset.codec_added_dbfs is None
                                else f"{onset.codec_added_dbfs:.3f}"
                            ),
                            "source_dominance_db": (
                                ""
                                if onset.source_dominance_db is None
                                else f"{onset.source_dominance_db:.3f}"
                            ),
                            "level_change_db": (
                                ""
                                if onset.level_change_db is None
                                else f"{onset.level_change_db:.3f}"
                            ),
                            "reference_pre_to_attack_db": (
                                f"{onset.reference_pre_to_attack_db:.3f}"
                            ),
                            "decoded_pre_to_attack_db": (
                                f"{onset.decoded_pre_to_attack_db:.3f}"
                            ),
                            "classification": onset.classification,
                            "delay_samples": codec.delay.delay_samples,
                            "delay_ms": f"{codec.delay.delay_ms:.4f}",
                            "delay_confidence": f"{codec.delay.confidence:.6f}",
                        }
                    )
