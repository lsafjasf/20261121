"""Command-line entry point: render and verify both layouts offline.

Run with::

    python3 -m spatial_export.main --output-dir out

No network access or third-party package is used at any point.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .audio import RenderConfig
from .exporter import export_layout
from .layouts import LAYOUT_ASYM, LAYOUT_CUBE
from .panner import VbapPanner
from .trajectory import demo_trajectory
from .verifier import verify_all


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline multi-channel VBAP WAV exporter and verifier"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out"),
        help="directory for WAV files, manifests and reports",
    )
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--bit-depth", type=int, default=24, choices=(16, 24, 32))
    args = parser.parse_args()

    config = RenderConfig(
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        bit_depth=args.bit_depth,
    )
    trajectory = demo_trajectory()
    output_dir = args.output_dir

    summary: dict = {
        "method": "3D VBAP (Vector Base Amplitude Panning), Pulkki 1997",
        "trajectory": trajectory.name,
        "config": asdict(config),
        "layouts": {},
    }

    all_passed = True
    for layout in (LAYOUT_CUBE, LAYOUT_ASYM):
        panner = VbapPanner(layout)
        coverage = panner.coverage_sample()
        result = export_layout(layout, trajectory, config, output_dir)
        report = verify_all(layout, trajectory, config, result)
        all_passed = all_passed and report.passed
        summary["layouts"][layout.name] = {
            "description": layout.description,
            "channel_order": list(layout.channel_order),
            "sphere_coverage_scan": coverage,
            "verification": report.checks,
            "passed": report.passed,
            "artifacts": {
                "multichannel_wav": result.multichannel_wav.name,
                "manifest": result.manifest_path.name,
                "per_channel_wav_count": len(result.per_channel_wavs),
            },
        }

    summary["all_checks_passed"] = all_passed
    report_path = output_dir / "verification_report.json"
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"report written to {report_path}")
    print(f"all checks passed: {all_passed}")
    for name, data in summary["layouts"].items():
        cov = data["sphere_coverage_scan"]
        print(
            f"- {name}: coverage={cov['coverage_ratio']:.4f}, "
            f"degraded_samples={cov['degraded_samples']}, "
            f"max_pan_error={data['verification']['pan_direction']['max_angular_error_deg']} deg, "
            f"passed={data['passed']}"
        )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
