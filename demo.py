"""Generate the synthetic meeting and build a real relisten package.

Run from the repository root::

    python3 demo.py

Outputs land in ``artifacts/demo/`` and can be inspected directly.
"""

from __future__ import annotations

from pathlib import Path

from relisten.builder import build_package
from relisten.model import BuildConfig
from tests.fixtures import EXPECTED_GROUPS, build_meeting


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "artifacts" / "demo" / "package"
    meeting = build_meeting(root / "artifacts" / "demo")
    manifest = build_package(
        meeting["tracks"],
        meeting["utterances"],
        output,
        BuildConfig(),
    )

    print(f"annotations : {meeting['annotations_path']}")
    print(f"package dir : {output}")
    print(f"speakers    : {', '.join(manifest['speakers'])}")
    for speaker_id in manifest["speakers"]:
        count = EXPECTED_GROUPS[speaker_id]
        rate = next(
            track["sample_rate"]
            for track in manifest["tracks"]
            if track["speaker_id"] == speaker_id
        )
        print(f"  {speaker_id}: {count} clips @ {rate} Hz")
    print(f"manifest    : {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
