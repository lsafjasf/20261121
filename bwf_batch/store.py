"""Version store: traceable per-adjustment history for batch processing.

Directory layout under the output directory:

    project.json                         # files, version pointers
    <stem>/master.wav                    # current broadcast master (BWF)
    <stem>/envelope.json                 # current envelope state
    <stem>/versions/v0000/meta.json      # version metadata
    <stem>/versions/v0000/envelope.json  # envelope snapshot after the edit
    <stem>/versions/v0000/source_ref.wav # source samples the edit referenced

`source_ref.wav` holds a bit-exact slice of the source samples the
adjustment referenced (the affected range; the whole source for the
baseline version), so every adjustment can be audited and replayed
independently of the original source file.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from .bwf import BwfFile, read_bwf, write_bwf
from .envelope import Envelope


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def envelope_hash(env: Envelope) -> str:
    canonical = json.dumps(env.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class ProjectStore:
    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.project_path = os.path.join(outdir, "project.json")
        if os.path.exists(self.project_path):
            with open(self.project_path, "r", encoding="utf-8") as fh:
                self.project = json.load(fh)
        else:
            self.project = {"files": {}}
            self._save_project()

    def _save_project(self) -> None:
        with open(self.project_path, "w", encoding="utf-8") as fh:
            json.dump(self.project, fh, indent=2, sort_keys=True)

    # -- paths -------------------------------------------------------------

    def stem_dir(self, stem: str) -> str:
        return os.path.join(self.outdir, stem)

    def master_path(self, stem: str) -> str:
        return os.path.join(self.stem_dir(stem), "master.wav")

    def envelope_path(self, stem: str) -> str:
        return os.path.join(self.stem_dir(stem), "envelope.json")

    def versions_dir(self, stem: str) -> str:
        return os.path.join(self.stem_dir(stem), "versions")

    # -- file registration -------------------------------------------------

    def register_file(self, stem: str, source_path: str) -> dict:
        entry = self.project["files"].get(stem)
        if entry is None:
            entry = {
                "source": os.path.abspath(source_path),
                "source_sha256": _sha256_path(source_path),
                "versions": [],
                "current_version": None,
            }
            self.project["files"][stem] = entry
        else:
            digest = _sha256_path(source_path)
            if digest != entry["source_sha256"]:
                raise ValueError(
                    "source file changed since last run: %s" % source_path
                )
        self._save_project()
        return entry

    # -- envelope state ----------------------------------------------------

    def load_envelope(self, stem: str) -> Envelope:
        path = self.envelope_path(stem)
        if not os.path.exists(path):
            return Envelope()
        with open(path, "r", encoding="utf-8") as fh:
            return Envelope.from_dict(json.load(fh))

    def save_envelope(self, stem: str, env: Envelope) -> None:
        os.makedirs(self.stem_dir(stem), exist_ok=True)
        with open(self.envelope_path(stem), "w", encoding="utf-8") as fh:
            json.dump(env.to_dict(), fh, indent=2)

    # -- versions ----------------------------------------------------------

    def save_master(self, stem: str, master: BwfFile) -> None:
        os.makedirs(self.stem_dir(stem), exist_ok=True)
        write_bwf(self.master_path(stem), master)

    def load_master(self, stem: str) -> BwfFile:
        return read_bwf(self.master_path(stem))

    def add_version(
        self,
        stem: str,
        op: dict,
        env_before: Envelope,
        env_after: Envelope,
        affected_frames: tuple | None,
        source_ref: BwfFile | None,
        render_mode: str,
    ) -> str:
        """Persist one adjustment: metadata, envelope snapshot, source samples."""
        entry = self.project["files"][stem]
        version = "v%04d" % len(entry["versions"])
        vdir = os.path.join(self.versions_dir(stem), version)
        os.makedirs(vdir, exist_ok=True)

        source_ref_name = None
        if source_ref is not None:
            source_ref_name = "source_ref.wav"
            write_bwf(os.path.join(vdir, source_ref_name), source_ref)

        env_path = os.path.join(vdir, "envelope.json")
        with open(env_path, "w", encoding="utf-8") as fh:
            json.dump(env_after.to_dict(), fh, indent=2)

        meta = {
            "version": version,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "op": op,
            "merge_mode": env_after.merge_mode,
            "render_mode": render_mode,  # "full" | "incremental" | "none"
            "affected_frames": (
                {"start": affected_frames[0], "end": affected_frames[1]}
                if affected_frames
                else None
            ),
            "source": entry["source"],
            "source_sha256": entry["source_sha256"],
            "source_ref": source_ref_name,
            "envelope_sha256_before": envelope_hash(env_before),
            "envelope_sha256_after": envelope_hash(env_after),
        }
        meta_path = os.path.join(vdir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        entry["versions"].append(version)
        entry["current_version"] = version
        self._save_project()
        return version

    def iter_version_meta(self, stem: str):
        for version in self.project["files"][stem]["versions"]:
            path = os.path.join(self.versions_dir(stem), version, "meta.json")
            with open(path, "r", encoding="utf-8") as fh:
                yield json.load(fh)
