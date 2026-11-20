"""Local HTTP service for the ECHOLAB bat screening desk."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import parse_qs, quote, unquote, urlparse
import uuid

import numpy as np
import soundfile as sf

from acoustics import (
    MONITOR_RATE,
    SourceBank,
    ensure_builtin_sources,
    export_bundle,
    process_selection,
    run_selftest,
    spectrogram_image,
    waveform_overview,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"
EXPORT_DIR = ROOT / "exports"
RENDER_DIR = ROOT / "renders"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

BANK = SourceBank(DATA_DIR)
DEMO, CALIBRATION = ensure_builtin_sources(BANK)
RENDER_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SPECTROGRAM_CACHE: dict[tuple, dict] = {}
EXPORT_HISTORY: dict[str, dict] = {}

for upload_path in sorted(UPLOAD_DIR.glob("*")):
    if not upload_path.is_file():
        continue
    display_name = re.sub(r"^[0-9a-f]{8}_", "", upload_path.name)
    try:
        BANK.add(upload_path, display_name, kind="upload")
    except Exception:
        continue


def _default_selection(source) -> dict:
    if source.id == DEMO.id:
        start = int(0.10 * source.rate)
        end = int(2.60 * source.rate)
    else:
        start = 0
        end = source.frames
    return {
        "start_frame": start,
        "end_frame": end,
        "start_seconds": start / source.rate,
        "end_seconds": end / source.rate,
    }


def _source_payload(source) -> dict:
    return {
        **source.metadata(),
        "default_selection": _default_selection(source),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _load_export_history() -> None:
    for manifest_path in EXPORT_DIR.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            export_id = manifest.get("export_id", manifest_path.parent.name)
            EXPORT_HISTORY[export_id] = manifest
        except (OSError, ValueError):
            continue


def _export_payload(export_id: str, manifest: dict) -> dict:
    base = f"/exports/{quote(export_id)}/"
    manifest_name = (
        manifest.get("files", {}).get("manifest", {}).get("name", "manifest.json")
    )
    return {
        **manifest,
        "urls": {
            "raw": base + quote(manifest["files"]["raw_clip"]["name"]),
            "monitor": base + quote(manifest["files"]["monitor_clip"]["name"]),
            "manifest": base + quote(manifest_name),
            "mapping": base + quote(manifest["files"]["mapping_table"]["name"]),
        },
    }


_load_export_history()


def _query_int(query: dict, name: str, default: int) -> int:
    try:
        return int(float(query.get(name, [default])[0]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"参数 {name} 必须是整数") from exc


def _parse_multipart(content_type: str, body: bytes) -> tuple[str, bytes]:
    match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not match:
        raise ValueError("上传请求缺少 multipart boundary")
    boundary = ("--" + match.group(1)).encode("latin-1")
    for part in body.split(boundary)[1:-1]:
        part = part.strip(b"\r\n")
        headers, separator, payload = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        filename_match = re.search(rb'filename="([^"]*)"', headers)
        if not filename_match:
            continue
        filename = filename_match.group(1).decode("utf-8", "replace")
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        return filename, payload
    raise ValueError("未找到上传音频")


def _selection_from_request(payload: dict) -> tuple:
    source = BANK.get(str(payload.get("source_id", "")))
    try:
        start = int(payload["start_frame"])
        end = int(payload["end_frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("选区端点必须是源样本帧号") from exc
    start = max(0, min(start, source.frames - 1))
    end = max(start + 1, min(end, source.frames))
    if end - start < 64:
        raise ValueError("选区至少需要 64 个源样本")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("处理参数无效")
    return source, start, end, params


def _monitor_peaks(audio: np.ndarray, points: int = 180) -> dict:
    points = max(20, min(points, 600))
    if audio.size == 0:
        return {"min": [], "max": []}
    edges = np.linspace(0, len(audio), points + 1).astype(int)
    minimums = []
    maximums = []
    for index in range(points):
        section = audio[edges[index]:max(edges[index + 1], edges[index] + 1)]
        minimums.append(float(np.min(section)) if section.size else 0.0)
        maximums.append(float(np.max(section)) if section.size else 0.0)
    return {"min": minimums, "max": maximums}


def _render_payload(source, start: int, end: int, rendered: dict, render_id: str) -> dict:
    factor = rendered["time_factor"]
    duration = rendered["monitor_duration"]
    return {
        "ok": True,
        "render_id": render_id,
        "url": f"/render/{render_id}.wav",
        "source_id": source.id,
        "source_rate": source.rate,
        "source_frames": source.frames,
        "selection": {
            "start_frame": start,
            "end_frame": end,
            "frames": end - start,
            "duration": (end - start) / source.rate,
        },
        "monitor": {
            "rate": rendered["monitor_rate"],
            "frames": rendered["monitor_frames"],
            "duration": duration,
        },
        "time_factor": factor,
        "params": rendered["params"],
        "frequency_map": rendered["frequency_map"],
        "mapping": {
            "source_sample_at_monitor_seconds": (
                f"round({start} + monitor_seconds * {source.rate} / {factor:g})"
            ),
            "monitor_seconds_at_source_sample": (
                f"(source_sample - {start}) * {factor:g} / {source.rate}"
            ),
            "source_hz_at_heard_hz": rendered["frequency_map"]["source_hz"],
            "start": {"source_frame": start, "monitor_seconds": 0.0},
            "end": {"source_frame": end, "monitor_seconds": duration},
        },
        "wave_peaks": _monitor_peaks(rendered["audio"]),
    }


def _prune_renders(max_age_seconds: int = 6 * 60 * 60) -> None:
    cutoff = time.time() - max_age_seconds
    for path in RENDER_DIR.glob("*.wav"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


class EchoLabHandler(BaseHTTPRequestHandler):
    server_version = "EchoLab/1.0"

    def log_message(self, format_string: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": message}, status)

    def _send_file(
        self,
        path: Path,
        content_type: str | None = None,
        download_name: str | None = None,
    ) -> None:
        if not path.is_file():
            self._send_error_json("文件不存在", HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes=") and not download_name:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                start = max(0, start)
                end = min(size - 1, end)
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header(
            "Content-Type",
            content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            ascii_name = download_name.encode("ascii", "replace").decode("ascii")
            self.send_header(
                "Content-Disposition",
                f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(download_name)}",
            )
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def _send_binary(
        self,
        data: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _read_body(self, limit: int = MAX_UPLOAD_BYTES) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Content-Length 无效")
        if length <= 0:
            raise ValueError("请求体为空")
        if length > limit:
            raise ValueError("上传文件超过 512 MB 限制")
        return self.rfile.read(length)

    def _read_json(self) -> dict:
        try:
            return json.loads(self._read_body(8 * 1024 * 1024).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("JSON 请求体无效") from exc

    def do_GET(self) -> None:
        try:
            self._handle_get()
        except ValueError as exc:
            self._send_error_json(str(exc))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            print(f"GET error: {exc!r}")
            self._send_error_json(f"服务错误：{exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            self._handle_post()
        except ValueError as exc:
            self._send_error_json(str(exc))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            print(f"POST error: {exc!r}")
            self._send_error_json(f"处理失败：{exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_HEAD(self) -> None:
        try:
            self._handle_get()
        except ValueError as exc:
            self._send_error_json(str(exc))

    def _handle_get(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if path == "/api/health":
            self._send_json({"ok": True, "monitor_rate": MONITOR_RATE})
            return
        if path == "/api/bootstrap":
            sources = [_source_payload(source) for source in BANK.sources.values()]
            exports = sorted(
                EXPORT_HISTORY.values(), key=lambda item: item.get("created_utc", ""), reverse=True
            )
            self._send_json(
                {
                    "ok": True,
                    "app": "ECHOLAB",
                    "version": "1.0",
                    "monitor_rate": MONITOR_RATE,
                    "sources": sources,
                    "default_source_id": DEMO.id,
                    "calibration_source_id": CALIBRATION.id,
                    "defaults": {
                        "mode": "heterodyne",
                        "center_hz": 70_000,
                        "bandwidth_hz": 16_000,
                        "sideband": "USB",
                        "factor": 10,
                        "f_min": 10_000,
                        "f_max": 96_000,
                    },
                    "exports": [
                        _export_payload(item.get("export_id", ""), item) for item in exports[:8]
                    ],
                }
            )
            return
        if path.startswith("/api/source/"):
            source = BANK.get(path.rsplit("/", 1)[-1])
            self._send_json({"ok": True, "source": _source_payload(source)})
            return
        if path == "/api/selftest":
            source_id = query.get("source_id", [CALIBRATION.id])[0]
            source = BANK.get(source_id)
            self._send_json({"ok": True, "result": run_selftest(BANK, source)})
            return
        if path == "/api/spectrogram":
            self._serve_spectrogram(query)
            return
        if path == "/api/waveform":
            source = BANK.get(query.get("source", [""])[0])
            start = _query_int(query, "start", 0)
            end = _query_int(query, "end", source.frames)
            width = _query_int(query, "width", 1400)
            self._send_json(
                waveform_overview(BANK, source, start, end, width)
            )
            return
        if path == "/api/exports":
            exports = sorted(
                EXPORT_HISTORY.values(), key=lambda item: item.get("created_utc", ""), reverse=True
            )
            self._send_json(
                {
                    "ok": True,
                    "exports": [
                        _export_payload(item["export_id"], item) for item in exports
                    ],
                }
            )
            return
        if path.startswith("/render/"):
            filename = Path(path).name
            self._send_file(RENDER_DIR / filename, "audio/wav")
            return
        if path.startswith("/exports/"):
            self._serve_export_file(path, query)
            return
        if path.startswith("/assets/") or path == "/":
            relative = "index.html" if path == "/" else path[len("/assets/"):]
            target = (STATIC_DIR / relative).resolve()
            if target != STATIC_DIR and STATIC_DIR not in target.parents:
                raise ValueError("非法静态路径")
            if not target.is_file():
                self._send_error_json("页面不存在", HTTPStatus.NOT_FOUND)
                return
            self._send_file(target)
            return
        self._send_error_json("接口不存在", HTTPStatus.NOT_FOUND)

    def _serve_spectrogram(self, query: dict) -> None:
        source = BANK.get(query.get("source", [""])[0])
        start = _query_int(query, "start", 0)
        end = _query_int(query, "end", source.frames)
        f_min = float(query.get("fmin", ["0"])[0])
        f_max = float(query.get("fmax", [str(source.rate / 2)])[0])
        width = _query_int(query, "width", 1400)
        height = _query_int(query, "height", 420)
        if end - start > 16_000_000:
            raise ValueError("该视图超过 1600 万源样本，请先放大时间范围")
        key = (source.id, start, end, round(f_min, 2), round(f_max, 2), width, height)
        result = SPECTROGRAM_CACHE.get(key)
        if result is None:
            result = spectrogram_image(
                BANK, source, start, end, f_min, f_max, width, height
            )
            if len(SPECTROGRAM_CACHE) >= 28:
                SPECTROGRAM_CACHE.pop(next(iter(SPECTROGRAM_CACHE)))
            SPECTROGRAM_CACHE[key] = result
        headers = {
            "X-Spec-Width": str(result["width"]),
            "X-Spec-Height": str(result["height"]),
            "X-Spec-Min-Db": f"{result['min_db']:.3f}",
            "X-Spec-Max-Db": f"{result['max_db']:.3f}",
            "X-Spec-Start-Frame": str(result["start_frame"]),
            "X-Spec-End-Frame": str(result["end_frame"]),
            "X-Spec-Fmin": str(result["f_min"]),
            "X-Spec-Fmax": str(result["f_max"]),
        }
        self._send_binary(result["bytes"], "application/octet-stream", headers)

    def _serve_export_file(self, path: str, query: dict) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) != 3:
            self._send_error_json("导出路径无效", HTTPStatus.NOT_FOUND)
            return
        export_id = parts[1]
        filename = Path(parts[2]).name
        folder = (EXPORT_DIR / export_id).resolve()
        target = (folder / filename).resolve()
        if EXPORT_DIR.resolve() not in folder.parents or target.parent != folder:
            self._send_error_json("导出路径无效", HTTPStatus.NOT_FOUND)
            return
        download = query.get("download", ["0"])[0] in {"1", "true"}
        self._send_file(
            target,
            mimetypes.guess_type(filename)[0],
            download_name=filename if download else None,
        )

    def _handle_post(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/upload":
            self._handle_upload()
            return
        if path == "/api/render":
            self._handle_render()
            return
        if path == "/api/export":
            self._handle_export()
            return
        self._send_error_json("接口不存在", HTTPStatus.NOT_FOUND)

    def _handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("请使用 multipart/form-data 上传")
        body = self._read_body()
        filename, payload = _parse_multipart(content_type, body)
        if not payload:
            raise ValueError("上传音频为空")
        safe_name = re.sub(r"[^\w.\- ()]+", "_", Path(filename).name, flags=re.UNICODE)
        safe_name = safe_name or "source.wav"
        suffix = Path(safe_name).suffix.lower()
        if suffix not in {".wav", ".flac", ".aiff", ".aif"}:
            raise ValueError("支持 WAV、FLAC 或 AIFF 声源")
        upload_dir = DATA_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)
        destination = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        destination.write_bytes(payload)
        try:
            info = sf.info(str(destination))
            if info.frames < 1 or info.samplerate < 8000:
                raise ValueError("音频没有有效样本")
        except Exception:
            destination.unlink(missing_ok=True)
            raise ValueError("无法读取该音频文件")
        source = BANK.add(destination, safe_name, kind="upload")
        self._send_json({"ok": True, "source": _source_payload(source)})

    def _handle_render(self) -> None:
        payload = self._read_json()
        source, start, end, params = _selection_from_request(payload)
        rendered = process_selection(BANK, source, start, end, params)
        if len(rendered["audio"]) == 0:
            raise ValueError("选区过短，无法生成监听")
        render_id = uuid.uuid4().hex
        output_path = RENDER_DIR / f"{render_id}.wav"
        sf.write(
            str(output_path),
            rendered["audio"],
            rendered["monitor_rate"],
            subtype="PCM_16",
            format="WAV",
        )
        _prune_renders()
        self._send_json(_render_payload(source, start, end, rendered, render_id))

    def _handle_export(self) -> None:
        payload = self._read_json()
        source, start, end, params = _selection_from_request(payload)
        rendered = process_selection(BANK, source, start, end, params)
        if len(rendered["audio"]) == 0:
            raise ValueError("选区过短，无法导出")
        bundle = export_bundle(
            BANK, source, start, end, rendered, EXPORT_DIR
        )
        manifest = bundle["manifest"]
        EXPORT_HISTORY[manifest["export_id"]] = manifest
        export_payload = _export_payload(manifest["export_id"], manifest)
        export_payload["folder_name"] = bundle["folder"].name
        self._send_json({"ok": True, "export": export_payload})


def main() -> None:
    parser = argparse.ArgumentParser(description="ECHOLAB local bat acoustic screening desk")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8787")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), EchoLabHandler)
    print(f"ECHOLAB running at http://{args.host}:{args.port}")
    print(f"Demo source: {DEMO.name} · {DEMO.rate / 1000:.1f} kHz · {DEMO.duration:.2f} s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ECHOLAB")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

