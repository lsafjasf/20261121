"""Local-only HTTP UI for the recognizer. No third-party dependencies."""

from __future__ import annotations

import base64
import json
import mimetypes
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .detector import analyze_file
from .report import attach_ground_truth, generate_demo


MAX_BODY_BYTES = 48 * 1024 * 1024


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def build_handler(root: Path):
    web_root = root / "web"
    report_path = root / "reports" / "museum_sample_report.json"
    sample_audio = root / "samples" / "museum_sample.wav"

    class Handler(BaseHTTPRequestHandler):
        server_version = "AssetTone/1.0"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict) -> None:
            self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

        def _serve_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "文件不存在"})
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, path.read_bytes(), content_type)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._serve_file(web_root / "index.html")
            elif path in ("/styles.css", "/app.js", "/sample-report.js"):
                self._serve_file(web_root / path.removeprefix("/"))
            elif path.startswith("/static/"):
                name = Path(path.removeprefix("/static/")).name
                self._serve_file(web_root / name)
            elif path == "/api/sample":
                if not report_path.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "样例尚未生成"})
                    return
                self._send(
                    HTTPStatus.OK,
                    report_path.read_bytes(),
                    "application/json; charset=utf-8",
                )
            elif path == "/api/health":
                self._send_json(HTTPStatus.OK, {"ok": True, "offline": True})
            elif path == "/samples/museum_sample.wav":
                self._serve_file(sample_audio)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "未知路径"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/analyze":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "未知路径"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ValueError("上传内容为空或超过 48 MB 限制")
                payload = json.loads(self.rfile.read(length))
                encoded = payload.get("audio_base64", "")
                if "," in encoded:
                    encoded = encoded.split(",", 1)[1]
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) < 44:
                    raise ValueError("WAV 文件无效")
                filename = Path(str(payload.get("filename", "recording.wav"))).name
                with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
                    handle.write(raw)
                    handle.flush()
                    report = analyze_file(handle.name, filename)
                manifest = payload.get("manifest")
                if isinstance(manifest, dict):
                    report = attach_ground_truth(report, manifest)
                report["audio_url"] = ""
                report["generated_at"] = None
                self._send_json(HTTPStatus.OK, report)
            except (ValueError, OSError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"分析失败: {error}"},
                )

        def log_message(self, format: str, *args) -> None:
            print(f"[ui] {self.address_string()} {format % args}")

    return Handler


def serve(host: str, port: int, root: Path, regenerate: bool = False) -> int:
    report_path = root / "reports" / "museum_sample_report.json"
    if regenerate or not report_path.is_file():
        print("正在生成馆藏样例与检测报告...")
        generate_demo(root)
    handler = build_handler(root)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"离线识别台已启动: http://{host}:{port}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()
    return 0
