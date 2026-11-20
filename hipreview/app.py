"""本地审片 Web 服务。

只用标准库 http.server，避免引入 Flask 等额外依赖。
路由：
  GET /                 审片页面
  GET /api/report       审片 JSON
  GET /audio/<file>     整卷 WAV（主/助听）
  GET /clips/<file>     问题片段 WAV
"""

from __future__ import annotations

import functools
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .report import run_audit

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


class AuditHandler(SimpleHTTPRequestHandler):
    """带 /api/report 与目录白名单的静态处理器。"""

    def __init__(self, *args, audit_dir: str, data_dir: str, **kwargs):
        self.audit_dir = audit_dir
        self.data_dir = data_dir
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (http.server 命名)
        parsed = urlparse(self.path)
        if parsed.path == "/api/report":
            self._serve_json(os.path.join(self.audit_dir, "report.json"))
            return
        if parsed.path.startswith("/audio/"):
            self._serve_file(self.data_dir, parsed.path[len("/audio/"):])
            return
        if parsed.path.startswith("/clips/"):
            self._serve_file(self.audit_dir, parsed.path[len("/"):])
            return
        super().do_GET()

    def _serve_json(self, path: str) -> None:
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            self.send_error(404, "报告不存在，请先运行审片")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, base: str, rel: str) -> None:
        # 防目录穿越：解析后必须仍在白名单目录内
        root = os.path.abspath(base)
        target = os.path.abspath(os.path.join(root, rel))
        if not target.startswith(root + os.sep):
            self.send_error(403)
            return
        if not os.path.isfile(target):
            self.send_error(404)
            return
        ctype = "audio/wav" if target.endswith(".wav") else "application/octet-stream"
        with open(target, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "none")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # 安静日志
        sys.stderr.write("[hipreview] %s\n" % (fmt % args))


def serve(audit_dir: str, data_dir: str, host: str = "127.0.0.1",
          port: int = 8765) -> None:
    handler = functools.partial(
        AuditHandler, audit_dir=audit_dir, data_dir=data_dir)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"审片服务已启动: {url}（Ctrl+C 退出）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()


def audit_and_serve(playlist_path: str, audit_dir: str,
                    host: str = "127.0.0.1", port: int = 8765) -> None:
    payload = run_audit(playlist_path, audit_dir)
    issue_count = sum(len(r["issues"]) for r in payload["results"])
    print(f"审片完成：{len(payload['results'])} 卷，发现 {issue_count} 条问题")
    data_dir = os.path.dirname(os.path.abspath(playlist_path))
    serve(audit_dir, data_dir, host, port)
