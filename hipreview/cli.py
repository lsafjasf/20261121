"""hipreview 命令行入口。

用法：
  python -m hipreview.cli gen-data [--out data]
  python -m hipreview.cli audit   --playlist data/playlist.json --out audit
  python -m hipreview.cli serve   --playlist data/playlist.json --out audit
  python -m hipreview.cli demo                       # 一键生成+审片+起服务
"""

from __future__ import annotations

import argparse
import os

from .app import audit_and_serve
from .dataset import generate
from .report import run_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hipreview",
                                     description="本地助听轨审片程序")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen-data", help="生成多卷测试素材与播放清单")
    p_gen.add_argument("--out", default="data")

    p_audit = sub.add_parser("audit", help="执行审片并输出报告")
    p_audit.add_argument("--playlist", required=True)
    p_audit.add_argument("--out", default="audit")

    p_serve = sub.add_parser("serve", help="审片并启动本地 Web 界面")
    p_serve.add_argument("--playlist", required=True)
    p_serve.add_argument("--out", default="audit")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)

    p_demo = sub.add_parser("demo", help="一键生成素材、审片并启动界面")
    p_demo.add_argument("--data", default="data")
    p_demo.add_argument("--out", default="audit")
    p_demo.add_argument("--host", default="127.0.0.1")
    p_demo.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    if args.cmd == "gen-data":
        playlist = generate(args.out)
        print(f"已生成 {len(playlist['reels'])} 卷素材 -> "
              f"{os.path.join(args.out, 'playlist.json')}")
        return 0

    if args.cmd == "audit":
        payload = run_audit(args.playlist, args.out)
        total = sum(len(r["issues"]) for r in payload["results"])
        print(f"审片完成：{len(payload['results'])} 卷，{total} 条问题，"
              f"报告见 {os.path.join(args.out, 'report.txt')}")
        return 0

    if args.cmd == "serve":
        audit_and_serve(args.playlist, args.out, args.host, args.port)
        return 0

    if args.cmd == "demo":
        generate(args.data)
        playlist_path = os.path.join(args.data, "playlist.json")
        audit_and_serve(playlist_path, args.out, args.host, args.port)
        return 0

    parser.error(f"未知命令 {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
