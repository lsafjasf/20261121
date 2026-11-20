"""Simulated capture endpoint for one camera track.

Runs a live videotestsrc and publishes raw frames on an shm socket, exactly
like a remote capture box would.  The recorder service attaches to the
socket.  Killing this process simulates a capture-side outage; starting it
again simulates the endpoint reconnecting.
"""

import argparse
import os
import signal

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

Gst.init(None)

PATTERNS = ["smpte", "ball", "snow", "checkers-8", "gradient", "colors"]


def main():
    parser = argparse.ArgumentParser(description="Simulated capture endpoint")
    parser.add_argument("--track", type=int, required=True)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds to wait before the endpoint comes up")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--run-dir", default="run")
    args = parser.parse_args()

    socket_path = os.path.join(args.run_dir, f"cam{args.track}.shm")
    if args.delay > 0:
        print(f"[cam{args.track}] coming up in {args.delay}s", flush=True)
        GLib.usleep(int(args.delay * 1_000_000))
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    pattern = PATTERNS[args.track % len(PATTERNS)]
    pipe = Gst.parse_launch(
        f"videotestsrc is-live=true pattern={pattern} ! "
        f"video/x-raw,format=I420,width={args.width},height={args.height},"
        f"framerate={args.fps}/1 ! "
        f"shmsink name=shm socket-path={socket_path} "
        f"wait-for-connection=false shm-size=20000000 sync=false"
    )
    loop = GLib.MainLoop()
    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message::eos", lambda *_: loop.quit())
    bus.connect(
        "message::error",
        lambda *_: (print(f"[cam{args.track}] error", flush=True),
                    loop.quit()),
    )
    pipe.set_state(Gst.State.PLAYING)
    print(f"[cam{args.track}] live on {socket_path} (pattern={pattern})",
          flush=True)

    def stop(signum, frame):
        loop.quit()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    loop.run()
    pipe.set_state(Gst.State.NULL)
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    print(f"[cam{args.track}] stopped", flush=True)


if __name__ == "__main__":
    main()
