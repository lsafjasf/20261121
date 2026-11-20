"""Unified show clock shared by all tracks and across process restarts.

The show clock is plain UNIX epoch time (ns).  Its origin (the moment the
show/performance started) is persisted in ``state/show.json`` the first time
the service runs, so a restarted process keeps using the exact same timeline.

Each GStreamer pipeline gets a ``base_time`` chosen so that its *running
time* equals nanoseconds since the show origin.  Buffer PTS values produced
by live, do-timestamp sources are therefore directly comparable between
tracks, no matter when each track (or the service itself) was started.
"""

import json
import os
import time

NS_PER_SEC = 1_000_000_000


class ShowClock:
    def __init__(self, state_dir, fps=30):
        self.state_dir = state_dir
        self.path = os.path.join(state_dir, "show.json")
        self.fps = fps
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.origin_epoch_ns = int(data["show_epoch_ns"])
            self.fps = int(data.get("fps", fps))
        else:
            self.origin_epoch_ns = time.time_ns()
            os.makedirs(state_dir, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "show_epoch_ns": self.origin_epoch_ns,
                        "fps": self.fps,
                        "created_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    },
                    fh,
                    indent=2,
                )
            os.replace(tmp, self.path)

    def show_time_ns(self):
        """Nanoseconds since the show origin (the unified timeline)."""
        return time.time_ns() - self.origin_epoch_ns

    def pipeline_base_time(self):
        """base_time so that pipeline running_time == unified show time."""
        return time.monotonic_ns() - self.show_time_ns()

    def to_epoch_ns(self, show_ns):
        return show_ns + self.origin_epoch_ns

    def frame_of(self, show_ns):
        """Frame index on the global fps grid for a show-time in ns."""
        return show_ns * self.fps // NS_PER_SEC
