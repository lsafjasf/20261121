"""GStreamer four-track recording service on a unified show clock.

Each track reads from a capture endpoint (shm socket written by a separate
capture process, e.g. capture_sim.py), encodes H.264 and writes short,
stackable MP4 segments via splitmuxsink.  All segment timestamps come from
the shared ShowClock, so segments from different tracks line up on one
timeline even though every track started at a different moment.

If a capture endpoint goes away, the track supervisor marks an explicit gap
interval in the manifest, keeps retrying, and resumes recording (closing the
gap) when the endpoint reconnects.  The other tracks are unaffected.
"""

import argparse
import os
import signal
import sys
import threading
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from .clockutil import NS_PER_SEC, ShowClock
from .manifest import Manifest

Gst.init(None)


class TrackSupervisor(threading.Thread):
    """Owns one track: build pipeline, record, detect loss, reconnect."""

    RETRY_S = 1.0
    NO_DATA_S = 3.0

    def __init__(self, track, args, clock, manifest):
        super().__init__(name=f"track-{track}", daemon=True)
        self.track = track
        self.args = args
        self.clock = clock
        self.manifest = manifest
        self.socket_path = os.path.join(args.run_dir, f"cam{track}.shm")
        self.stop_event = threading.Event()
        self.pipeline = None
        self.generation = 0
        self.ever_streamed = False
        self.in_gap = False
        self.pending_seg = None  # (generation, seg_id, start_ns)
        self._seg_counter = 0
        # x264enc re-bases output PTS by a per-stream constant offset, so
        # encoded PTS cannot be used directly.  Measure the offset between
        # the raw source PTS (running time == unified show time) and the
        # encoded PTS once per pipeline generation.
        self._first_src_pts = None
        self._first_enc_pts = None
        self._last_src_pts = None
        self._last_buffer_mono = None

    # -- pipeline --------------------------------------------------------

    def _build(self):
        fps = self.clock.fps
        seg_ns = int(self.args.seg_seconds * NS_PER_SEC)
        src = Gst.ElementFactory.make("shmsrc", None)
        src.set_property("socket-path", self.socket_path)
        src.set_property("is-live", True)
        src.set_property("do-timestamp", True)
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=I420,width={self.args.width},"
            f"height={self.args.height},framerate={fps}/1"
        )
        convert = Gst.ElementFactory.make("videoconvert", None)
        enc = Gst.ElementFactory.make("x264enc", None)
        enc.set_property("tune", "zerolatency")
        enc.set_property("speed-preset", "ultrafast")
        enc.set_property("bitrate", 600)
        enc.set_property("key-int-max", int(self.args.seg_seconds * fps))
        parse = Gst.ElementFactory.make("h264parse", None)
        mux = Gst.ElementFactory.make("mp4mux", None)
        # Fragmented MP4: a killed process still leaves a readable file.
        mux.set_property("fragment-duration", 1000)
        sink = Gst.ElementFactory.make("splitmuxsink", None)
        sink.set_property("muxer", mux)
        sink.set_property("max-size-time", seg_ns)
        sink.connect("format-location-full", self._on_format_location)

        pipe = Gst.Pipeline.new(f"rec-track{self.track}")
        for el in (src, convert, enc, parse, sink):
            pipe.add(el)
        if not src.link_filtered(convert, caps):
            raise RuntimeError("caps link failed")
        convert.link(enc)
        enc.link(parse)
        parse.link(sink)

        pipe.set_base_time(self.clock.pipeline_base_time())
        pipe.use_clock(Gst.SystemClock.obtain())
        # Keep our base_time: otherwise GstPipeline recomputes it when the
        # pipeline goes to PLAYING and running time would restart at 0.
        pipe.set_start_time(Gst.CLOCK_TIME_NONE)
        self._first_src_pts = None
        self._first_enc_pts = None
        self._last_buffer_mono = None
        self._last_src_pts = None
        src.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._on_src_buffer)
        parse.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._on_enc_buffer)
        bus = pipe.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self._on_eos)
        bus.connect("message::error", self._on_error)
        return pipe

    def _on_src_buffer(self, pad, info):
        self._last_buffer_mono = time.monotonic()
        buf = info.get_buffer()
        if buf.pts != Gst.CLOCK_TIME_NONE:
            self._last_src_pts = int(buf.pts)
            if self._first_src_pts is None:
                self._first_src_pts = int(buf.pts)
        return Gst.PadProbeReturn.OK

    def _on_enc_buffer(self, pad, info):
        if self._first_enc_pts is None:
            buf = info.get_buffer()
            if buf.pts != Gst.CLOCK_TIME_NONE:
                self._first_enc_pts = int(buf.pts)
        return Gst.PadProbeReturn.OK

    def _unified_time(self, enc_pts):
        """Map an encoded PTS back onto the unified show clock (ns)."""
        if self._first_src_pts is None or self._first_enc_pts is None:
            return self.clock.show_time_ns()
        return int(enc_pts) - self._first_enc_pts + self._first_src_pts

    # -- splitmuxsink callback (streaming thread) -------------------------

    def _on_format_location(self, sink, fragment_id, first_sample):
        buf = first_sample.get_buffer()
        pts = buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else 0
        start_ns = self._unified_time(pts)
        seg_id = self._seg_counter
        self._seg_counter += 1
        fname = f"track{self.track}_t{start_ns}.mp4"
        path = os.path.join(self.args.run_dir, fname)

        # Close the previous fragment of this pipeline generation.
        if self.pending_seg and self.pending_seg[0] == self.generation:
            _, prev_id, _ = self.pending_seg
            self.manifest.emit("segment_close", start_ns, track=self.track,
                               seg=prev_id)
        self.pending_seg = (self.generation, seg_id, start_ns)
        self.manifest.emit("segment_open", start_ns, track=self.track,
                           seg=seg_id, file=fname)

        if not self.ever_streamed:
            self.ever_streamed = True
            # If a previous process already recorded this track, the window
            # between its last frame and our first frame is an explicit
            # service-restart gap.
            prev = self.manifest.segments(self.track)
            last_end = max((s["end_ns"] or 0 for s in prev), default=0)
            if last_end and last_end < start_ns:
                self.manifest.emit("gap_start", last_end, track=self.track,
                                   reason="service-restart")
                self.manifest.emit("gap_end", start_ns, track=self.track)
            self.manifest.emit("track_up", start_ns, track=self.track)
        if self.in_gap:
            self.in_gap = False
            self.manifest.emit("gap_end", start_ns, track=self.track)
        return path

    # -- bus callbacks -----------------------------------------------------

    def _on_eos(self, bus, msg):
        self._pipeline_finished("eos")

    def _on_error(self, bus, msg):
        err, _ = msg.parse_error()
        self._pipeline_finished(f"error: {err.message}")

    def _pipeline_finished(self, reason):
        # Runs on the GLib main-context thread; just flag the supervisor.
        self._finished_reason = reason

    # -- supervisor loop ---------------------------------------------------

    def run(self):
        self.manifest.emit("track_waiting", self.clock.show_time_ns(),
                           track=self.track)
        while not self.stop_event.is_set():
            self.generation += 1
            self._finished_reason = None
            try:
                self.pipeline = self._build()
            except Exception as exc:  # pragma: no cover - defensive
                self._mark_gap(f"build-failed: {exc}")
                self.stop_event.wait(self.RETRY_S)
                continue
            self.pipeline.set_state(Gst.State.PLAYING)
            gen_start = time.monotonic()
            # Poll until the pipeline ends (EOS/error) or we are stopped.
            while (self._finished_reason is None
                   and not self.stop_event.is_set()):
                time.sleep(0.1)
                last = self._last_buffer_mono
                idle = time.monotonic() - (last if last else gen_start)
                if idle > self.NO_DATA_S:
                    self._finished_reason = (
                        "capture-stalled" if last else "capture-absent")
            if self.stop_event.is_set() and self._finished_reason is None:
                # Graceful stop: push EOS so splitmuxsink finalizes the
                # current fragment and the muxer writes a clean trailer.
                self.pipeline.send_event(Gst.Event.new_eos())
                deadline = time.monotonic() + 5.0
                while (self._finished_reason is None
                       and time.monotonic() < deadline):
                    time.sleep(0.05)
            reason = self._finished_reason or "service-stop"
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            if self.stop_event.is_set():
                break
            self._mark_gap(reason)
            self.stop_event.wait(self.RETRY_S)
        self._close_pending(self.clock.show_time_ns())
        self.manifest.emit("track_down", self.clock.show_time_ns(),
                           track=self.track)

    def _mark_gap(self, reason):
        now = self.clock.show_time_ns()
        self._close_pending(now)
        if self.ever_streamed and not self.in_gap:
            self.in_gap = True
            self.manifest.emit("gap_start", now, track=self.track,
                               reason=reason.split(":")[0])

    def _close_pending(self, t_ns):
        if self.pending_seg:
            _, seg_id, _ = self.pending_seg
            self.pending_seg = None
            # Do not claim coverage past the last frame we actually saw.
            if self._last_src_pts is not None:
                frame_ns = NS_PER_SEC // self.clock.fps
                t_ns = min(t_ns, self._last_src_pts + frame_ns)
            self.manifest.emit("segment_close", t_ns, track=self.track,
                               seg=seg_id)

    def stop(self):
        self.stop_event.set()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Four-track GStreamer recorder on a unified show clock")
    parser.add_argument("--tracks", type=int, default=4)
    parser.add_argument("--seg-seconds", type=float, default=2.0,
                        help="target length of each stackable segment")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--run-dir", default="run",
                        help="directory for shm sockets and segment files")
    parser.add_argument("--state-dir", default="state",
                        help="directory for the persistent manifest")
    args = parser.parse_args(argv)

    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)

    clock = ShowClock(args.state_dir, fps=args.fps)
    manifest = Manifest(args.state_dir)
    manifest.emit("service_start", clock.show_time_ns(), pid=os.getpid())

    ctx = GLib.MainContext.default()
    stop_all = threading.Event()

    def pump():
        # Daemon thread: keep dispatching bus messages until process exit,
        # including the EOS that finalizes segments during shutdown.
        while True:
            ctx.iteration(True)

    spinner = threading.Thread(target=pump, daemon=True)
    spinner.start()

    supervisors = [
        TrackSupervisor(t, args, clock, manifest) for t in range(args.tracks)
    ]
    for sup in supervisors:
        sup.start()
    print(f"[service] recording {args.tracks} tracks, "
          f"show origin epoch_ns={clock.origin_epoch_ns}", flush=True)

    def handle_signal(signum, frame):
        stop_all.set()
        for sup in supervisors:
            sup.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for sup in supervisors:
        sup.join()
    manifest.emit("service_stop", clock.show_time_ns(), pid=os.getpid())
    manifest.close()
    print("[service] stopped, manifest flushed", flush=True)


if __name__ == "__main__":
    main()
