from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIDTH = 160
HEIGHT = 90
FPS = 25
SAMPLE_RATE = 48000
RED = (230, 30, 20)
GREEN = (20, 180, 40)
BLUE = (20, 50, 220)
NOTICE_YELLOW = (240, 210, 20)


def _write_video(path: Path, frames: list[tuple[int, int, int]]) -> None:
    payload = b"".join(bytes(color) * (WIDTH * HEIGHT) for color in frames)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "rgb24",
        str(path),
    ]
    subprocess.run(command, input=payload, check=True)


def _write_tone(path: Path, frequency: int, frames: int) -> None:
    sample_count = round(frames * SAMPLE_RATE / FPS)
    samples = bytearray()
    for index in range(sample_count):
        value = int(0.48 * 32767 * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE))
        samples.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(bytes(samples))


def _decode_frames(path: Path) -> list[bytes]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    image = subprocess.run(command, check=True, capture_output=True).stdout
    frame_size = WIDTH * HEIGHT * 3
    if len(image) % frame_size:
        raise AssertionError(f"decoded video has {len(image)} bytes")
    return [image[i : i + frame_size] for i in range(0, len(image), frame_size)]


def _decode_audio(path: Path) -> list[int]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    pcm = subprocess.run(command, check=True, capture_output=True).stdout
    return list(struct.unpack(f"<{len(pcm) // 2}h", pcm))


def _pixel(frame: bytes) -> tuple[int, int, int]:
    center = (HEIGHT // 2 * WIDTH + WIDTH // 2) * 3
    return tuple(frame[center : center + 3])


def _frequency(samples: list[int], start_seconds: float, duration: float = 0.08) -> float:
    begin = round(start_seconds * SAMPLE_RATE)
    end = begin + round(duration * SAMPLE_RATE)
    window = samples[begin:end]
    crossings = sum(1 for left, right in zip(window, window[1:]) if (left < 0) != (right < 0))
    return crossings / 2 / duration


class RenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="langcut-tests-")
        cls.media = Path(cls.temp.name)
        program_frames = [RED] * 10 + [GREEN] * 10 + [BLUE] * 10
        _write_video(cls.media / "program.mkv", program_frames)
        _write_tone(cls.media / "program.wav", 1000, len(program_frames))
        _write_video(cls.media / "notice_exact.mkv", [NOTICE_YELLOW] * 25)
        _write_tone(cls.media / "notice_exact.wav", 300, 25)
        _write_video(cls.media / "notice_long.mkv", [NOTICE_YELLOW] * 35)
        _write_tone(cls.media / "notice_long.wav", 300, 35)
        _write_video(cls.media / "notice_short.mkv", [NOTICE_YELLOW] * 15)
        _write_tone(cls.media / "notice_short.wav", 300, 15)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def _manifest(
        self,
        *,
        name: str,
        leader_frames: int,
        notice_video: str,
        notice_audio: str,
        second_video: str | None = None,
        second_audio: str | None = None,
        output: dict[str, object] | None = None,
    ) -> Path:
        payload = {
            "version": 1,
            "name": "episode",
            "fps": FPS,
            "canvas": {"width": WIDTH, "height": HEIGHT},
            "opening_leader_frames": leader_frames,
            "program": {"video": "program.mkv", "audio": "program.wav"},
            "notices": {
                "en": {
                    "label": "English",
                    "video": notice_video,
                    "audio": notice_audio,
                },
                "zh": {
                    "label": "中文",
                    "video": second_video or notice_video,
                    "audio": second_audio or notice_audio,
                },
            },
            "chapters": [
                {"id": "program-start", "title": "Program start", "program_frame": 0},
                {"id": "green", "title": "Green section", "program_frame": 10},
                {"id": "blue", "title": "Blue section", "program_frame": 20},
            ],
            "cut_points": [10, 20],
            "output": output
            or {
                "filename": "{name}.{language}",
                "extension": "mkv",
                "video_codec": "ffv1",
                "audio_codec": "pcm_s16le",
                "pixel_format": "rgb24",
                "audio_sample_rate": SAMPLE_RATE,
            },
        }
        path = self.media / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def _render(self, manifest: Path, out_name: str) -> tuple[subprocess.CompletedProcess[str], Path]:
        out_dir = self.media / out_name
        result = subprocess.run(
            [sys.executable, "-m", "langcut", "render", str(manifest), "--out", str(out_dir)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result, out_dir

    def _assert_program_boundary(
        self, video: Path, program_start: int, expected_total: int
    ) -> None:
        frames = _decode_frames(video)
        self.assertEqual(len(frames), expected_total)
        expected = [
            (program_start - 1, NOTICE_YELLOW),
            (program_start, RED),
            (program_start + 10, GREEN),
            (program_start + 20, BLUE),
        ]
        for frame_number, color in expected:
            actual = _pixel(frames[frame_number])
            self.assertTrue(
                all(abs(left - right) <= 12 for left, right in zip(actual, color)),
                f"frame {frame_number}: expected {color}, got {actual}",
            )

    def _assert_program_audio(self, video: Path, program_start: int) -> None:
        samples = _decode_audio(video)
        start_seconds = program_start / FPS
        self.assertGreater(_frequency(samples, start_seconds), 800)
        self.assertLess(_frequency(samples, start_seconds - 0.02, 0.012), 450)

    def test_exact_fit_keeps_program_frame_and_audio_at_leader_boundary(self) -> None:
        manifest = self._manifest(
            name="exact.json",
            leader_frames=25,
            notice_video="notice_exact.mkv",
            notice_audio="notice_exact.wav",
            output={
                "filename": "{name}.{language}",
                "extension": "mp4",
                "video_codec": "libx264",
                "audio_codec": "aac",
                "pixel_format": "yuv420p",
                "crf": 18,
                "audio_sample_rate": SAMPLE_RATE,
            },
        )
        result, out_dir = self._render(manifest, "out-exact")
        self.assertEqual(result.returncode, 0, result.stderr)
        video = out_dir / "episode.en.mp4"
        self.assertTrue((out_dir / "episode.zh.mp4").is_file())
        self.assertTrue((out_dir / "chapter-map.csv").is_file())
        mapping = json.loads((out_dir / "chapter-map.json").read_text(encoding="utf-8"))
        english = mapping["languages"]["en"]
        self.assertEqual(english["notice_frames"], 25)
        self.assertEqual(english["new_program_start_frame"], 25)
        self.assertEqual(english["extension_frames"], 0)
        self.assertEqual(english["shift_frames"], 0)
        green = next(item for item in english["chapters"] if item["id"] == "green")
        self.assertEqual((green["old"]["frame"], green["new"]["frame"]), (35, 35))
        self._assert_program_boundary(video, 25, 55)
        self._assert_program_audio(video, 25)

    def test_long_notice_extends_timeline_and_preserves_relative_cuts(self) -> None:
        manifest = self._manifest(
            name="extended.json",
            leader_frames=15,
            notice_video="notice_exact.mkv",
            notice_audio="notice_exact.wav",
            second_video="notice_long.mkv",
            second_audio="notice_long.wav",
        )
        result, out_dir = self._render(manifest, "out-extended")
        self.assertEqual(result.returncode, 0, result.stderr)
        video = out_dir / "episode.zh.mkv"
        self.assertTrue((out_dir / "episode.en.mkv").is_file())
        mapping = json.loads((out_dir / "chapter-map.json").read_text(encoding="utf-8"))
        chinese = mapping["languages"]["zh"]
        english = mapping["languages"]["en"]
        self.assertEqual(english["new_program_start_frame"], 25)
        self.assertEqual(english["shift_frames"], 10)
        self.assertEqual(chinese["notice_frames"], 35)
        self.assertEqual(chinese["new_program_start_frame"], 35)
        self.assertEqual(chinese["extension_frames"], 20)
        self.assertEqual(chinese["shift_frames"], 20)
        self.assertAlmostEqual(chinese["shift_seconds"], 0.8)
        frames = chinese["chapters"]
        self.assertEqual(
            [(item["program_frame"], item["old"]["frame"], item["new"]["frame"]) for item in frames],
            [(0, 15, 35), (10, 25, 45), (20, 35, 55)],
        )
        self.assertTrue(all(item["delta_frames"] == 20 for item in frames))
        self.assertEqual(
            [(item["old"]["frame"], item["new"]["frame"]) for item in chinese["cut_points"]],
            [(25, 45), (35, 55)],
        )
        self._assert_program_boundary(video, 35, 65)
        self._assert_program_audio(video, 35)

    def test_short_notice_leaves_black_and_silence_before_program(self) -> None:
        manifest = self._manifest(
            name="short.json",
            leader_frames=25,
            notice_video="notice_short.mkv",
            notice_audio="notice_short.wav",
        )
        result, out_dir = self._render(manifest, "out-short")
        self.assertEqual(result.returncode, 0, result.stderr)
        video = out_dir / "episode.en.mkv"
        mapping = json.loads((out_dir / "chapter-map.json").read_text(encoding="utf-8"))
        english = mapping["languages"]["en"]
        self.assertEqual(english["notice_frames"], 15)
        self.assertEqual(english["new_program_start_frame"], 25)
        self.assertEqual(english["trailing_blank_frames"], 10)
        self.assertEqual(english["shift_frames"], 0)
        frames = _decode_frames(video)
        self.assertEqual(len(frames), 55)
        self.assertEqual(_pixel(frames[14]), NOTICE_YELLOW)
        self.assertEqual(_pixel(frames[15]), (0, 0, 0))
        self.assertEqual(_pixel(frames[24]), (0, 0, 0))
        self.assertEqual(_pixel(frames[25]), RED)
        samples = _decode_audio(video)
        self.assertLess(_frequency(samples, 0.1, 0.012), 450)
        self.assertLess(abs(sum(samples[round(0.65 * SAMPLE_RATE) : round(0.75 * SAMPLE_RATE)])), 1000)
        self.assertGreater(_frequency(samples, 1.0), 800)

    def test_missing_notice_asset_fails_before_writing_outputs(self) -> None:
        manifest = self._manifest(
            name="missing.json",
            leader_frames=25,
            notice_video="missing_notice.mkv",
            notice_audio="notice_exact.wav",
        )
        result, out_dir = self._render(manifest, "out-missing")
        self.assertEqual(result.returncode, 2)
        self.assertIn("ASSET_MISSING", result.stderr)
        self.assertIn("missing_notice.mkv", result.stderr)
        self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
