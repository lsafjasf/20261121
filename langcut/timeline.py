from __future__ import annotations

from dataclasses import dataclass

from .errors import LangcutError
from .manifest import Chapter


def seconds(frame: int, fps: int) -> float:
    return round(frame / fps, 6)


def timecode(frame: int, fps: int) -> str:
    total_ms = round(frame * 1000 / fps)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


@dataclass(frozen=True)
class Timeline:
    fps: int
    opening_leader_frames: int
    notice_frames: int
    program_frames: int

    @property
    def program_start_frame(self) -> int:
        return max(self.opening_leader_frames, self.notice_frames)

    @property
    def extension_frames(self) -> int:
        return max(0, self.notice_frames - self.opening_leader_frames)

    @property
    def trailing_blank_frames(self) -> int:
        return max(0, self.opening_leader_frames - self.notice_frames)

    @property
    def shift_frames(self) -> int:
        return self.program_start_frame - self.opening_leader_frames

    @property
    def total_frames(self) -> int:
        return self.program_start_frame + self.program_frames

    def map_point(self, program_frame: int) -> dict[str, object]:
        if program_frame >= self.program_frames:
            raise LangcutError(
                "TIMELINE_INVALID",
                f"program frame {program_frame} is outside the program "
                f"({self.program_frames} frames)",
            )
        old_frame = self.opening_leader_frames + program_frame
        new_frame = self.program_start_frame + program_frame
        return {
            "program_frame": program_frame,
            "old": {
                "frame": old_frame,
                "seconds": seconds(old_frame, self.fps),
                "timecode": timecode(old_frame, self.fps),
            },
            "new": {
                "frame": new_frame,
                "seconds": seconds(new_frame, self.fps),
                "timecode": timecode(new_frame, self.fps),
            },
            "delta_frames": self.shift_frames,
            "delta_seconds": seconds(self.shift_frames, self.fps),
        }

    def map_chapters(self, chapters: tuple[Chapter, ...]) -> list[dict[str, object]]:
        result = []
        for chapter in chapters:
            mapped = self.map_point(chapter.program_frame)
            mapped["id"] = chapter.id
            mapped["title"] = chapter.title
            result.append(mapped)
        return result

    def map_cut_points(self, cut_points: tuple[int, ...]) -> list[dict[str, object]]:
        result = []
        for index, frame in enumerate(cut_points, start=1):
            mapped = self.map_point(frame)
            mapped["id"] = f"cut-{index:04d}"
            result.append(mapped)
        return result
