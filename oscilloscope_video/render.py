"""Dependency-free oscilloscope frame rasterizer."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .font5x7 import GLYPH_HEIGHT, GLYPH_WIDTH, display_unit, glyph, text_width
from .model import Event, Timeline, TraceData

Color = tuple[int, int, int]

BG = (4, 11, 15)
PANEL = (7, 22, 29)
GRID = (21, 48, 57)
GRID_MAJOR = (31, 70, 82)
BORDER = (48, 112, 122)
INK = (206, 234, 236)
MUTED = (105, 146, 151)
CYAN_GLOW = (15, 83, 90)
CYAN = (71, 236, 217)
CYAN_HOT = (171, 255, 238)
AMBER = (238, 173, 61)
RED = (255, 40, 40)
GAP_FILL = (31, 18, 22)
GAP_INK = (184, 99, 107)


@dataclass(frozen=True)
class RenderConfig:
    width: int = 960
    height: int = 540
    supersample: int = 2
    title: str = "OSCILLOSCOPE TRACE"
    subtitle: str = ""
    y_min: float | None = None
    y_max: float | None = None
    flash_frames: int = 2

    def __post_init__(self) -> None:
        if self.width < 320 or self.height < 180:
            raise ValueError("video dimensions must be at least 320x180")
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 yuv420p output requires even width and height")
        if self.supersample < 1 or self.supersample > 4:
            raise ValueError("supersample must be between 1 and 4")
        if self.flash_frames < 1:
            raise ValueError("flash_frames must be positive")
        if (self.y_min is None) != (self.y_max is None):
            raise ValueError("y_min and y_max must be supplied together")
        if self.y_min is not None and self.y_max is not None and self.y_max <= self.y_min:
            raise ValueError("y_max must exceed y_min")

    @property
    def internal_size(self) -> tuple[int, int]:
        return self.width * self.supersample, self.height * self.supersample


class TraceRenderer:
    """Render one raw RGB24 frame at a time."""

    def __init__(self, data: TraceData, timeline: Timeline, config: RenderConfig):
        self.data = data
        self.timeline = timeline
        self.config = config
        self.ss = config.supersample
        self.width, self.height = config.internal_size
        self.times = [float(value) for value in data.times_s]
        self.voltages = [
            None if value is None else float(value) for value in data.voltages
        ]
        self.runs = data.sample_runs()
        self.gaps = data.effective_unknown_intervals()
        self.drawable = self._build_drawable_mask()
        self.event_first_frames = [
            timeline.first_frame(event.time_s) for event in data.events
        ]
        self.y_min, self.y_max = self._choose_y_range()
        self._set_layout()

    def _choose_y_range(self) -> tuple[float, float]:
        if self.config.y_min is not None and self.config.y_max is not None:
            return float(self.config.y_min), float(self.config.y_max)
        voltages = [
            voltage
            for voltage, drawable in zip(self.data.voltages, self.drawable)
            if voltage is not None and drawable
        ]
        if not voltages:
            return -1.0, 1.0
        low, high = min(voltages), max(voltages)
        if high <= low:
            delta = max(1.0, abs(low) * 0.1)
            return low - delta, high + delta
        padding = (high - low) * 0.08
        return low - padding, high + padding

    def _build_drawable_mask(self) -> list[bool]:
        drawable: list[bool] = []
        gap_index = 0
        for time_s, voltage in zip(self.times, self.voltages):
            while gap_index < len(self.gaps) and self.gaps[gap_index][1] <= time_s:
                gap_index += 1
            inside_gap = (
                gap_index < len(self.gaps)
                and self.gaps[gap_index][0] < time_s < self.gaps[gap_index][1]
            )
            drawable.append(voltage is not None and not inside_gap)
        return drawable

    def _set_layout(self) -> None:
        ss = self.ss
        self.left = 86 * ss
        self.right = 26 * ss
        self.top = 76 * ss
        self.bottom = 60 * ss
        self.plot_x0 = self.left
        self.plot_y0 = self.top
        self.plot_x1 = self.width - self.right
        self.plot_y1 = self.height - self.bottom
        self.plot_w = self.plot_x1 - self.plot_x0
        self.plot_h = self.plot_y1 - self.plot_y0
        self.font_small = max(1, ss)
        self.font_normal = max(1, ss * 2)
        self.font_large = max(2, ss * 3)
        self.font_axis = self.font_normal if self.config.width >= 640 else self.font_small
        self.font_header = self.font_normal if self.config.width >= 640 else self.font_small
        self.y_tick_count = 5
        self.x_tick_count = 5 if self.config.width >= 900 else 3
        self.frame = bytearray(self.width * self.height * 3)

    def render(self, frame_index: int) -> bytes:
        if not 0 <= frame_index < self.timeline.frame_count:
            raise ValueError(f"frame {frame_index} outside video")
        self._rendering_frame = frame_index
        self.frame[:] = bytes(BG) * (self.width * self.height)
        self._draw_header(frame_index)
        self._draw_plot_background()
        self._draw_unknown_regions(frame_index)
        self._draw_event_guides(frame_index)
        self._draw_trace(frame_index)
        self._draw_event_annotations(frame_index)
        self._draw_footer(frame_index)
        return bytes(self.frame)

    def _set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            index = (y * self.width + x) * 3
            self.frame[index : index + 3] = bytes(color)

    def _fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        x0, x1 = max(0, x0), min(self.width, x1)
        y0, y1 = max(0, y0), min(self.height, y1)
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes(color) * (x1 - x0)
        for y in range(y0, y1):
            start = (y * self.width + x0) * 3
            self.frame[start : start + len(row)] = row

    def _hline(self, x0: int, x1: int, y: int, color: Color) -> None:
        self._fill_rect(min(x0, x1), y, max(x0, x1) + 1, y + 1, color)

    def _vline(self, x: int, y0: int, y1: int, color: Color) -> None:
        self._fill_rect(x, min(y0, y1), x + 1, max(y0, y1) + 1, color)

    def _line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        """Integer Bresenham line; supersampling supplies edge smoothing."""

        dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
        dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            self._set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def _draw_text(
        self,
        x: int,
        y: int,
        text: str,
        color: Color,
        dot: int,
        *,
        align: str = "left",
    ) -> int:
        text = str(text)
        width = text_width(text, dot)
        if align == "right":
            x -= width
        elif align == "center":
            x -= width // 2
        for character_index, character in enumerate(text):
            origin = x + character_index * (GLYPH_WIDTH * dot + dot)
            pattern = glyph(character)
            for row, bits in enumerate(pattern):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        self._fill_rect(
                            origin + column * dot,
                            y + row * dot,
                            origin + (column + 1) * dot,
                            y + (row + 1) * dot,
                            color,
                        )
        return width

    def _draw_header(self, frame_index: int) -> None:
        ss = self.ss
        self._fill_rect(0, 0, self.width, 64 * ss, (3, 9, 12))
        self._fill_rect(16 * ss, 13 * ss, 19 * ss, 52 * ss, CYAN)
        self._draw_text(
            28 * ss,
            13 * ss,
            self.config.title,
            INK,
            self.font_header,
        )
        subtitle = self.config.subtitle
        if subtitle and self.config.width >= 640:
            available = self.width - 28 * ss - 340 * ss
            subtitle = subtitle.upper()
            while subtitle and text_width(subtitle, self.font_small) > available:
                subtitle = subtitle[:-1]
            self._draw_text(
                28 * ss,
                40 * ss,
                subtitle,
                MUTED,
                self.font_small,
            )
        frame_text = f"FRAME {frame_index:04d} / {self.timeline.frame_count:04d}"
        self._draw_text(
            self.width - 18 * ss,
            14 * ss,
            frame_text,
            INK,
            self.font_small,
            align="right",
        )
        if self.config.width >= 640:
            cfr_text = f"CFR {self.timeline.fps} FPS  H.264  AUDIO 48 kHz"
            self._draw_text(
                self.width - 18 * ss,
                40 * ss,
                cfr_text,
                MUTED,
                self.font_small,
                align="right",
            )
        self._hline(18 * ss, self.width - 18 * ss, 63 * ss, BORDER)

    def _draw_plot_background(self) -> None:
        ss = self.ss
        self._fill_rect(
            self.plot_x0,
            self.plot_y0,
            self.plot_x1,
            self.plot_y1,
            PANEL,
        )
        for step in range(self.y_tick_count):
            fraction = step / (self.y_tick_count - 1)
            y = round(self.plot_y0 + fraction * self.plot_h)
            self._hline(
                self.plot_x0,
                self.plot_x1,
                y,
                GRID_MAJOR if step in (0, self.y_tick_count - 1) else GRID,
            )
        for step in range(self.y_tick_count):
            fraction = step / (self.y_tick_count - 1)
            x = round(self.plot_x0 + fraction * self.plot_w)
            self._vline(
                x,
                self.plot_y0,
                self.plot_y1,
                GRID_MAJOR if step in (0, self.x_tick_count - 1) else GRID,
            )
        self._fill_rect(
            self.plot_x0 - ss,
            self.plot_y0 - ss,
            self.plot_x1 + ss,
            self.plot_y0,
            BORDER,
        )
        self._fill_rect(
            self.plot_x0 - ss,
            self.plot_y1,
            self.plot_x1 + ss,
            self.plot_y1 + ss,
            BORDER,
        )
        self._fill_rect(
            self.plot_x0 - ss,
            self.plot_y0,
            self.plot_x0,
            self.plot_y1,
            BORDER,
        )
        self._fill_rect(
            self.plot_x1,
            self.plot_y0,
            self.plot_x1 + ss,
            self.plot_y1,
            BORDER,
        )
        unit = display_unit(self.data.unit)
        self._draw_text(
            self.plot_x0,
            self.plot_y0 - 26 * ss,
            f"VOLTAGE / {unit}",
            MUTED,
            self.font_small,
        )
        if self.config.width >= 640:
            self._draw_text(
                self.plot_x1,
                self.plot_y0 - 26 * ss,
                f"{self.timeline.window_span_s:g} s WINDOW",
                MUTED,
                self.font_small,
                align="right",
            )
        self._draw_y_ticks()
        self._draw_time_ticks()

    def _draw_y_ticks(self) -> None:
        ss = self.ss
        value_span = self.y_max - self.y_min
        for step in range(self.y_tick_count):
            fraction = step / (self.y_tick_count - 1)
            value = self.y_max - fraction * value_span
            y = round(self.plot_y0 + fraction * self.plot_h)
            label = self._format_axis_value(value, value_span)
            self._draw_text(
                self.plot_x0 - 8 * ss,
                y - (GLYPH_HEIGHT * self.font_axis) // 2,
                label,
                MUTED,
                self.font_axis,
                align="right",
            )

    def _draw_time_ticks(self) -> None:
        ss = self.ss
        view_start = self.timeline.window_start(self._rendering_frame)
        for step in range(self.x_tick_count):
            fraction = step / (self.x_tick_count - 1)
            value = view_start + fraction * self.timeline.window_span_s
            x = round(self.plot_x0 + fraction * self.plot_w)
            label = self._format_time_axis(value)
            align = "center"
            if step == 0:
                align = "left"
            elif step == self.x_tick_count - 1:
                align = "right"
            self._draw_text(
                x,
                self.plot_y1 + 10 * ss,
                label,
                MUTED,
                self.font_axis,
                align=align,
            )

    @staticmethod
    def _format_axis_value(value: float, span: float) -> str:
        magnitude = max(abs(value), span)
        if magnitude >= 100:
            return f"{value:.0f}"
        if magnitude >= 10:
            return f"{value:.1f}"
        if magnitude >= 1:
            return f"{value:.2f}"
        if magnitude >= 0.1:
            return f"{value:.3f}"
        return f"{value:.4f}"

    def _format_time_axis(self, value: float) -> str:
        span = self.timeline.window_span_s
        if span >= 100:
            return f"{value:.1f} s"
        if span >= 10:
            return f"{value:.3f} s"
        if span >= 1:
            return f"{value:.4f} s"
        if span >= 0.1:
            return f"{value:.5f} s"
        return f"{value:.6f} s"

    def _draw_unknown_regions(self, frame_index: int) -> None:
        ss = self.ss
        view_start = self.timeline.window_start(frame_index)
        view_end = view_start + self.timeline.window_span_s
        for gap_start, gap_end in self.gaps:
            if gap_end < view_start or gap_start > view_end:
                continue
            left = max(gap_start, view_start)
            right = min(gap_end, view_end)
            x0 = self._time_to_x(left)
            x1 = self._time_to_x(right)
            if x1 <= x0:
                x1 = x0 + ss
            self._fill_rect(x0, self.plot_y0, x1, self.plot_y1, GAP_FILL)
            for x in range(x0 + 8 * ss, x1, 18 * ss):
                self._vline(x, self.plot_y0, self.plot_y1, (52, 29, 34))
            self._vline(x0, self.plot_y0, self.plot_y1, (91, 48, 54))
            self._vline(x1 - 1, self.plot_y0, self.plot_y1, (91, 48, 54))
            if x1 - x0 > 74 * ss:
                label = "NO DATA"
                width = text_width(label, self.font_small)
                center = (x0 + x1) // 2
                box_y = self.plot_y0 + 22 * ss
                self._fill_rect(
                    center - width // 2 - 5 * ss,
                    box_y - 3 * ss,
                    center + width // 2 + 5 * ss,
                    box_y + GLYPH_HEIGHT * self.font_small + 3 * ss,
                    (18, 10, 13),
                )
                self._draw_text(
                    center,
                    box_y,
                    label,
                    GAP_INK,
                    self.font_small,
                    align="center",
                )

    def _time_to_x(self, time_s: float) -> int:
        view_start = self.timeline.window_start(self._rendering_frame)
        fraction = (time_s - view_start) / self.timeline.window_span_s
        return round(self.plot_x0 + fraction * self.plot_w)

    def _voltage_to_y(self, voltage: float) -> int:
        fraction = (self.y_max - voltage) / (self.y_max - self.y_min)
        y = round(self.plot_y0 + fraction * self.plot_h)
        return max(self.plot_y0, min(self.plot_y1 - 1, y))

    def _draw_trace(self, frame_index: int) -> None:
        ss = self.ss
        view_start = self.timeline.window_start(frame_index)
        view_end = view_start + self.timeline.window_span_s
        low = bisect_left(self.times, view_start)
        high = bisect_right(self.times, view_end)
        low = max(0, low - 1)
        high = min(len(self.times), high + 1)
        if low >= high:
            return

        bins: list[dict[int, list[int]] | None] = [None] * self.plot_w
        for index in range(low, high):
            voltage = self.voltages[index]
            if voltage is None or not self.drawable[index]:
                continue
            x = self._time_to_x(self.times[index])
            x = max(self.plot_x0, min(self.plot_x1 - 1, x)) - self.plot_x0
            y = self._voltage_to_y(voltage)
            bucket = bins[x]
            if bucket is None:
                bucket = {}
                bins[x] = bucket
            state = bucket.get(self.runs[index])
            if state is None:
                bucket[self.runs[index]] = [y, y, 1, y, y]
            else:
                state[0] = min(state[0], y)
                state[1] = max(state[1], y)
                state[2] += 1
                state[4] = y

        for x, bucket in enumerate(bins):
            if bucket is None:
                continue
            screen_x = self.plot_x0 + x
            for state in bucket.values():
                y0 = max(self.plot_y0, state[0] - 2 * ss)
                y1 = min(self.plot_y1 - 1, state[1] + 2 * ss)
                self._vline(screen_x - ss, y0, y1, CYAN_GLOW)
                self._vline(screen_x + ss, y0, y1, CYAN_GLOW)

        connections: list[tuple[int, int, int, int, int]] = []
        previous: dict[int, tuple[int, int]] = {}
        for x, bucket in enumerate(bins):
            if bucket is None:
                continue
            for run, state in bucket.items():
                if run in previous:
                    previous_x, previous_y = previous[run]
                    if x > previous_x:
                        connections.append((previous_x, previous_y, x, state[3], run))
                previous[run] = (x, state[4])
        for x0, y0, x1, y1, _run in connections:
            self._line(
                self.plot_x0 + x0,
                y0,
                self.plot_x0 + x1,
                y1,
                CYAN,
            )
        for x, bucket in enumerate(bins):
            if bucket is None:
                continue
            screen_x = self.plot_x0 + x
            for state in bucket.values():
                y0 = max(self.plot_y0, state[0])
                y1 = min(self.plot_y1 - 1, state[1])
                self._vline(screen_x, y0, y1, CYAN)
                if state[0] == state[1]:
                    self._set_pixel(screen_x, y0, CYAN_HOT)
                else:
                    self._set_pixel(screen_x, y0, CYAN_HOT)
                    self._set_pixel(screen_x, y1, CYAN_HOT)

    def _draw_event_guides(self, frame_index: int) -> None:
        ss = self.ss
        view_start = self.timeline.window_start(frame_index)
        view_end = view_start + self.timeline.window_span_s
        for index, event in enumerate(self.data.events):
            first = self.event_first_frames[index]
            is_flash = first <= frame_index < first + self.config.flash_frames
            if not (view_start <= event.time_s <= view_end):
                continue
            x = self._time_to_x(event.time_s)
            color = (238, 173, 61) if is_flash else (116, 91, 43)
            for y in range(self.plot_y0 + 8 * ss, self.plot_y1, 10 * ss):
                self._vline(x, y, min(y + 4 * ss, self.plot_y1), color)

    def _draw_event_annotations(self, frame_index: int) -> None:
        ss = self.ss
        view_start = self.timeline.window_start(frame_index)
        view_end = view_start + self.timeline.window_span_s
        flashing: list[tuple[int, Event]] = []
        for index, event in enumerate(self.data.events):
            first = self.event_first_frames[index]
            is_flash = first <= frame_index < first + self.config.flash_frames
            if is_flash:
                flashing.append((index, event))
            if not (view_start <= event.time_s <= view_end):
                continue
            x = self._time_to_x(event.time_s)
            color = (255, 219, 126) if is_flash else AMBER
            self._line(x - 5 * ss, self.plot_y0, x + 5 * ss, self.plot_y0, color)
            self._line(x - 5 * ss, self.plot_y0, x, self.plot_y0 + 7 * ss, color)
            self._line(x + 5 * ss, self.plot_y0, x, self.plot_y0 + 7 * ss, color)
            label = event.safe_label
            width = text_width(label, self.font_small)
            label_x = max(
                self.plot_x0 + 4 * ss,
                min(self.plot_x1 - width - 4 * ss, x - width // 2),
            )
            self._fill_rect(
                label_x - 3 * ss,
                self.plot_y0 + 11 * ss,
                label_x + width + 3 * ss,
                self.plot_y0 + 11 * ss + GLYPH_HEIGHT * self.font_small + 3 * ss,
                PANEL,
            )
            self._draw_text(
                label_x,
                self.plot_y0 + 12 * ss,
                label,
                color,
                self.font_small,
            )
        if flashing:
            indexes = [index for index, _event in flashing]
            text = "EVENT " + ",".join(str(index + 1) for index in indexes)
            badge_font = self.font_normal if self.config.width >= 640 else self.font_small
            width = text_width(text, badge_font)
            x1 = self.plot_x1 - 8 * ss
            x0 = x1 - width - 12 * ss
            y0 = self.plot_y0 + 8 * ss
            y1 = y0 + GLYPH_HEIGHT * badge_font + 8 * ss
            self._fill_rect(x0, y0, x1, y1, RED)
            self._draw_text(
                (x0 + x1) // 2,
                y0 + 4 * ss,
                text,
                INK,
                badge_font,
                align="center",
            )

    def _draw_footer(self, frame_index: int) -> None:
        ss = self.ss
        footer_y = self.height - 30 * ss
        self._fill_rect(0, footer_y, self.width, self.height, (3, 9, 12))
        self._hline(18 * ss, self.width - 18 * ss, footer_y, BORDER)
        scale = self.timeline.time_scale
        if self.config.width >= 640:
            left = (
                f"WINDOW {self.timeline.window_span_s:.6g} s"
                f"  SCALE {scale:.3f}x"
                f"  SAMPLES {len(self.times)}"
            )
            footer_font = self.font_small
        else:
            left = f"W {self.timeline.window_span_s:.4g} s"
            footer_font = self.font_small
        center = self.timeline.window_center(frame_index)
        right = f"T = {center:+.6f} s"
        self._draw_text(
            18 * ss,
            footer_y + 9 * ss,
            left,
            MUTED,
            footer_font,
        )
        self._draw_text(
            self.width - 18 * ss,
            footer_y + (8 * ss if self.config.width >= 640 else 9 * ss),
            right,
            INK,
            self.font_axis,
            align="right",
        )
