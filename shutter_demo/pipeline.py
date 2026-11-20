"""把光源、快门、VFR 调度、积分与接触表组装成可复现实验。"""
from __future__ import annotations

import os

import numpy as np

from .color import linear_to_uint8
from .contact import ContactRow, render_contact_sheet
from .integrator import integrate_schedule, point_sample_schedule
from .metrics import count_duplicates
from .report import Experiment, build_report, write_json, write_text_summary
from .shutter import make_shutter
from .sources import BlendingDemo, FlashField, MovingBars
from .vfr import cfr_schedule, pulldown_2_1_schedule


def _save_png(path: str, exposure: np.ndarray) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(linear_to_uint8(exposure), mode="RGB").save(path)


def moving_bars_experiment(
    out_dir: str,
    samples_per_second: float = 40000.0,
) -> Experiment:
    """不同快门角与开合曲线下的移动黑白条：拖影长度与曝光重心对比。"""
    source = MovingBars(
        width=160,
        height=90,
        bar_period_px=48.0,
        speed_px_s=160.0,
        initial_offset_px=10.0,
    )
    schedule = cfr_schedule(fps=24.0, frame_count=8)
    configs = [
        ("90° 矩形", make_shutter(90.0, "rect")),
        ("180° 矩形", make_shutter(180.0, "rect")),
        ("360° 矩形", make_shutter(360.0, "rect")),
        ("180° 梯形缓开", make_shutter(180.0, "trapezoid")),
        ("180° 高斯软快门", make_shutter(180.0, "gaussian")),
        ("180° 双狭缝", make_shutter(180.0, "double_slot")),
    ]
    rows = []
    for title, shutter in configs:
        result = integrate_schedule(source, shutter, schedule, samples_per_second)
        rows.append(ContactRow(title=title, result=result, shutter=shutter))
    experiment = Experiment(
        name="移动黑白条：快门角与开合曲线",
        rows=rows,
        source_description={
            "type": "MovingBars",
            "speed_px_s": source.speed_px_s,
            "bar_period_px": source.bar_period_px,
            "output_fps": 24.0,
        },
    )
    _emit(experiment, os.path.join(out_dir, "moving_bars"), samples_per_second)
    return experiment


def flash_experiment(
    out_dir: str,
    samples_per_second: float = 200000.0,
) -> Experiment:
    """稀疏闪光：完整捕获、部分捕获、漏捕与快门边界。"""
    flash_times = (0.1005, 0.3005, 0.5005)
    source = FlashField(
        width=160,
        height=90,
        flash_times_s=flash_times,
        flash_duration_s=0.001,
        flash_radiance=4.0,
        base_radiance=0.05,
    )
    schedule = cfr_schedule(fps=24.0, frame_count=16)
    configs = [
        ("90° 窄窗(易漏闪)", make_shutter(90.0, "rect")),
        ("180° 标准", make_shutter(180.0, "rect")),
        ("360° 全帧窗", make_shutter(360.0, "rect")),
        ("180° 居中窗(相位0.5)", make_shutter(180.0, "rect", phase_offset_ratio=0.5)),
    ]
    rows = []
    for title, shutter in configs:
        result = integrate_schedule(source, shutter, schedule, samples_per_second)
        floor = source.base_radiance * schedule.durations[0]
        rows.append(
            ContactRow(
                title=title,
                result=result,
                shutter=shutter,
                energy_floor=floor * 1.5,
            )
        )
    experiment = Experiment(
        name="稀疏闪光：能量与快门边界",
        rows=rows,
        source_description={
            "type": "FlashField",
            "flash_times_s": list(flash_times),
            "flash_duration_s": source.flash_duration_s,
            "flash_radiance": source.flash_radiance,
            "base_radiance": source.base_radiance,
            "pulse_energy": source.pulse_energy(),
        },
    )
    _emit(experiment, os.path.join(out_dir, "flash"), samples_per_second)
    return experiment


def vfr_experiment(
    out_dir: str,
    hfr_fps: float = 240.0,
    samples_per_second: float = 40000.0,
) -> Experiment:
    """CFR 与 2:1 VFR 对比，并演示高输出帧率下的重复采样。"""
    source = MovingBars(
        width=160,
        height=90,
        bar_period_px=48.0,
        speed_px_s=160.0,
        initial_offset_px=10.0,
    )
    cfr = cfr_schedule(fps=24.0, frame_count=9)
    vfr = pulldown_2_1_schedule(base_fps=24.0, cycle_count=3)
    shutter = make_shutter(180.0, "rect")
    rows = [
        ContactRow("CFR 24fps 积分", integrate_schedule(source, shutter, cfr, samples_per_second), shutter),
        ContactRow("VFR 2:1 积分", integrate_schedule(source, shutter, vfr, samples_per_second), shutter),
    ]
    # 点采样：输出 60fps 但输入只有 24fps 量级的 HFR 网格 -> 出现重复采样
    oversampled_output = cfr_schedule(fps=60.0, frame_count=16)
    point_result = point_sample_schedule(source, oversampled_output, hfr_fps=24.0)
    point_shutter = make_shutter(0.0, "rect")
    rows.append(
        ContactRow(
            f"点采样 60fps<-24fps 网格 (重复{count_duplicates(point_result.frames)}帧)",
            point_result,
            point_shutter,
        )
    )
    experiment = Experiment(
        name="VFR 与重复采样",
        rows=rows,
        source_description={
            "type": "MovingBars",
            "speed_px_s": source.speed_px_s,
            "hfr_input_fps": hfr_fps,
            "point_sample_grid_fps": 24.0,
        },
    )
    _emit(experiment, os.path.join(out_dir, "vfr"), samples_per_second)
    return experiment


def blending_experiment(out_dir: str) -> Experiment:
    """线性光混合与 sRGB 伽马域直接混合的逐帧交叉溶解对比。

    左右两个亮斑按互补权重做溶解：左斑权重从 1->0、右斑从 0->1。
    同一权重分别在线性域与 sRGB 伽马域混合，第三行给出逐帧差异，
    可清楚看到伽马域混合在中间调系统性偏暗。
    """
    demo = BlendingDemo(width=160, height=90)
    duration = 1.0
    frame_count = 8
    schedule = cfr_schedule(fps=1.0, frame_count=frame_count)
    linear_frames, gamma_frames, diffs = [], [], []
    for idx in range(frame_count):
        # 左斑取 t=0（位置 A），右斑取 t=duration（位置 B），权重互补
        alpha = (idx + 0.5) / frame_count
        linear = demo.linear_blend(0.0, duration, duration, alpha)
        gamma = demo.gamma_blend(0.0, duration, duration, alpha)
        linear_frames.append(linear)
        gamma_frames.append(gamma)
        diffs.append(np.abs(linear - gamma))
    max_diff = max(float(diff.max()) for diff in diffs)
    diff_frames = [diff / max(max_diff, 1e-9) for diff in diffs]
    shutter = make_shutter(180.0, "rect")

    def _wrap(frames: list[np.ndarray], title: str) -> ContactRow:
        source = _SequenceSource(np.stack(frames, axis=0))
        result = integrate_schedule(source, shutter, schedule, samples_per_second=1000.0)
        return ContactRow(title, result, shutter, highlight_energy=False)

    rows = [
        _wrap(linear_frames, "线性光混合 (物理正确)"),
        _wrap(gamma_frames, "sRGB 伽马域混合 (对照)"),
        _wrap(diff_frames, "逐帧差异 (归一化放大)"),
    ]
    experiment = Experiment(
        name="线性光混合 vs 伽马域混合",
        rows=rows,
        source_description={
            "type": "BlendingDemo",
            "mode": "cross_dissolve_complementary_weights",
            "frame_count": frame_count,
            "max_abs_diff_linear": max_diff,
            "mean_abs_diff_linear": float(np.mean(np.stack(diffs, axis=0))),
        },
    )
    _emit(experiment, os.path.join(out_dir, "blending"), samples_per_second=1000.0)
    _save_png(
        os.path.join(out_dir, "blending", "linear_blend.png"), linear_frames[frame_count // 2]
    )
    _save_png(
        os.path.join(out_dir, "blending", "gamma_blend.png"), gamma_frames[frame_count // 2]
    )
    return experiment


class _StaticSource:
    """把一张固定曝光图包装成光源，供接触表管线复用。"""

    def __init__(self, exposure: np.ndarray):
        self._exposure = exposure
        self.height, self.width = exposure.shape[:2]

    def radiance(self, t: float) -> np.ndarray:
        return self._exposure.copy()

    def radiance_batch(self, times: np.ndarray) -> np.ndarray:
        return np.repeat(self._exposure[None, ...], np.asarray(times).size, axis=0)


class _SequenceSource:
    """把一组离散帧包装成积分器可用的光源，按 1fps 等间隔时间取帧。"""

    def __init__(self, frames: np.ndarray):
        self._frames = frames
        self.count, self.height, self.width = frames.shape[:3]

    def radiance(self, t: float) -> np.ndarray:
        idx = int(np.clip(np.floor(t), 0, self.count - 1))
        return self._frames[idx].copy()

    def radiance_batch(self, times: np.ndarray) -> np.ndarray:
        return np.stack([self.radiance(float(t)) for t in times], axis=0)


def _emit(experiment: Experiment, sub_dir: str, samples_per_second: float) -> None:
    sheet_path = os.path.join(sub_dir, "contact_sheet.png")
    render_contact_sheet(
        experiment.rows,
        sheet_path,
        title=experiment.name,
    )
    report = build_report(experiment)
    write_json(report, os.path.join(sub_dir, "report.json"))
    write_text_summary(report, os.path.join(sub_dir, "report.txt"))


def run_all_demos(out_dir: str) -> list[Experiment]:
    """运行全部预置实验并落盘接触表与报告。"""
    return [
        moving_bars_experiment(out_dir),
        flash_experiment(out_dir),
        vfr_experiment(out_dir),
        blending_experiment(out_dir),
    ]
