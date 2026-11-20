"""快门积分管线的解析级校验。"""
from __future__ import annotations

import os

import numpy as np

from shutter_demo.color import srgb_to_linear, linear_to_srgb
from shutter_demo.integrator import (
    integrate_schedule,
    point_sample_schedule,
)
from shutter_demo.metrics import count_duplicates, edge_spread_width, spatial_centroid_x
from shutter_demo.shutter import Shutter, make_shutter, load_custom_waveform
from shutter_demo.sources import BlendingDemo, FlashField, MovingBars
from shutter_demo.vfr import cfr_schedule, explicit_schedule, pulldown_2_1_schedule


def test_srgb_roundtrip_and_linearity():
    values = np.linspace(0.0, 1.0, 64)
    recovered = linear_to_srgb(srgb_to_linear(values))
    assert np.allclose(recovered, values, atol=1e-12)
    # 中间调在线性域更暗（伽马编码会抬高暗部）
    assert srgb_to_linear(0.5) < 0.5


def test_shutter_duty_and_window():
    shutter = make_shutter(180.0, "rect")
    assert np.isclose(shutter.duty, 0.5)
    w0, w1 = shutter.window(0.0, 1.0)
    assert np.isclose(w0, 0.0) and np.isclose(w1, 0.5)
    centered = make_shutter(180.0, "rect", phase_offset_ratio=0.5)
    w0, w1 = centered.window(0.0, 1.0)
    assert np.isclose(w0, 0.25) and np.isclose(w1, 0.75)
    times = np.array([0.0, 0.25, 0.3, 0.8])
    assert np.allclose(centered.weights(times, 0.0, 1.0), [0.0, 1.0, 1.0, 0.0])


def test_constant_scene_energy_normalized():
    """恒定光源下，无论快门角/曲线如何，归一化曝光都应等于光源本身。"""

    class Constant:
        height, width = 12, 16

        def radiance(self, t):
            return np.full((12, 16, 3), 0.7)

    schedule = cfr_schedule(10.0, 4)
    for angle in (45.0, 180.0, 360.0):
        for name in ("rect", "trapezoid", "gaussian", "double_slot"):
            result = integrate_schedule(
                Constant(), make_shutter(angle, name), schedule, 50_000.0
            )
            for frame in result.frames:
                assert np.allclose(frame.exposure, 0.7, atol=1e-9)


def test_motion_centroid_matches_window_center():
    """矩形快门下，亮斑空间重心应对应快门窗口中点时刻的位置。"""
    source = MovingBars(
        width=400,
        height=40,
        speed_px_s=120.0,
        bar_period_px=10000.0,  # 单脉冲：画面内只有一个完整亮条，重心即亮条中心
        bar_duty=0.005,  # 亮条宽 50px，远窄于画面，避免边缘裁切拉偏重心
        initial_offset_px=120.0,
    )
    shutter = make_shutter(180.0, "rect")
    schedule = cfr_schedule(24.0, 6)
    result = integrate_schedule(source, shutter, schedule, 100_000.0)
    for frame in result.frames:
        center_t = 0.5 * (frame.window_start + frame.window_end)
        # edge_position 是亮条左沿，重心在左沿加半个条宽
        half_stripe = 0.5 * source.bar_duty * source.bar_period_px
        expected_x = source.edge_position(center_t) + half_stripe
        measured_x = spatial_centroid_x(frame.exposure)
        assert abs(expected_x - measured_x) < 1.5
        assert np.isclose(frame.centroid_t, center_t, atol=1e-6)


def test_longer_shutter_angle_widens_motion_blur():
    source = MovingBars(width=200, height=40, speed_px_s=300.0, bar_period_px=120.0)
    schedule = cfr_schedule(24.0, 4)
    narrow = integrate_schedule(source, make_shutter(45.0, "rect"), schedule, 100_000.0)
    wide = integrate_schedule(source, make_shutter(360.0, "rect"), schedule, 100_000.0)
    narrow_width = np.nanmedian([edge_spread_width(f.exposure) for f in narrow.frames])
    wide_width = np.nanmedian([edge_spread_width(f.exposure) for f in wide.frames])
    assert wide_width > narrow_width * 2.0


def test_flash_full_partial_and_missed_capture():
    """闪光完全落入窗口得满能量，跨边界得部分能量，落在窗外则漏捕。"""
    flash_t = 0.1  # 位于 24fps 第 2 帧 [0.04167, 0.08333) 之后
    source = FlashField(
        width=40,
        height=20,
        flash_times_s=(flash_t,),
        flash_duration_s=0.002,
        flash_radiance=5.0,
        base_radiance=0.0,
    )
    schedule = cfr_schedule(24.0, 8)
    # 360 度窗口：第 3 帧 [0.08333, 0.125) 完整包含闪光
    full = integrate_schedule(source, make_shutter(360.0, "rect"), schedule, 500_000.0)
    hit_frame = full.frames[2]
    expected_energy = source.pulse_energy() / (1.0 / 24.0)
    bright_region = hit_frame.exposure[:, source.width // 2 :, :]
    assert np.isclose(float(bright_region.mean()), expected_energy, rtol=0.02)

    # 让闪光跨在窗口边界上：脉冲 1ms 在窗内、1ms 在窗外，即半捕获
    boundary_schedule = explicit_schedule(
        [flash_t - 0.02, flash_t + 0.001, flash_t + 0.021]
    )
    partial = integrate_schedule(
        source, make_shutter(360.0, "rect"), boundary_schedule, 500_000.0
    )
    partial_bright = partial.frames[0].exposure[:, source.width // 2 :, :].mean()
    # 窗内 1ms 脉冲 / 20ms 帧窗，理论归一化能量 = 5*0.001/0.02 = 0.25
    assert np.isclose(float(partial_bright), 0.25, rtol=0.05)
    # 第二帧捕获另一半，能量量级相同
    second_bright = partial.frames[1].exposure[:, source.width // 2 :, :].mean()
    assert np.isclose(float(second_bright), 0.25, rtol=0.05)

    # 90 度窄窗在该时刻漏捕：帧窗 [0.0833, 0.09375) 不覆盖 0.1
    miss = integrate_schedule(source, make_shutter(90.0, "rect"), schedule, 500_000.0)
    assert all(frame.energy < 1e-6 for frame in miss.frames)


def test_vfr_schedule_monotonic_and_pulldown_cycle():
    schedule = pulldown_2_1_schedule(24.0, cycle_count=2)
    assert schedule.count == 6
    assert np.all(np.diff(schedule.start_times) > 0)
    cycle_duration = float(np.sum(schedule.durations[:3]))
    assert np.isclose(cycle_duration, 1.0 / 24.0)
    long_dt, short_dt = schedule.durations[0], schedule.durations[2]
    assert np.isclose(long_dt / short_dt, 2.0)
    assert np.isclose(2 * long_dt + short_dt, 1.0 / 24.0)


def test_point_sampling_detects_duplicates():
    source = MovingBars(width=60, height=30, speed_px_s=100.0)
    # 输出 60fps，量化到 24fps 输入网格 -> 每 5 个输出帧约有 2 个重复
    schedule = cfr_schedule(60.0, frame_count=15)
    result = point_sample_schedule(source, schedule, hfr_fps=24.0)
    assert count_duplicates(result.frames) > 0
    duplicate_indices = [f.index for f in result.frames if f.duplicate_of is not None]
    # 重复帧内容必须与被重复帧完全一致
    for idx in duplicate_indices:
        original = result.frames[result.frames[idx].duplicate_of]
        assert np.array_equal(original.exposure, result.frames[idx].exposure)


def test_linear_vs_gamma_blend_differ():
    demo = BlendingDemo(width=80, height=40)
    linear = demo.linear_blend(0.25, 0.75, 1.0, 0.5)
    gamma = demo.gamma_blend(0.25, 0.75, 1.0, 0.5)
    assert not np.allclose(linear, gamma)
    # 在两斑重叠的中间区域：线性域平均保持物理亮度，伽马域平均偏暗
    overlap = (linear > 0) & (gamma > 0)
    assert float(gamma[overlap].mean()) < float(linear[overlap].mean())


def test_custom_waveform_loader():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples",
        "my_shutter.py",
    )
    waveform = load_custom_waveform(path)
    outside = waveform(np.array([-0.1, 1.2]))
    assert np.allclose(outside, 0.0)
    inside = waveform(np.linspace(0.0, 0.999, 20))
    assert np.all(inside >= 0.0) and np.all(inside <= 1.0)
    assert inside.max() > 0.9


def test_explicit_pts_validation():
    try:
        explicit_schedule([0.0, 0.1, 0.1])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("非严格递增 PTS 必须报错")


def test_gaussian_softer_than_rect_edges():
    source = MovingBars(width=200, height=40, speed_px_s=260.0, bar_period_px=120.0)
    schedule = cfr_schedule(24.0, 4)
    rect = integrate_schedule(source, make_shutter(180.0, "rect"), schedule, 100_000.0)
    gauss = integrate_schedule(source, make_shutter(180.0, "gaussian"), schedule, 100_000.0)
    rect_width = np.nanmedian([edge_spread_width(f.exposure) for f in rect.frames])
    gauss_width = np.nanmedian([edge_spread_width(f.exposure) for f in gauss.frames])
    # 高斯软快门等效曝光更集中，边沿更锐
    assert gauss_width < rect_width
