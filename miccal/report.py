from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .ambiguity import LayoutAmbiguity, audit_scene, catalog_layouts
from .beam import BeamPattern
from .config import Scene
from .estimate import CalibrationResult


def write_results(
    result: CalibrationResult,
    pattern: BeamPattern,
    output_dir: str | Path,
    truth: dict[str, Any] | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "calibration.json"
    csv_path = output / "channels.csv"
    audit_path = output / "ambiguity.json"
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "channel",
                "mapped_microphone",
                "relative_delay_us",
                "delay_residual_mm",
                "signed_gain_relative",
                "gain_error_db",
                "polarity",
            ]
        )
        for channel in range(4):
            writer.writerow(
                [
                    channel,
                    result.best_map[channel],
                    f"{result.relative_delays_s[channel] * 1e6:.3f}",
                    f"{result.delay_residual_mm[channel]:.3f}",
                    f"{result.gain_relative[channel]:.6f}",
                    f"{result.gain_error_db[channel]:.3f}",
                    int(result.polarity[channel]),
                ]
            )
    current_audit = audit_scene(
        result.scene, result_map=result.best_map, name=result.scene.name
    )
    catalog = catalog_layouts()
    audit_path.write_text(
        json.dumps(
            {
                "current": _audit_dict(current_audit),
                "catalog": [_audit_dict(item) for item in catalog],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "calibration_json": str(json_path),
        "channels_csv": str(csv_path),
        "ambiguity_json": str(audit_path),
    }


def write_markdown_report(
    result: CalibrationResult,
    pattern: BeamPattern,
    path: str | Path,
    truth: dict[str, Any] | None = None,
) -> None:
    scene = result.scene
    current_audit = audit_scene(scene, result_map=result.best_map, name=scene.name)
    catalog = catalog_layouts()
    lines = [
        f"# 四麦克风校准报告：{scene.name}",
        "",
        f"**总体状态：`{result.status}`**",
        "",
        "估计依据是四通道直达声的几何一致性，不依据“最大峰必为直达声”。"
        "因此单路 1.0 倍以上的强一次反射会被记录并跳过。",
        "",
        "## 场景",
        "",
        f"- 采样率：`{scene.sample_rate} Hz`，声速：`{scene.sound_speed:.1f} m/s`",
        f"- 参考通道：`{scene.reference_channel}`（其相对延时固定为 0）",
        f"- 声源位置：`{', '.join(str(list(s.position)) for s in scene.sources)}`",
        f"- 标称麦克风：`{scene.microphones}`",
        (
            f"- 一次反射幅度（按输出通道）："
            f"`{list(scene.wall.reflection_amplitudes)}`"
            if scene.wall
            else "- 未配置一次反射"
        ),
        "",
        "## 估计结果",
        "",
        "| 通道 | 匹配麦克风 | 相对延时 (µs) | 延时残差 (mm) | 相对增益 (dB) | 极性 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for channel in range(4):
        lines.append(
            f"| {channel} | {result.best_map[channel]} | "
            f"{result.relative_delays_s[channel] * 1e6:.2f} | "
            f"{result.delay_residual_mm[channel]:+.2f} | "
            f"{result.gain_error_db[channel]:+.2f} | "
            f"{int(result.polarity[channel]):+d} |"
        )
    lines.extend(
        [
            "",
            f"最佳通道/麦克风映射：`{list(result.best_map)}`；"
            f"映射唯一：`{not result.map_ambiguous}`。",
            "",
            "候选峰与反射证据：",
            "",
        ]
    )
    for estimate in result.source_estimates:
        details = []
        for channel, (chosen, candidates) in enumerate(
            zip(estimate.chosen, estimate.candidates)
        ):
            later = [
                peak
                for peak in candidates
                if peak.delay_s > chosen.delay_s
                and peak.envelope > chosen.envelope * 1.02
            ]
            ratio = (
                max(peak.envelope for peak in later) / chosen.envelope
                if later
                else 1.0
            )
            details.append(
                f"CH{channel}: 选中 {chosen.delay_s * 1e3:.3f} ms，"
                f"{len(candidates)} 峰，最强后峰/直达 {ratio:.2f}x"
            )
        lines.append(f"- `{estimate.source_name}`：" + "；".join(details))
    lines.extend(["", "### 安装诊断", ""])
    for finding in result.findings:
        lines.append(
            f"- `{finding['severity']}` / `{finding['code']}`：{finding['message']}"
        )
    if truth:
        lines.extend(_truth_section(truth))
    lines.extend(
        [
            "",
            "## 校准前后的主瓣",
            "",
            "![校准前后波束主瓣](beam_before_after.svg)",
            "",
            f"- 校准前主瓣峰值：`{pattern.before_peak_deg:+.1f}°`",
            f"- 校准后主瓣峰值：`{pattern.after_peak_deg:+.1f}°`",
            (
                f"- 校准后 -6 dB 主瓣宽度："
                f"`{pattern.after_minus6db_width_deg:.1f}°`"
                if pattern.after_minus6db_width_deg is not None
                else "- 校准后 -6 dB 主瓣宽度：`n/a`"
            ),
            "- 远场假设不成立时使用固定距离的近场极坐标切面；仿真图使用已知安装位置。",
            "",
            "## 当前布局可辨识性",
            "",
            f"- 通道排列等价类数量：`{len(current_audit.equivalent_orders)}`",
            f"- 强反射替代解数量：`{len(current_audit.reflection_orders)}`",
            f"- 同布局直达/反射组合歧义数：`{current_audit.reflection_selections}`",
        ]
    )
    for note in current_audit.notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## 无法唯一判断的布局",
            "",
            "以下巡检只使用布局、声源位置、墙面一次反射及容差；"
            "“等价排列 > 1”表示仅凭延时无法确认接线顺序。",
            "",
            "方向秩是声源方向在四麦克风投影空间中的秩；小于 3 时，"
            "阵列平面法向的镜像或离面几何存在额外盲区。",
            "",
            "| 布局 | 方向秩 | 等价排列 | 反射替代排列 | 唯一可判 |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for item in catalog:
        lines.append(
            f"| {item.name} | {item.source_rank} | "
            f"{len(item.equivalent_orders)} | {len(item.reflection_orders)} | "
            f"{'是' if item.identifiable else '否'} |"
        )
    ambiguous = [item for item in catalog if not item.identifiable]
    lines.extend(["", "明确的不可唯一判断布局："])
    if ambiguous:
        for item in ambiguous:
            reason = "；".join(item.notes) or "存在等价几何解"
            lines.append(f"- `{item.name}`：{reason}")
    else:
        lines.append("- 巡检集合中未发现不可辨识布局。")
    lines.extend(
        [
            "",
            "## 判读边界",
            "",
            "- 只能测得**相对极性**；参考通道极性被定义为 +1，全部通道同时反相不可由声学测量区分。",
            "- 增益和延时均相对参考通道；共同电子延时、共同灵敏度不会进入结果。",
            "- 一个声源位置只能约束该方向上的投影，接线与镜像布局需要多方向声源才能分开。",
            "- 若直达峰与反射峰落在同一容差等价类内，报告会把映射标为歧义，而不是强行给出接线结论。",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _truth_section(truth: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "### 仿真真值对照",
        "",
        "| 通道 | 真值增益 | 真值极性 | 仿真物理麦克风 |",
        "|---:|---:|---:|---:|",
    ]
    for item in truth.get("channels", []):
        lines.append(
            f"| {item['channel']} | {item['gain']:.4f} | "
            f"{int(item['polarity']):+d} | {item['nominal_microphone']} |"
        )
    reflected = []
    for source in truth.get("sources", []):
        for channel, details in enumerate(source.get("channels", [])):
            direct = abs(details.get("direct_amplitude") or 0.0)
            reflection = abs(details.get("reflection_amplitude") or 0.0)
            if direct > 0 and reflection > direct:
                reflected.append(
                    f"{source['name']} CH{channel} ({reflection / direct:.2f}x)"
                )
    lines.append("")
    lines.append(
        "- 强于直达声的仿真反射："
        + (", ".join(reflected) if reflected else "无")
    )
    return lines


def _audit_dict(item: LayoutAmbiguity) -> dict[str, Any]:
    return {
        "name": item.name,
        "source_rank": item.source_rank,
        "mapping_ambiguous": item.mapping_ambiguous,
        "reflection_ambiguous": item.reflection_ambiguous,
        "identifiable": item.identifiable,
        "equivalent_orders": [list(order) for order in item.equivalent_orders],
        "reflection_orders": [list(order) for order in item.reflection_orders],
        "reflection_selections": item.reflection_selections,
        "notes": item.notes,
    }
