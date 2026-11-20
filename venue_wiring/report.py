"""Render machine-readable and human-reviewable analysis reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


STATUS_ZH = {
    "ok": "正常",
    "inverted": "反相",
    "duplicate": "重复接线",
    "open": "断路/无测试信号",
    "review": "人工复核",
}

POLARITY_ZH = {
    "normal": "正常",
    "inverted": "反相",
    "unknown": "未知",
    "undetermined": "待定",
}

CONFIDENCE_ZH = {"high": "高", "medium": "中", "low": "低"}


def _channel_heading(channel: dict[str, Any]) -> str:
    status = STATUS_ZH.get(channel["status"], channel["status"])
    if channel["mapped_source"]:
        return f"{channel['return_channel']} → {channel['mapped_source']} ({status})"
    if channel["candidate_source"]:
        return f"{channel['return_channel']} → 候选 {channel['candidate_source']} ({status})"
    return f"{channel['return_channel']} ({status})"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 八路回录接线分析报告",
        "",
        "## 结论摘要",
        "",
        f"- 自动结论：{summary['automatic_claims']} / {summary['channel_count']} 路",
        f"- 正常 {summary['ok']}，反相 {summary['inverted']}，"
        f"重复 {summary['duplicate']}，断路 {summary['open']}，"
        f"待复核 {summary['review']}",
        f"- 重复接线组：{summary['duplicate_group_count']}",
        f"- 未观测到回录的源：{', '.join(summary['unobserved_sources']) or '无'}",
        f"- 需要人工复核：{'是' if summary['manual_review_required'] else '否'}",
        "",
    ]
    if report["manual_review"]:
        lines.extend(["## 人工复核", ""])
        for item in report["manual_review"]:
            lines.append(
                f"- `{item['return_channel']}`：候选 `{item['candidate_source']}`；"
                f"{item['reason']}"
            )
        lines.append("")

    lines.extend(
        [
            "## 映射与证据",
            "",
            "| 回录通道 | 状态 | 映射源 | 候选极性 | 相关峰 | 延迟 ms | 源裕量 | 峰-次峰 dB | 置信 |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for channel in report["channels"]:
        evidence = channel["evidence"]
        lines.append(
            "| {return_channel} | {status} | {source} | {polarity} | {corr:.3f} | "
            "{lag:.2f} | {margin:.3f} | {psr:.1f} | {level} ({score:.2f}) |".format(
                return_channel=channel["return_channel"],
                status=STATUS_ZH.get(channel["status"], channel["status"]),
                source=channel["mapped_source"] or channel["candidate_source"] or "—",
                polarity=POLARITY_ZH.get(channel["polarity"], channel["polarity"]),
                corr=evidence["signed_correlation"],
                lag=evidence["lag_ms"],
                margin=evidence["source_margin"],
                psr=evidence["peak_to_secondary_db"],
                level=CONFIDENCE_ZH.get(
                    channel["confidence"]["level"], channel["confidence"]["level"]
                ),
                score=channel["confidence"]["score"],
            )
        )
    lines.append("")

    if report["duplicate_groups"]:
        lines.extend(["## 重复接线组", ""])
        for group in report["duplicate_groups"]:
            lines.append(
                f"- `{group['id']}`：源 `{group['source']}`；参考 "
                f"`{group['reference_channel']}`；成员 "
                f"{', '.join(group['members'])}"
            )
        lines.append("")

    lines.extend(["## 逐通道证据", ""])
    for channel in report["channels"]:
        evidence = channel["evidence"]
        lines.extend(
            [
                f"### {_channel_heading(channel)}",
                "",
                f"- 判定：{STATUS_ZH.get(channel['status'], channel['status'])}；"
                f"自动结论：{'是' if channel['automatic'] else '否'}",
                f"- 最强相关：{evidence['signed_correlation']:.4f}"
                f"（绝对 {evidence['absolute_correlation']:.4f}，"
                f"延迟 {evidence['lag_ms']:.2f} ms，重叠 {evidence['overlap_fraction']:.0%}）",
                f"- 次优源：`{evidence['second_source']}`；裕量 "
                f"{evidence['source_margin']:.4f}",
                f"- 峰突出度：{evidence['peak_to_secondary_db']:.2f} dB；"
                f"峰-噪声底 {evidence['peak_to_noise']:.1f}×",
                f"- 置信：{CONFIDENCE_ZH.get(channel['confidence']['level'], channel['confidence']['level'])} "
                f"({channel['confidence']['score']:.3f})；"
                + "；".join(channel["confidence"]["basis"]),
                f"- 依据：{'；'.join(channel['reasons']) or '无'}",
                "",
            ]
        )
        top_three = evidence["all_source_peaks"][:3]
        lines.append("| 源候选 | 相关系数 | 延迟 ms | 极性 |")
        lines.append("|---|---:|---:|---|")
        for peak in top_three:
            lines.append(
                f"| {peak['source']} | {peak['signed_correlation']:.4f} | "
                f"{peak['lag_ms']:.2f} | "
                f"{POLARITY_ZH.get(peak['polarity'], peak['polarity'])} |"
            )
        lines.append("")

    config = report["config"]
    lines.extend(
        [
            "## 判定阈值",
            "",
            f"- 自动接受相关阈值：{config['accept_threshold']:.3f}",
            f"- 开路阈值：{config['open_threshold']:.3f}",
            f"- 最小源裕量：{config['margin_threshold']:.3f}",
            f"- 峰突出度置信参考点：{config['min_prominence_db']:.2f} dB",
            f"- 最小有效重叠：{config['min_overlap']:.0%}",
            f"- 重复波形相似度阈值：{config['duplicate_similarity']:.3f}",
            "",
            "峰突出度用于置信度展示，不作为硬否决条件，因为重复发射或混响拖尾"
            "可能产生合法的次峰；源裕量和重叠仍是自动接受的必要条件。"
            "任何未同时满足相关强度、源裕量和重叠要求的通道都会进入人工复核。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_evidence_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "return_channel",
        "decision",
        "rank",
        "source",
        "signed_correlation",
        "absolute_correlation",
        "lag_ms",
        "polarity",
        "overlap_fraction",
        "peak_to_secondary_db",
        "peak_to_noise",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for channel in report["channels"]:
            for rank, peak in enumerate(channel["evidence"]["all_source_peaks"], start=1):
                writer.writerow(
                    {
                        "return_channel": channel["return_channel"],
                        "decision": channel["status"],
                        "rank": rank,
                        "source": peak["source"],
                        "signed_correlation": peak["signed_correlation"],
                        "absolute_correlation": peak["absolute_correlation"],
                        "lag_ms": peak["lag_ms"],
                        "polarity": peak["polarity"],
                        "overlap_fraction": peak["overlap_fraction"],
                        "peak_to_secondary_db": peak["peak_to_secondary_db"],
                        "peak_to_noise": peak["peak_to_noise"],
                    }
                )


def write_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output / "report.json",
        "markdown": output / "report.md",
        "evidence_csv": output / "evidence.csv",
    }
    paths["json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    _write_evidence_csv(report, paths["evidence_csv"])
    return paths
