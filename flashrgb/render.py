from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .color import blank_rgb, compose_channels, clear_rect
from .draw import draw_border, draw_text, side_by_side, stack_vertical, text_width
from .imageio import GrayImage, RGBImage, VideoWriter, read_gray, write_png
from .metrics import (
    GroupEvaluation,
    aggregate_summary,
    evaluate_group,
    load_truth,
    load_truth_assets,
    truth_group_index,
)
from .motion import MotionField, estimate_motion, warp_gray
from .sync import (
    FrameRecord,
    build_sync_report,
    decode_flash,
    discover_frames,
    group_exposures,
    parse_roi,
    write_json,
)


def _copy_rgb(image: RGBImage) -> RGBImage:
    return RGBImage(image.width, image.height, image.pixels)


def _label_frame(
    image: RGBImage,
    text: str,
    color=(255, 255, 255),
    missing: bool = False,
) -> RGBImage:
    result = _copy_rgb(image)
    scale = 2
    width = text_width(text, scale) + 10
    clear_rect(result, 0, 0, min(image.width, max(112, width)), 28, (0, 0, 0))
    background = (104, 0, 0) if missing else (0, 0, 0)
    draw_text(result, 4, 4, text, color, scale=scale, background=background)
    if missing:
        draw_border(result, 0, 0, image.width, image.height, 4, (242, 42, 42))
    return result


def _blank_heatmap(width: int, height: int, label: str) -> RGBImage:
    image = blank_rgb(width, height, 8)
    draw_text(image, 4, 4, label, (120, 150, 190), scale=2, background=(0, 0, 0))
    return image


def _comparison_frame(
    before: RGBImage,
    after: RGBImage,
    before_label: str,
    after_label: str,
    evaluation: GroupEvaluation | None,
    missing: bool,
) -> RGBImage:
    before_panel = _label_frame(
        before, before_label, (250, 220, 120) if missing else (120, 210, 255), missing
    )
    after_panel = _label_frame(
        after, after_label, (250, 220, 120) if missing else (130, 245, 170), missing
    )
    top = side_by_side([before_panel, after_panel], gap=2, background=(24, 24, 24))
    if evaluation is not None:
        heat_before = _label_frame(
            evaluation.before_heatmap, "EDGE BEF", (255, 170, 80), missing
        )
        heat_after = _label_frame(
            evaluation.after_heatmap, "EDGE AFT", (255, 170, 80), missing
        )
    else:
        heat_before = _blank_heatmap(before.width, before.height, "EDGE BEF N/A")
        heat_after = _blank_heatmap(after.width, after.height, "EDGE AFT N/A")
    bottom = side_by_side([heat_before, heat_after], gap=2, background=(24, 24, 24))
    return stack_vertical([top, bottom], gap=2, background=(24, 24, 24))


def _target_mode_motion(
    field: MotionField, target_box: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    x, y, width, height = target_box
    vectors = [
        (tile.dx, tile.dy)
        for tile in field.tiles
        if tile.valid
        and (tile.dx or tile.dy)
        and x <= tile.cx < x + width
        and y <= tile.cy < y + height
    ]
    if not vectors:
        return None
    return Counter(vectors).most_common(1)[0][0]


def _write_metrics_csv(path: Path, evaluations: list[GroupEvaluation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group_id",
                "edge_pixels",
                "visible_edge_pixels",
                "before_mean_chroma_error",
                "before_p95_chroma_error",
                "after_mean_chroma_error",
                "after_p95_chroma_error",
                "improvement_percent",
            ]
        )
        for evaluation in evaluations:
            before = evaluation.before["visible_edge_chroma_error"]
            after = evaluation.after["visible_edge_chroma_error"]
            writer.writerow(
                [
                    evaluation.group_id,
                    evaluation.edge_pixels,
                    evaluation.visible_edge_pixels,
                    before["mean"],
                    before["p95"],
                    after["mean"],
                    after["p95"],
                    round(evaluation.improvement_percent, 4),
                ]
            )


def _write_pair_index_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "group_id",
        "status",
        "missing_count",
        "missing_channels",
        "red_source_frame",
        "green_source_frame",
        "blue_source_frame",
        "red_source_index",
        "green_source_index",
        "blue_source_index",
        "video_frame",
        "motion_R_to_G",
        "motion_B_to_G",
        "compensation_valid_ratio",
        "edge_chroma_before",
        "edge_chroma_after",
        "edge_improvement_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _find_truth_group(
    group,
    truth_index: dict[int, dict],
) -> dict | None:
    candidate = truth_index.get(group.group_id)
    actual_files = {record.path.name for record in group.frames.values()}
    if candidate is not None:
        expected_files = set(candidate.get("frames", {}).values())
        if expected_files == actual_files:
            return candidate
    for item in truth_index.values():
        if set(item.get("frames", {}).values()) == actual_files:
            return item
    return None


def render_sequence(
    input_path: str,
    output_dir: str | Path,
    flash_roi_value: str | None = None,
    search_radius: int = 16,
    fps: float = 4.0,
    truth_path: str | None = None,
    compensate: bool = True,
) -> dict:
    output = Path(output_dir)
    preview_dir = output / "previews"
    output.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for stale in preview_dir.glob("group_*.png"):
        stale.unlink()
    stale_comparison = output / "edge_comparison.png"
    if stale_comparison.exists():
        stale_comparison.unlink()

    frame_paths = discover_frames(input_path)
    images = [read_gray(path) for path in frame_paths]
    width = images[0].width
    height = images[0].height
    if any((image.width, image.height) != (width, height) for image in images):
        raise ValueError("all input frames must have the same dimensions")

    truth = load_truth(truth_path) if truth_path else None
    if flash_roi_value is not None:
        roi = parse_roi(flash_roi_value, width, height)
    elif truth and truth.get("flash_roi"):
        roi = parse_roi(",".join(str(value) for value in truth["flash_roi"]), width, height)
    else:
        roi = parse_roi("auto", width, height)

    records: list[FrameRecord] = []
    for order, (path, image) in enumerate(zip(frame_paths, images)):
        channel, confidence, means = decode_flash(image, roi)
        records.append(
            FrameRecord(
                path=path,
                order=order,
                channel=channel,
                confidence=confidence,
                cell_means=means,
            )
        )
    groups = group_exposures(records)
    if not groups:
        raise ValueError("no synchronized exposure groups were decoded")
    write_json(output / "sync_report.json", build_sync_report(records, groups, roi))

    truth_index = truth_group_index(truth) if truth else {}
    truth_root = truth["_root"] if truth else None
    pair_rows: list[dict] = []
    pair_json: list[dict] = []
    evaluations: list[GroupEvaluation] = []
    group_details: list[dict] = []
    best_evaluation: GroupEvaluation | None = None
    best_panels: tuple[RGBImage, RGBImage] | None = None

    before_writer = VideoWriter(output / "color_before.mp4", width, height, fps)
    after_writer = VideoWriter(output / "color_after.mp4", width, height, fps)
    comparison_writer = VideoWriter(
        output / "comparison.mp4", width * 2 + 2, height * 2 + 2, fps
    )
    writers = [before_writer, after_writer, comparison_writer]
    try:
        for group in groups:
            channels = {
                channel: images[record.order] for channel, record in group.frames.items()
            }
            before_rgb = compose_channels(channels, width, height)
            after_rgb = _copy_rgb(before_rgb)
            fields: dict[str, MotionField] = {}
            target_motions: dict[str, tuple[int, int] | None] = {}
            truth_group = _find_truth_group(group, truth_index) if truth else None
            complete = not group.missing

            if complete:
                reference = channels["G"]
                aligned: dict[str, GrayImage] = {"G": reference}
                if compensate:
                    for channel in ("R", "B"):
                        field = estimate_motion(
                            channels[channel],
                            reference,
                            channel,
                            "G",
                            search_radius=search_radius,
                            roi=roi,
                        )
                        fields[channel] = field
                        aligned[channel] = warp_gray(channels[channel], field)
                else:
                    fields["R"] = MotionField(
                        width, height, 0, 0, [], [], [], [], "R", "G"
                    )
                    fields["B"] = MotionField(
                        width, height, 0, 0, [], [], [], [], "B", "G"
                    )
                    aligned["R"] = channels["R"]
                    aligned["B"] = channels["B"]
                after_rgb = compose_channels(aligned, width, height)

            clear_rect(before_rgb, *roi)
            clear_rect(after_rgb, *roi)

            evaluation = None
            if (
                complete
                and truth is not None
                and truth_group is not None
                and truth_group.get("status") == "complete"
            ):
                truth_image, visible_mask, edge_mask = load_truth_assets(
                    truth_root, truth_group
                )
                evaluation = evaluate_group(
                    group.group_id,
                    before_rgb,
                    after_rgb,
                    truth_image,
                    edge_mask,
                    visible_mask,
                )
                evaluations.append(evaluation)

            if truth_group is not None:
                target_box = truth_group.get("target_positions", {}).get("G")
                if target_box:
                    box = (target_box[0], target_box[1], 72, 60)
                    for channel in ("R", "B"):
                        if channel in fields:
                            target_motions[channel] = _target_mode_motion(
                                fields[channel], box
                            )
                known_motion = truth_group.get("known_motion", {})
            else:
                known_motion = {}

            if complete:
                before_label = f"G{group.group_id:03d} BEF"
                after_label = f"G{group.group_id:03d} AFT"
            else:
                before_label = (
                    f"G{group.group_id:03d} BEF MISSING "
                    f"{len(group.missing)}:{''.join(group.missing)}"
                )
                after_label = (
                    f"G{group.group_id:03d} AFT MISSING "
                    f"{len(group.missing)}:{''.join(group.missing)}"
                )
            before_video = _label_frame(
                before_rgb, before_label, missing=not complete
            )
            after_video = _label_frame(
                after_rgb, after_label, missing=not complete
            )
            comparison = _comparison_frame(
                before_rgb,
                after_rgb,
                before_label,
                after_label,
                evaluation,
                missing=not complete,
            )
            before_writer.write(before_video)
            after_writer.write(after_video)
            comparison_writer.write(comparison)

            if evaluation is not None and (
                best_evaluation is None
                or evaluation.improvement_percent > best_evaluation.improvement_percent
            ):
                best_evaluation = evaluation
                best_panels = (_copy_rgb(before_rgb), _copy_rgb(after_rgb))

            write_png(
                preview_dir / f"group_{group.group_id:03d}_before.png",
                before_video,
            )
            write_png(
                preview_dir / f"group_{group.group_id:03d}_after.png",
                after_video,
            )
            if evaluation is not None:
                write_png(
                    preview_dir / f"group_{group.group_id:03d}_edge_before.png",
                    evaluation.before_heatmap,
                )
                write_png(
                    preview_dir / f"group_{group.group_id:03d}_edge_after.png",
                    evaluation.after_heatmap,
                )

            before_mean = (
                evaluation.before["visible_edge_chroma_error"]["mean"]
                if evaluation is not None
                else None
            )
            after_mean = (
                evaluation.after["visible_edge_chroma_error"]["mean"]
                if evaluation is not None
                else None
            )
            valid_ratios = [field.valid_ratio for field in fields.values()]
            row = {
                "group_id": group.group_id,
                "status": group.status,
                "missing_count": len(group.missing),
                "missing_channels": "".join(group.missing),
                "red_source_frame": group.frames.get("R").path.name
                if group.frames.get("R")
                else "",
                "green_source_frame": group.frames.get("G").path.name
                if group.frames.get("G")
                else "",
                "blue_source_frame": group.frames.get("B").path.name
                if group.frames.get("B")
                else "",
                "red_source_index": group.frames.get("R").order
                if group.frames.get("R")
                else "",
                "green_source_index": group.frames.get("G").order
                if group.frames.get("G")
                else "",
                "blue_source_index": group.frames.get("B").order
                if group.frames.get("B")
                else "",
                "video_frame": group.group_id,
                "motion_R_to_G": (
                    f"{target_motions['R'][0]},{target_motions['R'][1]}"
                    if target_motions.get("R")
                    else ""
                ),
                "motion_B_to_G": (
                    f"{target_motions['B'][0]},{target_motions['B'][1]}"
                    if target_motions.get("B")
                    else ""
                ),
                "compensation_valid_ratio": round(
                    sum(valid_ratios) / len(valid_ratios), 4
                )
                if valid_ratios
                else "",
                "edge_chroma_before": round(before_mean, 4)
                if before_mean is not None
                else "",
                "edge_chroma_after": round(after_mean, 4)
                if after_mean is not None
                else "",
                "edge_improvement_percent": round(evaluation.improvement_percent, 2)
                if evaluation is not None
                else "",
            }
            pair_rows.append(row)
            detail = dict(row)
            detail["motion_fields"] = {
                channel: field.to_dict() for channel, field in fields.items()
            }
            detail["known_motion"] = known_motion
            detail["estimated_target_motion"] = {
                channel: list(value) if value else None
                for channel, value in target_motions.items()
            }
            if evaluation is not None:
                detail["edge_metrics"] = evaluation.to_dict()
            pair_json.append(detail)
            group_details.append(detail)
    finally:
        for writer in writers:
            writer.close()

    _write_pair_index_csv(output / "pair_index.csv", pair_rows)
    write_json(output / "pair_index.json", {"groups": pair_json})
    _write_metrics_csv(output / "edge_metrics.csv", evaluations)

    if best_evaluation is not None and best_panels is not None:
        comparison = _comparison_frame(
            best_panels[0],
            best_panels[1],
            f"G{best_evaluation.group_id:03d} BEF",
            f"G{best_evaluation.group_id:03d} AFT",
            best_evaluation,
            missing=False,
        )
        write_png(output / "edge_comparison.png", comparison)

    summary = aggregate_summary(evaluations)
    metric_payload = {
        "summary": summary,
        "groups": [evaluation.to_dict() for evaluation in evaluations],
    }
    write_json(output / "edge_metrics.json", metric_payload)
    incomplete = [group for group in groups if group.missing]
    missing_two = [group for group in incomplete if len(group.missing) == 2]
    report = {
        "input": str(Path(input_path)),
        "output": str(output),
        "frame_size": [width, height],
        "frame_count": len(records),
        "flash_roi": list(roi),
        "compensation_enabled": compensate,
        "search_radius_px": search_radius,
        "fps": fps,
        "group_count": len(groups),
        "complete_groups": len(groups) - len(incomplete),
        "incomplete_groups": len(incomplete),
        "groups_missing_two_exposures": len(missing_two),
        "undecoded_frames": sum(record.channel is None for record in records),
        "edge_metrics": summary,
        "outputs": {
            "before_video": "color_before.mp4",
            "after_video": "color_after.mp4",
            "comparison_video": "comparison.mp4",
            "pair_index_csv": "pair_index.csv",
            "pair_index_json": "pair_index.json",
            "sync_report": "sync_report.json",
            "edge_metrics_csv": "edge_metrics.csv",
            "edge_metrics_json": "edge_metrics.json",
            "edge_comparison_png": "edge_comparison.png"
            if best_evaluation is not None
            else None,
        },
    }
    write_json(output / "run_report.json", report)
    return report
