from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from typing import Any, Sequence

from .config import Scene
from .geometry import (
    all_permutations,
    delay_rank,
    matrix_distance,
    predicted_relative_delays,
)
from .simulate import channel_events


@dataclass
class LayoutAmbiguity:
    name: str
    source_rank: int
    equivalent_orders: list[tuple[int, int, int, int]] = field(default_factory=list)
    reflection_orders: list[tuple[int, int, int, int]] = field(default_factory=list)
    reflection_selections: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def mapping_ambiguous(self) -> bool:
        return len(self.equivalent_orders) > 1

    @property
    def reflection_ambiguous(self) -> bool:
        return len(self.reflection_orders) > 1 or self.reflection_selections > 1

    @property
    def identifiable(self) -> bool:
        return not self.mapping_ambiguous and not self.reflection_ambiguous


def _delay_matrices(scene: Scene) -> dict[tuple[int, int, int, int], list[list[float]]]:
    matrices: dict[tuple[int, int, int, int], list[list[float]]] = {}
    sources = [source.position for source in scene.sources]
    for order in all_permutations():
        mics = tuple(scene.microphones[index] for index in order)
        matrices[order] = [
            list(row)
            for row in predicted_relative_delays(
                mics, sources, scene.reference_channel, scene.sound_speed
            )
        ]
    return matrices


def _reflection_events(scene: Scene) -> list[list[list[tuple[float, float]]]]:
    output: list[list[list[tuple[float, float]]]] = []
    for source in scene.sources:
        per_source: list[list[tuple[float, float]]] = []
        for channel in range(4):
            events, _ = channel_events(scene, source.position, channel)
            per_source.append([(time, amplitude) for time, amplitude in events])
        output.append(per_source)
    return output


def _selection_scores(
    scene: Scene,
    events: Sequence[Sequence[Sequence[tuple[float, float]]]],
    order: tuple[int, int, int, int],
) -> tuple[float, int]:
    reference = scene.reference_channel
    mics = tuple(scene.microphones[index] for index in order)
    total_squared = 0.0
    count = 0
    ambiguous_selections = 0
    for source, per_channel in zip(scene.sources, events):
        predicted = predicted_relative_delays(
            mics, [source.position], reference, scene.sound_speed
        )[0]
        choices: list[tuple[float, tuple[float, ...]]] = []
        for combination in itertools.product(*per_channel):
            times = tuple(item[0] for item in combination)
            observed = tuple(
                value - times[reference] for value in times
            )
            residuals = [
                (obs - pred) * scene.sample_rate
                for obs, pred in zip(observed, predicted)
            ]
            score = math.sqrt(
                sum(value * value for value in residuals) / len(residuals)
            )
            choices.append((score, observed))
        if not choices:
            return math.inf, 0
        best = min(score for score, _ in choices)
        tolerance = scene.analysis.map_tolerance_samples
        best_vectors = {
            tuple(round(value, 10) for value in observed)
            for score, observed in choices
            if score <= best + tolerance
        }
        ambiguous_selections += max(0, len(best_vectors) - 1)
        total_squared += best * best
        count += 1
    return math.sqrt(total_squared / max(count, 1)), ambiguous_selections


def audit_scene(
    scene: Scene,
    result_map: tuple[int, int, int, int] | None = None,
    name: str | None = None,
) -> LayoutAmbiguity:
    base_order = result_map or (0, 1, 2, 3)
    tolerance_s = scene.analysis.map_tolerance_samples / scene.sample_rate
    matrices = _delay_matrices(scene)
    base = matrices[base_order]
    equivalent = [
        order
        for order, matrix in matrices.items()
        if matrix_distance(base, matrix) <= tolerance_s
    ]
    reflection_orders: list[tuple[int, int, int, int]] = []
    reflection_selections = 0
    if scene.wall is not None:
        events = _reflection_events(scene)
        scores = {
            order: _selection_scores(scene, events, order)[0]
            for order in matrices
        }
        best_score = min(scores.values())
        reflection_orders = [
            order
            for order, score in scores.items()
            if score <= best_score + scene.analysis.map_tolerance_samples
        ]
        reflection_selections = max(
            _selection_scores(scene, events, order)[1]
            for order in reflection_orders
        )
    rank = delay_rank(
        scene.microphones, [source.position for source in scene.sources]
    )
    notes: list[str] = []
    if equivalent:
        if len(equivalent) > 1:
            notes.append(
                "多个通道排列给出相同的四通道延时签名，接线顺序不可唯一判断"
            )
        else:
            notes.append("在当前声源位置上，通道排列唯一")
    if len(reflection_orders) > 1:
        notes.append("选择较强反射也可组成另一组几何一致解")
    if reflection_selections:
        notes.append("同一布局存在多个直达/反射峰组合满足容差")
    if rank < min(3, len(scene.sources)):
        notes.append("声源方向张成的秩不足，镜像或离面位置不可辨识")
    return LayoutAmbiguity(
        name=name or scene.name,
        source_rank=rank,
        equivalent_orders=equivalent,
        reflection_orders=reflection_orders,
        reflection_selections=reflection_selections,
        notes=notes,
    )


def _scene_from_layout(
    name: str,
    microphones: list[list[float]],
    sources: list[list[float]],
    *,
    wall: dict[str, Any] | None = None,
) -> Scene:
    data = {
        "name": name,
        "sample_rate": 16000,
        "recording_seconds": 0.58,
        "microphones": microphones,
        "sources": [
            {"name": f"source_{index + 1}", "position": source}
            for index, source in enumerate(sources)
        ],
        "analysis": {
            "map_tolerance_samples": 0.12,
            "beam_frequencies": [1200, 1800, 2500, 3500, 4800],
        },
        "wall": wall,
    }
    return Scene.from_dict(data)


def catalog_layouts() -> list[LayoutAmbiguity]:
    square = [
        [-0.12, -0.12, 0.0],
        [0.12, -0.12, 0.0],
        [-0.12, 0.12, 0.0],
        [0.12, 0.12, 0.0],
    ]
    collinear = [
        [-0.18, 0.0, 0.0],
        [-0.06, 0.0, 0.0],
        [0.06, 0.0, 0.0],
        [0.18, 0.0, 0.0],
    ]
    asymmetric = [
        [-0.16, -0.06, 0.0],
        [0.16, -0.06, 0.0],
        [-0.14, 0.08, 0.0],
        [0.14, 0.08, 0.0],
    ]
    tetra = [
        [0.1, 0.1, 0.1],
        [0.1, -0.1, -0.1],
        [-0.1, 0.1, -0.1],
        [-0.1, -0.1, 0.1],
    ]
    wall = {
        "point": [0.0, 2.2, 0.3],
        "normal": [0.0, 1.0, 0.0],
        "reflection_amplitudes": [0.55, 0.7, 2.2, 0.6],
    }
    scenarios = [
        _scene_from_layout(
            "asymmetric-two-source",
            asymmetric,
            [[-0.9, 1.6, 0.35], [1.1, 1.5, 0.45]],
        ),
        _scene_from_layout(
            "square-source-on-axis",
            square,
            [[0.0, 2.0, 0.0]],
        ),
        _scene_from_layout(
            "square-centered-source",
            square,
            [[0.0, 0.0, 1.5]],
        ),
        _scene_from_layout(
            "collinear-broadside",
            collinear,
            [[0.0, 2.0, 0.0]],
        ),
        _scene_from_layout(
            "tetrahedron-centered-source",
            tetra,
            [[0.0, 0.0, 0.0]],
        ),
        _scene_from_layout(
            "coplanar-array-and-sources",
            square,
            [[-0.8, 1.5, 0.0], [0.8, 1.5, 0.0]],
        ),
        _scene_from_layout(
            "asymmetric-strong-reflection",
            asymmetric,
            [[1.1, 1.5, 0.45]],
            wall=wall,
        ),
    ]
    return [audit_scene(scene) for scene in scenarios]


def catalog_dicts() -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "source_rank": item.source_rank,
            "mapping_ambiguous": item.mapping_ambiguous,
            "reflection_ambiguous": item.reflection_ambiguous,
            "equivalent_order_count": len(item.equivalent_orders),
            "reflection_order_count": len(item.reflection_orders),
            "reflection_selections": item.reflection_selections,
            "notes": item.notes,
        }
        for item in catalog_layouts()
    ]
