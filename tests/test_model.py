from __future__ import annotations

import unittest

from color_audit.model import (
    LayerTags,
    TagValue,
    find_conflicts,
    parse_decisions,
    parse_tag_code,
    resolve_tags,
    tags_from_codes,
)


class ModelTests(unittest.TestCase):
    def test_codes_map_to_canonical_names(self) -> None:
        self.assertEqual(TagValue.from_code("primaries", 9).name, "bt2020")
        self.assertEqual(TagValue.from_code("transfer", 16).name, "smpte2084")
        self.assertEqual(TagValue.from_code("matrix", 9).name, "bt2020nc")
        self.assertEqual(TagValue.from_code("range", 1).name, "full")

    def test_parse_tag_aliases(self) -> None:
        self.assertEqual(parse_tag_code("range", "tv"), 0)
        self.assertEqual(parse_tag_code("range", "pc"), 1)
        self.assertEqual(parse_tag_code("transfer", "pq"), 16)
        self.assertEqual(parse_tag_code("matrix", "bt2020-ncl"), 9)

    def test_conflicts_require_a_decision(self) -> None:
        container = LayerTags(
            "container",
            tags_from_codes({"primaries": 9, "transfer": 16, "matrix": 9, "range": 1}),
        )
        codec = LayerTags(
            "codec",
            tags_from_codes({"primaries": 1, "transfer": 1, "matrix": 1, "range": 0}),
        )
        self.assertEqual(len(find_conflicts(container, codec)), 4)
        _, errors = resolve_tags(container, codec, {})
        self.assertEqual(len(errors), 4)
        resolved, errors = resolve_tags(
            container,
            codec,
            {"primaries": "codec", "transfer": "codec", "matrix": "codec", "range": "codec"},
        )
        self.assertEqual(errors, [])
        self.assertEqual(resolved["range"].value.name, "limited")

    def test_parse_decisions_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValueError):
            parse_decisions(["dither=codec"])


if __name__ == "__main__":
    unittest.main()

