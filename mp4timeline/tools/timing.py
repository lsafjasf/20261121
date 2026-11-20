"""Three-layer timing model for an ISO-BMFF movie.

Layer 1 - sample duration (stts): every sample's decode duration.
Layer 2 - composition offset (ctts): DTS->PTS shift, exposing the
          decode-order vs display-order reordering caused by B-frames.
Layer 3 - edit list (elst): how media time is mapped onto movie time
          (pre-roll empty edits, delayed-start edits, recording gaps).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import mp4box


@dataclass
class Sample:
    index: int                 # decode order, 0 based
    dts: int                   # in media timescale units
    duration: int              # stts delta, media timescale units
    composition_offset: int    # signed ctts delta (v1 may be negative)
    pts: int                   # dts + composition_offset, media timescale units
    is_keyframe: bool
    size: int
    data_offset: int           # absolute file offset of sample bytes


@dataclass
class Edit:
    index: int
    segment_duration: int      # movie timescale units
    media_time: int            # media timescale units; -1 = empty edit
    media_rate: float

    @property
    def is_empty(self) -> bool:
        return self.media_time == -1


@dataclass
class Track:
    track_id: int
    handler: str               # 'vide' / 'soun' / ...
    media_timescale: int
    media_duration: int
    movie_timescale: int
    width: int = 0
    height: int = 0
    samples: list[Sample] = field(default_factory=list)
    edits: list[Edit] = field(default_factory=list)
    display_order: list[int] = field(default_factory=list)

    @property
    def is_video(self) -> bool:
        return self.handler == "vide"

    def to_seconds(self, media_units: float) -> float:
        return media_units / self.media_timescale

    def to_movie_seconds(self, movie_units: float) -> float:
        return movie_units / self.movie_timescale


def _expand_run_length(pairs: list[tuple[int, int]]) -> list[int]:
    out: list[int] = []
    for count, value in pairs:
        out.extend([value] * count)
    return out


def _read_stts(box, data) -> list[tuple[int, int]]:
    p = box.payload_start + 4  # version/flags
    count = mp4box.u32(data, p)
    p += 4
    return [(mp4box.u32(data, p + 8 * i), mp4box.u32(data, p + 8 * i + 4))
            for i in range(count)]


def _read_ctts(box, data) -> list[int]:
    if box is None:
        return []
    version, _ = box.fullbox(data)
    p = box.payload_start + 4
    count = mp4box.u32(data, p)
    p += 4
    offsets: list[int] = []
    for i in range(count):
        run = mp4box.u32(data, p + 8 * i)
        if version == 1:
            value = mp4box.i32(data, p + 8 * i + 4)
        else:
            value = mp4box.u32(data, p + 8 * i + 4)
        offsets.extend([value] * run)
    return offsets


def _read_stss(box, data) -> set[int]:
    if box is None:
        return set()
    p = box.payload_start + 4
    count = mp4box.u32(data, p)
    p += 4
    return {mp4box.u32(data, p + 4 * i) - 1 for i in range(count)}


def _read_stsz(box, data) -> list[int]:
    p = box.payload_start + 4
    sample_size = mp4box.u32(data, p)
    count = mp4box.u32(data, p + 4)
    p += 8
    if sample_size != 0:
        return [sample_size] * count
    return [mp4box.u32(data, p + 4 * i) for i in range(count)]


def _read_chunk_offsets(stbl, data) -> tuple[list[int], bool]:
    co64 = stbl.find("co64")
    if co64 is not None:
        p = co64.payload_start + 4
        count = mp4box.u32(data, p)
        p += 4
        return [mp4box.u64(data, p + 8 * i) for i in range(count)], True
    stco = stbl.find("stco")
    p = stco.payload_start + 4
    count = mp4box.u32(data, p)
    p += 4
    return [mp4box.u32(data, p + 4 * i) for i in range(count)], False


def _read_stsc(box, data) -> list[tuple[int, int, int]]:
    p = box.payload_start + 4
    count = mp4box.u32(data, p)
    p += 4
    return [(mp4box.u32(data, p + 12 * i),
             mp4box.u32(data, p + 12 * i + 4),
             mp4box.u32(data, p + 12 * i + 8)) for i in range(count)]


def _sample_data_offsets(stbl, data, sample_count: int) -> list[int]:
    """Resolve each sample's absolute byte offset using stsc/stco/stsz."""
    chunk_offsets, _ = _read_chunk_offsets(stbl, data)
    stsc = _read_stsc(stbl.find("stsc"), data)
    sizes = _read_stsz(stbl.find("stsz"), data)

    samples_per_chunk: list[int] = []
    entry = 0
    for chunk_number in range(1, len(chunk_offsets) + 1):
        while entry + 1 < len(stsc) and stsc[entry + 1][0] <= chunk_number:
            entry += 1
        samples_per_chunk.append(stsc[entry][1])

    offsets: list[int] = []
    sample_cursor = 0
    for chunk_index, chunk_offset in enumerate(chunk_offsets):
        cursor = chunk_offset
        for _ in range(samples_per_chunk[chunk_index]):
            offsets.append(cursor)
            cursor += sizes[sample_cursor]
            sample_cursor += 1
            if sample_cursor >= sample_count:
                return offsets
    return offsets


def _read_elst(edts, data, movie_timescale: int) -> list[Edit]:
    if edts is None:
        return []
    elst = edts.find("elst")
    if elst is None:
        return []
    version, _ = elst.fullbox(data)
    p = elst.payload_start + 4
    count = mp4box.u32(data, p)
    p += 4
    edits: list[Edit] = []
    for i in range(count):
        if version == 1:
            segment = mp4box.u64(data, p)
            media_time = mp4box.u64(data, p + 8)
            rate_int = mp4box.u16(data, p + 16)
            rate_frac = mp4box.u16(data, p + 18)
            p += 20
        else:
            segment = mp4box.u32(data, p)
            media_time = mp4box.i32(data, p + 4)
            rate_int = mp4box.u16(data, p + 8)
            rate_frac = mp4box.u16(data, p + 10)
            p += 12
        edits.append(Edit(i, segment, media_time,
                          rate_int + rate_frac / 65536.0))
    return edits


def _read_mdhd(mdhd, data) -> tuple[int, int]:
    version, _ = mdhd.fullbox(data)
    p = mdhd.payload_start + 4
    if version == 1:
        p += 8  # creation/modification
        timescale = mp4box.u32(data, p)
        duration = mp4box.u64(data, p + 4)
    else:
        p += 8
        timescale = mp4box.u32(data, p)
        duration = mp4box.u32(data, p + 4)
    return timescale, duration


def _read_tkhd_id(tkhd, data) -> int:
    version, _ = tkhd.fullbox(data)
    p = tkhd.payload_start + 4
    if version == 1:
        p += 16  # creation_time + modification_time (64 bit each)
    else:
        p += 8   # creation_time + modification_time (32 bit each)
    return mp4box.u32(data, p)


def _read_tkhd_dimensions(tkhd, data) -> tuple[int, int]:
    # width/height are the last two 16.16 fixed-point values in tkhd.
    width = mp4box.u32(data, tkhd.end - 8)
    height = mp4box.u32(data, tkhd.end - 4)
    return width >> 16, height >> 16


def parse_movie(data: bytes) -> tuple[int, list[Track], list]:
    roots = mp4box.scan_boxes(data, 0, len(data))
    moov = next(b for b in roots if b.type == "moov")
    mvhd = moov.find("mvhd")
    movie_version, _ = mvhd.fullbox(data)
    p = mvhd.payload_start + 4
    if movie_version == 1:
        p += 16
        movie_timescale = mp4box.u32(data, p)
    else:
        p += 8
        movie_timescale = mp4box.u32(data, p)

    tracks: list[Track] = []
    for trak in [c for c in moov.children if c.type == "trak"]:
        mdia = trak.find("mdia")
        mdhd = mdia.find("mdhd")
        hdlr = mdia.find("hdlr")
        handler = data[hdlr.payload_start + 8:hdlr.payload_start + 12].decode("latin1")
        timescale, media_duration = _read_mdhd(mdhd, data)
        track_id = _read_tkhd_id(trak.find("tkhd"), data)
        stbl = mdia.find("minf", "stbl")

        durations = _expand_run_length(_read_stts(stbl.find("stts"), data))
        offsets = _read_ctts(stbl.find("ctts"), data)
        if not offsets:
            offsets = [0] * len(durations)
        keyframes = _read_stss(stbl.find("stss"), data)
        data_offsets = _sample_data_offsets(stbl, data, len(durations))
        sizes = _read_stsz(stbl.find("stsz"), data)
        edits = _read_elst(trak.find("edts"), data, movie_timescale)

        samples: list[Sample] = []
        dts = 0
        for i, duration in enumerate(durations):
            comp = offsets[i] if i < len(offsets) else 0
            samples.append(Sample(
                index=i,
                dts=dts,
                duration=duration,
                composition_offset=comp,
                pts=dts + comp,
                is_keyframe=(i in keyframes) if keyframes else True,
                size=sizes[i],
                data_offset=data_offsets[i],
            ))
            dts += duration

        track = Track(
            track_id=track_id,
            handler=handler,
            media_timescale=timescale,
            media_duration=media_duration,
            movie_timescale=movie_timescale,
            width=_read_tkhd_dimensions(trak.find("tkhd"), data)[0],
            height=_read_tkhd_dimensions(trak.find("tkhd"), data)[1],
            samples=samples,
            edits=edits,
        )
        track.display_order = sorted(range(len(samples)),
                                     key=lambda i: (samples[i].pts, i))
        tracks.append(track)
    return movie_timescale, tracks, roots


def presentation_map(track: Track) -> list[dict]:
    """Apply the edit list, yielding movie-time placement per displayed sample."""
    rows: list[dict] = []
    movie_cursor = 0
    for edit in track.edits:
        if edit.is_empty:
            movie_cursor += edit.segment_duration
            continue
        media_rate = edit.media_rate or 1.0
        segment_end = edit.media_time + int(
            edit.segment_duration * track.media_timescale
            / track.movie_timescale * media_rate
        )
        for display_pos, sample_index in enumerate(track.display_order):
            sample = track.samples[sample_index]
            if sample.pts < edit.media_time or sample.pts >= segment_end:
                continue
            relative_media = sample.pts - edit.media_time
            relative_movie = (
                relative_media * track.movie_timescale
                / track.media_timescale / media_rate
            )
            movie_pts = movie_cursor + relative_movie
            rows.append({
                "edit_index": edit.index,
                "display_position": display_pos,
                "sample_index": sample_index,
                "media_pts": sample.pts,
                "movie_pts": movie_pts,
                "movie_pts_seconds": movie_pts / track.movie_timescale,
            })
        movie_cursor += edit.segment_duration
    return rows
