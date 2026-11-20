# TTML -> IMSC migration report

## `01_cascade.ttml`

Migrated document: `out/01_cascade.imsc.ttml`

No degradations: every effect maps to the IMSC subset.

### Rendered-frame verification

- frame diff (source vs migrated): **none**
- overlap incident groups: **0**
- fractional in-points: **0/0 passed**
- flash-frame risks: **none**

Key frames: `out/verify/01_cascade/frames`

## `02_timecontainers.ttml`

Migrated document: `out/02_timecontainers.imsc.ttml`

### Degradations (9)

| Severity | Element | Attribute | Value | Reason | Suggestion |
|---|---|---|---|---|---|
| info | `/tt/body` | `timeContainer` | `seq` | IMSC does not allow seq time containers | flattened to absolute par timing on each <p> |
| warning | `/tt/body/div[1]/p[2]` | `begin/end` | `3.001500s` | time is not representable exactly at the document frame rate | quantise cue times to whole frames or milliseconds |
| warning | `/tt/body/div[1]/p[3]` | `begin/end` | `3.351850s` | time is not representable exactly at the document frame rate | quantise cue times to whole frames or milliseconds |
| warning | `/tt/body/div[1]/p[3]` | `begin/end` | `5.351850s` | time is not representable exactly at the document frame rate | quantise cue times to whole frames or milliseconds |
| warning | `/tt/body/div[1]/div[1]/p[1]` | `begin/end` | `5.351850s` | time is not representable exactly at the document frame rate | quantise cue times to whole frames or milliseconds |
| warning | `/tt/body/div[1]/div[1]/p[1]` | `begin/end` | `6.351850s` | time is not representable exactly at the document frame rate | quantise cue times to whole frames or milliseconds |
| warning | `/tt/body/div[1]/div[1]/p[2]` | `begin/end` | `5.360192s` | time is not representable exactly at the document frame rate | quantise cue times to whole frames or milliseconds |
| warning | `/tt/body/div[1]/div[1]/p[2]` | `begin/end` | `6.360192s` | time is not representable exactly at the document frame rate | quantise cue times to whole frames or milliseconds |
| warning | `/tt/body/div[2]/p[2]/span[1]` | `begin/end` | `begin='1s', end='1.5s'` | timing on <span> is not permitted in IMSC | split the text into separate <p> elements carrying the timing, or move the timing up to the parent <p> |

### Rendered-frame verification

- frame diff (source vs migrated): **none**
- overlap incident groups: **2**
- fractional in-points: **6/6 passed**
- flash-frame risks: **none**

Key frames: `out/verify/02_timecontainers/frames`

## `03_unsupported.ttml`

Migrated document: `out/03_unsupported.imsc.ttml`

### Degradations (8)

| Severity | Element | Attribute | Value | Reason | Suggestion |
|---|---|---|---|---|---|
| warning | `/tt/head/styling/style[1]` | `tts:fontSize` | `50% 80%` | two-component fontSize (width height) is not in IMSC | kept the height component (80%); use a single length |
| warning | `/tt/head/styling/style[2]` | `tts:textShadow` | `2px 2px 3px black` | tts:textShadow is not in the IMSC text profile | approximate with tts:textOutline (supported in IMSC 1.1), e.g. a small dark outline, or pre-render the effect |
| info | `/tt/head/styling/style[3]` | `tts:writingMode` | `tb` | TTML1 writing-mode alias is not in IMSC | rewritten to canonical 'tbrl' |
| warning | `/tt/body/div[1]/p[1]/set[1]` | `set` | `1s` | discrete animation <set> is not in the IMSC text profile | split the caption into consecutive timed <p> elements that change tts:visibility at the same offsets |
| warning | `/tt/body/div[1]/p[1]/set[2]` | `set` | `3s` | discrete animation <set> is not in the IMSC text profile | split the caption into consecutive timed <p> elements that change tts:color at the same offsets |
| warning | `/tt/body/div[1]/p[4]` | `ebutts:multiRowAlign` | `center` | ebu multiRowAlign is not in IMSC | mapped to tts:textAlign='center' (per-line alignment of ragged lines is lost) |
| warning | `/tt/body/div[1]/p[5]` | `smpte:backgroundImage` | `sign.png` | SMPTE-TT image/attribute is not in the IMSC text profile | use the IMSC image profile for background images, or replace the image with text |
| info | `/tt/body/div[1]/p[7]/metadata[1]` | `metadata` | `rating` | foreign metadata vocabulary is not in IMSC | keep it in the sidecar workflow or convert to ttm:desc |

### Rendered-frame verification

- frame diff (source vs migrated): **none**
- overlap incident groups: **0**
- fractional in-points: **0/0 passed**
- flash-frame risks: **none**

Key frames: `out/verify/03_unsupported/frames`

## `04_overlap.ttml`

Migrated document: `out/04_overlap.imsc.ttml`

### Degradations (1)

| Severity | Element | Attribute | Value | Reason | Suggestion |
|---|---|---|---|---|---|
| warning | `/tt/body/div[1]/p[4]` | `begin/end` | `5.033300s` | time is not representable exactly at the document frame rate | quantise cue times to whole frames or milliseconds |

### Rendered-frame verification

- frame diff (source vs migrated): **none**
- overlap incident groups: **1**
- fractional in-points: **0/0 passed**
- flash-frame risks: **o4@150**

Errors:
- cross-region overlap: cues o1 and o2 intersect (225 px peak) on frames 30-89

Key frames: `out/verify/04_overlap/frames`

