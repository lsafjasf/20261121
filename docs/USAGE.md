# 本地 OTIO/FFmpeg 渲染入口

`local_render` 把 OpenTimelineIO 时间线翻译为 FFmpeg 过滤器图，并在本地完成
渲染。它面向剪辑组的常见短工程场景：两条视频轨（V1/V2）及其音频轨
（A1/A2）、负源起点产生的显式黑场、叠化所需的真实媒体 handle。

## 能力与约束

- 两条视频轨通过 `overlay=eof_action=pass` 合成，短轨结束后自动露出底轨。
- 负源起点（`source_range.start_time < 0`）渲染为显式黑场（视频）与静音
  （音频），而不是冻结首帧或重复媒体。
- SMPTE Dissolve 转场要求真实媒体 handle；handle 不足时抛出
  `MissingHandleError`，`validate` 与 `render` 均以退出码 2 失败并逐条列出
  缺失的 head/tail handle，绝不静默用黑场顶替。
- 混合帧率素材先统一到时间线帧率，再按时间线整帧 trim，避免半帧取整误差。
- 每条轨道先输出无损中间片（视频 FFV1/Matroska，音频 PCM/WAV），再合成
  H.264/AAC 成片，便于逐轨核对。

## 命令

```bash
# 1. 生成合成素材（24 fps red/blue，30 fps green，均为 FFV1 + PCM）
.venv/bin/python scripts/make_media.py work/media

# 2. 生成 OTIO 工程与缺 handle 反例工程
.venv/bin/python scripts/make_timeline.py work/project work/media

# 3. 仅校验 handle
.venv/bin/python -m local_render validate work/project/demo.otio

# 4. 渲染（中间片与贡献清单写入 --workdir）
.venv/bin/python -m local_render render work/project/demo.otio \
  work/out/demo.mp4 --workdir work/render

# 5. 像素、音频交叉淡化与清单交叉核对
.venv/bin/python scripts/verify.py work/out/demo.mp4 \
  work/render/track_V1.mkv work/render/track_A1.wav \
  work/render/contrib.json work/verify
```

## 渲染模型

- 视频叠化：参与转场的两条流被补齐为等长（出流接真实 tail handle，入流接
  真实 head handle 与 tail handle，不足才补黑），`xfade` 在
  `cut - duration` 处开始线性叠化，输出经 trim 精确回到可见时间线帧数。
- 音频交叉淡化：每条 clip 前置静音对齐到时间线位置，出流在重叠窗口
  `afade=t=out`、入流 `afade=t=in`，再用 `amix` 混合，淡化中心精确落在
  时间线切点上。
- 贡献清单 `contrib.json` 记录每个源的帧率、黑场长度、head/tail handle、
  分段贡献摘要与逐帧权重；`contrib.md` 是其可读版本。
