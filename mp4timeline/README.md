# MP4 片段时序调查工具

把「多次录制暂停后导出的一条连续时间线」拆解成三层时序视图，并生成可辨认
画面、解码/展示顺序对照以及供第三方比对的表格与逐帧小样。纯 Python 标准库
解析 ISO-BMFF 盒子，画面处理全部交给 ffmpeg（无需 Pillow）。

## 三层时序模型

1. **Sample duration（`stts`）**：每个采样的解码时长与 DTS，决定解码节奏。
2. **Composition offset（`ctts`）**：`PTS = DTS + offset`。B 帧使该偏移非零，
   于是解码顺序（DTS）与展示顺序（PTS）不一致——这正是对照图要标出的差异。
3. **Edit list（`elst`）**：把媒体时间映射到电影时间。空编辑（`media_time=-1`）
   表示预卷/暂停间隙；普通编辑的 `media_time` 表示从媒体的哪一刻开始播放，
   可用于跳过编码器预卷、制造音频晚起或录制暂停的空档。

解析器同时读取 `stss`（关键帧）、`stsz`（大小）、`stsc/stco/co64`（字节偏移），
支持 v0/v1 的 `ctts`、`elst`、`mdhd`、`tkhd`、`mvhd`。

## 参考夹具构造

`make_fixture` 生成 320x240@25fps、含 3 个连续 B 帧的 H.264 视频，三段可辨认
画面（红/绿/蓝标题带 + 段名 + 帧号 + 时间戳），再用盒子改写器注入：

- 700 ms 空编辑预卷（画面在电影时间 0.7s 才出现）；
- 两段各 400 ms 的暂停间隙（三段画面起点 0.7s / 2.1s / 3.5s）；
- 音频在 1.3s 才起（预卷 0.7s + 额外晚起 0.6s），并保留 AAC 原生预卷；
- B 帧重排，前几个展示采样为 `[0, 3, 2, 4, 1, 7]`。

## 用法

一键完整流程（生成夹具 → 调查 → 自检）：

```bash
cd mp4timeline
./run_demo.sh                # 默认输出到 output/
./run_demo.sh other_out      # 自定义输出目录
```

单独使用各模块：

```bash
# 只分析任意 MP4（不限于自带夹具）
python3 -m tools.investigate path/to/movie.mp4 out_dir

# 只生成夹具
python3 -m tools.make_fixture out_dir

# 自检（与 ffprobe 的 PTS/DTS 交叉验证）
python3 -m tools.selftest out_dir/fixture.mp4
```

## 产物

```
out_dir/investigation/
├── tables/
│   ├── <track>_layer12_samples.csv   # 第一层+第二层：逐采样 DTS/PTS/offset/duration
│   ├── <track>_layer3_edits.csv      # 第三层：逐条 edit list
│   ├── <track>_presentation.csv      # 套用 edit list 后的电影时间落点
│   ├── summary.json                  # 机器可读汇总（含展示顺序、重排标志）
│   └── report.md                     # 人读报告
├── contact_decode_order.jpg          # 解码顺序：DEC# -> DISP#、PTS、offset、KEY
├── contact_display_order.jpg         # 展示顺序：DISP# <- DEC#、电影时间戳
└── frames_labeled/frame_*.jpg        # 逐帧带标注小样（展示顺序）
```

解码顺序抽帧使用 `ffmpeg -ignore_editlist 1 -vsync 0`，确保每个编码采样对应
一张原始解码帧，而不是被空编辑补齐后的重复帧。

## 模块

- `tools/mp4box.py`：依赖无关的盒子扫描器（支持 32/64 位 size、容器递归）。
- `tools/timing.py`：三层时序模型与 edit list 套用逻辑。
- `tools/boxpatcher.py`：递归重写 `moov`，替换 edit list 与时长，不动 `mdat`。
- `tools/report.py`：CSV / JSON / Markdown 导出。
- `tools/frames.py`：ffmpeg 抽帧、ASS 标注、tile 对照表与逐帧小样。
- `tools/ass.py`：逐帧 ASS 字幕生成（本环境 ffmpeg 无 drawtext，改用 libass）。
- `tools/selftest.py`：断言式回归，并与 ffprobe 交叉验证。
