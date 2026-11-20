# riaa-eq — 档案室 RIAA 均衡渲染器

纯 Rust、零外部依赖的命令行工具，用于在两种信号表示之间做确定性的
RIAA 均衡渲染：

- `flat`：唱放平坦输出（未经 RIAA 处理的刻片前信号）
- `riaa`：已经做过 RIAA 播放均衡的唱片录音

操作员必须通过 `--from` / `--to` 显式声明输入与输出表示，工具据此选择
播放曲线、逆曲线或已均衡旁路。

## 三条处理路径

| `--from` | `--to` | 处理 | 滤波器 |
| --- | --- | --- | --- |
| `riaa` | `flat` | 播放（重放）曲线 | 一个双二阶节 |
| `flat` | `riaa` | 逆（录音）曲线 | 一个带保护极点的双二阶节 |
| 相同 | 相同 | 已均衡旁路 | 无状态，样本原样通过 |

## 数学模型

RIAA 标准时间常数（IEC 60098）：

- `tau1 = 3180 us`（约 50.05 Hz）
- `tau2 = 318 us`（约 500.49 Hz）
- `tau3 = 75 us`（约 2122.07 Hz）

播放原型传输函数（直流增益归一化为 1）：

```text
H_play(s) = (1 + s*tau2) / ((1 + s*tau1)(1 + s*tau3))
```

两个滤波器都用同一次双线性变换离散化：

```text
s = 2*fs*(z-1)/(z+1)
```

因此播放与逆曲线的数字级联在数学上等价于单个保护极点的双线性变换，
没有额外的实现失配。

### 为什么逆曲线需要保护极点

纯逆滤波器 `1/H_play(s)` 在 `tau2` 处的极点经双线性变换后恰好落在
单位圆 `z = -1`（奈奎斯特点），成为临界稳定滤波器，任何量化激励都会
产生永不衰减的奈奎斯特振荡。本工具在模拟域加入一个保护极点
`1/(1 + s/omega_guard)`，把所有极点严格拉进单位圆：

- 默认位置为采样率的 `0.45`（48 kHz 下约 21.6 kHz），高于音频通带
- 可用 `--guard <分数>` 调整，取值被钳制在 `(0, 0.5]`
- 保护极点对 4 kHz 以下的影响小于 0.2 dB；其高频滚降在分析报告中
  显式列出，属于稳定逆滤波的必要取舍

## 滤波状态约定

- 滤波器结构为直接 II 型转置（DF2T），状态为每通道两个延迟量 `s1/s2`
- 首状态：冷启动，恒为零状态（每次渲染新建滤波器链）
- 尾状态：输入最后一个样本处理完后、flush 之前的状态，渲染时逐通道打印
- `--flush`：从尾状态开始向整条滤波器链喂零帧，收集末级输出作为尾响
  追加到 WAV 末尾，直到每级状态绝对值低于 `1e-12`（上限 20 万帧）；
  不加 `--flush` 时信号在最后一个样本处直接截断，尾状态保持非零
- 多声道文件逐声道独立维护状态

## 超满刻度处理

内部全程使用 `f64`，滤波本身不做任何限幅。输出编码时：

- `float32`：保留超过 `|1.0|` 的值，只计数、不削波（建议用于中间产物）
- `pcm16/pcm24/pcm32`：四舍五入后硬饱和到编码极值，报告
  `over_full_scale`（超满刻度样本数）与 `clipped`（实际钳位数）

逆曲线在 1 kHz 附近有约 +19.7 dB 的提升，平坦满刻度输入必然超满刻度，
这是 RIAA 录音曲线的物理特性而非缺陷。

## 命令用法

```text
riaa-eq analyze  [--sample-rate HZ] [--guard FRACTION] [--json]
riaa-eq render   -i IN.wav -o OUT.wav --from flat|riaa --to flat|riaa
                 [--format pcm16|pcm24|pcm32|float32] [--guard FRACTION] [--flush]
riaa-eq transient -o OUT.wav [--kind impulse|tone] [--duration-ms MS]
                 [--channels N] [--amplitude A] [--tone-hz HZ] [--format FMT]
riaa-eq verify   [--sample-rate HZ] [--guard FRACTION] [--json]
```

### analyze — 可重复的系数分析

- 打印时间常数、三个转折频点、保护极点、完整双二阶节系数（17 位有效
  数字）与直流增益校验
- 参考频响表覆盖 20 Hz–20 kHz，并包含三个 RIAA 转折频点；同时给出
  数字频响、双线性变换映射后的模拟频点、模拟原型值，以及以 1 kHz 为
  0 dB 的行业惯例列
- 每个系数集附 FNV-1a 指纹：相同参数与常数必然产生相同指纹，可用于
  归档审计和跨环境复核

### transient — 短瞬态 WAV

- `impulse`：第 0 样本为单位脉冲、其余为零（默认 100 ms）
- `tone`：整段 Hann 窗门控的正弦音爆，无硬边界点击
- 默认输出 `float32` 以保留全部动态

### verify — 播放后再逆变的闭环验证

频响验证：

- 对播放、逆、级联分别在固定审计频点计算数字频响，并与模拟原型对照
- 判据为 4 kHz 核心通带内级联偏差小于 0.2 dB

时域验证：

- 1 秒确定性多音探测信号（30 Hz–4 kHz，共 7 个分量，首尾 8 ms Hann
  淡变），依次通过播放链与逆链
- 用单频 DFT 相关估计每个分量的稳态幅度（对 IIR 群延迟造成的样本
  错位不敏感），相对误差判据小于 2%
- 同时报告含群延迟的逐点 RMS，以及截断/flush 两种尾部策略的结果

## 交付物（`artifacts/`）

- `analysis.txt` / `analysis.json`：系数与参考频响分析
- `verification.txt` / `verification.json`：闭环验证报告
- `impulse_48k_float32.wav`：单位脉冲短瞬态
- `toneburst_1k_float32.wav` / `toneburst_1k_pcm16.wav`：1 kHz 音爆
- `render_playback_riaa-to-flat.wav`：播放曲线渲染（含 flush 尾响）
- `render_inverse_flat-to-riaa.wav`：逆曲线渲染（PCM16，含硬饱和计数）
- `render_bypass_riaa-to-riaa.wav`：已均衡旁路
- 三个 `render_*.log`：对应渲染的首尾状态与溢出报告

## 构建与测试

```text
cargo build --release
cargo test
cargo clippy --all-targets
```

支持的 WAV 格式：PCM 16/24/32 位整数与 IEEE 32 位浮点，单声道或多声道，
解析时跳过未知 chunk。工具不依赖任何第三方 crate。
