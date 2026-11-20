# 滚动快门标定工具（LED 闪烁反演）

从已知频率 LED 的闪烁录影，估计车载滚动快门相机的：

- **逐行读出时间（line time, `t_line`）**：相邻两行曝光起始时刻之差；
- **逐行曝光时间（exposure, `t_exp`）**：单行积分窗口长度；
- **扫描方向（scan direction）**：自上而下 `+1` 或自下而上 `-1`；
- **传感器旋转角**：条纹相对图像行的倾角并自动校正。

同时内置合成视频生成器，可调节噪声、旋转、扫描方向，并把估计值与
生成器真值逐帧对照，输出残差图与误差报告。

## 物理原理

LED 以频率 `f_led`、占空比 `duty` 驱动方波，周期 `T = 1/f_led`。滚动快门下
第 `r` 行的曝光起始时刻为

```
t0(r) = frame_time - s * r * t_line          # s = +1 自上而下, -1 自下而上
```

该行对亮度在窗口 `[t0, t0 + t_exp]` 内积分。把方波与曝光窗都写成傅里叶级数后，
第 `k` 次谐波的观测幅度为

```
A_k = G * |c_led(k, duty)| * |sinc(k * t_exp / T)|
```

于是：

- 条纹的**基波空间频率** `f0 = f_led * t_line`（单位：周期/行）直接给出 `t_line`；
- 谐波幅度随 `|sinc(k t_exp / T)|` 衰减，拟合高次谐波比给出 `t_exp`；
- 已知相位基准（genlock，LED 在帧触发处相位为 0）时，基波空间频率的**符号**
  给出扫描方向。

### 何时解不出来

工具不会在病态情形下硬给数字，而是把帧标记为不可解并写入原因：

- **曝光覆盖整数个闪烁周期**（`t_exp/T` 为整数）：`sinc` 严格为零，
  携带信息的谐波消失在噪声中（如 1 kHz LED、1000 µs 曝光）；
- **接近 sinc 零点的长曝光**：可用谐波的零区余量（null margin）低于门限；
- **条纹周期数太少**（默认每帧 < 1.5 个周期）或条纹相干性太低；
- **基波信噪比不足**（强噪声 / 无闪烁）；
- **sinc 波瓣简并**：谐波幅度比在 `t_exp` 与其更长的整数倍波瓣上相同，
  自由亮度尺度无法区分；提供标定好的亮暗电平跨度 `brightness_span` 可打破简并；
- **无 genlock 的自由拍摄**：扫描方向存在全局相位翻转歧义，
  报告 `direction_ambiguous=true`，但 `t_line/t_exp` 仍可估计。

## 安装

```bash
pip install -r requirements.txt
```

## 命令行用法

生成合成视频（真值写到同名 `.truth.json`）：

```bash
python -m rolling_shutter_calib.cli synthesize \
  --out artifacts/demo.avi --frames 12 --rotation-deg 12 \
  --scan-direction 1 --t-line-us 20 --t-exp-us 120 --genlock
```

标定并与真值对照（输出到 `--out-dir`）：

```bash
python -m rolling_shutter_calib.cli calibrate artifacts/demo.avi \
  --f-led 1000 --genlock \
  --truth artifacts/demo.truth.json --out-dir artifacts/demo_out
```

长曝光（多周期）需提供亮暗电平跨度以消除波瓣简并：

```bash
python -m rolling_shutter_calib.cli calibrate clip.avi --f-led 1000 \
  --genlock --brightness-span 212 --out-dir out
```

## 输出产物

- `estimate.json`：逐帧与整段估计、谐波 SNR、可解性原因/警告、候选行时间；
- `residual_frame_XX.png`：观测 / 拟合模型 / 带符号残差三联图；
- `profile_frame_XX.png`：行剖面实测 vs 拟合曲线；
- `truth_comparison.png` 与 `truth_report.json`：真值对照、百分比误差与方向判定。

## 代码结构

| 模块 | 职责 |
| --- | --- |
| `model.py` | LED 方波、曝光积分、傅里叶谐波与 sinc 传递函数 |
| `synth.py` | 可调噪声/旋转/方向的合成视频生成与真值导出 |
| `geometry.py` | 结构张量测角、旋转校正、行剖面、有符号空间频率解调 |
| `estimator.py` | 谐波拟合、行时间混叠枚举、曝光波瓣选择与零区判定 |
| `calibrate.py` | 逐帧流水线、可解性检查、多帧聚合、扫描方向解析 |
| `visualize.py` | 残差图、剖面对比、真值对照（仅依赖 OpenCV） |
| `cli.py` | `synthesize` / `calibrate` 命令行入口 |

## 测试

```bash
python -m pytest rolling_shutter_calib/tests -q
```

覆盖物理模型边界、合成确定性、几何测角、有符号频率方向、端到端精度以及
整周期/近零区/强噪声等不可解场景。
