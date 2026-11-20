# Transparent Proxy Toolkit (alpha_proxy)

合成师用的透明视频代理生成与检查脚本。显式选择 **straight** /
**premultiplied** 解释，输出带 alpha 的代理与棋盘格预览，并能通过
黑/白双底合成对比自动判定代理实际的解释方式、报告错误声明。

依赖：`ffmpeg`/`ffprobe`（含 `prores_ks`、`libx264`）、Python 3 + numpy。

## 脚本

- `make_masters.py` — 生成三个 straight-alpha 测试母版（PNG 序列）：
  - `gradient`：半透明渐变（alpha 0→1 连续过渡，边缘误差敏感）
  - `edge_glow`：暗底上的细发光边缘（低 alpha 细线，最易暴露解释错误）
  - `transparent_color`：全透明但带颜色的像素（检验 straight 下隐藏 RGB
    是否保留；premultiplied 下必然被乘成 0）
- `make_proxy.py` — 从母版生成代理：
  - `--interpretation straight|premultiplied` 显式选择编码解释
    （premultiplied 会在编码前把 RGB 乘以 alpha）
  - `--codec prores|qtrle|png`（默认 ProRes 4444，`.mov` 带 alpha）
  - 同时输出 `<name>.checker.mp4` 棋盘格预览（从**编码后的**代理解码
    再按声明解释合成，所见即所得）和 `<name>.json` 元数据
- `check_proxy.py` — 检查代理：
  - 解码代理，在**黑色和白色**两种底色上分别按 straight / premultiplied
    合成，与母版的真值合成比较，统计边缘带（半透明像素及其邻域）的
    MAE / 最大值
  - 自动判定代理实际解释方式；与声明（sidecar json 或 `--declared`）
    不一致时报告 `MISMATCH` 并以非零码退出
  - alpha 为二值（无半透明边缘）时合成无法区分两种解释，自动改用
    「全透明区域隐藏 RGB」判定（report 中 `detection_method: hidden_rgb`）
  - `--report-dir` 输出 `report.json` 与两种解释的边缘误差热图
    `edge_error_{straight,premultiplied}.png`（误差 ×4 放大）

## 用法

```bash
cd alpha_proxy
python3 make_masters.py                                  # 生成母版
python3 make_proxy.py --master out/masters/gradient \
    --interpretation straight --out out/proxies/gradient_straight.mov
python3 check_proxy.py --proxy out/proxies/gradient_straight.mov \
    --master out/masters/gradient --report-dir out/reports/gradient_straight
```

一键完整演示（含故意错误声明的负例测试）：

```bash
bash run_demo.sh
```

## 判定逻辑

真值合成固定为「母版按 straight 合成」：`out = rgb*a + bg*(1-a)`。
代理若按某解释合成后在黑、白底上都逼近真值，则该解释正确：

- premultiplied 数据被当作 straight：边缘被二次乘 alpha，黑底白底都偏暗
- straight 数据被当作 premultiplied：边缘发亮、出现亮边光晕
- 两种解释误差接近且 alpha 二值化时，看全透明区的隐藏 RGB：
  保留 → straight；被清零 → premultiplied
