"""把测量结果汇总成 Markdown 报告（无第三方模板依赖）。"""

from __future__ import annotations

from .audiogram import STD_FREQS, BAND_NAMES

STATUS_CN = {
    "measured": "实测",
    "interpolated": "插值",
    "clamped_low": "低端钳位",
    "clamped_high": "高端钳位",
}


def _table(headers, rows):
    line = "| " + " | ".join(headers) + " |\n"
    line += "| " + " | ".join("---" for _ in headers) + " |\n"
    for r in rows:
        line += "| " + " | ".join(str(c) for c in r) + " |\n"
    return line


def build_report(cfg, meas, outputs) -> str:
    lines = ["# 听力演示工具实测报告", ""]
    lines.append(f"- 工具版本: {cfg['tool_version']}，采样率: {cfg['sample_rate_hz']} Hz")
    lines.append(f"- 分频点: {cfg['crossover_hz']} Hz（4 频段一阶互补分频，全带平坦重组）")
    lines.append("- 用途: 设备原型检查；处方为简化线性规则，非临床验配")
    lines.append("")

    lines.append("## 1. 听阈测点与空白点插值")
    for side_cn, side in (("左耳", "left"), ("右耳", "right")):
        ear = cfg["ears"][side]
        lines.append(f"### {side_cn}")
        rows = []
        for f in STD_FREQS:
            st = ear["point_status"][str(f)]
            rows.append([f"{f} Hz", ear["thresholds_dbhl"][str(f)],
                         STATUS_CN.get(st, st)])
        lines.append(_table(["频率", "听阈 dB HL", "来源"], rows))
    lines.append("插值规则：标准点之间按对数频率轴线性插值；")
    lines.append("实测范围之外钳位到最近实测值，不做趋势外推。")
    lines.append("")

    lines.append("## 2. 压缩配置（每耳每段）")
    for side_cn, side in (("左耳", "left"), ("右耳", "right")):
        lines.append(f"### {side_cn}")
        rows = []
        for i, p in enumerate(cfg["ears"][side]["bands"]):
            rows.append([BAND_NAMES[i], p["hl_band_db"], p["insertion_gain_db"],
                         f'{p["ratio"]}:1', p["knee_dbfs"], p["ceiling_dbfs"],
                         f'{p["attack_ms"]}/{p["release_ms"]}'])
        lines.append(_table(
            ["频段", "段听阈", "增益dB", "压缩比", "软膝dBFS", "限顶dBFS", "攻/放 ms"],
            rows))
    lines.append("")

    lines.append("## 3. 交叉频带附近纯音实测")
    for side_cn, side in (("左耳", "left"), ("右耳", "right")):
        m = meas[side]
        lines.append(f"### {side_cn} / 压缩开启")
        rows = []
        for r in m["tone_active"]:
            rows.append([r["tone_hz"], r["input_dbfs"],
                         "/".join(str(v) for v in r["band_input_dbfs"]),
                         "/".join(str(v) for v in r["band_output_dbfs"]),
                         r["sum_output_dbfs"]])
        lines.append(_table(
            ["纯音Hz", "输入dBFS", "四段输入", "四段输出", "合成输出"], rows))
        rows = []
        for r in m["tone_bypass"]:
            rows.append([r["tone_hz"], r["input_dbfs"],
                         "/".join(str(v) for v in r["band_input_dbfs"]),
                         "/".join(str(v) for v in r["band_output_dbfs"]),
                         r["sum_output_dbfs"]])
        lines.append(f"### {side_cn} / 旁路（直通，输出=分频输入）")
        lines.append(_table(
            ["纯音Hz", "输入dBFS", "四段输入", "四段输出", "合成输出"], rows))
        lines.append(f"- 分频重组误差（全带 RMS 相对误差）: {m['recomb_error_db']} dB")
        lines.append("")

    lines.append("## 4. 快速声级跃迁实测（1 kHz，-30/-12/-8 dBFS 阶跃）")
    for side_cn, side in (("左耳", "left"), ("右耳", "right")):
        t = meas[side]["transition"]
        lines.append(f"### {side_cn}")
        lines.append(_table(["指标", "数值"], [
            ["包络稳态(低电平) dBFS", t["env_db_steady_low"]],
            ["包络稳态(高电平) dBFS", t["env_db_steady_high"]],
            ["增益(低电平) dB", t["gain_db_steady_low"]],
            ["增益(高电平) dB", t["gain_db_steady_high"]],
            ["攻击时间 实测/配置 ms",
             f"{t['attack_ms_measured']} / {t['attack_ms_config']}"],
            ["释放时间 实测/配置 ms",
             f"{t['release_ms_measured']} / {t['release_ms_config']}"],
        ]))
    lines.append("")

    lines.append("## 5. 旁路对比与分段输入输出电平（演示节目）")
    for side_cn, side in (("左耳", "left"), ("右耳", "right")):
        for mode in ("active", "bypass"):
            d = meas[side]["demo"][mode]
            lines.append(f"### {side_cn} / {('压缩' if mode == 'active' else '旁路')}")
            rows = [[BAND_NAMES[r["band"]], r["input_rms_dbfs"],
                     r["output_rms_dbfs"], r["input_peak_dbfs"],
                     r["output_peak_dbfs"]] for r in d["band_summary"]]
            lines.append(_table(
                ["频段", "输入RMS", "输出RMS", "输入峰值", "输出峰值"], rows))
        lines.append("")

    lines.append("## 6. 导出文件")
    for k, v in outputs.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## 7. 复现方式")
    lines.append("```")
    lines.append("python3 -m hearing_demo.cli all --audiogram data/sample_audiogram.json --out out")
    lines.append("```")
    return "\n".join(lines) + "\n"

