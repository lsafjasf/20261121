import { runDelayedSwitchRace, summarizeRace } from "./race-scenario.ts";

const comparison = runDelayedSwitchRace();

console.log("可控延时换源回放：A 的 12.000s 帧延迟 300ms，在切到 B 并开始新区间后到达。\n");
for (const line of summarizeRace(comparison)) {
  console.log(line);
}
console.log("\n修复版按帧票据拒绝跨源/跨跳转代的迟到回调；旧逻辑会把它解释成当前源的呈现位置。");
