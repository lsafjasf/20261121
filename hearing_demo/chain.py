"""多频段压缩处理链：分频 -> 分段包络/静态增益/限幅 -> 重组 -> 输出限幅。"""

from __future__ import annotations

import math

from .envelope import EnvelopeFollower, static_gain_db, db_to_gain, gain_to_db
from .filters import Crossover4


class BandCompressor:
    __slots__ = ("index", "params", "env", "last_gain_db", "last_env_db")

    def __init__(self, index: int, params: dict, fs: int):
        self.index = index
        self.params = params
        self.env = EnvelopeFollower(params["attack_ms"], params["release_ms"], fs)
        self.last_gain_db = 0.0
        self.last_env_db = -120.0

    def reset(self):
        self.env.reset()
        self.last_gain_db = 0.0
        self.last_env_db = -120.0

    def process(self, x):
        p = self.params
        env = self.env.process(x)
        env_db = gain_to_db(env)
        self.last_env_db = env_db
        gr = static_gain_db(env_db, p["knee_dbfs"], p["ratio"])
        makeup = p["insertion_gain_db"]
        # 压缩/增益级（慢，包络驱动）
        gain_db = makeup + gr
        # 瞬时峰值限幅：保证该段每样本输出不超过 ceiling（dBFS）
        inst_db = gain_to_db(abs(x))
        peak_limit = min(0.0, p["ceiling_dbfs"] - (inst_db + gain_db))
        total_db = gain_db + peak_limit
        self.last_gain_db = total_db
        return x * db_to_gain(total_db)


class ProcessingChain:
    """完整 4 频段链；bypass=True 时直通，旁路各段仍记录分频后电平。"""

    def __init__(self, band_params: list, fs: int,
                 crossovers=(500.0, 2000.0, 6000.0), bypass=False):
        self.fs = fs
        self.bypass = bypass
        self.crossover = Crossover4(*crossovers, fs)
        self.bands = [BandCompressor(i, band_params[i], fs) for i in range(4)]
        self.output_ceiling = -3.0
        self._out_env = EnvelopeFollower(1.0, 20.0, fs)

    def reset(self):
        self.crossover.reset()
        for b in self.bands:
            b.reset()
        self._out_env.reset()

    def process_sample(self, x):
        parts = self.crossover.split_sample(x)
        if self.bypass:
            processed = parts
        else:
            processed = tuple(self.bands[i].process(parts[i]) for i in range(4))
        y = sum(processed)
        if not self.bypass:
            # 重组后总保护：瞬时峰值超过 ceiling 时按比例衰减
            peak_db = gain_to_db(abs(y))
            if peak_db > self.output_ceiling:
                y *= db_to_gain(self.output_ceiling - peak_db)
        return y, parts, processed

    def process_block(self, samples):
        """返回 (输出 list, 每段输入 list[4], 每段输出 list[4])。"""
        self.reset()
        out = []
        band_in = [[] for _ in range(4)]
        band_out = [[] for _ in range(4)]
        for x in samples:
            y, parts, processed = self.process_sample(x)
            out.append(y)
            for i in range(4):
                band_in[i].append(parts[i])
                band_out[i].append(processed[i])
        return out, band_in, band_out

