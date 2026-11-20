"""hipreview - 本地助听轨审片工具包。

模块划分：
  synth    生成带语言代号与可辨识测试词的替代音 WAV
  audioio  ffmpeg 解码/重采样与波形缩略
  analyzer 主声轨/助听轨对齐、重复与语言标记核查
  report   审片结果 JSON/文本报告与波形片段导出
  app      本地 Web 审片界面（一键切换监听、保留播放位置）
"""

__version__ = "1.0.0"
