"""离线快门积分演示包：连续光源按快门角/开合曲线积分，支持 VFR 与线性光混合。"""

__version__ = "1.0.0"

from .shutter import Shutter, make_shutter
from .vfr import cfr_schedule, explicit_schedule, pulldown_2_1_schedule

__all__ = [
    "Shutter",
    "make_shutter",
    "cfr_schedule",
    "explicit_schedule",
    "pulldown_2_1_schedule",
    "__version__",
]
