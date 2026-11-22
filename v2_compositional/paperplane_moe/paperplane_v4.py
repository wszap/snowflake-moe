# -*- coding: utf-8 -*-
"""paperplane-Moe 别名模块：旧 snowflake_v4 保持不动，新名字对外统一。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V2 = os.path.join(_ROOT, "v2_compositional")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

from snowflake_v4 import *                      # noqa: F401,F403
from snowflake_v4 import SnowflakeLMV4, SnowflakeB  # noqa: F401

PaperplaneLMV4 = SnowflakeLMV4                  # 新名
PaperplaneB = SnowflakeB

__all__ = ["PaperplaneLMV4", "PaperplaneB", "SnowflakeLMV4", "SnowflakeB"]
