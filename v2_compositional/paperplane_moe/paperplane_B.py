# -*- coding: utf-8 -*-
"""paperplane-Moe 别名模块：B 档主干（历史逐位对照的地基）。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V2 = os.path.join(_ROOT, "v2_compositional")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

from snowflake_B import *                      # noqa: F401,F403
from snowflake_B import SnowflakeB              # noqa: F401

PaperplaneB = SnowflakeB

__all__ = ["PaperplaneB", "SnowflakeB"]
