# -*- coding: utf-8 -*-
"""paperplane-Moe 别名模块：V1.001 残差版真拼核心的对外入口。"""
from .compose import ResidualComposeCell, compose_width, param_overhead

PaperplaneResidualComposeCell = ResidualComposeCell

__all__ = ["PaperplaneResidualComposeCell", "ResidualComposeCell",
           "compose_width", "param_overhead"]
