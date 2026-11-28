# -*- coding: utf-8 -*-
"""paperplane-Moe · V1.001 统一命名入口（生物学思想 → 可执行代码）。

本包是「对外统一改名」的落地点：所有【新代码】都用 paperplane_* 名字，
旧 snowflake_* 文件保留为兼容层（见 legacy_shim.py 与 06_改名执行_V1.001.sh），
从而不破坏 train_split.py / train_compose.py 的 import 与 20+ 个历史 json 的路径。

红线（手写笔记规格 12）：
    一切挂在  Σ_i w_i · silu(x·E1_i)·E2_i  或其【残差增量】之上；
    wiring 只分配份额、不筛选细胞；默认 gate=0 → 严格恒等、零扰动起步。

本包所有模块默认关闭（BioConfig.enable=False）：
    关闭时前向逐位恒等、不消耗全局 RNG、不进 optimizer，
    只有观测层 TelemetryRecorder 往 events 写字段（规格 11/13）。
"""
__version__ = "V1.001"
__project__ = "paperplane-Moe"

from .biology import (
    BioConfig,
    MemoryPoint,
    Bloodstream,
    Receptor,
    Nucleoskeleton,
    SparseRecall,
    TempExpertModule,
    ATPBudget,
    Enzyme,
    GradientGuard,
    Mitochondria,
    Chloroplast,
    Autophagy,
    Apoptosis,
    CellLineage,
    assert_compose_redline,
)
from .telemetry import TelemetryRecorder, REQUIRED_EVENT_KINDS, replay_lineage
from .compose import ResidualComposeCell, compose_width, param_overhead
from .cell_environment import (
    ENVIRONMENT_AXES,
    ChemicalCompartment,
    CellEnvironment,
    audit_taxonomic_coherence,
)
from .legacy_shim import ALIAS_TABLE, alias_table

__all__ = [
    "__version__", "__project__",
    "BioConfig", "MemoryPoint", "Bloodstream", "Receptor",
    "Nucleoskeleton", "SparseRecall", "TempExpertModule", "ATPBudget",
    "Enzyme", "GradientGuard", "Mitochondria", "Chloroplast",
    "Autophagy", "Apoptosis", "CellLineage", "assert_compose_redline",
    "TelemetryRecorder", "REQUIRED_EVENT_KINDS", "replay_lineage",
    "ResidualComposeCell", "compose_width", "param_overhead",
    "ENVIRONMENT_AXES", "ChemicalCompartment", "CellEnvironment",
    "audit_taxonomic_coherence",
    "ALIAS_TABLE", "alias_table",
]
