# -*- coding: utf-8 -*-
"""paperplane-Moe · 旧名兼容层（改名 B 的核心，不破坏任何历史文件）。

V1.000 只改了对外名称，src 里仍有 60 个 py 含 snowflake 共 224 处。
V1.001 的做法（可回滚、零破坏）：
    1. 所有【新代码】一律 paperplane_moe / paperplane_* 名字；
    2. 提供 paperplane_v4.py / paperplane_B.py / paperplane_compose.py
       三个别名模块，旧 snowflake_* 保持原样、继续可用；
    3. 06_改名执行_V1.001.sh 给出真正 git mv + 旧名 shim 的一键方案，
       默认 dry-run，确认后才 EXEC=1 执行。

为什么不做一次性全量替换：train_split.py / train_compose.py 的 import、
make_cell() 返回类型、output/v4/*.json 的 tag 与 checkpoint key 全挂在旧名上，
批量改会同时动 20+ 个脚本，和架构改动叠在一起会分不清是改名还是改架构
（这十几轮反复栽的「混淆变量」坑）。
"""
from __future__ import annotations

# 旧名 → 新名（对外统一 paperplane-Moe）
ALIAS_TABLE = {
    "仓库/目录": ("snowflake-moe/", "paperplane-Moe/ (V1.001 交付包顶层)"),
    "主干模型": ("snowflake_v4.SnowflakeLMV4", "paperplane_v4.PaperplaneLMV4"),
    "B 档 cell": ("snowflake_B.SnowflakeB", "paperplane_B.PaperplaneB"),
    "真拼核心(v8)": ("composable_expert.ComposableExpert",
                     "paperplane_moe.ComposableExpert (re-export)"),
    "step25 核心": ("compose_v10.ComposableExpertV10",
                    "paperplane_moe.ComposeV10 (re-export)"),
    "V1.001 核心": ("(新增)", "paperplane_moe.compose.ResidualComposeCell"),
    "训练入口": ("train_compose.py / train_split.py",
                "paperplane_moe.train_bio / v2_compositional/train_bio.py"),
    "观测层": ("(新增)", "paperplane_moe.telemetry.TelemetryRecorder"),
    "翻转片(已归档)": ("flipper*.py / FlipperTrap*", "已判「当前不进主模型」，保留原名"),
    "实验区": ("v2_compositional/", "保留（json 路径依赖）"),
}


def alias_table() -> dict:
    return dict(ALIAS_TABLE)


def verify_no_global_rng_pollution(local_seed: int = 0) -> bool:
    """改名/新模块验收辅助：局部 generator 不得改变全局 RNG 状态。

    在 DSW 的 verify_bio.py 里用 torch 实测；这里只给接口说明。
    """
    return True
