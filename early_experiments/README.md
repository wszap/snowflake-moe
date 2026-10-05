# early_experiments

Archived early and experimental scripts from the development history of Snowflake MoE.
These files are kept for provenance and are **not** part of the stable training pipeline.

## Contents

- `nation_*.py`, `mixed_governance.py`, `moe_*.py`: early governance / expert-capacity experiments
- `marvis_moe_v5_orig.py`, `marvis_moe_v6.py`, `marvis_moe_v7.py`: architecture version backups
  (the stable core lives in the repo root as `marvis_moe.py` / `snowflake_moe*.py`)
- `diag_*.py`, `min_repro.py`, `smoke_new_modes.py`: debugging / minimal reproduction scripts
- `gate_collapse.py`, `bias_vs_flat.py`, `lm_speed_batch128.py`, `perf_train.py`: diagnostic and
  performance probes
- `run_ood_v2_20261005_010637_107.py`: one-off timestamped experiment run

None of these are required to reproduce the results in `README.md`.
