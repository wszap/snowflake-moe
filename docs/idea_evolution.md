# Idea Evolution: Snowflake MoE

A record of how the project evolved from initial experiments to the stable architecture.

## Stage 1: Baseline Sparse MoE

- First from-scratch MoE: fine-grained experts + top-k routing.
- Learned a dense-ish gate; suffered from load imbalance and router collapse.
- Scripts: `run_stage1.py`.

## Stage 2: Shared Expert + Memory Points

- Introduced a shared expert that all tokens attend to, acting as a "memory point".
- Decoupled factual storage (memory values) from routing weights.
- Scripts: `run_stage2.py`.

## Stage 3: Compositional Routing

- Combined fine-grained experts with sparse activation and a composition bonus.
- The router learned to compose expert subsets, improving OOD generalization.
- Scripts: `run_stage3.py`.

## Stage 4: Erasable Memory

- Demonstrated targeted erasure: zeroing memory-value slots at inference.
- Erasing 50% slots caused only mild PPL degradation; 100% erasure collapsed the model,
  proving knowledge localization.
- Scripts: `run_stage4.py`, `run_erase_tinystories.py`, `run_erase_mr2.py`.

## Stage 5: Lifelong Learning & Stability

- Sequential domain addition (TinyStories -> Bible) with negligible regression on old domains.
- Stabilized training with fixed routes / learned gate variants and multi-seed evaluation.
- Scripts: `run_stage5.py`, `run_stage5_fast.py`, `run_stage5_fixed.py`, `run_lifelong_tinystories.py`.

## Stable Architecture (v7 / CellMoE)

- `marvis_moe.py` + `snowflake_moe.py` + `snowflake_moe_improved.py` form the stable core.
- Earlier versions (v5/v6/v7) and experimental scripts are archived in `early_experiments/`.
