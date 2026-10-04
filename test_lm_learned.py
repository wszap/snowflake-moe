import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lm import load_shakespeare, train_lm
from marvis_moe import Config
vs, tr, va, ch = load_shakespeare()
cfg = Config(d=64, h=128, E=4, S=1, L=2, topk=2, out_dim=vs)
train_lm(cfg, vs, tr, va, council_mode='learned', seed=2026, epochs=1, verbose=True)
print("LEARNED_EP1_DONE")
