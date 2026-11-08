import os, statistics as st
import torch
import torch.nn as nn
import train_split
from flipper_decoupled import FlipperTrapDecoupled

MODE = os.environ.get("FLIP_MODE", "")
assert MODE in ("iso_f1", "iso_d0"), MODE

FLIP_IDS = set()
STATS = []
_orig_clip = torch.nn.utils.clip_grad_norm_

def clip_split(parameters, max_norm=1.0, *args, **kwargs):
    ps = list(parameters)                      # generator 只能消费一次
    main = [p for p in ps if id(p) not in FLIP_IDS]
    flip = [p for p in ps if id(p) in FLIP_IDS]
    nm = _orig_clip(main, max_norm, *args, **kwargs) if main else 0.0
    nf = _orig_clip(flip, max_norm, *args, **kwargs) if flip else 0.0
    STATS.append((float(nm), float(nf)))
    if len(STATS) % 200 == 0:
        a = [q[0] for q in STATS]; b = [q[1] for q in STATS]
        print("[clip_iso] n=%d main=%.3f flip=%.3f main>1占比=%.1f%%" %
              (len(STATS), st.mean(a), st.mean(b),
               100.0*sum(v > 1 for v in a)/len(a)), flush=True)
    return nm

torch.nn.utils.clip_grad_norm_ = clip_split
if hasattr(train_split, "clip_grad_norm_"):        # 双覆盖：属性 / from-import
    train_split.clip_grad_norm_ = clip_split

class DummyFlip(nn.Module):                        # 不能 import step22（会触发 main）
    def __init__(self, d):
        super().__init__()
        self.inner = FlipperTrapDecoupled(d)
    def forward(self, x, carry=None):
        y, c, dg = self.inner(x, carry)
        return x + (y - y.detach()), c, dg

_orig_build = train_split.build
def build_patched(a, vocab, n_start, device):
    m = _orig_build(a, vocab, n_start, device)
    cpu_st = torch.get_rng_state()
    cuda_st = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    dev = next(m.parameters()).device
    cls = FlipperTrapDecoupled if MODE == "iso_f1" else DummyFlip
    m.flippers = nn.ModuleList([cls(a.d).to(dev) for _ in range(a.L)])
    FLIP_IDS.update(id(p) for mod in m.flippers for p in mod.parameters())
    torch.set_rng_state(cpu_st)
    if cuda_st is not None: torch.cuda.set_rng_state_all(cuda_st)
    print("[clip_iso] MODE=%s L=%s flip_ids=%d" %
          (MODE, a.L, len(FLIP_IDS)), flush=True)
    return m
train_split.build = build_patched
train_split.main()
