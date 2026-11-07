# -*- coding: utf-8 -*-
"""翻盖 A/B 曲线深挖：只读两个 json，不改源码、不重跑。"""
import json, os, statistics as st

def interp_reach(h, thr):
    """history 每 10 步一条 → 线性插值估首次 loss<=thr 的步数。"""
    for i in range(1, len(h)):
        a, b = h[i-1], h[i]
        if b['loss'] <= thr:
            if a['loss'] <= thr: return float(a['step'])
            f = (a['loss']-thr)/(a['loss']-b['loss'])
            return a['step'] + f*(b['step']-a['step'])
    return None

A = json.load(open('output/v4/flip_F0_off.json'))
B = json.load(open('output/v4/flip_F1_decoupled.json'))
print('【插值到达步数（比 10 步粒度细）】')
for thr in [1.6, 1.5, 1.4, 1.3, 1.25]:
    ra, rb = interp_reach(A['history'], thr), interp_reach(B['history'], thr)
    d = None if ra is None or rb is None else rb-ra
    print(f'  thr={thr:<5} F0={ra}  F1={rb}  Δ={d}')

ha = {x['step']: x['loss'] for x in A['history']}
hb = {x['step']: x['loss'] for x in B['history']}
common = sorted(set(ha) & set(hb))
print(f'\n【逐点配对 Δloss = F1 − F0】 对齐点数={len(common)}')
d_all = [hb[s]-ha[s] for s in common]
print(f'  全程    mean={st.mean(d_all):+.5f}  std={st.pstdev(d_all):.5f}  '
      f'负号比例={sum(q<0 for q in d_all)/len(d_all):.1%}')
for lo, hi in [(0,500),(500,1500),(1500,2000),(2000,3000)]:
    seg = [hb[s]-ha[s] for s in common if lo <= s < hi]
    if seg:
        print(f'  [{lo:>4},{hi:>4}) n={len(seg):>3} mean={st.mean(seg):+.5f} '
              f'负号={sum(q<0 for q in seg)/len(seg):.1%}')
print('\n【判读】')
print('  逐点 mean≈0 且负号比例≈50% → 全程无系统性差异，「加速」证据不足，如实写')
print('  逐点 mean 显著<0 且负号比例>80% → 有系统性微弱加速，但需多 seed 配对确认')
print('  分段看差异从哪一步开始 → 判断 alpha_net.weight 是否真被训练推开')
