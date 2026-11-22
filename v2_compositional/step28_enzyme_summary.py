# -*- coding: utf-8 -*-
"""三档判据：A 对照 / B 纯残差 / C 残差+酶。只读 json，纯标准库。"""
import json, os, statistics as st
from collections import Counter
V4 = os.environ.get('V4DIR', 'output/v4')
def L(t):
    p = os.path.join(V4, t + '.json')
    if not os.path.exists(p): raise SystemExit('缺 ' + p + ' → 三档还没跑完')
    return json.load(open(p))
def late(a, b):
    h0 = {x['step']: x['loss'] for x in a['history']}
    h1 = {x['step']: x['loss'] for x in b['history']}
    cs = sorted(set(h0) & set(h1))
    return [h1[s]-h0[s] for s in cs if s >= 500]

A, B, C = L('bio_s0_B'), L('bio_s0_residual'), L('bio_v1003_enz')
H = L('b5_E1_split')
ha = {x['step']: x for x in H['history']}; hb = {x['step']: x for x in A['history']}
cs = sorted(set(ha) & set(hb))
mx_loss = max(abs(ha[s]['loss']-hb[s]['loss']) for s in cs)
mx_eff  = max(abs(ha[s]['eff'] -hb[s]['eff'])  for s in cs)
print('[0] 零污染硬前置  对齐点=%d  loss max|Δ|=%.3e  eff max|Δ|=%.3e  Δval=%+.3e'
      % (len(cs), mx_loss, mx_eff, A['val_loss']-H['val_loss']))
print('    ' + ('PASS' if mx_loss == 0 and mx_eff == 0 and A['val_loss'] == H['val_loss'] else 'FAIL → 停止，别往下读'))
enz_off = [e for e in A.get('events', []) if str(e.get('event','')).startswith('enzyme_')]
print('    OFF 档 enzyme_* 事件数 = %d （应为 0）' % len(enz_off))

print('\n[1] 三档终态')
print('%-22s %9s %12s %9s %11s' % ('tag','val','Δval vs A','eff','var_norm'))
for name, d in [('A bio_s0_B', A), ('B bio_s0_residual', B), ('C bio_v1003_enz', C)]:
    print('%-22s %9.4f %+12.4f %9.2f %11.4f' % (
        name, d['val_loss'], d['val_loss']-A['val_loss'],
        d['eff_organs_last'], d.get('var_norm_last', float('nan'))))

print('\n[2] 两个净效应（后段 s>=500 逐点配对）')
for nm, ref, x in [('残差骨架 B−A', A, B), ('酶 ON  C−B', B, C)]:
    ds = late(ref, x)
    print('  %-16s n=%d  mean=%+.5f  负号=%.1f%%' %
          (nm, len(ds), st.mean(ds), 100*sum(q < 0 for q in ds)/len(ds)))

print('\n[3] 酶谱系与观测层（规格 11/13）')
ev = C.get('events', [])
cnt = Counter(str(e.get('event','')) for e in ev)
for k in sorted(k for k in cnt if k.startswith('enzyme_') or k in ('substrate','atp_supply','product_inhibition')):
    print('  %-24s %d' % (k, cnt[k]))
ids = set()
for e in ev:
    for key in ('lineage_id','enzyme_id'):
        if key in e: ids.add((key, str(e[key])))
print('  谱系标识种类 = %d （>0 才可重放）' % len(ids))

print('\n【判读】')
print('  [0] 不过 → 回滚 *.bak_v1002，别进 A/B')
print('  B−A ≈ 0 且 C−B ≈ 0  → 残差+酶都不扰动，如实写「零扰动、暂无收益」')
print('  B−A < 0             → 残差版真拼第一次赢过 B 档，才谈拼专家路线有效')
print('  C−B < 0             → 酶催化在拼专家内部有净贡献（单 seed 仅方向，不叫显著）')
print('  永远不写「加速」：到达步数另需 history 插值验；终点等价不等于提速')
