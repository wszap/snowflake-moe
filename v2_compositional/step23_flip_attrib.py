# -*- coding: utf-8 -*-
"""翻盖归因三 seed 配对：F1(旋转) / D0(inert dummy) 相对 F0。只读 json。"""
import json, os, statistics as st
V4 = os.environ.get('V4DIR', 'output/v4')
TRI = [('flip_F0_off','flip_F1_decoupled','flip_D0_dummy'),
       ('ms_s1_E1',   'flip_s1_F1',      'flip_s1_D0'),
       ('ms_s2_E1',   'flip_s2_F1',      'flip_s2_D0')]
def L(tag):
    p = os.path.join(V4, tag+'.json')
    if not os.path.exists(p): raise SystemExit('缺 '+p+' → D0 补 seed 还没跑完')
    return json.load(open(p))
def late_diff(a, b):
    h0 = {x['step']:x['loss'] for x in a['history']}
    h1 = {x['step']:x['loss'] for x in b['history']}
    cs = sorted(set(h0) & set(h1))
    ds = [h1[s]-h0[s] for s in cs if s >= 500]
    assert ds, '后段为空'
    return st.mean(ds), len(ds)
rows = []
print('seed   F1−F0(val)   D0−F0(val)   F1−D0(val)   后段F1    后段D0    后段(F1−D0)  n')
for i,(t0,t1,td) in enumerate(TRI):
    f0,f1,d0 = L(t0),L(t1),L(td)
    a = f1['val_loss']-f0['val_loss']; b = d0['val_loss']-f0['val_loss']
    c = f1['val_loss']-d0['val_loss']
    la,n1 = late_diff(f0,f1); lb,n2 = late_diff(f0,d0)
    assert n1 == n2
    rows.append((a,b,c,la,lb,la-lb))
    print(f'  {i}    {a:+.4f}      {b:+.4f}      {c:+.4f}    {la:+.5f}  {lb:+.5f}   {la-lb:+.5f}   {n1}')
cs = [r[2] for r in rows]; ls = [r[5] for r in rows]
print('\n===== 汇总 =====')
print(f'  F1−D0 (val) = {st.mean(cs):+.4f} ± {st.stdev(cs):.4f}   '
      f'{"3/3 同号" if all(q<0 for q in cs) or all(q>0 for q in cs) else "有反号"}')
print(f'  后段 F1−D0  = ' + '  '.join(f'{q:+.5f}' for q in ls) + f'   3/3<0={all(q<0 for q in ls)}')
print(f'  D0−F0 三档  = ' + '  '.join(f'{r[1]:+.4f}' for r in rows) +
      f'   3/3>0(副作用有害)={all(r[1]>0 for r in rows)}')
