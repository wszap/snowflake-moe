# -*- coding: utf-8 -*-
"""确认 FLIP_MODE=off 与直接 train_split 逐位相同 → seed1/2 的 F0 可复用 ms_s*_E1。"""
import json, os
V4 = os.environ.get('V4DIR', 'output/v4')
a = json.load(open(os.path.join(V4, 'flip_F0_off.json')))
b = json.load(open(os.path.join(V4, 'b5_E1_split.json')))
ha = {x['step']: x['loss'] for x in a['history']}
hb = {x['step']: x['loss'] for x in b['history']}
cs = sorted(set(ha) & set(hb))
assert cs, 'history 无对齐点'
mx = max(abs(ha[s]-hb[s]) for s in cs)
print(f'  对齐点 = {len(cs)}')
print(f'  val  差 = {a["val_loss"]-b["val_loss"]:+.3e}')
print(f'  history max|Δ| = {mx:.3e}')
print(f'  eff  差 = {a["eff_organs_last"]-b["eff_organs_last"]:+.3e}')
print(f'  params {a["params_start"]:,}→{a["params_end"]:,}  vs  '
      f'{b["params_start"]:,}→{b["params_end"]:,}')
ok = mx == 0 and a['val_loss'] == b['val_loss']
print('  ' + ('PASS → off 档零改动，seed1/2 复用 ms_s*_E1 合法'
             if ok else 'FAIL → 不可复用，F0 必须逐 seed 重跑'))
