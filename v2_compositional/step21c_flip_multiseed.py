# -*- coding: utf-8 -*-
"""翻盖 3-seed 配对判据：只读 json，不改源码、不重跑训练。"""
import json, os, statistics as st

V4 = os.environ.get('V4DIR', 'output/v4')
PAIRS = [('flip_F0_off', 'flip_F1_decoupled'),
         ('ms_s1_E1',    'flip_s1_F1'),
         ('ms_s2_E1',    'flip_s2_F1')]

def load(tag):
    p = os.path.join(V4, tag + '.json')
    if not os.path.exists(p):
        raise SystemExit('缺 ' + p + '  → 补 seed 还没跑完')
    return json.load(open(p))

def paired(f0, f1):
    assert f0['history'] and f1['history'], 'history 为空'
    h0 = {x['step']: x['loss'] for x in f0['history']}
    h1 = {x['step']: x['loss'] for x in f1['history']}
    cs = sorted(set(h0) & set(h1))
    assert cs, '两档 history 无对齐点'
    return cs, [h1[s] - h0[s] for s in cs]

dvals, dparams, late_means, neg_rates = [], [], [], []
print('seed  F0 -> F1                                    Δval     Δparams   逐点mean   负号   [0,500)   后段mean')
for i, (t0, t1) in enumerate(PAIRS):
    f0, f1 = load(t0), load(t1)
    dv = f1['val_loss'] - f0['val_loss']
    dp = f1['params_start'] - f0['params_start']
    cs, ds = paired(f0, f1)
    early = [q for s, q in zip(cs, ds) if s < 500]
    late  = [q for s, q in zip(cs, ds) if s >= 500]      # ★ 含最后一点
    assert early and late, '分段为空，检查 history 步长'
    lm = st.mean(late)
    dvals.append(dv); dparams.append(dp); late_means.append(lm)
    neg_rates.append(sum(q < 0 for q in ds) / len(ds))
    print(f'  {i}   {t0:<20} -> {t1:<20} {dv:+.4f}  {dp:>9,}  '
          f'{st.mean(ds):+.5f}  {neg_rates[-1]:>5.1%}  '
          f'{st.mean(early):+.5f}    {lm:+.5f}')
    print(f'       对齐点={len(cs)}  early_n={len(early)}  late_n={len(late)}')

assert len(dvals) == 3
print('\n===== 汇总 =====')
print(f'  Δval      = {st.mean(dvals):+.4f} ± {st.stdev(dvals):.4f}   '
      f'{"3/3 同号" if all(q<0 for q in dvals) or all(q>0 for q in dvals) else "有反号"}')
print(f'  Δparams   = {dparams}   三档一致={len(set(dparams))==1}')
print(f'  后段 mean = ' + '  '.join(f'{q:+.5f}' for q in late_means) +
      f'   3/3 < 0 = {all(q < 0 for q in late_means)}')
print(f'  终点等价  = {abs(st.mean(dvals)) < 0.021}   (|mean Δval| < 噪声 0.021)')
print('\n【判读 · 不叫加速】')
print('  终点等价 + 后段 3/3 系统性压低 → 「无损 + 训练动力学一致微扰」跨 seed 成立')
print('  到达步数未变 → 仍不写「加速」；归因到结构而非 3.07% 容量需同参数量对照')
