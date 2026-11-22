#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step28f 判据：分类 (a)/(b)/(c) + 证明探针数值惰性。纯标准库，不依赖 torch。"""
import json, os, statistics as st
V4 = 'output/v4'

def L(t):
    p = os.path.join(V4, t + '.json')
    return json.load(open(p)) if os.path.exists(p) else None

print('===== [1] 归档三档（结论不重算）=====')
for t in ['bio_s0_B', 'bio_s0_residual', 'bio_v1003_enz']:
    d = L(t)
    if d:
        print('  %-22s val=%.6f  eff=%.4f' %
              (t, d['val_loss'], d.get('eff_organs_last', 0.0)))

new = L('bio_v1003_enz_probe')
old = L('bio_v1003_enz') or L('bio_v1003_enz.preprobe')

print('===== [2] 探针 C 档：gate_comp 轨迹 =====')
if not new:
    print('  缺 bio_v1003_enz_probe.json → 还没补跑 C 档')
else:
    h = new['history']
    g = [(x['step'], float(x['gate_comp'])) for x in h if 'gate_comp' in x]
    c = [(x['step'], float(x['comp_norm'])) for x in h if 'comp_norm' in x]
    z = [(x['step'], float(x['ci_mean'])) for x in h if 'ci_mean' in x]
    print('  落盘点 = %d' % len(g))
    if g:
        print('  gate_comp  step0=%+.3e  mid=%+.3e  last=%+.3e  max|g|=%.3e' %
              (g[0][1], g[len(g)//2][1], g[-1][1], max(abs(v) for _, v in g)))
        for target in [0, 500, 1500, 2500, 3000]:
            near = min(g, key=lambda q: abs(q[0] - target))
            print('    step~%-5d gate_comp=%+.6e' % (near[0], near[1]))
    if c:
        print('  comp_norm   last=%.6f  max=%.6f' % (c[-1][1], max(v for _, v in c)))
    if z:
        nz = sum(1 for _, v in z if abs(v - 1.0) > 1e-9)
        print('  ci_mean    min=%.6f max=%.6f 偏离1的点=%d/%d' %
              (min(v for _, v in z), max(v for _, v in z), nz, len(z)))

print('===== [3] 分类（预先写死，防事后挑数）=====')
if new and g and z:
    grew = max(abs(v) for _, v in g) > 1e-6
    ci_on = any(abs(v - 1.0) > 1e-9 for _, v in z)
    sentinel = any(v == -9.0 for _, v in g)
    if sentinel:
        print('  探针异常（-9.0 哨兵）→ 贴 traceback，先看 diag / model 变量名')
    elif not grew:
        print('  (b) gate_comp 全程≈0 → 残差分支休眠，酶有没有用根本没被测过')
        print('      → 下一步是让 gate 有可学起点，但会破 301 点逐位恒等，属新实验')
    elif not ci_on:
        print('  (c) gate!=0 但 ci≡1 → 酶只跑了谱系状态机，没进前向')
    else:
        print('  (a) gate!=0 且 ci 偏离1 → 催化确实进了前向')
        print('      → C-B≈0 就是干净负结果：拼专家 + 酶当前无收益，如实收口')
else:
    print('  数据不足，不分类')

print('===== [4] 探针数值惰性（新 C vs 归档 C，301 点必须全 0）=====')
if new and old:
    ha = {x['step']: x for x in old['history']}
    hb = {x['step']: x for x in new['history']}
    cs = sorted(set(ha) & set(hb))
    print('  对齐点 = %d   Δval = %+.3e' % (len(cs), new['val_loss'] - old['val_loss']))
    mxs = {}
    for k in ['loss', 'eff']:
        if k in ha[cs[0]] and k in hb[cs[0]]:
            mxs[k] = max(abs(ha[s][k] - hb[s][k]) for s in cs)
            print('  %-4s max|Δ| = %.3e' % (k, mxs[k]))
    if all(v == 0.0 for v in mxs.values()) and abs(new['val_loss'] - old['val_loss']) == 0.0:
        print('  PASS → 探针纯只读，补跑与归档逐位一致，gate 轨迹可信')
    else:
        print('  FAIL → 探针或重跑引入差异，先查，别用它下结论')
