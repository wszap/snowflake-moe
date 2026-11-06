import json, glob, os
def g(d, *ks):
    for k in ks:
        if isinstance(d.get(k), (int, float)): return d[k]
    return None
def f(v, w=10, p=4):
    return '-'.rjust(w) if v is None else (f'{v:{w}.{p}f}' if isinstance(v, float) else f'{v:>{w},}')
print('file'.ljust(32), 'val'.rjust(10), 'eff'.rjust(8), 'var'.rjust(10), 'params'.rjust(14))
print('-' * 78)
for pat in ['output/v3/*.json','output/v4/*.json','output/v8/*.json','output/dense/*.json']:
    for fn in sorted(glob.glob(pat)):
        try: d = json.load(open(fn))
        except Exception: continue
        print(os.path.basename(fn).ljust(32),
              f(g(d,'val_loss','val')),
              f(g(d,'eff_organs_last','eff','eff_organs'), 8, 2),
              f(g(d,'var_norm_last','var_norm')),
              f(g(d,'params','n_params','param_count'), 14, 0))
