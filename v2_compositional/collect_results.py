import json, glob, os, math

def flatten(d, out=None, pre=""):
    if out is None: out = {}
    for k, v in d.items():
        kk = pre + k
        if isinstance(v, dict):
            flatten(v, out, kk + "_")
        elif isinstance(v, (int, float, str)):
            out.setdefault(kk, v)
    return out

def pick(d, *keys):
    for k, v in d.items():
        lk = k.lower()
        for key in keys:
            if key in lk and isinstance(v, (int, float)):
                return v
    return None

def fmt(v, w=9, p=4):
    if v is None: return "-" .rjust(w)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return "nan".rjust(w)
    if isinstance(v, float): return f"{v:{w}.{p}f}"
    return str(v).rjust(w)

rows = []
for f in sorted(glob.glob("output/**/*.json", recursive=True)):
    try: raw = json.load(open(f))
    except Exception: continue
    if not isinstance(raw, dict): continue
    d = flatten(raw)
    parts = f.split(os.sep)
    grp = parts[1] if len(parts) > 2 else "root"
    tag = str(pick_raw(d) if False else d.get("tag", os.path.basename(f).replace(".json","")))
    rows.append((grp, tag[:18],
        pick(d,'val_loss','val','best_val'), pick(d,'ppl'),
        pick(d,'eff_organs_last','eff'), pick(d,'var_norm_last','var_norm'),
        pick(d,'theta','alpha'), pick(d,'params','n_params'),
        pick(d,'time_sec','time','elapsed')))

rows.sort(key=lambda r: (r[0], r[1]))
print("共解析", len(rows), "条\n")
h = f"{'组':<7}{'tag':<19}{'val_loss':>10}{'ppl':>9}{'eff':>8}{'var_norm':>10}{'角度':>9}{'参数':>12}{'用时s':>8}"
cur = None
for r in rows:
    if r[0] != cur:
        cur = r[0]
        print("=" * len(h)); print(f"【{cur}】"); print("-" * len(h)); print(h); print("-" * len(h))
    print(f"{r[0]:<7}{r[1]:<19}{fmt(r[2],10)}{fmt(r[3],9,2)}{fmt(r[4],8,2)}{fmt(r[5],10)}{fmt(r[6],9)}{fmt(r[7],12,0)}{fmt(r[8],8,0)}")

with open("results_all.csv","w",encoding="utf-8") as fh:
    fh.write("group,tag,val_loss,ppl,eff,var_norm,angle,params,time\n")
    for r in rows:
        fh.write(",".join("" if x is None else str(x) for x in r) + "\n")
print("\n已写出 results_all.csv")
