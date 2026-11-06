import json, glob

N = 384
VAR_MAX = (1.0/N) * (1.0 - 1.0/N)

def analyze(path):
    r = json.load(open(path))
    vn  = r["var_norm_last"]
    eff = r["eff_organs_last"]
    sum_var  = vn * VAR_MAX * N
    e_sum_sq = 1.0 / eff
    sum_m2   = e_sum_sq - sum_var
    if sum_m2 <= 0:
        return None
    marginal = 1.0 / sum_m2
    return (r["tag"], eff, sum_var, sum_m2, marginal, marginal/eff)

rows = [a for a in (analyze(f) for f in sorted(glob.glob("output/v3/*.json"))) if a]
print("%-14s %6s %9s %9s %9s %8s" % ("tag","eff","sum_var","sum_m2","marginal","pool"))
print("-"*60)
for t, e, sv, sm, mg, pl in rows:
    print("%-14s %6.2f %9.4f %9.4f %9.1f %7.1fx" % (t, e, sv, sm, mg, pl))
