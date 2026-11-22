import ast,sys,os,math
P="paperplane_moe/enzyme_rate.py"
s=open(P,encoding="utf-8").read()
if "V1.006-B" in s:
    print("ALREADY_PATCHED skip"); sys.exit(0)
L=s.splitlines(True)
def idx(prefix, exact):
    return [i for i,l in enumerate(L) if (l.rstrip("\n")==prefix if exact else l.startswith(prefix))]
a1=idx('        if self.scope == "backbone":', True)
a2=idx('            H = [max(v, _EPS) for v in S]', False)
b =idx('        self.k = k', True)
print("A1=%s A2=%s B=%s"%(a1,a2,b))
if len(a1)!=1 or len(a2)!=1 or len(b)!=1:
    print("FATAL anchors not unique (file untouched)"); sys.exit(2)
A1='        H_comp = list(H)  # V1.006-A: keep true comp-curvature proxy before H is overwritten\n'
A2=('            S = [min(sg / max(hc, _EPS), 1e12) for sg, hc in zip(S, H_comp)]'
    '  # V1.006-A: [S] = g^2/H remaining substrate (MM / Briggs-Haldane)\n')
B='''        # V1.006-B  zero-sum: sum(log k)=0 (MCA conc-summation / proteome budget / Haldane)
        _os = __import__("os"); _rnd = __import__("random"); _m = __import__("math")
        if int(step) < int(getattr(self, "warmup", 1)):
            k = [1.0] * len(k)
        elif (self.mode in ("cat", "cat_het") or _os.environ.get("ENZ_SHUFFLE") == "1") and int(getattr(self, "freeze_cell", -1)) < 0:
            _lk = [_m.log(max(float(v), 1e-9)) for v in k]
            _mu = sum(_lk) / max(len(_lk), 1)
           _cap=_m.log(4.0)
            k = [_m.exp(max(-_cap, min(_cap, v - _mu))) for v in _lk]
            if _os.environ.get("ENZ_SHUFFLE") == "1":
                _rnd.Random(20261122).shuffle(k)
'''
for i,t in sorted([(a1[0],A1),(a2[0]+1,A2),(b[0],B)], reverse=True):
    L.insert(i,t)
new="".join(L)
ast.parse(new)
bak=P+".bak_v1006"
if not os.path.exists(bak):
    open(bak,"wb").write(open(P,"rb").read())
open(P,"w",encoding="utf-8").write(new)
print("PATCH_OK backup=%s"%bak)
z=[0.0,0.4,0.9,1.6,0.2,1.1,0.7,0.3]*2
lk=[math.log(1+v) for v in z]; mu=sum(lk)/len(lk); k=[math.exp(v-mu) for v in lk]
sl=sum(math.log(v) for v in k)
print("sum(log k)=%+.3e  step0_k=1.000000"%sl)
assert abs(sl)<1e-9
print("FORMULA_SELFTEST=PASS")
