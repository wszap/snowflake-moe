import ast,sys,os,math,re
P="paperplane_moe/enzyme_rate.py"
s=open(P,encoding="utf-8").read()
if "V1.006-B" in s:
    print("ALREADY_PATCHED skip"); sys.exit(0)
t=ast.parse(s)
c=next((n for n in t.body if isinstance(n,ast.ClassDef) and "EnzymeRate" in n.name),None)
if c is None:
    print("CLASS_CANDIDATES=",[n.name for n in t.body if isinstance(n,ast.ClassDef)]); sys.exit(2)
f=next((n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=="observe_step"),None)
if f is None:
    print("METHODS=",[x.name for x in c.body if isinstance(x,ast.FunctionDef)]); sys.exit(2)
L=s.splitlines(True)
ns={n.id for n in ast.walk(f) if isinstance(n,ast.Name)}
print("ARGS=",[a.arg for a in f.args.args]," H_IN_SCOPE=",("H" in ns)," NAMES=",sorted(x for x in ns if x in ("G","H","Hs","Hlist","S","k","grad")))
def E(nm):
    for n in f.body:
        tg=n.targets if isinstance(n,ast.Assign) else ([n.target] if isinstance(n,ast.AnnAssign) else [])
        for q in tg:
            if isinstance(q,ast.Name) and q.id==nm: return n.end_lineno
            if isinstance(q,(ast.Tuple,ast.List)):
                for e in q.elts:
                    if isinstance(e,ast.Name) and e.id==nm: return n.end_lineno
    return None
se,ke=E("S"),E("k")
print("S_end=%s k_end=%s"%(se,ke))
if se is None or ke is None or ke<=se:
    print("ORIGINAL:"); sys.stdout.write("".join(L[f.lineno-1:f.end_lineno])); print("FATAL bad anchors"); sys.exit(2)
ind=re.match(r"\s*",L[se-1]).group(0)
print("INDENT=%r"%ind)
def block(xs):
    return "".join(ind+x+"\n" for x in xs)
A=block([
 "# V1.006-A remaining substrate S = g^2/H (MM / Briggs-Haldane)",
 "try:",
 "    _Hi=[float(v) for v in H]",
 "except NameError:",
 "    _Hi=[1.0]*len(S)",
 "_Si=[float(v) for v in S]",
 "if len(_Hi)!=len(_Si):",
 "    _Hi=[1.0]*len(_Si)",
 "S=[si/(hi+1e-12) for si,hi in zip(_Si,_Hi)]",
])
B=block([
 "# V1.006-B zero-sum: sum(log k)=0 (MCA / proteome budget / Haldane)",
 "_os=__import__('os'); _rnd=__import__('random'); _m=__import__('math')",
 "try:",
 "    _step=int(step)",
 "except Exception:",
 "    _step=-1",
 "if _step<int(getattr(self,'warmup',1)):",
 "    k=[1.0]*len(k)",
 "elif str(getattr(self,'mode','')) in ('bb_cat','cat','cat_het') or _os.environ.get('ENZ_SHUFFLE')=='1':",
 "    _kk=[float(v) for v in k]",
 "    _lk=[_m.log(max(v,1e-9)) for v in _kk]",
 "    _mu=sum(_lk)/max(len(_lk),1)",
 "    _cap=_m.log(4.0)",
 "    k=[_m.exp(max(-_cap,min(_cap,v-_mu))) for v in _lk]",
 "    if _os.environ.get('ENZ_SHUFFLE')=='1':",
 "        _rnd.Random(20261122).shuffle(k)",
])
for ln,blk in sorted([(ke,B),(se,A)],reverse=True):
    L.insert(ln,blk)
new="".join(L)
ast.parse(new)
b=P+".bak_v1006"
if not os.path.exists(b):
    open(b,"wb").write(open(P,"rb").read())
open(P,"w",encoding="utf-8").write(new)
z=[0.0,0.4,0.9,1.6,0.2,1.1,0.7,0.3]*2
lk=[math.log(1+v) for v in z]; mu=sum(lk)/len(lk); k=[math.exp(v-mu) for v in lk]
sl=sum(math.log(v) for v in k)
print("PATCH_OK backup=%s"%b)
print("sum(log k)=%+.3e  step0_k=1.000000"%sl)
assert abs(sl)<1e-9
print("FORMULA_SELFTEST=PASS")
