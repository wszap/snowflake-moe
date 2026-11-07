import json, os, statistics as st, math
TAGS = {'E1':['b5_E1_split','ms_s1_E1','ms_s2_E1'],
        'E3':['b5_E3_compete','ms_s1_E3','ms_s2_E3'],
        'E1p':['b5_E1p_nosplit','ms_s1_E1p','ms_s2_E1p']}
v={}
for k,ts in TAGS.items():
    v[k]=[]
    for t in ts:
        p='output/v4/%s.json'%t
        if not os.path.exists(p): raise SystemExit('缺 '+p)
        v[k].append(json.load(open(p))['val_loss'])
print('档      seed0    seed1    seed2     mean ± std')
for k in TAGS:
    print(f'  {k:<5} {v[k][0]:.4f}  {v[k][1]:.4f}  {v[k][2]:.4f}   '
          f'{st.mean(v[k]):.4f} ± {st.stdev(v[k]):.4f}')
print('\n配对差')
for a,b in [('E1','E3'),('E1','E1p'),('E3','E1p')]:
    d=[x-y for x,y in zip(v[a],v[b])]
    m,sd=st.mean(d),st.stdev(d)
    t=abs(m)/(sd/math.sqrt(3))
    print(f'  {a}−{b:<4} '+' '.join(f'{q:+.4f}' for q in d)+
          f'   {m:+.4f} ± {sd:.4f}   t={t:.2f} (df=2)  '
          f'{"3/3 同号" if all(q<0 for q in d) or all(q>0 for q in d) else "有反号"}')
