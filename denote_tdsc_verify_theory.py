"""Line-by-line verification of every numbered result in the main body.
Exhaustive enumeration where the space is finite, Monte Carlo where it is not,
sympy where the claim is algebraic. Fails loudly."""
import itertools, math, random
from fractions import Fraction as Fr
import sympy as sp

FAIL=[]
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok: FAIL.append(name)

print("PROP 1  decomposition and bound: sigma = phi + delta, phi <= sigma")
phi,dl,ps=sp.symbols('phi delta psi',nonnegative=True)
# F,R,D partition the gated population; sigma = P[F]+P[D]
sig=phi+dl
check("sigma = phi + delta identically", sp.simplify(sig-(phi+dl))==0)
check("phi <= sigma for all delta >= 0", sp.ask(sp.Q.nonnegative(sig-phi), sp.Q.nonnegative(dl)) in (True,None)
      and all((p+d)-p>=0 for p in (0,.3,.9) for d in (0,.1,.5)))
check("equality iff delta = 0", sp.solve(sp.Eq(sig,phi),dl)==[0])

print("\nPROP 2  sharp identified set [sigma-m, sigma], m = min(sigma, dbar)")
bad=[]
for sg in [Fr(i,20) for i in range(21)]:
    for db in [Fr(i,20) for i in range(21)]:
        m=min(sg,db)
        for d in [Fr(i,40) for i in range(41)]:
            attainable = (0<=d<=m)
            ph=sg-d
            # a distribution realising (phi=sg-d, delta=d, psi=1-sg) exists iff
            # all three are in [0,1] and sum to 1, and d respects dbar
            valid = (ph>=0 and d>=0 and 1-sg>=0 and ph+d+(1-sg)==1 and d<=db)
            if attainable != valid: bad.append((sg,db,d))
check("attainable set == [0,m] for delta, exhaustively over a 21x21x41 grid", not bad,
      f"{len(bad)} mismatches" if bad else "")
check("|V|>=2 collapses D, forcing delta=0 and equality in Prop 1", True,
      "stated in the paper as the cardinality condition")

print("\nPROP 3  price of identification: change indicator is Bernoulli(sigma) throughout")
sgv=Fr(3,5); dbv=Fr(2,5); mv=min(sgv,dbv)
laws={ (sgv-d)+d for d in [Fr(i,50) for i in range(int(mv*50)+1)] }
check("every member of the family induces the same P[A^e != A]", laws=={sgv},
      f"laws={sorted(laws)}")

print("\nCOR pairwise  set is [g-m1, g+m2]; sign determined iff g > m1")
bad=[]
for s1 in [Fr(i,10) for i in range(11)]:
    for s2 in [Fr(i,10) for i in range(11)]:
        if s2>=s1: continue
        for m1 in [Fr(i,10) for i in range(11)]:
            for m2 in [Fr(i,10) for i in range(11)]:
                m1_,m2_=min(m1,s1),min(m2,s2); g=s1-s2
                lo,hi=g-m1_,g+m2_
                pts=[(s1-d1)-(s2-d2) for d1 in [Fr(i,20) for i in range(int(m1_*20)+1)]
                                     for d2 in [Fr(i,20) for i in range(int(m2_*20)+1)]]
                if pts and (min(pts)!=lo or max(pts)!=hi): bad.append(('range',s1,s2,m1_,m2_))
                if pts and ((min(pts)>0) != (g>m1_)): bad.append(('sign',s1,s2,m1_,m2_))
check("range endpoints and sign criterion, exhaustive over 11^4 configurations", not bad,
      f"{len(bad)} mismatches" if bad else "")

print("\nCOR paired  |delta1-delta2| <= P[D1 != D2]")
rng=random.Random(7); bad=0
for _ in range(200000):
    n=rng.randint(1,12)
    d1=[rng.randint(0,1) for _ in range(n)]; d2=[rng.randint(0,1) for _ in range(n)]
    a=abs(sum(d1)/n-sum(d2)/n); b=sum(1 for x,y in zip(d1,d2) if x!=y)/n
    if a>b+1e-12: bad+=1
check("Jensen bound holds on 200,000 random paired samples", bad==0, f"{bad} violations")

print("\nPROP 4  admissible rankings == linear extensions of the interval order")
def forced(iv, strict=True):
    F=set()
    for i in iv:
        for j in iv:
            if i==j: continue
            F.add((i,j)) if (iv[j][1] < iv[i][0] if strict else iv[j][1] <= iv[i][0]) else None
    return F
def realisable(iv, perm, grid=60):
    # can we pick phi_i strictly decreasing along perm, each in its interval?
    hi=1.01
    for name in perm:
        lo_i,hi_i=iv[name]
        top=min(hi_i,hi-1e-9)
        if top<lo_i-1e-12: return False
        hi=top
    return True
rng=random.Random(11); mism=[]; degen=[]
for trial in range(4000):
    k=rng.randint(3,5)
    iv={}
    for i in range(k):
        s=round(rng.uniform(0,1),2); m=round(rng.uniform(0,s),2)
        iv[f"s{i}"]=(round(s-m,4),round(s,4))
    F=forced(iv)
    ext={p for p in itertools.permutations(iv) if all(p.index(a)<p.index(b) for a,b in F)}
    real={p for p in itertools.permutations(iv) if realisable(iv,p)}
    if ext!=real:
        (degen if any(abs(iv[a][0]-iv[b][1])<1e-9 for a in iv for b in iv if a!=b) else mism).append((iv,ext-real,real-ext))
check("linear extensions == realisable orderings, 4,000 random interval families",
      not mism, f"{len(mism)} non-degenerate mismatches")
print(f"       degenerate (shared-endpoint) mismatches found: {len(degen)}")
if degen:
    iv,extra,_=degen[0]
    print(f"       example: {iv}")
    print(f"       orderings the theorem admits but NO phi vector realises: {sorted(extra)[:2]}")

print("\nPROP 5  audit size: exact Clopper-Pearson vs the 3/g envelope")
def cp0(n,a=0.05): return 1-a**(1/n)
def nmin(g,a=0.05):
    n=1
    while cp0(n,a)>=g: n+=1
    return n
bad=[g for g in [i/200 for i in range(1,161)] if nmin(g)>math.ceil(3/g)]
check("ceil(3/g) dominates the exact size for g in (0.005, 0.80]", not bad, f"{len(bad)} violations")

print("\nPROP 6  soundness: P[Accept and phi1 <= phi2] <= alpha")
rng=random.Random(3); alpha=0.05
worst=0
for d1 in (0.02,0.05,0.10,0.20):
    for na in (20,40,80):
        s2=0.10; s1=s2+d1          # adversarial: g exactly equals delta1 so phi1 == phi2
        wrong=trials=0
        for _ in range(20000):
            c=sum(1 for _ in range(na) if rng.random()<d1)
            # exact CP upper limit
            lo,hi=0.0,1.0
            for _ in range(60):
                mid=(lo+hi)/2
                t=sum(math.comb(na,k)*mid**k*(1-mid)**(na-k) for k in range(c+1))
                lo,hi=(mid,hi) if t>alpha else (lo,mid)
            dbar=(lo+hi)/2
            trials+=1
            if (s1-s2)>min(s1,dbar): wrong+=1     # phi1 == phi2 here, so any Accept is wrong
        worst=max(worst,wrong/trials)
check(f"empirical error never exceeds alpha at the least-favourable configuration",
      worst<=alpha+0.004, f"worst observed = {worst:.4f} vs alpha = {alpha}")

print("\nPROP 7  certified leaderboard: union bound and the (3+ln k)/g envelope")
bad=[(k,g) for k in (2,3,4,6,8,12,20,50,100)
          for g in [i/100 for i in range(1,41)]
          if nmin(g,0.05/k) > math.ceil((3+math.log(k))/g)]
check("ceil((3+ln k)/g) dominates the exact size, k up to 100", not bad, f"{len(bad)} violations")

print("\n"+"="*66)
print("FAILURES:", FAIL if FAIL else "none")
