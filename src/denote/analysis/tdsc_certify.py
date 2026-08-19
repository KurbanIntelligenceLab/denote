#!/usr/bin/env python3
"""Experiments E1-E3 for the certification procedure, computed from the
released records. No model calls. Standard library only."""
import json, os, math, random
from itertools import permutations
D='results/raw'
SHORT={'meta-llama/llama-3.1-70b-instruct':'Llama-70B','meta-llama/llama-3.1-8b-instruct':'Llama-8B',
 'qwen/qwen-2.5-72b-instruct':'Qwen-72B','qwen/qwen-2.5-7b-instruct':'Qwen-7B',
 'anthropic/claude-sonnet-5':'Claude-5','openai/gpt-5.6-luna':'GPT-5.6'}
def close(a,b): return abs(a-b)<1e-9
def cp(c,n,alpha=0.05):
    if n<=0 or c>=n: return 1.0
    if c==0: return 1.0-alpha**(1.0/n)
    lo,hi=0.0,1.0
    for _ in range(120):
        m=(lo+hi)/2
        t=sum(math.comb(n,k)*m**k*(1-m)**(n-k) for k in range(c+1))
        lo,hi=(m,hi) if t>alpha else (lo,m)
    return (lo+hi)/2
def items(recs):
    """per-item outcome F/R/D on the complete-case policy, with problem id"""
    out=[]
    for r in recs:
        if r.get('status')!='built': continue
        a=r.get('answers') or {}; dc=r.get('den_clean')
        if a.get('control') is None or dc is None or not close(a['control'],dc): continue
        at=a.get('treat')
        if at is None: continue
        o='F' if close(at,r['den_treat']) else ('R' if close(at,dc) else 'D')
        out.append((r['problem_id'],o))
    return out
P={}
for fn in sorted(os.listdir(D)):
    # Exclude *_grammar.json: it shares the same `model` field as its
    # canonical counterpart, so without this filter P[short_name] gets
    # silently clobbered by whichever file sorts last (found 2026-08-11,
    # same bug as denote_tdsc_verify.py's load_panel/mode_grid).
    if fn.startswith('full_') and '_grammar' not in fn:
        b=json.load(open(os.path.join(D,fn), encoding="utf-8"))
        if SHORT.get(b['model']): P[SHORT[b['model']]]=items(b['records'])
HEAD=['Llama-70B','Llama-8B','Qwen-72B','Qwen-7B','Claude-5','GPT-5.6']
truth={m:{'n':len(P[m]),'phi':sum(1 for _,o in P[m] if o=='F')/len(P[m]),
          'delta':sum(1 for _,o in P[m] if o=='D')/len(P[m]),
          'sigma':sum(1 for _,o in P[m] if o in 'FD')/len(P[m])} for m in HEAD}

print("="*72); print("E1  CERTIFY evaluated as a procedure (audit subsamples, one item per problem)")
print("="*72)
rng=random.Random(20260809); B=2000
print(f"  {'budget':>7}{'certified':>11}{'of pairs':>10}{'mis-certified':>15}")
for na in (10,20,30,50,100):
    acc=0; tot=0; wrong=0
    for _ in range(B):
        for i in range(len(HEAD)):
            for j in range(i+1,len(HEAD)):
                a,b_=HEAD[i],HEAD[j]
                if truth[a]['sigma']<=truth[b_]['sigma']: a,b_=b_,a
                pool={}
                for pid,o in P[a]: pool.setdefault(pid,[]).append(o)
                pids=list(pool)
                if len(pids)<na: continue
                samp=[rng.choice(pool[p]) for p in rng.sample(pids,na)]
                c=sum(1 for o in samp if o=='D')
                dbar=cp(c,na); g=truth[a]['sigma']-truth[b_]['sigma']
                tot+=1
                if g>min(truth[a]['sigma'],dbar):
                    acc+=1
                    if truth[a]['phi']<=truth[b_]['phi']: wrong+=1
    print(f"  {na:>7}{100*acc/max(tot,1):>10.1f}%{tot//B:>10}{wrong:>15}")

print()
print("="*72); print("E2  FAMILY-WISE COVERAGE of a k-system certified leaderboard")
print("="*72)
for k in (2,6,12):
    print(f"  k={k:>2}: union bound on joint coverage at per-system alpha=0.05 -> "
          f"{max(0,1-k*0.05):.2f};  at alpha/k -> 0.95")
print(f"\n  {'g':>7}{'n at a=.05':>12}{'n at a/k, k=6':>15}{'n at a/k, k=12':>16}{'ceil((3+ln k)/g), k=6':>24}")
for g in (0.186,0.102,0.082,0.043):
    def need(al):
        n=1
        while 1-al**(1.0/n)>=g: n+=1
        return n
    print(f"  {g:>7.3f}{need(0.05):>12}{need(0.05/6):>15}{need(0.05/12):>16}"
          f"{math.ceil((3+math.log(6))/g):>24}")
print("\n  envelope check, n = ceil(ln(k/alpha)/g) suffices at alpha=0.05:",
      all((lambda n: 1-(0.05/k)**(1.0/n)<g)(math.ceil(math.log(k/0.05)/g))
          for k in (1,2,6,12,50) for g in (0.2,0.1,0.05,0.02,0.005)))

print()
print("="*72); print("E3  EVASION: substituting derailment for stability, on real outcomes")
print("="*72)
tgt='Qwen-72B'; top='Llama-70B'
t=truth[tgt]; psi=1-t['sigma']
need_delta=truth[top]['sigma']-t['phi']
conv=(need_delta-t['delta'])/psi
print(f"  {tgt}: phi={t['phi']:.3f} delta={t['delta']:.3f} sigma={t['sigma']:.3f} psi={psi:.3f}")
print(f"  {top} reports sigma={truth[top]['sigma']:.3f} (rank 1 of 6)")
print(f"  To reach that sigma with phi UNCHANGED, {tgt} needs delta={need_delta:.3f},")
print(f"  i.e. converting {100*conv:.1f}% of its stable (restore) responses into derailments.")
print(f"  Its true propagation rate stays at {t['phi']:.3f}, BELOW {top}'s {truth[top]['phi']:.3f}.")
print()
print(f"  {'delta':>7}{'sigma':>8}{'phi':>7}{'sigma rank':>12}{'phi rank':>10}{'audit n to expose':>19}")
sig_all=sorted((truth[m]['sigma'] for m in HEAD),reverse=True)
phi_all=sorted((truth[m]['phi'] for m in HEAD),reverse=True)
for d in (0.024,0.10,0.15,0.20,0.224,0.30):
    s=t['phi']+d
    sr=1+sum(1 for m in HEAD if m!=tgt and truth[m]['sigma']>s)
    pr=1+sum(1 for m in HEAD if m!=tgt and truth[m]['phi']>t['phi'])
    n=1
    while cp(0,n)>=d: n+=1
    print(f"  {d:>7.3f}{s:>8.3f}{t['phi']:>7.3f}{sr:>12}{pr:>10}{n:>19}")
print("\n  A corruption test reports the sigma rank. No corruption protocol")
print("  separates the two columns. An audit of the sizes above does.")
