"""Final analysis. Usage: python3 analysis.py end|closed
end    = price taken N days before the market's scheduled end date (set in advance)
closed = price taken N days before the recorded close time (can be set after the outcome is known)"""
import json, math, bisect, random, sys, collections, statistics as st
from datetime import datetime
BASIS=sys.argv[1] if len(sys.argv)>1 else 'end'
random.seed(5)
def ts(s):
    if not s: return None
    s=s.replace('Z','+00:00')
    if s.endswith('+00'): s+=':00'
    try: return datetime.fromisoformat(s).timestamp()
    except Exception: return None
meta={}
for l in open('events_raw.jsonl'):
    e=json.loads(l)
    for m in e['markets']: meta[m['id']]={'end':ts(m['endDate']) or ts(e['endDate']),'mclosed':ts(m['closedTime']),'eclosed':ts(e['closedTime'])}
R=[json.loads(l) for l in open('hist_full.jsonl')]
for r in R:
    mt=meta.get(r['mid'],{})
    r['t0']= mt.get('end') if BASIS=='end' else (mt.get('mclosed') or mt.get('eclosed'))
def price_at(h,t,tol=2*86400):
    if not h or t is None: return None
    xs=[x[0] for x in h]; i=bisect.bisect_right(xs,t)-1
    if i<0 or t-xs[i]>tol: return None
    return h[i][1]
A=collections.defaultdict(list)
for r in R:
    if r['kind']=='A': A[r['eid']].append(r)
print('BASIS',BASIS,'| markets',len(R),'| multi outcome events',len(A))
for lead in [1,7,30]:
    contract=[]; evs=[]; dropped=0
    for eid,ms in A.items():
        ps=[price_at(m['hist'],(m['t0'] or 0)-lead*86400 if m['t0'] else None) for m in ms]
        if any(p is None for p in ps): dropped+=1; continue
        s=sum(ps)
        if s<=0.5 or s>1.5: dropped+=1; continue
        probs=[p/s for p in ps]; wins=[m['won'] for m in ms]
        contract+=list(zip(probs,wins))
        pw=max(sum(p for p,w in zip(probs,wins) if w),1e-4); top=max(range(len(probs)),key=lambda i:probs[i])
        evs.append({'n':len(probs),'pw':pw,'top_p':probs[top],'top_won':wins[top]})
    if not evs: print('lead',lead,'none'); continue
    br=lambda c: st.mean((p-(1 if w else 0))**2 for p,w in c)
    big=[(p,w) for p,w in contract if p>=0.05]; low=[(p,w) for p,w in contract if p<0.05]
    print(f"\n== EVENTS, {lead}d before: {len(evs)} used, {dropped} dropped, {len(contract)} contracts")
    print(f" contracts <5%: {len(low)} ({len(low)/len(contract):.0%}); wins {sum(w for _,w in low)} ({sum(w for _,w in low)/len(low):.2%}); mean price {st.mean(p for p,_ in low):.2%}")
    print(f" contract Brier all {br(contract):.4f} | contracts >=5% only {br(big):.4f}")
    print(f" event: winner's mean prob {st.mean(e['pw'] for e in evs):.2f} vs uniform {st.mean(1/e['n'] for e in evs):.2f}; bits better {st.mean(math.log2(e['pw']*e['n']) for e in evs):.2f}; favourite hit {sum(e['top_won'] for e in evs)/len(evs):.0%} at mean price {st.mean(e['top_p'] for e in evs):.2f}")
    for lo,hi in [(0,.3),(.3,.5),(.5,.7),(.7,.9),(.9,1.01)]:
        b=[e for e in evs if lo<=e['top_p']<hi]
        if b: print(f"   favourite {lo:.1f}-{min(hi,1):.1f}: {len(b)} ev, won {sum(e['top_won'] for e in b)/len(b):.0%}, mean price {st.mean(e['top_p'] for e in b):.2f}")
def sides(lead):
    rows=[]
    for r in R:
        p=price_at(r['hist'],(r['t0']-lead*86400) if r['t0'] else None)
        if p is None: continue
        q=max(p,1-p)
        if q<0.99: continue
        rows.append({'q':q,'won':(r['won'] if p>=0.5 else not r['won']),'eid':r['eid'],'qt':r['q'],'yes':p>=0.5,'kind':r['kind']})
    return rows
def ret(x,rate=0.0):
    cost=x['q']+rate*x['q']*(1-x['q'])
    return ((1-cost)/cost) if x['won'] else -1.0
def boot(rows,B=2000):
    by=collections.defaultdict(list)
    for x in rows: by[x['eid']].append(x)
    ids=list(by); out=[]
    for _ in range(B):
        out.append(st.mean(ret(x) for i in random.choices(ids,k=len(ids)) for x in by[i]))
    out.sort(); return out[int(.025*B)],out[int(.975*B)]
for lead in [1,7,30]:
    rows=sides(lead)
    print(f"\n== NEAR CERTAIN SIDES (99%+), {lead}d before: {len(rows)} sides, {len({x['eid'] for x in rows})} events, {sum(not x['won'] for x in rows)} losses, loss events {len({x['eid'] for x in rows if not x['won']})}")
    for lo,hi,name in [(0.99,0.995,'99.0 to 99.5'),(0.995,0.999,'99.5 to 99.9'),(0.999,1.01,'99.9 plus'),(0.99,1.01,'all 99+')]:
        b=[x for x in rows if lo<=x['q']<hi]
        if not b: continue
        L=sum(not x['won'] for x in b); a,c=boot(b)
        print(f" {name}: n={len(b)} events={len({x['eid'] for x in b})} losses={L} ({L/len(b):.2%}) priced loss {1-st.mean(x['q'] for x in b):.2%} | mean return {st.mean(ret(x) for x in b):+.3%} CI {a:+.3%} to {c:+.3%} | after fees(0.05) {st.mean(ret(x,0.05) for x in b):+.3%}")
print('\nLosing sides at 1 day:')
for x in sides(1):
    if not x['won']: print(f"  {x['q']:.4f} {'Yes' if x['yes'] else 'No'} {x['kind']} | {x['qt'][:90]}")
