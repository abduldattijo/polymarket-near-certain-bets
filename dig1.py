import json,bisect,math,random,collections
import numpy as np
from datetime import datetime
def ts(s):
    if not s: return None
    s=s.replace('Z','+00:00')
    if s.endswith('+00'): s+=':00'
    try: return datetime.fromisoformat(s).timestamp()
    except Exception: return None
meta={}
for l in open('events_raw.jsonl'):
    e=json.loads(l)
    for m in e['markets']: meta[m['id']]={'end':ts(m['endDate']) or ts(e['endDate']),'cl':ts(m['closedTime']) or ts(e['closedTime']),'vol':m.get('volumeNum') or 0,'title':e['title']}
rows=[]
for l in open('hist_full.jsonl'):
    r=json.loads(l); mt=meta.get(r['mid'])
    if not mt or not mt['end'] or not r['hist']: continue
    h=r['hist']; xs=[x[0] for x in h]; t=mt['end']-86400
    i=bisect.bisect_right(xs,t)-1
    if i<0 or t-xs[i]>2*86400: continue
    p=h[i][1]
    last=h[-1]
    rows.append(dict(p=p,won=r['won'],kind=r['kind'],eid=r['eid'],vol=mt['vol'],q=r['q'],title=mt['title'],end=mt['end'],cl=mt['cl'],age=t-xs[i],lastts=last[0],lastp=last[1],n=len(h),first=xs[0]))
print(len(rows))
mid=[r for r in rows if 0.2<=r['p']<=0.8 and r['kind']!='A']
print('mid priced non-multi markets',len(mid),'events',len({r['eid'] for r in mid}))
no_ret=[((1-r['won'])-(1-r['p']))/(1-r['p']) for r in mid]
print('buy No on every one: mean return',round(float(np.mean(no_ret)),3),'yes rate',round(np.mean([r['won'] for r in mid]),3),'mean price',round(np.mean([r['p'] for r in mid]),3))
def boot(vals,ids):
    ue=sorted(set(ids));ix=collections.defaultdict(list)
    for i,e in enumerate(ids): ix[e].append(i)
    v=np.array(vals); bs=[v[np.concatenate([ix[e] for e in random.choices(ue,k=len(ue))])].mean() for _ in range(500)]
    return np.percentile(bs,2.5),np.percentile(bs,97.5)
print('  95% range',[round(x,3) for x in boot(no_ret,[r['eid'] for r in mid])])
for name,f in [('volume<10k',lambda r:r['vol']<1e4),('10k-100k',lambda r:1e4<=r['vol']<1e5),('>=100k',lambda r:r['vol']>=1e5)]:
    s=[r for r in mid if f(r)]
    if s: print(f"  {name:12s} {len(s):4d} mkts {len({r['eid'] for r in s}):3d} ev  No return {np.mean([((1-r['won'])-(1-r['p']))/(1-r['p']) for r in s]):+.3f}  yes rate {np.mean([r['won'] for r in s]):.2f} price {np.mean([r['p'] for r in s]):.2f}")
# by year
by=collections.defaultdict(list)
for r in mid: by[datetime.fromtimestamp(r['end']).year].append(((1-r['won'])-(1-r['p']))/(1-r['p']))
print('by year',{y:(len(v),round(float(np.mean(v)),3)) for y,v in sorted(by.items())})
# is the price at t stale? how old and does series end before t+1d
print('price age (days) median',np.median([r['age'] for r in mid])/86400,' series ends before scheduled end:',np.mean([r['lastts']<r['end']-86400*0.5 for r in mid]))
c=collections.Counter(r['title'][:50] for r in mid); print(c.most_common(6))
random.seed(2)
for r in random.sample(mid,8): print(round(r['p'],2),r['won'],round(r['vol']),r['q'][:60],'| hist pts',r['n'],'| days before end the series ends',round((r['end']-r['lastts'])/86400,1))

print('--- small market description')
small=[r for r in mid if r['vol']<1e4]
import re
prop=re.compile(r'spread|o/u|moneyline|\bvs\.?\b|game \d|team total|1h|2h|1q|2q|3q|4q|1st|2nd|map|rebounds|points|yards|safety|touchdown|both teams|slay',re.I)
print('small',len(small),'zero volume',sum(r['vol']==0 for r in small)/len(small),'props-like titles',sum(bool(prop.search(r['q'])) for r in small)/len(small),'median hist points',np.median([r['n'] for r in small]),'<=13 pts',np.mean([r['n']<=13 for r in small]))
big=[r for r in mid if r['vol']>=1e5]
print('big props-like',sum(bool(prop.search(r['q'])) for r in big)/len(big),'median pts',np.median([r['n'] for r in big]))
