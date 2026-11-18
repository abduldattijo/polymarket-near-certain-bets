import json, random, time, os, urllib.request as u
from concurrent.futures import ThreadPoolExecutor
random.seed(11)
def g(url):
    for i in range(4):
        try: return json.load(u.urlopen(u.Request(url,headers={'User-Agent':'research-script'}),timeout=30))
        except Exception: time.sleep(1.5*(i+1))
    return None
ev=[json.loads(l) for l in open('events_raw.jsonl')]
def clean(e):
    if not (e['negRisk'] and 3<=len(e['markets'])<=40): return False
    y=0
    for m in e['markets']:
        try: p=float(json.loads(m['outcomePrices'])[0])
        except: return False
        if p>0.99: y+=1
        elif p>0.01: return False
    return y==1
A=[(e,m,'A') for e in ev if clean(e) for m in e['markets']]
cleanids={e['id'] for e in ev if clean(e)}
rest=[]
for e in ev:
    if e['id'] in cleanids: continue
    for m in e['markets']:
        try: p=float(json.loads(m['outcomePrices'])[0])
        except: continue
        if (p>0.99 or p<0.01) and m.get('closedTime') or e['closedTime']:
            rest.append((e,m,'B'))
B=random.sample(rest,3000)
jobs=A+B
done=set()
if os.path.exists('hist_full.jsonl'):
    for l in open('hist_full.jsonl'): done.add(json.loads(l)['mid'])
jobs=[j for j in jobs if j[1]['id'] not in done]
print('A',len(A),'B',len(B),'todo',len(jobs),flush=True)
def one(j):
    e,m,k=j
    try: tok=json.loads(m['clobTokenIds'])[0]; won=float(json.loads(m['outcomePrices'])[0])>0.99
    except: return None
    h=g(f"https://clob.polymarket.com/prices-history?market={tok}&interval=max&fidelity=1440")
    if h is None: return None
    return {'eid':e['id'],'mid':m['id'],'kind':k,'q':m['question'],'won':won,'closed':m.get('closedTime') or e['closedTime'],'ev_closed':e['closedTime'],'hist':[(p['t'],p['p']) for p in h.get('history',[])]}
n=0
with ThreadPoolExecutor(12) as ex, open('hist_full.jsonl','a') as f:
    for r in ex.map(one,jobs):
        if r: f.write(json.dumps(r)+'\n')
        n+=1
        if n%500==0: print(n,flush=True); f.flush()
print('done',flush=True)
