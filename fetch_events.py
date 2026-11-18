import json, time, urllib.request as u, sys
def g(url):
    for i in range(4):
        try:
            return json.load(u.urlopen(u.Request(url,headers={'User-Agent':'research-script'}),timeout=40))
        except Exception as e:
            time.sleep(2*(i+1))
    return None
KEEP_M=['id','question','outcomePrices','outcomes','clobTokenIds','closedTime','umaResolutionStatus','volumeNum','groupItemTitle','negRisk','endDate','startDate','closed']
KEEP_E=['id','title','negRisk','negRiskMarketID','closedTime','volume','endDate','startDate']
MAXP=int(sys.argv[1]) if len(sys.argv)>1 else 40
n=0
with open('events_raw.jsonl','w') as f:
    for p in range(MAXP):
        ev=g(f"https://gamma-api.polymarket.com/events?closed=true&limit=50&offset={p*50}&order=volume&ascending=false")
        if not ev: print('stop at page',p); break
        for e in ev:
            r={k:e.get(k) for k in KEEP_E}
            r['markets']=[{k:m.get(k) for k in KEEP_M} for m in e.get('markets',[])]
            f.write(json.dumps(r)+'\n'); n+=1
        print(p,n,flush=True)
        time.sleep(0.3)
