"""Can a model beat the market price? Predict at LEAD days before scheduled end using only earlier price history.
Grouped by event cross validation. Usage: python model_test.py [LEAD]"""
import json, bisect, sys, math, random
import numpy as np
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
LEAD=int(sys.argv[1]) if len(sys.argv)>1 else 7
random.seed(1); np.random.seed(1)
def ts(s):
    if not s: return None
    s=s.replace('Z','+00:00')
    if s.endswith('+00'): s+=':00'
    try: return datetime.fromisoformat(s).timestamp()
    except Exception: return None
meta={}
for l in open('events_raw.jsonl'):
    e=json.loads(l)
    for m in e['markets']: meta[m['id']]={'end':ts(m['endDate']) or ts(e['endDate']),'n':len(e['markets']),'neg':bool(e.get('negRisk'))}
def at(h,xs,t,tol=2*86400):
    i=bisect.bisect_right(xs,t)-1
    return h[i][1] if i>=0 and t-xs[i]<=tol else None
lg=lambda p: math.log(min(max(p,0.005),0.995)/(1-min(max(p,0.005),0.995)))
X=[];y=[];g=[];P=[];K=[];EARLY=[];T=[];names=['logit_p','logit_p_7d_earlier','logit_p_23d_earlier','change_7d','swing_14d','history_days','n_outcomes','neg_risk']
for l in open('hist_full.jsonl'):
    r=json.loads(l); mt=meta.get(r['mid'])
    if not mt or not mt['end'] or not r['hist']: continue
    h=r['hist']; xs=[x[0] for x in h]; t=mt['end']-LEAD*86400
    p=at(h,xs,t)
    if p is None: continue
    p1=at(h,xs,t-7*86400); p2=at(h,xs,t-23*86400)
    recent=[v for tt,v in h if t-14*86400<=tt<=t]
    X.append([lg(p),lg(p1) if p1 is not None else lg(p),lg(p2) if p2 is not None else lg(p),
              lg(p)-(lg(p1) if p1 is not None else lg(p)), (max(recent)-min(recent)) if recent else 0,
              (t-xs[0])/86400, mt['n'], 1 if mt['neg'] else 0])
    y.append(1 if r['won'] else 0); g.append(r['eid']); P.append(p); K.append(r['kind']); T.append(mt['end'])
    cl=ts(r.get('closed')); EARLY.append(1 if cl and cl<mt['end']-86400 else 0)
K=np.array(K);EARLY=np.array(EARLY);T=np.array(T);X=np.array(X);y=np.array(y);g=np.array(g);P=np.array(P)
print('LEAD',LEAD,'rows',len(y),'events',len(set(g)),'yes rate',round(y.mean(),3))
brier=lambda p:float(np.mean((p-y)**2))
def ll(p):
    p=np.clip(p,1e-3,1-1e-3); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
oof={'logistic':np.zeros(len(y)),'boosting':np.zeros(len(y)),'price+recalibrate':np.zeros(len(y))}
for tr,te in GroupKFold(5).split(X,y,g):
    m=LogisticRegression(C=1.0,max_iter=2000).fit(X[tr],y[tr]); oof['logistic'][te]=m.predict_proba(X[te])[:,1]
    b=GradientBoostingClassifier(n_estimators=150,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0).fit(X[tr],y[tr]); oof['boosting'][te]=b.predict_proba(X[te])[:,1]
    m0=LogisticRegression(C=1e6,max_iter=2000).fit(X[tr][:,:1],y[tr]); oof['price+recalibrate'][te]=m0.predict_proba(X[te][:,:1])[:,1]
print(f"{'market price':20s} Brier {brier(P):.4f} logloss {ll(P):.4f}")
for k,v in oof.items(): print(f"{k:20s} Brier {brier(v):.4f} logloss {ll(v):.4f}")
# event bootstrap of Brier difference (model minus price)
ev=sorted(set(g)); idx={e:np.where(g==e)[0] for e in ev}
def diff(sel,key):
    ii=np.concatenate([idx[e] for e in sel]); return float(np.mean((oof[key][ii]-y[ii])**2)-np.mean((P[ii]-y[ii])**2))
for key in ['logistic','boosting']:
    bs=[diff(random.choices(ev,k=len(ev)),key) for _ in range(500)]
    print(f"{key} minus price Brier: {diff(ev,key):+.5f}, 95% range {np.percentile(bs,2.5):+.5f} to {np.percentile(bs,97.5):+.5f}")
# betting: buy Yes if model > price+gap, buy No if model < price-gap; return per 1 staked
for key in ['boosting']:
    for gap in [0.05,0.10]:
        ret=[];eid=[]
        for i in range(len(y)):
            d=oof[key][i]-P[i]
            if d>gap and 0.02<P[i]<0.98: ret.append((y[i]-P[i])/P[i]); eid.append(g[i])
            elif d<-gap and 0.02<P[i]<0.98: ret.append(((1-y[i])-(1-P[i]))/(1-P[i])); eid.append(g[i])
        if not ret: print(key,gap,'no bets'); continue
        ret=np.array(ret);eid=np.array(eid); ue=sorted(set(eid)); ix={e:np.where(eid==e)[0] for e in ue}
        bs=[]
        for _ in range(500):
            s=np.concatenate([ix[e] for e in random.choices(ue,k=len(ue))]); bs.append(ret[s].mean())
        print(f"{key} bets when model differs from price by >{gap}: {len(ret)} bets, {len(ue)} events, mean return per 1 staked {ret.mean():+.3f}, 95% range {np.percentile(bs,2.5):+.3f} to {np.percentile(bs,97.5):+.3f}")

print("--- where do the boosting bets (gap>0.10) make money?")
rows=[]
for i in range(len(y)):
    d=oof['boosting'][i]-P[i]
    if abs(d)>0.10 and 0.02<P[i]<0.98:
        side='Yes' if d>0 else 'No'
        r_=(y[i]-P[i])/P[i] if d>0 else ((1-y[i])-(1-P[i]))/(1-P[i])
        rows.append((side,K[i],EARLY[i],P[i],r_,T[i],g[i]))
import collections
def show(name,f):
    sel=[r for r in rows if f(r)]
    if sel: print(f"{name:28s} {len(sel):5d} bets {len({r[6] for r in sel}):4d} events mean {np.mean([r[4] for r in sel]):+.3f}")
show('all',lambda r:True)
show('buy Yes',lambda r:r[0]=='Yes'); show('buy No',lambda r:r[0]=='No')
show('kind A (multi outcome)',lambda r:r[1]=='A'); show('other markets',lambda r:r[1]!='A')
show('closed >1d before sched end',lambda r:r[2]==1); show('not early closed',lambda r:r[2]==0)
show('Yes price <0.2',lambda r:r[3]<0.2); show('Yes price 0.2-0.8',lambda r:0.2<=r[3]<=0.8); show('Yes price >0.8',lambda r:r[3]>0.8)

print("--- time split: train on events ending before the median date, test on later ones (boosting, gap>0.10, other markets+all)")
med=np.median(T); tr=T<med; te=~tr
b=GradientBoostingClassifier(n_estimators=150,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0).fit(X[tr],y[tr])
pt=b.predict_proba(X[te])[:,1]; Pt=P[te]; yt=y[te]; gt=g[te]
print('test rows',te.sum(),'Brier price',round(float(np.mean((Pt-yt)**2)),4),'model',round(float(np.mean((pt-yt)**2)),4))
for spread in [0.0,0.02]:
    ret=[];ev_=[]
    for p_,m_,y_,e_ in zip(Pt,pt,yt,gt):
        d=m_-p_
        if abs(d)>0.10 and 0.02<p_<0.98:
            if d>0: ret.append((y_-(p_+spread))/(p_+spread))
            else: ret.append(((1-y_)-((1-p_)+spread))/((1-p_)+spread))
            ev_.append(e_)
    ret=np.array(ret);ev_=np.array(ev_);ue=sorted(set(ev_));ix={e:np.where(ev_==e)[0] for e in ue}
    bs=[ret[np.concatenate([ix[e] for e in random.choices(ue,k=len(ue))])].mean() for _ in range(500)]
    print(f"spread {spread}: {len(ret)} bets {len(ue)} events mean {ret.mean():+.3f} range {np.percentile(bs,2.5):+.3f} to {np.percentile(bs,97.5):+.3f}")
Q={}
for l in open('hist_full.jsonl'):
    r=json.loads(l); Q[r['eid']]=r['q']
c=collections.Counter()
for r in rows:
    if r[1]!='A' and r[0]=='No': c[Q.get(r[6],'?')[:70]]+=1
print('top events behind the No bets:'); [print(' ',n,q) for q,n in c.most_common(8)]

print("--- lead check: time-split test bets by feature value (history_days bucket) and side")
d_=pt-Pt; sel=(np.abs(d_)>0.10)&(Pt>0.02)&(Pt<0.98); Xt=X[te]
rets=np.where(d_>0,(yt-Pt)/Pt,((1-yt)-(1-Pt))/(1-Pt))
for name,f in [('history <2 days',Xt[:,5]<2),('history 2-14 days',(Xt[:,5]>=2)&(Xt[:,5]<14)),('history >=14 days',Xt[:,5]>=14),('buy No',d_<0),('buy Yes',d_>0)]:
    m_=sel&f
    if m_.sum(): print(f"{name:20s} {m_.sum():4d} bets mean {rets[m_].mean():+.3f}")
