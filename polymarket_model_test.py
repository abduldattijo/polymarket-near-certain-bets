import bisect, json, math, random, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

PAGES, SAMPLE, LEAD_DAYS, GAP = 40, 5000, 7, 0.10
random.seed(1); np.random.seed(1)

def get(url):
    for wait in (1, 3, 6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except Exception:
            time.sleep(wait)

def to_ts(text):
    text = (text or "").replace("Z", "+00:00")
    text += ":00" if text.endswith("+00") else ""
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None

# 1. Resolved markets from the biggest closed events.
markets = []
for page in range(PAGES):
    url = ("https://gamma-api.polymarket.com/events?closed=true&limit=50"
           f"&offset={page * 50}&order=volume&ascending=false")
    for event in get(url) or []:
        for m in event["markets"]:
            try:
                p_yes = float(json.loads(m["outcomePrices"])[0])
                token = json.loads(m["clobTokenIds"])[0]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            end = to_ts(m.get("endDate")) or to_ts(event.get("endDate"))
            if end and (p_yes > 0.99 or p_yes < 0.01):
                markets.append({"event": event["id"], "token": token, "yes_won": p_yes > 0.99, "end": end,
                                "n_out": len(event["markets"]), "volume": float(m.get("volumeNum") or 0)})
markets = random.sample(markets, min(SAMPLE, len(markets)))

def history(m):
    data = get("https://clob.polymarket.com/prices-history"
               f"?market={m['token']}&interval=max&fidelity=1440") or {}
    return [(p["t"], p["p"]) for p in data.get("history", [])]

with ThreadPoolExecutor(12) as pool:
    histories = list(pool.map(history, markets))

# 2. Features known LEAD_DAYS before the end date, from earlier prices only.
def logit(p):
    p = min(max(p, 0.005), 0.995)
    return math.log(p / (1 - p))

def price_at(times, points, when):
    i = bisect.bisect_right(times, when) - 1
    return points[i][1] if i >= 0 and when - times[i] <= 2 * 86400 else None

X, y, price, group, end_time, volume = [], [], [], [], [], []
for m, h in zip(markets, histories):
    if not h:
        continue
    times = [t for t, _ in h]
    when = m["end"] - LEAD_DAYS * 86400
    p = price_at(times, h, when)
    if p is None:
        continue
    p1 = price_at(times, h, when - 7 * 86400)
    p2 = price_at(times, h, when - 23 * 86400)
    p1 = p if p1 is None else p1
    p2 = p if p2 is None else p2
    recent = [v for t, v in h if when - 14 * 86400 <= t <= when]
    X.append([logit(p), logit(p1), logit(p2), logit(p) - logit(p1), max(recent) - min(recent),
              (when - times[0]) / 86400, m["n_out"]])
    y.append(int(m["yes_won"])); price.append(p); group.append(m["event"])
    end_time.append(m["end"]); volume.append(m["volume"])
X, y, price, group = np.array(X), np.array(y), np.array(price), np.array(group)
end_time, volume = np.array(end_time), np.array(volume)
print(f"{len(y)} markets with a price {LEAD_DAYS} days before the end, {len(set(group))} events")

def brier(p, target):
    return float(np.mean((p - target) ** 2))

def boosting():
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=0)

def bet_returns(model_p, p, won, volume_mask=None):
    """Buy Yes if the model is above the price by GAP, buy No if below. Return per 1 staked."""
    out = []
    for i in range(len(p)):
        d = model_p[i] - p[i]
        if abs(d) > GAP and 0.02 < p[i] < 0.98 and (volume_mask is None or volume_mask[i]):
            out.append((won[i] - p[i]) / p[i] if d > 0 else ((1 - won[i]) - (1 - p[i])) / (1 - p[i]))
    return np.array(out)

# 3. Random split by event: models against the market price.
oof = np.zeros(len(y)); oof_b = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, group):
    oof[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    oof_b[te] = boosting().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
print(f"random split by event: Brier price {brier(price, y):.4f}, logistic {brier(oof, y):.4f}, "
      f"boosting {brier(oof_b, y):.4f}")
r = bet_returns(oof_b, price, y)
print(f"  boosting bets: {len(r)}, average return per 1 staked {r.mean():+.3f}")

# 4. Time split: train on the earlier half of the markets, test on the later half.
early = end_time < np.median(end_time)
pred = boosting().fit(X[early], y[early]).predict_proba(X[~early])[:, 1]
r = bet_returns(pred, price[~early], y[~early])
print(f"time split: Brier price {brier(price[~early], y[~early]):.4f}, boosting {brier(pred, y[~early]):.4f}")
print(f"  boosting bets: {len(r)}, average return per 1 staked {r.mean():+.3f}")

# 5. Where do the random split bets make money? By final trading volume.
for name, mask in [("under 10,000", volume < 1e4), ("10,000 to 100,000", (volume >= 1e4) & (volume < 1e5)),
                   ("100,000 and above", volume >= 1e5)]:
    r = bet_returns(oof_b, price, y, mask)
    if len(r):
        print(f"  volume {name}: {len(r)} bets, average return {r.mean():+.3f}")
