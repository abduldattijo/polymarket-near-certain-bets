import bisect, json, random, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

PAGES, SAMPLE, LEAD_DAYS = 10, 1500, 1
random.seed(1)

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

# 1. Closed events with their markets (Gamma API, 50 events per page, biggest first).
markets = []
for page in range(PAGES):
    url = ("https://gamma-api.polymarket.com/events?closed=true&limit=50"
           f"&offset={page * 50}&order=volume&ascending=false")
    for event in get(url) or []:
        for m in event["markets"]:
            try:
                yes_won = float(json.loads(m["outcomePrices"])[0]) > 0.99
                no_won = float(json.loads(m["outcomePrices"])[0]) < 0.01
                token = json.loads(m["clobTokenIds"])[0]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            end = to_ts(m.get("endDate")) or to_ts(event.get("endDate"))
            if end and (yes_won or no_won):
                markets.append({"event": event["id"], "token": token, "yes_won": yes_won, "end": end})
markets = random.sample(markets, min(SAMPLE, len(markets)))

# 2. Daily price history for the Yes token (CLOB API), then the price LEAD_DAYS before the end date.
def price_before_end(m):
    data = get("https://clob.polymarket.com/prices-history"
               f"?market={m['token']}&interval=max&fidelity=1440") or {}
    points = data.get("history", [])
    when = m["end"] - LEAD_DAYS * 86400
    times = [p["t"] for p in points]
    i = bisect.bisect_right(times, when) - 1
    if i < 0 or when - times[i] > 2 * 86400:
        return None
    return points[i]["p"]

with ThreadPoolExecutor(12) as pool:
    prices = list(pool.map(price_before_end, markets))

# 3. Near certain side = whichever side is priced at 99 percent or more. Return per 1 staked.
bands = {"99.0 to 99.5": [], "99.5 to 99.9": [], "99.9 and above": []}
for m, p in zip(markets, prices):
    if p is None or max(p, 1 - p) < 0.99:
        continue
    side_price = max(p, 1 - p)
    won = m["yes_won"] if p >= 0.5 else not m["yes_won"]
    gain = (1 - side_price) / side_price if won else -1.0
    name = ("99.0 to 99.5" if side_price < 0.995 else
            "99.5 to 99.9" if side_price < 0.999 else "99.9 and above")
    bands[name].append((m["event"], won, gain))

print(f"{len(markets)} markets sampled")
for name, rows in bands.items():
    if rows:
        losses = sum(1 for _, won, _ in rows if not won)
        mean = sum(g for _, _, g in rows) / len(rows)
        print(f"{name} US cents: {len(rows)} bets, {len({e for e, _, _ in rows})} events, "
              f"{losses} losses, average gain {mean * 100:+.2f} US cents per 100 staked")
