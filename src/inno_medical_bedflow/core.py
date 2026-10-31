"""Bed occupancy forecast (7 days), discharge candidates and overload alerts.

Lab prototype on SYNTHETIC data only (TRL 4).
"""
import numpy as np

WARDS = {"icu": (12, 4.5), "surgery": (30, 5.0), "internal": (40, 6.0), "ortho": (20, 7.0)}
DAYS = 400
ORIGINS = list(range(220, 380, 4))

CRITERIA = [
    ("mae_gain", ">=", 0.10, "7-day census MAE improvement over persistence"),
    ("overload_recall", ">=", 0.70, "recall of days above 95% occupancy (alert at 90% forecast)"),
    ("overload_precision", ">=", 0.50, "precision of overload alerts"),
]


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    dow = np.array([1.15, 1.2, 1.1, 1.05, 0.9, 0.7, 0.8])
    pts = []
    for w, (cap, mean_los) in WARDS.items():
        lam0 = cap * 0.9 / mean_los
        for d in range(DAYS):
            for _ in range(rng.poisson(lam0 * dow[d % 7])):
                los = int(max(1, np.ceil(rng.lognormal(np.log(mean_los) - 0.18, 0.6))))
                pts.append({"ward": w, "admit": d, "los": los, "discharge": d + los,
                            "ready_from": (d + los - int(rng.integers(0, 3))) if rng.random() < 0.85 else None})
    return pts


def census(pts, ward, day):
    return sum(1 for p in pts if p["ward"] == ward and p["admit"] <= day < p["discharge"])


def survival(pts, ward, upto):
    los = np.array([p["los"] for p in pts if p["ward"] == ward and p["discharge"] <= upto])
    mx = int(los.max()) + 30
    return np.array([(los > k).mean() for k in range(mx)])


def forecast_census(pts, ward, T, horizon=7):
    """Uses only information available at the end of day T (patients admitted <= T, stays discharged <= T)."""
    S = survival(pts, ward, T)
    s = lambda k: S[min(k, len(S) - 1)]
    cur = [T - p["admit"] for p in pts if p["ward"] == ward and p["admit"] <= T < p["discharge"]]
    hist = np.zeros(7)
    cnt = np.zeros(7)
    for d in range(max(0, T - 55), T + 1):
        hist[d % 7] += sum(1 for p in pts if p["ward"] == ward and p["admit"] == d); cnt[d % 7] += 1
    lam = hist / np.maximum(cnt, 1)
    out = []
    for k in range(1, horizon + 1):
        stay = sum(s(e + k) / max(s(e), 1e-6) if s(e) > 1e-6 else 0.0 for e in cur)
        # note: a patient with elapsed e (days since admit) is still present k days later if LOS > e + k
        new = sum(lam[(T + j) % 7] * s(k - j) for j in range(1, k + 1))
        out.append(stay + new)
    return np.array(out)


def discharge_candidates(pts, ward, T, p75=None):
    S = survival(pts, ward, T)
    los = np.array([p["los"] for p in pts if p["ward"] == ward and p["discharge"] <= T])
    thr = p75 if p75 is not None else float(np.percentile(los, 60))
    return [p for p in pts if p["ward"] == ward and p["admit"] <= T < p["discharge"] and (T - p["admit"] + 1) >= thr and p["ready_from"] is not None and p["ready_from"] <= T]


def overload_alerts(forecast, capacity, margin=0.0):
    return [k + 1 for k, v in enumerate(forecast) if v > capacity * (1 - margin)]


def evaluate(seed=0):
    pts = simulate(seed)
    em, eb = [], []
    tp = fn = fp = 0
    dp = dn = 0
    for w, (cap, _) in WARDS.items():
        for T in ORIGINS:
            f = forecast_census(pts, w, T); a = np.array([census(pts, w, T + k) for k in range(1, 8)])
            em.append(np.abs(f - a).mean()); eb.append(np.abs(census(pts, w, T) - a).mean())
            pred = np.array(f) > cap * 0.90; true = a > cap * 0.95
            tp += int((pred & true).sum()); fn += int((~pred & true).sum()); fp += int((pred & ~true).sum())
            for p in discharge_candidates(pts, w, T):
                dn += 1; dp += int(p["discharge"] - T <= 3)
    return {"mae_gain": float(1 - np.mean(em) / np.mean(eb)), "overload_recall": tp / max(tp + fn, 1),
            "overload_precision": tp / max(tp + fp, 1), "discharge_precision": dp / max(dn, 1), "n_overload_days": tp + fn}


def generate(seed=0):
    pts = simulate(seed)
    return {"stays": [dict(stay_id="S%05d" % i, **p) for i, p in enumerate(pts)],
            "wards": [dict(ward=w, beds=c, mean_los=m) for w, (c, m) in WARDS.items()]}


def demo(seed=0):
    pts = simulate(seed)
    f = forecast_census(pts, "icu", 300)
    return "ICU 7-day census forecast: %s (beds=%d)\nOverload days: %s" % (f.round(1).tolist(), WARDS["icu"][0], overload_alerts(f, WARDS["icu"][0]))
