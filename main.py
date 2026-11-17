"""gta6-twitch-ccu-forecast-01: forecast a launch day's hourly Twitch viewers, hours 2-24.

Model: pooled launch-day curves. Every comparable in the case is a past big launch with hourly
Twitch category counts around its own release day. For each comparable ("donor") whose release
day is fully sampled, take its release-day hourly counts as ratios to its own hour-1 print, in
UTC day coordinates: these launches unlock around the globe (midnight local, or a fixed UTC
hour), so their UTC day shapes carry the same anatomy - a first hour held up by the earliest
regions and the previous evening's audience, a build through the day, a peak in the late UTC
evening. Pool the donors' log-ratio curves with weights that favour donors whose pre-release
signature resembles the target's own (its launch "pop": hour 1 over its pre-release baseline,
and its "eve": how live the category already was in the hours before the day began, relative to
hour 1). The forecast is the pooled predictive distribution itself: per hour, the weighted
quantiles of the donors' log-ratios, smoothed over the day and widened by a calibration gain
fitted on held-out launches, then blended a little toward the persistence baseline as a hedge.
The target's pre-release baseline comes from its history (and, when the case's comparables
carry the target game's own pre-release category history, from that too).

Deterministic: pure functions of the input file, donors processed in name order, no randomness,
no clock. Robust: any failure falls back to the persistence baseline (flat at hour 1's print,
spread 0.15 * sqrt(h - 1)), and a valid output file is written for every case no matter what.

Run as the grader runs it: python3 -P -s /bot/main.py, standard library only, no network.
Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
BASELINE_SIGMA = 0.15

# Model constants, fitted on held-out launches (each comparable scored as a case against the
# remaining ones) and kept smooth on purpose: pooled, regularized choices, not per-case tweaks.
POP_SIGMA = 0.85       # weight sharpness in log-pop distance between donor and target
EVE_SIGMA = 1.0        # weight sharpness in log-eve distance
W_FLOOR = 0.05         # every donor keeps at least this share of its weight
SPREAD_GAIN = 1.1      # calibration multiplier on the pooled quantiles' log-deviations
BASE_BLEND = 0.06      # log-space blend toward the persistence baseline, as a hedge
TAIL_CAP = 3.0         # a quantile never sits further than this (log units) from the median
SMOOTH = (0.25, 0.5, 0.25)


def round2(x):
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def weighted_quantile(pairs, tau):
    """pairs: list of (value, weight). Weighted quantile, linear on the sorted values."""
    if not pairs:
        return None
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * total
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc >= target:
            if i == 0:
                return v
            prev_v = pairs[i - 1][0]
            prev_acc = acc - w
            if w <= 0:
                return v
            frac = (target - prev_acc) / w
            return prev_v + frac * (v - prev_v)
    return pairs[-1][0]


def parse_ts(s):
    """ISO 8601 UTC timestamp to epoch seconds; None when unparseable."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def hourly_values(points, day_start_ts):
    """Mean viewers per hour offset (in hours) from day_start, from a list of point dicts.

    Points carry 'from' (hourly rows) or 'at' (minute/history rows); each point stands for the
    hour it starts, so bucket by the point's own timestamp. Returns {offset: mean viewers}.
    """
    buckets = {}
    for p in points or []:
        try:
            t = parse_ts(p.get("from") or p.get("at") or "")
            v = p.get("viewers")
            if t is None or v is None:
                continue
            off = int((t - day_start_ts) // 3600)
            buckets.setdefault(off, []).append(float(v))
        except Exception:
            continue
    return {k: sum(vs) / len(vs) for k, vs in buckets.items()}


def pre_baseline(hour_map, lo=-96, hi=-25):
    """Median hourly viewers in the pre-release window [lo, hi) hours from day start."""
    vals = [v for k, v in hour_map.items() if lo <= k < hi and v > 0]
    return median(vals)


def eve_level(hour_map, h1):
    """Log of the mean viewers over the four hours before the day, relative to hour 1."""
    eve = [hour_map[k] for k in (-4, -3, -2, -1) if k in hour_map and hour_map[k] > 0]
    if not eve or h1 <= 0:
        return None
    return math.log((sum(eve) / len(eve)) / h1)


def donor_curve(comp):
    """A comparable's release-day log-ratio curve and pre-release signature, or None."""
    day = parse_ts(comp.get("day_start") or "")
    if day is None:
        return None
    hours = hourly_values(comp.get("hourly"), day)
    day_hours = {k: v for k, v in hours.items() if 0 <= k < 24}
    if len(day_hours) < 22 or 0 not in day_hours or day_hours[0] <= 0:
        return None
    base = pre_baseline(hours)
    if base is None or base <= 0:
        return None
    h1 = day_hours[0]
    logratios = {}
    for k in range(1, 24):
        if k in day_hours and day_hours[k] > 0:
            logratios[k] = math.log(day_hours[k] / h1)
    if len(logratios) < 20:
        return None
    return {"name": comp.get("name", ""), "logpop": math.log(h1 / base),
            "logeve": eve_level(hours, h1), "logratios": logratios}


def target_features(case, p1):
    """The target's (log pop, log eve) signature, from its own history where possible."""
    day = parse_ts(case.get("day_start") or "")
    base = None
    logeve = None
    if day is not None:
        hist = hourly_values(case.get("history_hourly") or case.get("history"), day)
        base = pre_baseline(hist)
        logeve = eve_level(hist, p1)
        if base is None:
            # Thin history (the final's recorder started two days out): use what there is
            # before the release day's own ramp (before hour -13, the first unlock's eve).
            vals = [v for k, v in hist.items() if k < -13 and v > 0]
            base = median(vals)
    if base is None:
        # The case's comparables may carry the target game's own pre-release category
        # history (the final case does, back to its extended-look premiere).
        name = ((case.get("game") or {}).get("name") or "").strip().lower()
        if name:
            for comp in case.get("comparables") or []:
                cname = (comp.get("name") or "").strip().lower()
                if cname and (name in cname or cname in name):
                    cday = parse_ts(comp.get("day_start") or "")
                    if cday is None:
                        continue
                    chours = hourly_values(comp.get("hourly"), cday)
                    vals = [v for v in chours.values() if v > 0]
                    b = median(vals[-24 * 21:]) if vals else None
                    if b:
                        base = b
                    break
    if base is None or base <= 0:
        return None
    return math.log(p1 / base), logeve


def smooth(xs):
    if len(xs) < 3:
        return list(xs)
    a, b, c = SMOOTH
    out = []
    for i in range(len(xs)):
        if i == 0:
            out.append((xs[0] * (b + a) + xs[1] * c) / (a + b + c))
        elif i == len(xs) - 1:
            out.append((xs[-2] * a + xs[-1] * (b + c)) / (a + b + c))
        else:
            out.append(a * xs[i - 1] + b * xs[i] + c * xs[i + 1])
    return out


def model_forecast(case, p1):
    """The pooled-curve forecast: 23 rows of quantiles, or None to fall back."""
    donors = []
    for comp in case.get("comparables") or []:
        d = donor_curve(comp)
        if d is not None:
            donors.append(d)
    if len(donors) < 6:
        return None
    donors.sort(key=lambda d: d["name"])
    feats = target_features(case, p1)
    weights = []
    for d in donors:
        if feats is None:
            w = 1.0
        else:
            logpop_t, logeve_t = feats
            dist2 = ((d["logpop"] - logpop_t) / POP_SIGMA) ** 2
            if logeve_t is not None and d["logeve"] is not None:
                dist2 += ((d["logeve"] - logeve_t) / EVE_SIGMA) ** 2
            w = W_FLOOR + (1.0 - W_FLOOR) * math.exp(-0.5 * dist2)
        weights.append(w)
    # Per hour: the weighted quantiles of the donors' log-ratios, at the five levels.
    per_hour = []
    for k in range(1, 24):
        pairs = [(d["logratios"][k], w) for d, w in zip(donors, weights) if k in d["logratios"]]
        if len(pairs) < 6:
            return None
        per_hour.append([weighted_quantile(pairs, tau) for tau in LEVELS])
    medians = smooth([row[2] for row in per_hour])
    out_rows = []
    # Smooth each level's deviation from the median across hours, then widen and blend.
    dev_by_level = []
    for li in range(len(LEVELS)):
        dev_by_level.append(smooth([per_hour[i][li] - per_hour[i][2] for i in range(23)]))
    for i, h in enumerate(range(2, 25)):
        med = medians[i]
        base_sig = BASELINE_SIGMA * math.sqrt(h - 1)
        qs = []
        for li, z in enumerate(Z):
            dev = dev_by_level[li][i] * SPREAD_GAIN
            dev = clamp(dev, -TAIL_CAP, TAIL_CAP)
            logq = med + dev
            base_logq = z * base_sig
            logq = (1.0 - BASE_BLEND) * logq + BASE_BLEND * base_logq
            qs.append(round2(clamp(p1 * math.exp(logq), 0.01, 1e9)))
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        out_rows.append({"hour": h, "q": qs})
    return {"hours": out_rows}


def baseline_forecast(p1, hours):
    rows = []
    for h in hours:
        qs = [round2(p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1))) for z in Z]
        rows.append({"hour": h, "q": [clamp(q, 0.01, 1e9) for q in qs]})
    return {"hours": rows}


def hour1_print(case):
    for k in case.get("known") or []:
        if k.get("hour") == 1 and k.get("viewers") is not None:
            return max(1.0, float(k["viewers"]))
    hh = case.get("history_hourly") or []
    if hh:
        try:
            return max(1.0, float(hh[-1]["viewers"]))
        except Exception:
            pass
    return 1.0


def forecast_case(case):
    p1 = hour1_print(case)
    hours = []
    for row in case.get("forecast") or []:
        try:
            hours.append(int(row["hour"]))
        except Exception:
            continue
    if not hours:
        hours = list(range(2, 25))
    try:
        out = model_forecast(case, p1)
    except Exception:
        out = None
    if out is None or len(out.get("hours", [])) != len(hours):
        return baseline_forecast(p1, hours)
    by_hour = {r["hour"]: r for r in out["hours"]}
    if all(h in by_hour for h in hours):
        return {"hours": [by_hour[h] for h in hours]}
    return baseline_forecast(p1, hours)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        out_path = OUTPUT_DIR / (path.stem + ".out")
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            forecast = baseline_forecast(1.0, list(range(2, 25)))
        try:
            out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")
        except Exception:
            pass


if __name__ == "__main__":
    main()
