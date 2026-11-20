"""GTA VI trophy rarity forecast — z-map forecaster.

Entry for make.fun competition gta6-trophy-rarity-forecast-01: forecast
each trophy's Exophase earn rate at the case target as five quantiles
(5/25/50/75/95%), scored by mean pinball loss on the logit scale.

Method (full write-up in README.md):
  * collapse consecutive identical house reads into distinct reads;
  * Branch 2 (the real forecast): a trophy's earn rate is the share of
    tracked players whose played hours exceed that trophy's
    requirement, played hours are approximately lognormal, so the
    probit transform of each rate is affine in the trophy's log
    requirement — the whole z-vector therefore maps linearly from
    one read to another. Regress each read's z-vector on the
    checkpoint read's, fit the map coefficients linearly in log-time
    over the recent reads, and read the map at the target;
  * Branch 3: per-trophy logit-linear fits, slopes shrunk toward the
    median slope of trophies in the same last-rate band, re-anchored
    on the last observed rate;
  * persistence fallback when neither fits.

Python 3.13, standard library only, offline, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

RATE_LO, RATE_HI = 0.01, 99.99


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(when):
    return (parse_time(when) - RELEASE).total_seconds() / 3600.0


def clamp_rate(rate):
    return min(RATE_HI, max(RATE_LO, rate))


def logit_pct(rate):
    r = clamp_rate(rate)
    return math.log(r / (100.0 - r))


def inv_logit_pct(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam's rational approximation to the inverse normal CDF.
_P_LOW = 0.02425
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)


def norm_ppf(p):
    p = min(1.0 - 1e-12, max(1e-12, p))
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        den = ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
        return num / den
    if p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
        den = (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
    den = ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    return -num / den


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def quantiles(median, spread):
    base = logit_pct(median)
    out = []
    prev = None
    for z in Z:
        q = round4(inv_logit_pct(base + z * spread))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def distinct_reads(history):
    """Collapse consecutive identical (players, rates) reads into one."""
    distinct = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if distinct and distinct[-1][0] == key:
            continue
        distinct.append((key, read))
    return [read for _, read in distinct]


def ols(xs, ys):
    """Ordinary least squares; returns (slope, intercept) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def last_rates(reads, n_trophies):
    """Last non-null rate per trophy, scanning reads newest first."""
    out = [None] * n_trophies
    for read in reversed(reads):
        rates = read.get("rates") or []
        for i in range(min(n_trophies, len(rates))):
            if out[i] is None and rates[i] is not None:
                out[i] = rates[i]
        if all(v is not None for v in out):
            break
    return [0.0 if v is None else v for v in out]


def z_of_rate(rate):
    return norm_ppf(1.0 - clamp_rate(rate) / 100.0)


FLOOD_ALPHA = 4.44
FLOOD_SPREAD = 0.6


def flood_prior(reads, last, t_c, t_t):
    """Branch 1 — launch-flood dilution for flat early histories.

    Returns (medians, spreads) or None when the history has more
    than one distinct read or the checkpoint is past 12 h.
    """
    if len(reads) > 1 or t_c > 12.0 or t_c <= 0:
        return None
    shift = -FLOOD_ALPHA * math.log(t_t / t_c)
    medians, spreads = [], []
    for rate in last:
        if rate >= 99.99:
            medians.append(100.0)
            spreads.append(0.05)
        else:
            medians.append(min(RATE_HI, max(RATE_LO, inv_logit_pct(logit_pct(rate) + shift))))
            spreads.append(FLOOD_SPREAD)
    return medians, spreads


def zmap(reads, last, t_c, t_t):
    """Branch 2 — the z-map. Returns (medians, spreads) or None."""
    n = len(last)
    z_c = [z_of_rate(r) for r in last]
    points = []  # (t_hours, A, B) per distinct read except the last
    for read in reads[:-1]:
        rates = read.get("rates") or []
        xs, ys = [], []
        for i in range(min(n, len(rates))):
            if rates[i] is None:
                continue
            xs.append(z_c[i])
            ys.append(z_of_rate(rates[i]))
        fit = ols(xs, ys)
        if fit is None:
            continue
        points.append((hours_since_release(read["at"]), fit[0], fit[1]))
    recent = None
    for window_h in (12.0, 24.0, 48.0, float("inf")):
        cand = [p for p in points if p[0] >= t_c - window_h]
        if len(cand) >= 3:
            recent = cand
            break
        recent = cand
    if recent is None or len(recent) < 2:
        return None
    xs = [math.log1p(p[0]) for p in recent]
    a_fit = ols(xs, [p[1] for p in recent])
    b_fit = ols(xs, [p[2] for p in recent])
    if a_fit is None or b_fit is None:
        return None
    x_t = math.log1p(t_t)
    a_t = min(1.3, max(0.5, a_fit[0] * x_t + a_fit[1]))
    b_t = min(0.5, max(-3.0, b_fit[0] * x_t + b_fit[1]))
    spread = min(1.20, max(0.30, 0.5 + 0.3 * (math.log1p(t_t) - math.log1p(t_c))))
    medians, spreads = [], []
    for i in range(n):
        if last[i] >= 99.99:
            medians.append(100.0)
            spreads.append(0.05)
        else:
            medians.append(min(RATE_HI, max(RATE_LO, 100.0 * (1.0 - norm_cdf(a_t * z_c[i] + b_t)))))
            spreads.append(spread)
    return medians, spreads


def band_index(rate):
    if rate < 1.0:
        return 0
    if rate < 3.0:
        return 1
    if rate < 10.0:
        return 2
    if rate < 30.0:
        return 3
    return 4


def band_forecast(reads, last, t_c, t_t):
    """Branch 3 — per-trophy logit-linear fits with band shrinkage.

    Returns (medians, spreads) or None when too few distinct reads.
    """
    nd = len(reads)
    if nd < 3:
        return None
    n = len(last)
    dx = math.log1p(t_t) - math.log1p(t_c)
    xs_all = [math.log1p(hours_since_release(r["at"])) for r in reads]
    slopes, resid_std = [], []
    for i in range(n):
        xs, ys = [], []
        for read, x in zip(reads, xs_all):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                xs.append(x)
                ys.append(logit_pct(rates[i]))
        fit = ols(xs, ys) if len(xs) >= 3 else None
        if fit is None:
            slopes.append(None)
            resid_std.append(None)
            continue
        slope, intercept = fit
        slopes.append(slope)
        ss = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        resid_std.append(math.sqrt(ss / (len(xs) - 2)) if len(xs) > 2 else 0.0)
    fitted = [s for s in slopes if s is not None]
    if not fitted:
        return None
    overall = median_of(fitted)
    band_medians = []
    for b in range(5):
        vals = [slopes[i] for i in range(n)
                if slopes[i] is not None and band_index(last[i]) == b]
        band_medians.append(median_of(vals) if vals else overall)
    w = nd / (nd + 3.0)
    medians, spreads = [], []
    for i in range(n):
        if slopes[i] is None:
            medians.append(clamp_rate(last[i]))
            spreads.append(0.5)
            continue
        shrunk = w * slopes[i] + (1.0 - w) * band_medians[band_index(last[i])]
        med = inv_logit_pct(logit_pct(last[i]) + shrunk * dx)
        medians.append(min(RATE_HI, max(RATE_LO, med)))
        sd = resid_std[i] if resid_std[i] is not None else 0.0
        spreads.append(min(1.2, max(0.25, 1.0 * sd * math.sqrt(1.0 + dx))))
    return medians, spreads


def persistence(last, horizon_minutes):
    """Fallback: last rate, spread widening like a random walk."""
    spread = 0.5 * math.sqrt(horizon_minutes / 1440.0)
    return [clamp_rate(r) for r in last], [spread] * len(last)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    reads = distinct_reads(case["history"])
    n = len(case["trophies"])
    last = last_rates(reads, n)
    t_c = hours_since_release(case["at"])
    t_t = hours_since_release(case["target"])
    result = flood_prior(reads, last, t_c, t_t)
    if result is None:
        result = zmap(reads, last, t_c, t_t)
    if result is None:
        result = band_forecast(reads, last, t_c, t_t)
    if result is None:
        result = persistence(last, case["horizon_minutes"])
    medians, spreads = result
    rows = []
    for trophy, med, spr in zip(case["trophies"], medians, spreads):
        rows.append({"id": trophy["id"], "q": quantiles(med, spr)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
