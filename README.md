# GTA VI trophy rarity forecast — week one

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`:
forecast each of the 55 GTA VI trophies' Exophase earn rate at
2026-11-26T00:00:00Z as five quantiles (5/25/50/75/95%). Scored by
mean pinball loss on the logit scale.

## Model

`main.py` (Python 3.13, stdlib only, offline, deterministic) reads
each `<id>.in` and writes `<id>.out`. Times are hours since the
2026-11-19T00:00:00Z release; `t_C` is the checkpoint, `t_T` the
target. Consecutive identical house reads (same players tracked and
same rates — Exophase refreshes only every few hours) are collapsed
into one distinct read before any fitting.

**Branch 1 — flat history, early checkpoint.** If there is at most
one distinct read and `t_C ≤ 12 h`, the page has not refreshed yet
and the next refresh dilutes every non-100% rate. A launch-flood
dilution shift in logit space, `−4.44 · ln(t_T / t_C)`, is applied
to every trophy below 99.99% (4.44 from the kit's published first
refresh: 638 → 3,868 players between 3 h and 4.5 h after release).
Spread 0.6 logit units; trophies at ≥ 99.99% stay at 100% with
spread 0.05.

**Branch 2 — the z-map (the real forecast).** A trophy's earn rate
is the share of tracked players whose played hours exceed that
trophy's requirement, and played hours are approximately lognormal,
so the probit transform `z = Φ⁻¹(1 − r/100)` of each rate is an
affine function of the trophy's log requirement. The whole
z-vector therefore maps linearly from one read to another:
`z(t) = A(t)·z(C) + B(t)`, with the same A, B for all 55 trophies.
Each distinct read's z-vector is regressed on the checkpoint
read's (OLS over the 55 trophies) to get A(t), B(t); A and B are
each fit linearly against `ln(1+t)` over recent reads (12 h window,
widening to 24 h, 48 h, then all reads until ≥ 3 points), read at
the target, clamped to A ∈ [0.5, 1.3], B ∈ [−3.0, 0.5], and each
trophy is forecast as `r = 100·(1 − Φ(A_T·z_C + B_T))`. Spread is
`clamp(0.5 + 0.3·(ln(1+t_T) − ln(1+t_C)), 0.30, 1.20)` logit units.

**Branch 3 — fallback.** Per-trophy logit-linear fits on `ln(1+t)`,
slopes shrunk toward the median slope of trophies in the same
last-rate band (<1%, 1–3%, 3–10%, 10–30%, ≥30%) with weight
`nd/(nd+3)`, forecast re-anchored on the last observed rate.
If even that fails, persistence (last rate, spread
`0.5·sqrt(horizon days)`).

Quantiles: `q_τ = inv-logit(logit(median) + z_τ·spread)` with
`z = (−1.645, −0.674, 0, 0.674, 1.645)`, rounded half-up to 4
decimals, enforced non-decreasing. Φ⁻¹ is Acklam's rational
approximation; Φ uses `erf`.

## Validation (simulated launches)

On a player-level simulator built to match the two published
anchors (638 → 3,868 players at the first refresh; every non-100%
rate down at that refresh), mean pinball loss at the final
checkpoint (59.5 h → 168 h): **0.62** for this entry vs 1.73 for
the house curve baseline and 1.96 for persistence; on practice
cuts (3 h → 4.5 h): **0.43** vs 1.59 for the house baseline. The
z-map beats the house curve in every simulated world tested.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
