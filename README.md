# GTA VI Twitch CCU Forecaster

[![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/onnyh/gta6forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/onnyh/gta6forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for make.fun's [GTA VI Twitch CCU forecast](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast, hour by hour, how many people watch Grand Theft Auto VI live on Twitch over its
release day, 2026-11-19 (UTC), after the first hour — five quantiles (5%, 25%, 50%, 75%, 95%)
for each of hours 2 to 24.

## How it works

`main.py` (Python 3, standard library only) is run once per case. It reads the case file —
hour 1's print, the recorded history up to that checkpoint, and 38–42 past big launches with
their full hourly Twitch counts around their release days — and writes one forecast file.

The model pools the past launches' release-day curves:

- Each comparable with a fully sampled release day becomes a **donor**: its day's hourly counts
  taken as ratios to its own hour-1 print, in UTC day coordinates (these launches unlock around
  the globe, so their UTC shapes share one anatomy: earliest regions first, a build through the
  day, a late-evening peak).
- Donors are **weighted by resemblance to the target's pre-release signature** — its launch
  "pop" (hour 1 over its pre-release baseline) and its "eve" (how live the category already was
  in the four hours before the day began, relative to hour 1).
- The forecast is the **pooled predictive distribution**: per hour, the weighted quantiles of
  the donors' log-ratios, smoothed over the day, widened by a calibration gain fitted on
  held-out launches, blended slightly toward the persistence baseline as a hedge, with the
  outer quantiles winsorized so no line sits more than e³ ≈ 20× from the median — no launch in
  the donor set whose first hour was a real audience ever ranged further.

The program is deterministic (same case in, same forecast out, byte for byte) and defensive:
any failure falls back to the persistence baseline — flat at hour 1's print with a
0.15·√(h−1) log spread — so a valid forecast file is written for every case no matter what.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `<id>.out` per case to the output
directory. No network, no dependencies.
