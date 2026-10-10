# AIWQ CIDM Single Flagship v0.2 — L4 competition-investment contract

**Supersedes v0.1 four-model architecture screening.** One architecture and one seed only; trained neural controls are intentionally omitted.

## Stage Q0 — real data and model provenance
- ECMWF AI-WQ-package annual ERA5 7-day weekly mean: MSLP and TAS, 1984–2025, separate persistent annual caches.
- Strict Thursday issue, inputs 8 complete historical weeks ending at issue-1; prediction targets Monday at +18 and +25 days = days 19–25 and 26–32.
- Native 121×240 global grid, MSLP quintile output 2×5×121×240.
- 20-year historical climatology **proxy** generated exclusively from earlier years; it is NOT the official quantile climatology.
- Data split: train 2005–2016, val 2017–2019, independent proxy test 2020–2022; 2023 embargo; 2024–2025 independent official-mirror issue dates.
- L4-only preflight, time/coordinate contracts, Flux/SOC gradient smoke.

## Stage Q1 — train one strong CIDM model
- CIDM-S2S Flagship, approx. 1.019M trainable parameters (D=48, hidden=96); archived core, phase+time+space, antisymmetric Flux, Skew, Hidden, SOC, slow-fast conditioning and season/geography.
- BER disabled because no real forcing/port input exists. No extra neural baselines.
- Full 1.5° grid, 8 history weeks, 4 state transitions and separate week 3 / week 4 categorical heads.
- Loss: cosine-latitude weighted RPS. Validation-based early stop and one temperature-calibration parameter.
- Full optimizer/scheduler/RNG checkpoints every epoch, optional HF MODEL sync.

## Stage Q2 — only independent historical proxy, not leaderboard comparison
- Evaluate the *frozen validation-selected* checkpoint on 2020–2022 and record RPSS for both windows, by issue and year.

## Stage Q3 — ECMWF official independent historical mirror
- Run 10 issue Thursdays per year in 2024–2025, full official weekly observations, official 20-yr quintiles and official RPSS.
- Save per-week/per-window scores. Missing official data means BLOCKED, not PASS.
- Do not tune architecture or calibrator on official 2024–2025 once examined. B-route redesign requires fresh holdout/live weeks.

## Budget triage (project thresholds, NOT ECMWF rules)
- A: W3 mean RPSS >= .27 and W4 >= .20, >=70% positive issues per window, each year/window >=.08 -> possible PRO 6000 investment with known leaderboard uncertainty.
- B: positive but short of A -> improve before scale-up and evaluate with new holdout periods.
- C: neither window skillfully beats climatology or highest window mean < .03 -> stop this candidate/input route.
- These historical thresholds do not predict medals, cash, exact rankings, or future seasons.
- **ECMWF competition terms explicitly rule out monetary or in-kind awards.** The aspiration is leaderboard standing/official recognition.

## Versioned repositories
- GitHub: `Mr-wuff/AI-Weather-Quest` source and reports, provided cloud runtime `GITHUB_TOKEN`.
- Hugging Face model repo: `wuff-mann/AI_Weather_Quest` optimizer-complete checkpoints and reports, provided write-capable cloud runtime `HF_TOKEN`.
- Raw weather caches never auto-published.

## Official docs
- https://aiweatherquest.ecmwf.int/evaluation-system/
- https://aiweatherquest.ecmwf.int/leaderboards/
- https://aiweatherquest.ecmwf.int/terms-of-ai-weather-quest/
