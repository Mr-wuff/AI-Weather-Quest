# CIDM-MHC A13-FIX3 → A14 WB2 conditional information gate (2026-10-17)

## Verified user-uploaded A13-FIX3 result

- ZIP SHA256 `6864cf65e9d34decc70505c9d703cf5681f5b9fa147f9068a40a89cda1d96f88`; ZIP CRC PASS, both submitted archives identical.
- Real public WeatherBench 2 ERA5 1.5° conservative GCS source works. 2012–2015: four dates/year, 16 individual fields each for Z500 and SST, 121×240 grid.
- `Z500 = geopotential(level=500)/9.80665` in metres; ocean SST in °C with conservative land-sea mask, SST valid grid fraction 0.598416 (not an HTTP download loss).
- Historical source Sundays 12UTC → following Thursday 00UTC issue, 84-hour margin.
- **NO continuous weather windows, NO W3/W4 truth labels, NO CIDM optimization, NO official AI Weather Quest RPSS.** A13 Q0 data-source PASS only.
- No claim that retrospective ERA5 data would have been released in time for an operational issue.

## Previously archived evidence

- A12-R1 six-feature conditional MJO regression fails preregistered increment criterion on reused 2015 developer.
- October 10 L4 flagship historical MSLP official mirror (2024–2025): W3 RPSS 0.001480, W4 RPSS 0.007201. This stops one candidate, not all CIDM-v2 variants.

## A14 preregistration: no long CIDM training

1. **Q0 offline smoke**: exact Thursday issue and day 19–25 / day 26–32 windows, 121×240 IO, 28 contiguous six-hour points per week, masks, source, cache integrity.
2. **Q0 online pilot**: read one 2012 genuine retrospective input+weekly MSLP/TAS truth and one 2015 **input-only** case through the same public WB2 GCS. No NOAA dependencies.
3. **Q1** (only after Q0): train years 2008–2011, tune only on 2012, report development 2013–2014; 2015 is input-only, 2016+ remain unopened for future independent work. Purge labels crossing boundaries. Default 26 evenly sampled issues per year.
4. Inputs: surface MSLP/TAS snapshots Sunday 12UTC at lead -4 and -11 calendar days relative to issue Thursday, plus latest Sunday 12UTC Z500 and SST. Targets are 6-hourly ERA5 W3/W4 seven-day means, **not** future input predictors.
5. Coarsen to 3×4 large-area diagnostics. Compare season-only climatology, surface, surface+Z500, surface+SST, surface+both, and time-permuted sham-both Ridge probes. Ridge lambda [1,10,100,1000], amplitude [0,0.25,0.5,1] chosen on 2012 only. Train-only normalization.
6. Preliminary mechanism candidate only if at least 20 2012 and 40 2013–2014 cases; both W3 and W4 for at least one variable improve conditionally over surface by 0.5% and over sham, no other variable/window regression >2%. This gate *does not* authorize CIDM-v2 long training.
7. **Not official AIWQ scoring**: diagnostic 3×4-area deterministic latitude-weighted MSE, no PR weekly accumulation, no 121×240 five-quintile probability RPSS. A15 must use fresh independent periods and official scoring/provenance.

## Reproduction and sync

- Deliverable Notebook `CIDM_MHC_REAL_A14_P0_P1_WB2_ConditionalInfo_Colab_STANDALONE.ipynb` and stand-alone `a14_wb2_conditional_gate.py` provided in project handoff ZIP.
- Offline synthetic smoke and 121×240 xarray synthetic IO tests passed before PR. Actual A14 remote pilot and Q1 have not been run, so **there is no A14 predictive result yet**.
- Notebook can optionally publish *lightweight* source/reports to GitHub `Mr-wuff/AI-Weather-Quest` and HF model repo `wuff-mann/AI_Weather_Quest` after the user enables explicit runtime token sync. No raw annual ERA5, caches, secrets, or private datasets are to be published.
