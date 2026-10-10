# CIDM-S2S · ECMWF AI Weather Quest

Research repository for the CIDM (Closed Information Dynamics Model) subseasonal probabilistic forecasting entry.

## First stage: L4 qualification before RTX PRO 6000 investment

Qualification targets **MSLP** quintile-probability forecasts for days **19–25** and **26–32**, evaluated as ranked probability skill score (RPSS). Real training data and official evaluation are sourced from ECMWF's AI-WQ-package.

The L4 pilot tests four architectures:
- CNN reference (~47.7K trainable parameters)
- CIDM Local-only (~31.1K; capacity-reduced ablation)
- CIDM Generic capacity replacement (~49.2K)
- CIDM Structured Flux / Hidden (~48.1K)

It uses strictly historical Thursday issues, train/validation/test chronology, three random seeds, a causally constructed 20-year **historical quantile proxy** for architecture screening, and a separately gated **official 2024–2025 evaluation** using ECMWF observations/quantiles and `work_out_RPSS`.

**Important:** Proxy RPSS is not the official competition score. An official test failure or missing registration credentials automatically blocks the PRO 6000 formal-training decision. No forecasts are automatically submitted to ECMWF.

## Where experiment artifacts go

- GitHub: code, notebooks, configuration, gates, lightweight results and reproducibility reports.
- Hugging Face **model**: [wuff-mann/AI_Weather_Quest](https://huggingface.co/wuff-mann/AI_Weather_Quest), training checkpoints, weights, and versioned experiment artifacts.
- Persistent cloud volume: raw ECMWF ERA5 annual NetCDF and local sample caches (not uploaded to the model repository).

At the outset the complete self-contained Notebook and accompanying `src/` files are distributed as `CIDM_AIWQ_Q0_Q3_L4_OfficialQualification_v01.ipynb` and the matching qualification release ZIP. Run once on the cloud L4 with write-capable environment credentials; it will sync results to GitHub/HF when those credentials are available.

## Official references
- https://aiweatherquest.ecmwf.int/
- https://aiweatherquest.ecmwf.int/leaderboards/
- https://aiweatherquest.ecmwf.int/evaluation-system/
- https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/training_data.html
- https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/forecast_evaluation.html

## Security

Never commit `AIWQ_PASSWORD`, `HF_TOKEN`, or `GITHUB_TOKEN`. Use cloud secrets.