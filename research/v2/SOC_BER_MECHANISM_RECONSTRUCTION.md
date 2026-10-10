# CIDM-v2 SOC and BER mechanism clarification (2026-10-10)

Status: conceptual research hypotheses; no experimental result or performance claim. CIDM-v1 paper remains frozen.

## SOC: cross-scale information propagation, **not merely Fourier low-frequency mixing**
The original SOC motivation is bidirectional coarse-to-fine and fine-to-coarse information exchange and evolution. The archived `LatentSpectralOperatorCore` computes a low-frequency FFT residual and does not on its own implement explicit scale-to-scale flux. A3's task-dependent SOC value therefore neither disproves the original concept nor validates the existing implementation.

V2 candidate: a hierarchy of scale-specific latent states with explicitly paired exchanges, conservative/metric-aware on only the designated conserved channels; local same-scale antisymmetric flux plus cross-scale transfer with restriction/prolongation operators. Stage Gate: perturb one scale and measure causal change at another; broadband vs low-band spectral error; physical budget audits; zero/neutral degeneration where coupling is unnecessary; spherical metrics for global AIWQ.

## BER: causal external exchange, **not an arbitrary residual patch**
The BER motivation is exchange with an incompletely observed exterior so internal prediction need not explain all change. The archived BER primarily learns bounded corrections from latent state and internal dynamics, with optional port. Without identifiable issued forcing, the external/internal contribution is not identifiable; task dependence may reflect insufficient abstraction/regularity.

V2 candidate: separately tracked internal latent z, externally available issue-time forcing u, and uncertainty/memory r; controlled exchange B(z,r,u). Gate: causal availability audit, zero-port neutrality, driver permutation and delay tests, budget/regularity/step stability, task-adaptive inactive mode, probabilistic rather than deterministic treatment of unknown external events.

## A0 experiment status
The Colab standalone fix addresses a `src/cidm_v2_a0.py` missing-file assertion by embedding the original A0 source inside a single notebook. It **does not implement** the new SOC/BER proposals and should not be treated as a successful AI Weather Quest qualification. Any synthetic A0 pass still leaves global RTX PRO 6000 training blocked pending real S2S data and RPSS.

Full Colab FIX1 notebook and design note are included in the chat release `CIDM_v2_A0_Colab_STANDALONE_FIX1_Package.zip`.
