# v1.0.0 — Frozen reproducibility release

Initial public reproducibility release for *Contagion or Common Factor? Instrumental Identification and Robust Inference in High-Frequency Spillover Networks*.

## Included

- Final manuscript source and compiled PDF.
- Confirmatory eight-window equal-weight application.
- Full PC1 training-frozen robustness application.
- HAC Hansen-J diagnostics.
- Corrected lead-innovation placebo.
- Rowwise structural-compatibility diagnostic for the 16 primary designs.
- Analytic/HAC Anderson–Rubin inference and certified spectral-radius projection code.
- 10,000-replication projection/membership audits.
- Monte Carlo code and frozen outputs.
- Section 6/Table 4 regeneration script and frozen numerical source files.
- SHA-256 release checksums.

## Frozen empirical state

- 192/192 equal-weight main first stages have `F_eff >= 10`.
- `rho_2SLS < rho_naive` in 48/48 lag-1, 48/48 lag-2, 42/48 lag-5, and 4/48 lag-10 designs.
- 192/192 equal-weight and 192/192 PC1 cone-restricted joint AR sets are empty under `B >= 0`.
- No nonempty unbounded main AR cases occur.
- 136/288 primary asset-design rows are incompatible with the maintained nonnegative structure.

## Data policy

Raw Binance trades are publicly available but are not redistributed. Frozen derived outputs sufficient to regenerate the reported application table and numerical audits are included.
