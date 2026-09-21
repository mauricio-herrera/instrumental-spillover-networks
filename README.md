# Instrumental Spillover Networks

Reproducibility materials for the manuscript:

> **Contagion or Common Factor? Instrumental Identification and Robust Inference in High-Frequency Spillover Networks**  
> Mauricio Herrera-Marín, Universidad del Desarrollo, Santiago, Chile

This repository develops and audits an instrumental-variable framework for high-frequency spillover networks when a latent common driver is observed only through external noisy proxies. The workflow combines 2SLS, weak-instrument-robust Anderson–Rubin inference, Hansen overidentification diagnostics, and certified projection of rowwise confidence regions onto the network spectral radius.

## Main empirical findings frozen in v1.0.0

The confirmatory application uses 18 cryptocurrency assets across three stress windows and five control windows, two event thresholds, three external-instrument splits, and lags of 1, 2, 5, and 10 minutes.

- Main first stages satisfy `F_eff >= 10` in **192/192** equal-weight designs.
- The 2SLS spectral radius is below the naive estimate in **48/48** designs at lag 1, **48/48** at lag 2, **42/48** at lag 5, and **4/48** at lag 10.
- All 16 primary S1/lag-1 corrected point estimates remain above one.
- The cone-restricted joint Anderson–Rubin set under the maintained nonnegative network restriction `B >= 0` is empty in **192/192** equal-weight and **192/192** PC1 main designs; there are no nonempty unbounded cases.
- The rowwise structural-compatibility diagnostic identifies **136/288** incompatible asset-design rows in the 16 primary window/threshold designs.
- Equal-weight Hansen-J rejects at the 5% level in **21/288** tests; PC1 rejects in **28/288**.
- The corrected lead-innovation placebo has `F_eff >= 10` in **12/48** equal-weight and **13/48** PC1 designs.

The empirical conclusion is therefore not a claim that instrumental correction recovers a uniquely identified subcritical network. Rather, the results indicate that common market activity inflates short-horizon naive spillover estimates and that, once the external-proxy restrictions are imposed, the pure-excitation 18-asset nonnegative network class is too restrictive for the full data.

## Repository structure

```text
manuscript/
  paper2_jbes.tex                 LaTeX source
  paper2_jbes.pdf                 compiled manuscript
  refs.bib                        bibliography
  generated/                      Table 4 and Section 6 frozen numbers

reproducibility/
  b5_instrument_two_baskets_v5_FINAL.ipynb
  cascade_cs.py
  pipeline_v4.py
  make_section6_final.py
  audit_final_results.py
  frozen_results/                 confirmatory equal-weight/PC1 outputs
  v6_single_proxy_audit/          cross-fitted single-proxy audit
  ...                             Monte Carlo and validation scripts

REPRODUCIBILITY_AUDIT.md          final scientific/computational audit
REPRODUCE.md                      practical reproduction guide
RELEASE_NOTES_v1.0.0.md          frozen release notes
CITATION.cff                      citation metadata
.zenodo.json                      Zenodo metadata template
SHA256SUMS.txt                    release checksums
```

## Fast verification from frozen outputs

Create an environment and install the Python requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reproducibility/requirements.txt
```

Run the frozen-output audit:

```bash
cd reproducibility
python audit_final_results.py
```

Regenerate the application table and numerical summary:

```bash
python make_section6_final.py \
  frozen_results/b5v4_instrument_results__equal_weight_full.csv \
  frozen_results/b5v4_overid_by_asset__equal_weight_full.csv \
  frozen_results/primary_ar_empty_row_diagnostic.csv \
  frozen_results/b5v4_instrument_results__pc1_train_full.csv \
  frozen_results/b5v4_overid_by_asset__pc1_train_full.csv \
  ../manuscript/generated
```

See [`REPRODUCE.md`](REPRODUCE.md) for the full-data workflow.

## Data

Raw Binance aggregate-trade data are **not redistributed**. They are publicly available from Binance Data Vision. The application notebook and pipeline contain the exact asset universe, windows, event thresholds, and public-data URL pattern used by the analysis. Set the environment variable `CRYPTO_RAW_BINANCE` to a local directory containing the corresponding raw files before running the full application.

The frozen derived outputs used in the manuscript are included under `reproducibility/frozen_results/`, so the reported tables and numerical audits can be reproduced without redistributing raw trades.

## Related work

This repository is a companion, not a new version, of the earlier common-drive project:

- [`mauricio-herrera/apparent-criticality-cryptocon`](https://github.com/mauricio-herrera/apparent-criticality-cryptocon)

The earlier work studies apparent criticality and partial identification under latent common forcing. The present repository addresses the distinct problem of **external-proxy instrumental identification and weak-instrument-robust inference** when the common driver is not directly observed.

## Reproducibility status

Release v1.0.0 is a frozen submission artifact. The confirmatory specification is not altered after observing the final application results. In particular, incompatible assets are not removed and the maintained `B >= 0` restriction is not relaxed post hoc; such changes would constitute a separate exploratory analysis.

## Licenses

- Source code: MIT License (`LICENSE`).
- Documentation and frozen derived outputs: CC BY 4.0 (`LICENSE-DATA-DOCS`).
- The manuscript text remains © Mauricio Herrera-Marín; it is included here for scholarly reproducibility and version traceability.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A Zenodo DOI will be added after the v1.0.0 archival deposit is minted.
