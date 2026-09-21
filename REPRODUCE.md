# Reproduction guide

## 1. Frozen-output verification

This is the fastest path and reproduces the numerical claims reported in the manuscript without downloading raw market data.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reproducibility/requirements.txt
cd reproducibility
python audit_final_results.py
```

Expected terminal endpoint:

```text
FINAL RESULT AUDIT PASS
```

Regenerate Section 6 outputs:

```bash
python make_section6_final.py \
  frozen_results/b5v4_instrument_results__equal_weight_full.csv \
  frozen_results/b5v4_overid_by_asset__equal_weight_full.csv \
  frozen_results/primary_ar_empty_row_diagnostic.csv \
  frozen_results/b5v4_instrument_results__pc1_train_full.csv \
  frozen_results/b5v4_overid_by_asset__pc1_train_full.csv \
  ../manuscript/generated
```

## 2. Full application from public Binance data

Raw Binance aggregate trades are not included in the repository. The code expects the public daily aggregate-trade files used by the frozen analysis.

Set the raw-data directory before starting Jupyter:

```bash
export CRYPTO_RAW_BINANCE="/absolute/path/to/raw_binance"
jupyter lab reproducibility/b5_instrument_two_baskets_v5_FINAL.ipynb
```

For the frozen full equal-weight specification use:

```python
QUICK_MODE = False
RESUME = False
PROXY_METHOD = "equal_weight"
```

For the frozen PC1 robustness specification use:

```python
QUICK_MODE = False
RESUME = False
PROXY_METHOD = "pc1_train"
```

The final notebook contains the finite-value handling used in the PC1 training branch. PC1 loadings are estimated only on the training fraction and then frozen for projection.

## 3. Structural-compatibility diagnostic

With `CRYPTO_RAW_BINANCE` defined:

```bash
cd reproducibility
python diagnose_primary_structural_compatibility.py
```

This diagnostic evaluates the 16 primary S1/lag-1 designs across 18 assets. It does not alter the confirmatory model.

## 4. Anderson–Rubin projection audit

```bash
cd reproducibility
python audit_fast_ar_projection.py
```

The release audit verifies the certified outer spectral-radius projection against 10,000 feasible synthetic matrices.

## 5. Monte Carlo and membership audit

The repository contains the Monte Carlo generator (`sim_iv.py`), the frozen Monte Carlo table (`mc_table.csv`), and the 10,000-replication membership audit output (`membership_audit_10k.txt`).

## 6. Manuscript build

A compiled PDF is provided. To rebuild the manuscript with a standard LaTeX installation:

```bash
cd manuscript
pdflatex paper2_jbes.tex
bibtex paper2_jbes
pdflatex paper2_jbes.tex
pdflatex paper2_jbes.tex
```

Before journal submission, the final Zenodo DOI should replace the DOI placeholder in the Data and code availability statement.
