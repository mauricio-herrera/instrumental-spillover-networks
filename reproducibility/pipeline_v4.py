# pipeline_v4.py — minimal extraction from crypto_confirmatory_V4.ipynb; V4 cells 1--5.


# ---- V4 cell 1 ----

from pathlib import Path
from datetime import datetime, timedelta
import io
import json
import math
import time
import warnings
import zipfile
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

QUICK_MODE = False
RUN_DOWNLOAD = True
FORCE_REDOWNLOAD = False
RANDOM_SEED = 20260713

# A final run should use at least 500 bootstrap replicates.
BOOTSTRAP_REPS = 40 if QUICK_MODE else 500
NULL_REPS = 20 if QUICK_MODE else 200
BOOTSTRAP_BLOCK_MINUTES = 180
MIN_EVENT_COUNT_PER_ASSET = 12

ROOT = Path.cwd()
DATA = ROOT / "crypto_common_drive_data_V3"
RAW = DATA / "raw_binance"
OUT = ROOT / "crypto_common_drive_outputs_V3"
TABLES = OUT / "tables"
FIG = OUT / "figures"
CACHE = OUT / "cache"

for p in [RAW, OUT, TABLES, FIG, CACHE]:
    p.mkdir(parents=True, exist_ok=True)

SYMBOLS_QUICK = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
SYMBOLS_FULL = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
    "TRXUSDT", "DOTUSDT", "ATOMUSDT", "BCHUSDT", "ETCUSDT",
    "FILUSDT", "NEARUSDT", "UNIUSDT"
]
SYMBOLS = SYMBOLS_QUICK if QUICK_MODE else SYMBOLS_FULL

# Shock windows and several controls. The full run evaluates all windows.
WINDOWS_FULL = [
    {"name": "Terra_Luna", "kind": "shock", "start": "2022-05-09", "end": "2022-05-12"},
    {"name": "FTX_collapse", "kind": "shock", "start": "2022-11-08", "end": "2022-11-10"},
    {"name": "Banking_crypto", "kind": "shock", "start": "2023-03-10", "end": "2023-03-13"},
    {"name": "Control_2022_Aug_A", "kind": "control", "start": "2022-08-08", "end": "2022-08-10"},
    {"name": "Control_2022_Aug_B", "kind": "control", "start": "2022-08-22", "end": "2022-08-24"},
    {"name": "Control_2022_Oct", "kind": "control", "start": "2022-10-10", "end": "2022-10-12"},
    {"name": "Control_2023_Jan", "kind": "control", "start": "2023-01-16", "end": "2023-01-18"},
    {"name": "Control_2023_Feb", "kind": "control", "start": "2023-02-06", "end": "2023-02-08"},
]
WINDOWS = [
    next(w for w in WINDOWS_FULL if w["name"] == "FTX_collapse"),
    next(w for w in WINDOWS_FULL if w["name"] == "Control_2022_Aug_A"),
] if QUICK_MODE else WINDOWS_FULL

# Keep QUICK_MODE small but diagnostic. The full grid is intentionally broader.
DESIGN_GRID_QUICK = [
    dict(event_type="volume_burst", q=0.985, bin="1min", lags=60, half_life=10,
         drive_mode="loo_market_activity", drive_timing="contemporaneous"),
    dict(event_type="volume_burst", q=0.985, bin="1min", lags=60, half_life=10,
         drive_mode="loo_market_activity", drive_timing="lag1"),
    dict(event_type="volatility_burst", q=0.985, bin="1min", lags=60, half_life=10,
         drive_mode="loo_market_activity", drive_timing="contemporaneous"),
    dict(event_type="volatility_burst", q=0.985, bin="1min", lags=60, half_life=10,
         drive_mode="loo_market_activity", drive_timing="lag1"),
    dict(event_type="volume_burst", q=0.985, bin="1min", lags=60, half_life=10,
         drive_mode="btc_eth_reference", drive_timing="contemporaneous"),
]

DESIGN_GRID_FULL = []
for event_type in ["volume_burst", "volatility_burst", "combined_burst", "imbalance_burst"]:
    for q in [0.98, 0.985, 0.99]:
        for bin_size, lags, half_life in [("1min", 60, 10), ("5min", 36, 4)]:
            for drive_mode in ["loo_market_activity", "btc_eth_reference", "global_covariate_pca"]:
                for drive_timing in ["contemporaneous", "lag1"]:
                    DESIGN_GRID_FULL.append(dict(
                        event_type=event_type, q=q, bin=bin_size, lags=lags,
                        half_life=half_life, drive_mode=drive_mode,
                        drive_timing=drive_timing
                    ))

DESIGN_GRID = DESIGN_GRID_QUICK if QUICK_MODE else DESIGN_GRID_FULL

print("QUICK_MODE:", QUICK_MODE)
print("Symbols:", len(SYMBOLS), SYMBOLS)
print("Windows:", [w["name"] for w in WINDOWS])
print("Designs per window:", len(DESIGN_GRID))
print("Bootstrap reps:", BOOTSTRAP_REPS)


# ---- V4 cell 2 ----

# =============================================================================
# UTILITIES
# =============================================================================

def date_range_inclusive(start, end):
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    d = d0
    while d <= d1:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out

def simple_table_markdown(df, index=False, floatfmt=".6g"):
    if df is None or len(df) == 0:
        return "_No rows._"
    df2 = df.reset_index() if index else df.copy()
    cols = list(df2.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df2.iterrows():
        vals = []
        for x in row:
            if pd.isna(x):
                vals.append("")
            elif isinstance(x, (float, np.floating)):
                vals.append(format(float(x), floatfmt))
            else:
                vals.append(str(x))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def safe_standardize_df(df):
    df = df.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    med = df.median(axis=0)
    mad = (df - med).abs().median(axis=0).replace(0.0, np.nan)
    z = (df - med) / (1.4826 * mad)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10, 10)

def first_pc_score(df):
    X = safe_standardize_df(df).to_numpy(dtype=float)
    if X.shape[1] == 0:
        return np.zeros(X.shape[0])
    X -= X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    score = X @ vt[0]
    # Orient the score toward aggregate activity for interpretability.
    agg = X.mean(axis=1)
    if np.corrcoef(score, agg)[0, 1] < 0:
        score = -score
    score = (score - np.mean(score)) / (np.std(score) + 1e-12)
    return score

def spectral_radius(B):
    vals = np.linalg.eigvals(np.asarray(B, dtype=float))
    return float(np.max(np.abs(vals))) if len(vals) else np.nan

def rank1_share(A):
    s = np.linalg.svd(np.asarray(A, dtype=float), compute_uv=False)
    den = np.sum(s ** 2)
    return float((s[0] ** 2) / den) if den > 0 else np.nan

def exp_kernel_weights(lags, half_life):
    ell = np.arange(1, lags + 1)
    w = np.exp(-np.log(2.0) * ell / max(float(half_life), 1e-6))
    return w / (w.sum() + 1e-12)

def build_filtered_histories(Y, lags=60, half_life=10):
    Y = np.asarray(Y, dtype=float)
    T, K = Y.shape
    w = exp_kernel_weights(lags, half_life)
    H = np.zeros_like(Y, dtype=float)
    for ell in range(1, lags + 1):
        H[ell:] += w[ell - 1] * Y[:-ell]
    return H

def circular_shift_columns(Y, rng, min_shift=120):
    Y = np.asarray(Y)
    T, K = Y.shape
    out = np.empty_like(Y)
    for k in range(K):
        lo = min(min_shift, max(1, T // 4))
        shift = int(rng.integers(lo, max(lo + 1, T - lo)))
        out[:, k] = np.roll(Y[:, k], shift)
    return out


# ---- V4 cell 3 ----

# =============================================================================
# BINANCE AGGREGATE TRADES
# =============================================================================

AGGTRADE_COLUMNS = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker", "is_best_match"
]

def aggtrade_url(symbol, date_str):
    return (
        "https://data.binance.vision/data/spot/daily/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{date_str}.zip"
    )

def download_one_aggtrade(symbol, date_str, raw_dir=RAW, force=False, timeout=60):
    out = raw_dir / symbol / f"{symbol}-aggTrades-{date_str}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 100 and not force:
        return out
    url = aggtrade_url(symbol, date_str)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            out.write_bytes(response.read())
        return out
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"Download failed: {symbol} {date_str}: {exc}")
        return None

def read_aggtrade_zip(path):
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=AGGTRADE_COLUMNS)
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            return pd.DataFrame(columns=AGGTRADE_COLUMNS)
        with zf.open(names[0]) as fh:
            df = pd.read_csv(fh, header=None, names=AGGTRADE_COLUMNS)
    for c in ["price", "quantity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["transact_time"] = pd.to_numeric(df["transact_time"], errors="coerce")
    df = df.dropna(subset=["price", "quantity", "transact_time"])
    df["time"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().eq("true")
    return df

def aggregate_symbol_day(symbol, date_str, bin_size="1min"):
    cache_file = CACHE / f"{symbol}_{date_str}_{bin_size}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    path = download_one_aggtrade(
        symbol, date_str, force=FORCE_REDOWNLOAD
    ) if RUN_DOWNLOAD else RAW / symbol / f"{symbol}-aggTrades-{date_str}.zip"

    raw = read_aggtrade_zip(path)
    if raw.empty:
        return pd.DataFrame()

    raw["quote_volume"] = raw["price"] * raw["quantity"]
    raw["signed_quote_volume"] = np.where(
        raw["is_buyer_maker"], -raw["quote_volume"], raw["quote_volume"]
    )
    raw = raw.set_index("time").sort_index()

    agg = raw.resample(bin_size).agg(
        price=("price", "last"),
        volume=("quantity", "sum"),
        quote_volume=("quote_volume", "sum"),
        signed_quote_volume=("signed_quote_volume", "sum"),
        n_trades=("agg_trade_id", "count"),
    )
    agg["price"] = agg["price"].ffill()
    agg[["volume", "quote_volume", "signed_quote_volume", "n_trades"]] = (
        agg[["volume", "quote_volume", "signed_quote_volume", "n_trades"]].fillna(0.0)
    )
    agg["logret"] = np.log(agg["price"]).diff().fillna(0.0)
    agg["abs_logret"] = agg["logret"].abs()
    agg["imbalance"] = agg["signed_quote_volume"] / (agg["quote_volume"] + 1e-12)
    agg["abs_imbalance_x_volume"] = agg["imbalance"].abs() * agg["quote_volume"]
    agg["symbol"] = symbol
    agg = agg.reset_index()
    agg.to_parquet(cache_file, index=False)
    return agg

def load_window_panel(symbols, window, bin_size="1min"):
    frames = []
    for symbol in symbols:
        symbol_frames = []
        for date_str in date_range_inclusive(window["start"], window["end"]):
            x = aggregate_symbol_day(symbol, date_str, bin_size=bin_size)
            if not x.empty:
                symbol_frames.append(x)
        if symbol_frames:
            frames.append(pd.concat(symbol_frames, ignore_index=True))
        else:
            print("No usable data for", symbol, window["name"])
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["time", "symbol"]).reset_index(drop=True)


# ---- V4 cell 4 ----

# =============================================================================
# FEATURES, EVENTS, AND LEAKAGE-FREE DRIVES
# =============================================================================

def panel_to_wide_features(panel, symbols):
    feats = {}
    feature_names = [
        "volume", "quote_volume", "n_trades", "logret", "abs_logret",
        "imbalance", "abs_imbalance_x_volume"
    ]
    full_index = pd.DatetimeIndex(sorted(panel["time"].unique()))
    for feature in feature_names:
        wide = panel.pivot(index="time", columns="symbol", values=feature)
        wide = wide.reindex(index=full_index, columns=symbols)
        if feature == "price":
            wide = wide.ffill()
        else:
            wide = wide.fillna(0.0)
        feats[feature] = wide.sort_index()
    return feats

def build_events_from_features(feats, event_type="volume_burst", q=0.985):
    if event_type == "volume_burst":
        X = np.log1p(feats["quote_volume"])
    elif event_type == "volatility_burst":
        X = feats["abs_logret"]
    elif event_type == "imbalance_burst":
        X = np.log1p(feats["abs_imbalance_x_volume"])
    elif event_type == "combined_burst":
        zv = safe_standardize_df(np.log1p(feats["quote_volume"]))
        zr = safe_standardize_df(feats["abs_logret"])
        X = zv + zr
    else:
        raise ValueError(f"Unknown event_type: {event_type}")

    thresholds = X.quantile(q, axis=0)
    Y = X.gt(thresholds, axis=1).astype(float)
    return Y, thresholds

def continuous_activity_score(feats):
    """Asset-specific continuous activity; contains no event indicators."""
    z_volume = safe_standardize_df(np.log1p(feats["quote_volume"]))
    z_trades = safe_standardize_df(np.log1p(feats["n_trades"]))
    z_vol = safe_standardize_df(feats["abs_logret"])
    z_imb = safe_standardize_df(np.log1p(feats["abs_imbalance_x_volume"]))
    return (z_volume + z_trades + z_vol + z_imb) / 4.0

def build_drive_matrix(feats, symbols, mode="loo_market_activity"):
    """
    Returns a T x K matrix M. Column k is the drive used in equation k.
    No event indicator Y is used anywhere in this function.
    """
    activity = continuous_activity_score(feats).reindex(columns=symbols)
    T, K = activity.shape
    M = np.zeros((T, K), dtype=float)

    if mode == "loo_market_activity":
        A = activity.to_numpy(dtype=float)
        for k in range(K):
            other = np.delete(A, k, axis=1)
            M[:, k] = first_pc_score(pd.DataFrame(other, index=activity.index))

    elif mode == "global_covariate_pca":
        score = first_pc_score(activity)
        M[:] = score[:, None]

    elif mode == "btc_eth_reference":
        refs = [s for s in ["BTCUSDT", "ETHUSDT"] if s in symbols]
        if not refs:
            raise ValueError("btc_eth_reference requires BTCUSDT or ETHUSDT.")
        ref_features = []
        for name in ["quote_volume", "n_trades", "abs_logret", "abs_imbalance_x_volume"]:
            x = feats[name][refs].copy()
            if name in ["quote_volume", "n_trades", "abs_imbalance_x_volume"]:
                x = np.log1p(x)
            x.columns = [f"{name}_{c}" for c in x.columns]
            ref_features.append(x)
        ref_df = pd.concat(ref_features, axis=1)
        base_score = first_pc_score(ref_df)
        for k, symbol in enumerate(symbols):
            # For BTC and ETH equations, use the other reference where possible.
            if symbol in refs and len(refs) > 1:
                other_ref = [r for r in refs if r != symbol]
                local = []
                for name in ["quote_volume", "n_trades", "abs_logret", "abs_imbalance_x_volume"]:
                    x = feats[name][other_ref].copy()
                    if name in ["quote_volume", "n_trades", "abs_imbalance_x_volume"]:
                        x = np.log1p(x)
                    local.append(x)
                M[:, k] = first_pc_score(pd.concat(local, axis=1))
            else:
                M[:, k] = base_score
    else:
        raise ValueError(f"Unknown drive mode: {mode}")

    M = (M - M.mean(axis=0, keepdims=True)) / (M.std(axis=0, keepdims=True) + 1e-12)
    return pd.DataFrame(M, index=activity.index, columns=symbols)

def apply_drive_timing(M_df, timing):
    if timing == "contemporaneous":
        return M_df.copy()
    if timing == "lag1":
        return M_df.shift(1).fillna(0.0)
    raise ValueError(f"Unknown drive timing: {timing}")


# ---- V4 cell 5 ----

# =============================================================================
# NETWORK ESTIMATION
# =============================================================================

try:
    from scipy.optimize import lsq_linear
except ImportError as exc:
    raise ImportError("This notebook requires scipy. Install with: pip install scipy") from exc

def fit_row_lsq(y, H, m=None, nonnegative_B=True, ridge=1e-8):
    y = np.asarray(y, dtype=float)
    H = np.asarray(H, dtype=float)
    n, k = H.shape

    if m is None:
        X = np.column_stack([np.ones(n), H])
        lower = np.r_[-np.inf, np.zeros(k)] if nonnegative_B else np.full(k + 1, -np.inf)
        upper = np.full(k + 1, np.inf)
    else:
        m = np.asarray(m, dtype=float).reshape(-1)
        X = np.column_stack([np.ones(n), H, m])
        lower = np.r_[-np.inf, np.zeros(k), -np.inf] if nonnegative_B else np.full(k + 2, -np.inf)
        upper = np.full(k + 2, np.inf)

    if ridge > 0:
        p = X.shape[1]
        penalty = np.sqrt(ridge) * np.eye(p)
        penalty[0, 0] = 0.0
        X_aug = np.vstack([X, penalty])
        y_aug = np.r_[y, np.zeros(p)]
    else:
        X_aug, y_aug = X, y

    fit = lsq_linear(X_aug, y_aug, bounds=(lower, upper), lsmr_tol="auto")
    coef = fit.x
    mu = float(coef[0])
    B_row = coef[1:1 + k]
    c = float(coef[-1]) if m is not None else np.nan
    resid = y - X @ coef
    return mu, B_row, c, resid

def fit_network(Y, H, M=None, nonnegative_B=True):
    Y = np.asarray(Y, dtype=float)
    H = np.asarray(H, dtype=float)
    K = Y.shape[1]
    B = np.zeros((K, K))
    mu = np.zeros(K)
    c = np.full(K, np.nan)
    residuals = np.zeros_like(Y)

    for z in range(K):
        mz = None if M is None else np.asarray(M)[:, z]
        mu[z], B[z], c[z], residuals[:, z] = fit_row_lsq(
            Y[:, z], H, m=mz, nonnegative_B=nonnegative_B
        )

    return {"mu": mu, "B": B, "c": c, "residuals": residuals}

def effective_design(Y_df, M_df, lags, half_life):
    Y = Y_df.to_numpy(dtype=float)
    M = M_df.to_numpy(dtype=float)
    H = build_filtered_histories(Y, lags=lags, half_life=half_life)
    start = int(lags)
    return Y[start:], H[start:], M[start:], Y_df.index[start:]

def residual_loading_diagnostics(residuals, M, c_hat):
    residuals = np.asarray(residuals, dtype=float)
    M = np.asarray(M, dtype=float)
    K = residuals.shape[1]

    # Residual spike ratio
    cov_r = np.cov(residuals, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_r)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    spike_ratio = float(eigvals[0] / max(eigvals[1], 1e-12)) if K > 1 else np.nan

    # Alignment between estimated regression loading and residual PC1
    pc1 = eigvecs[:, 0]
    cvec = np.asarray(c_hat, dtype=float)
    cvec = cvec / (np.linalg.norm(cvec) + 1e-12)
    loading_alignment = float(abs(np.dot(cvec, pc1)))

    # Per-equation drive-history separation kappa = 1 - R2(m_z ~ H)
    kappas = []
    for z in range(K):
        mz = M[:, z]
        X = np.column_stack([np.ones(len(mz)), build_filtered_histories(
            np.eye(1)[np.zeros(len(mz), dtype=int)], lags=1, half_life=1
        )]) if False else None
        # Direct least-squares projection on the actual H is computed outside.
        kappas.append(np.nan)

    return {
        "spike_ratio": spike_ratio,
        "loading_alignment": loading_alignment,
        "residual_pc1": pc1,
        "residual_eigenvalues": eigvals,
    }

def compute_kappa_per_equation(H, M):
    H = np.asarray(H, dtype=float)
    M = np.asarray(M, dtype=float)
    X = np.column_stack([np.ones(len(H)), H])
    kappas = []
    for z in range(M.shape[1]):
        m = M[:, z]
        coef, *_ = np.linalg.lstsq(X, m, rcond=None)
        pred = X @ coef
        sse = np.sum((m - pred) ** 2)
        sst = np.sum((m - np.mean(m)) ** 2)
        r2 = 1.0 - sse / max(sst, 1e-12)
        kappas.append(max(0.0, 1.0 - r2))
    return np.asarray(kappas)

