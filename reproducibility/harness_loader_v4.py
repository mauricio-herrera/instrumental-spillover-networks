# V4 pipeline harness extracted from crypto_confirmatory_V4.ipynb (data download, features, events, histories).

import json, ast, sys
from pathlib import Path
import numpy as np, pandas as pd

# --- locate the V4 notebook (must sit next to this one) ---
CANDS = sorted(Path.cwd().glob("*V4*CONFIRMATORY*.ipynb")) + \
        sorted(Path.cwd().glob("*crypto*V4*.ipynb")) + \
        sorted(Path.cwd().glob("crypto_confirmatory_V4.ipynb"))
assert CANDS, ("Place crypto_confirmatory_V4.ipynb next to this notebook. "
               "Its functions (load_window_panel, build_events_from_features, "
               "build_drive_matrix, fit_network, build_filtered_histories, "
               "spectral_radius, rank1_share, ...) are reused here.")
V4 = CANDS[0]; print("V4 found:", V4.name)
nb_v4 = json.load(open(V4))
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb_v4["cells"])
              if c["cell_type"] == "code"]

def _is_def_cell(src):
    try: tree = ast.parse(src)
    except SyntaxError: return False
    has_def = any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body)
    top_calls = [n for n in tree.body
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    return has_def and not top_calls

to_exec = [code_cells[0]] + [(i, s) for i, s in code_cells[1:] if _is_def_cell(s)]
for i, src in to_exec:
    try: exec(compile(src, f"<V4 cell {i}>", "exec"), globals())
    except Exception as exc: print(f"  ! V4 cell {i} skipped ({type(exc).__name__}: {exc})")

# copy-on-write safe PC1 (same numeric result as V4)
_ssd = safe_standardize_df
def first_pc_score(df):
    X = np.array(_ssd(df).to_numpy(dtype=float), copy=True)
    if X.shape[1] == 0: return np.zeros(X.shape[0])
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    score = X @ vt[0]
    if np.corrcoef(score, X.mean(axis=1))[0, 1] < 0: score = -score
    return (score - score.mean()) / (score.std() + 1e-12)

for _f in ["load_window_panel","panel_to_wide_features","build_events_from_features",
           "continuous_activity_score","build_drive_matrix","apply_drive_timing",
           "fit_network","build_filtered_histories","spectral_radius","rank1_share",
           "effective_design"]:
    assert _f in globals(), f"missing V4 function: {_f}"
print("V4 pipeline loaded OK.")

# ---- robust downloader (retries + atomic write + zip validation) ----
import time as _time, zipfile as _zip, urllib.request, urllib.error
def _valid_zip(p):
    try:
        with _zip.ZipFile(p) as zf: return zf.testzip() is None and len(zf.namelist())>0
    except Exception: return False
def download_one_aggtrade(symbol, date_str, raw_dir=RAW, force=False, timeout=120, retries=5):
    out=raw_dir/symbol/f"{symbol}-aggTrades-{date_str}.zip"; out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and not force:
        if out.stat().st_size>100 and _valid_zip(out): return out
        try: out.unlink()
        except Exception: pass
    url=aggtrade_url(symbol,date_str); tmp=out.parent/(out.name+".part"); last=None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url,timeout=timeout) as r: data=r.read()
            tmp.write_bytes(data)
            if _valid_zip(tmp): tmp.replace(out); return out
            try: tmp.unlink()
            except Exception: pass
            last="corrupt payload"
        except urllib.error.HTTPError as e:
            if e.code==404: return None
            last=e
        except Exception as e: last=e
        _time.sleep(1.5*(attempt+1))
    try: tmp.unlink()
    except Exception: pass
    print(f"Download failed after {retries} tries: {symbol} {date_str}: {last}"); return None
print("robust downloader installed")
