"""Extract the minimal V4 data/feature/network pipeline needed by the JBES application.

Usage: python extract_v4_pipeline.py /path/to/crypto_confirmatory_V4.ipynb
Writes pipeline_v4.py from V4 code cells 1--5 only. Later confirmatory cells depend on
notebook-only constants and are intentionally excluded.
"""
import json, sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("Usage: python extract_v4_pipeline.py /path/to/crypto_confirmatory_V4.ipynb")
src = Path(sys.argv[1])
nb = json.load(open(src))
keep_idx = [1,2,3,4,5]
out = f"# pipeline_v4.py — minimal extraction from {src.name}; V4 cells 1--5.\n\n"
for i in keep_idx:
    c = nb["cells"][i]
    if c.get("cell_type") != "code":
        raise RuntimeError(f"Expected code cell {i} in {src.name}")
    out += f"\n# ---- V4 cell {i} ----\n" + "".join(c.get("source", [])) + "\n"
Path("pipeline_v4.py").write_text(out)
print(f"pipeline_v4.py written from V4 cells {keep_idx} of {src.name}")
