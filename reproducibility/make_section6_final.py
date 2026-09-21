"""Generate the final JBES application table and numerical audit.

Usage
-----
python make_section6_final.py \
  frozen_results/b5v4_instrument_results__equal_weight_full.csv \
  frozen_results/b5v4_overid_by_asset__equal_weight_full.csv \
  frozen_results/primary_ar_empty_row_diagnostic.csv \
  frozen_results/b5v4_instrument_results__pc1_train_full.csv \
  frozen_results/b5v4_overid_by_asset__pc1_train_full.csv \
  ../manuscript/generated

The script deliberately distinguishes EMPTY, UNBOUNDED and FINITE AR projections.
It never counts an empty set (NaN endpoints) as an unbounded interval.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if len(sys.argv) != 7:
    raise SystemExit(__doc__)
peq, poj, pdiag, ppc, ppj, pout = map(Path, sys.argv[1:])
out = pout
out.mkdir(parents=True, exist_ok=True)

eq = pd.read_csv(peq)
oj = pd.read_csv(poj)
diag = pd.read_csv(pdiag)
pc = pd.read_csv(ppc)
pj = pd.read_csv(ppj)

# ---------- structural checks ----------
for name, d, n in [("equal-weight", eq, 240), ("PC1", pc, 240)]:
    if len(d) != n:
        raise RuntimeError(f"{name}: expected {n} rows, got {len(d)}")
    if d.duplicated(["window","q","split","lag"]).any():
        raise RuntimeError(f"{name}: duplicate design keys")
for name, d, n in [("equal-weight J", oj, 288), ("PC1 J", pj, 288)]:
    if len(d) != n:
        raise RuntimeError(f"{name}: expected {n} rows, got {len(d)}")
if len(diag) != 288:
    raise RuntimeError(f"structural diagnostic: expected 288 row-designs, got {len(diag)}")

main = eq[~eq.placebo].copy()
pl = eq[eq.placebo].copy()
prim = main[(main["split"] == "S1") & (main.lag == 1)].copy()
pcmain = pc[~pc.placebo].copy()
pcpl = pc[pc.placebo].copy()
pcprim = pcmain[(pcmain["split"] == "S1") & (pcmain.lag == 1)].copy()

empty = main.ar_empty.fillna(False).astype(bool)
true_unbounded = (~empty) & (~np.isfinite(main.ar_hi))
finite = (~empty) & np.isfinite(main.ar_lo) & np.isfinite(main.ar_hi)

if int(empty.sum()) != 192 or int(true_unbounded.sum()) != 0:
    raise RuntimeError("Frozen AR status differs from the audited full run")

# Primary structural incompatibility counts: S1/lag1, 18 rows per window x q.
diag = diag.copy()
diag["incompat"] = diag.status != "FEASIBLE_AT_STATED_RADIUS"
dsum = (diag.groupby(["window","kind","q"], as_index=False)
        .agg(K=("asset","size"), incompat=("incompat","sum"),
             median_radius_ratio=("required_radius_ratio","median")))

# Hansen J counts by window/q.
jsum = (oj.groupby(["window","q"], as_index=False)
        .agg(J_n=("p","size"), J_rej05=("p", lambda s: int((s < .05).sum())),
             J_rej10=("p", lambda s: int((s < .10).sum())), J_pmin=("p","min")))

# Lead-innovation placebo summary across S1/S2/S3.
plsum = (pl.groupby(["window","kind","q"], as_index=False)
         .agg(placebo_F_med=("F_eff_MOP","median"),
              placebo_F_max=("F_eff_MOP","max"),
              placebo_F_ge10=("F_eff_MOP", lambda s: int((s >= 10).sum()))))

T = prim.merge(dsum, on=["window","kind","q"]).merge(jsum, on=["window","q"]).merge(plsum, on=["window","kind","q"])
T["delta_rho"] = T.rho_naive - T.rho_2sls

# ---------- Table 4 ----------
order = ["Terra_Luna","FTX_collapse","Banking_crypto","Control_2022_Aug_A","Control_2022_Aug_B","Control_2022_Oct","Control_2023_Jan","Control_2023_Feb"]
T["ord"] = pd.Categorical(T.window, categories=order, ordered=True)
T = T.sort_values(["ord","q"])
labels = {
    "Terra_Luna":"Terra/Luna",
    "FTX_collapse":"FTX collapse",
    "Banking_crypto":"Banking/crypto",
    "Control_2022_Aug_A":"Control Aug-A",
    "Control_2022_Aug_B":"Control Aug-B",
    "Control_2022_Oct":"Control Oct",
    "Control_2023_Jan":"Control Jan",
    "Control_2023_Feb":"Control Feb",
}
rows=[]
for _, r in T.iterrows():
    rows.append(
        f"{labels[r.window]} & {r.q:.2f} & {r.F_eff_MOP:.0f} & {r.rho_naive:.3f} & "
        f"{r.rho_2sls:.3f} & {r.delta_rho:.3f} & $\\varnothing$ & "
        f"{int(r.incompat)}/18 & {int(r.J_rej05)}/18 \\\\"
    )
tex = r"""\begin{table}[t]
\centering\small
\caption{Cryptocurrency spillover networks, primary equal-weight specification (split S1, lag 1). $F_{\rm eff}$ is the HAC-robust effective first-stage statistic; $\Delta\rho=\rho_{\rm naive}-\rho_{\rm 2SLS}$. $S^{\rm AR}_{.05}=\varnothing$ means that the joint Anderson--Rubin/reduced-form confidence restrictions have no intersection with the maintained nonnegative-excitation cone. ``Incompat.'' is the number of the 18 rowwise structural restrictions that are incompatible in the diagnostic decomposition; $J$ rej. is the number of asset-level HAC Hansen tests rejecting at 5\%.}
\label{tab:app}
\begin{tabular}{llrrrrlrr}
\toprule
window & $q$ & $F_{\rm eff}$ & $\rho_{\rm naive}$ & $\rho_{\rm 2SLS}$ & $\Delta\rho$ & $S^{\rm AR}_{.05}$ & Incompat. & $J$ rej.\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(out / "table_app.tex").write_text(tex)
T.drop(columns=["ord"]).to_csv(out / "table_app_source.csv", index=False)

# ---------- numerical narrative ----------
L=[]
P=L.append
P("# Section 6 final numerical audit\n")
P("## Frozen design")
P(f"- Equal-weight rows: {len(eq)} = {len(main)} main + {len(pl)} corrected lead-innovation placebo.")
P(f"- Equal-weight Hansen-J rows: {len(oj)}.")
P(f"- PC1 rows: {len(pc)}; PC1 Hansen-J rows: {len(pj)}.")
P(f"- Structural row diagnostic: {len(diag)} = 16 primary window/q designs x 18 assets.")
P("- Windows: 3 stress + 5 control; q in {0.95, 0.97}; splits S1/S2/S3; main lags 1/2/5/10.")

P("\n## Instrument strength")
P(f"- Main equal-weight designs with F_eff >= 10: {(main.F_eff_MOP>=10).sum()}/{len(main)}.")
P(f"- Main median F_eff: {main.F_eff_MOP.median():.1f}; range {main.F_eff_MOP.min():.1f}-{main.F_eff_MOP.max():.1f}.")
P(f"- Primary S1/lag1 median F_eff: {prim.F_eff_MOP.median():.1f}; range {prim.F_eff_MOP.min():.1f}-{prim.F_eff_MOP.max():.1f}.")
for kind,g in prim.groupby("kind"):
    P(f"- Primary {kind}: median F_eff {g.F_eff_MOP.median():.1f} (range {g.F_eff_MOP.min():.1f}-{g.F_eff_MOP.max():.1f}).")

P("\n## Lag profile of point correction")
for lag,g in main.groupby("lag"):
    delta = g.rho_naive-g.rho_2sls
    P(f"- lag {lag}: rho_2SLS < rho_naive in {(delta>0).sum()}/{len(g)}; mean delta-rho {delta.mean():.3f}; median {delta.median():.3f}; mean rho_2SLS {g.rho_2sls.mean():.3f}.")
P(f"- Primary S1/lag1: reduction in {(prim.rho_2sls<prim.rho_naive).sum()}/{len(prim)}; mean delta-rho {(prim.rho_naive-prim.rho_2sls).mean():.3f}; mean relative reduction {100*((prim.rho_naive-prim.rho_2sls)/prim.rho_naive).mean():.1f}%.")
P(f"- Primary S1/lag1 with rho_2SLS < 1: {(prim.rho_2sls<1).sum()}/{len(prim)}.")

P("\n## Robust AR / structural compatibility")
P(f"- Empty joint AR sets: {int(empty.sum())}/{len(main)} main designs.")
P(f"- Truly unbounded (nonempty) AR upper endpoints: {int(true_unbounded.sum())}/{len(main)}.")
P(f"- Finite nonempty AR intervals: {int(finite.sum())}/{len(main)}.")
P(f"- Rowwise analytic HAC-AR loading sets bounded in all rows: {int((main.hac_ar_frac_bounded==1).sum())}/{len(main)} designs.")
P(f"- Primary diagnostic incompatible rows: {int(diag.incompat.sum())}/{len(diag)} ({100*diag.incompat.mean():.1f}%).")
for kind,g in diag.groupby("kind"):
    rr=g.required_radius_ratio.replace([np.inf,-np.inf],np.nan)
    P(f"- {kind}: incompatible rows {int(g.incompat.sum())}/{len(g)} ({100*g.incompat.mean():.1f}%); median required-radius ratio {rr.median():.3f}.")
counts=diag[diag.incompat].groupby("asset").size().sort_values(ascending=False)
P("- Most frequent incompatible assets across 16 primary designs: " + ", ".join(f"{a} {n}/16" for a,n in counts.head(6).items()) + ".")

P("\n## Exclusion / placebo diagnostics")
P(f"- Equal-weight lead-innovation placebo: median F_eff {pl.F_eff_MOP.median():.2f}; F_eff >= 10 in {(pl.F_eff_MOP>=10).sum()}/{len(pl)} ({100*(pl.F_eff_MOP>=10).mean():.1f}%).")
# relative to main lag1 on matching design
m1=main[main.lag==1].merge(pl,on=["window","kind","q","split"],suffixes=("_main","_pl"))
ratio=m1.F_eff_MOP_pl/m1.F_eff_MOP_main
P(f"- Placebo/main-lag1 F_eff ratio: median {100*ratio.median():.2f}%; maximum {100*ratio.max():.2f}%.")
P(f"- Equal-weight Hansen J rejections: {(oj.p<.05).sum()}/{len(oj)} ({100*(oj.p<.05).mean():.1f}%) at 5%; {(oj.p<.10).sum()}/{len(oj)} ({100*(oj.p<.10).mean():.1f}%) at 10%.")

P("\n## PC1 robustness")
merge=main.merge(pcmain,on=["window","kind","q","split","lag"],suffixes=("_eq","_pc"))
adiff=(merge.rho_2sls_eq-merge.rho_2sls_pc).abs()
P(f"- Absolute difference in rho_2SLS, PC1 vs equal-weight: median {adiff.median():.4f}, mean {adiff.mean():.4f}, maximum {adiff.max():.4f}.")
P(f"- PC1 empty joint AR sets: {int(pcmain.ar_empty.sum())}/{len(pcmain)}.")
P(f"- PC1 lead placebo F_eff >= 10: {(pcpl.F_eff_MOP>=10).sum()}/{len(pcpl)}; median {pcpl.F_eff_MOP.median():.2f}.")
P(f"- PC1 Hansen J rejections: {(pj.p<.05).sum()}/{len(pj)} ({100*(pj.p<.05).mean():.1f}%) at 5%; {(pj.p<.10).sum()}/{len(pj)} ({100*(pj.p<.10).mean():.1f}%) at 10%.")

P("\n## Interpretation guardrails")
P("- The pre-registered strength scenario is A (strong instruments); this does NOT validate exclusion.")
P("- Empty is not unbounded. The full-network robust set is structurally incompatible with B>=0, rather than weakly identified.")
P("- Do not claim a uniform rho reduction over lags 1-10: it holds in 48/48 at lags 1 and 2, 42/48 at lag 5, and only 4/48 at lag 10.")
P("- Do not report response connectedness in the empirical primary table: all primary corrected radii exceed one, so the stable response matrix is not defined there.")
P("- Treat 2SLS radii as point diagnostics under the maintained equation system, not as a validated structural cascade-risk estimate once the joint cone-restricted confidence set is empty.")
P("- PC1 is a robustness check only; equal-weight remains primary.")

(out / "section6_numbers.md").write_text("\n".join(L))
print("\n".join(L))
print(f"\nWROTE {out/'table_app.tex'}, {out/'table_app_source.csv'}, {out/'section6_numbers.md'}")
