import sys, numpy as np, pandas as pd
res = pd.read_csv(sys.argv[1])
ov = pd.read_csv(sys.argv[2])

main = res[~res.placebo].copy()
pl = res[res.placebo].copy()
prim = main[(main.split=="S1") & (main.lag==1)].copy()

empty = main.ar_empty.fillna(False).astype(bool)
unbounded = (~empty) & (~np.isfinite(main.ar_hi))
finite = (~empty) & np.isfinite(main.ar_lo) & np.isfinite(main.ar_hi)

print("FULL V4 SUMMARY")
print("="*72)
print(f"rows={len(res)} main={len(main)} placebo={len(pl)} overid={len(ov)}")
print(f"windows={res.window.nunique()} q={sorted(res.q.unique())}")
print("\nAR status, mutually exclusive:")
print(f"  empty      : {int(empty.sum())}/{len(main)}")
print(f"  unbounded  : {int(unbounded.sum())}/{len(main)}")
print(f"  finite     : {int(finite.sum())}/{len(main)}")

print("\nPrimary S1 / lag 1:")
x = prim[["window","kind","q","F_eff_MOP","rho_naive","rho_2sls","ar_empty"]].copy()
x["delta_rho"] = x.rho_2sls - x.rho_naive
print(x.round(4).to_string(index=False))
print(f"\nPrimary F>=10: {int((prim.F_eff_MOP>=10).sum())}/{len(prim)}")
print(f"Primary rho reduced by 2SLS: {int((prim.rho_2sls<prim.rho_naive).sum())}/{len(prim)}")
print(f"Primary rho_2SLS<1: {int((prim.rho_2sls<1).sum())}/{len(prim)}")

print("\nMain first-stage by window:")
print(main.groupby(["window","kind"]).F_eff_MOP.agg(["median","min","max"]).round(2).to_string())

print("\nCorrected temporal placebo by window:")
p = pl.groupby(["window","kind"]).F_eff_MOP.agg(["median","min","max"])
p["F>=10"] = pl.groupby(["window","kind"]).F_eff_MOP.apply(lambda s:int((s>=10).sum()))
print(p.round(3).to_string())
print(f"\nPlacebo F>=10 overall: {int((pl.F_eff_MOP>=10).sum())}/{len(pl)} "
      f"({100*(pl.F_eff_MOP>=10).mean():.1f}%)")

print("\nHansen-J:")
print(f"  p<0.05: {int((ov.p<.05).sum())}/{len(ov)} ({100*(ov.p<.05).mean():.1f}%)")
print(f"  p<0.10: {int((ov.p<.10).sum())}/{len(ov)} ({100*(ov.p<.10).mean():.1f}%)")
print(ov.groupby("window").p.agg(
    n="size",
    rej05=lambda s:int((s<.05).sum()),
    rej10=lambda s:int((s<.10).sum()),
    pmin="min"
).round(4).to_string())
