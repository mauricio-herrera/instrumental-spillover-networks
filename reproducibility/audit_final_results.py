from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
f = root / 'frozen_results'
eq = pd.read_csv(f/'b5v4_instrument_results__equal_weight_full.csv')
ej = pd.read_csv(f/'b5v4_overid_by_asset__equal_weight_full.csv')
di = pd.read_csv(f/'primary_ar_empty_row_diagnostic.csv')
pc = pd.read_csv(f/'b5v4_instrument_results__pc1_train_full.csv')
pj = pd.read_csv(f/'b5v4_overid_by_asset__pc1_train_full.csv')

assert len(eq)==240 and len(pc)==240
assert len(ej)==288 and len(pj)==288 and len(di)==288
assert eq.duplicated(['window','q','split','lag']).sum()==0
assert pc.duplicated(['window','q','split','lag']).sum()==0
for d in (eq,pc):
    main=d[~d.placebo]
    pl=d[d.placebo]
    assert len(main)==192 and len(pl)==48
    assert np.isfinite(main.F_eff_MOP).all() and np.isfinite(pl.F_eff_MOP).all()
    assert main.ar_empty.fillna(False).astype(bool).all()
    # Empty is distinct from truly unbounded: no nonempty main set has an infinite hi.
    true_unbounded=(~main.ar_empty.fillna(False).astype(bool)) & (~np.isfinite(main.ar_hi))
    assert true_unbounded.sum()==0

m=eq[~eq.placebo]
assert (m[m.lag==1].rho_2sls < m[m.lag==1].rho_naive).all()
assert (m[m.lag==2].rho_2sls < m[m.lag==2].rho_naive).all()
assert int((m[m.lag==5].rho_2sls < m[m.lag==5].rho_naive).sum())==42
assert int((m[m.lag==10].rho_2sls < m[m.lag==10].rho_naive).sum())==4
prim=m[(m['split']=='S1')&(m.lag==1)]
assert len(prim)==16 and (prim.rho_2sls>1).all()
assert int((di.status!='FEASIBLE_AT_STATED_RADIUS').sum())==136
assert int((ej.p<.05).sum())==21 and int((pj.p<.05).sum())==28
assert int((eq[eq.placebo].F_eff_MOP>=10).sum())==12
assert int((pc[pc.placebo].F_eff_MOP>=10).sum())==13

print('FINAL RESULT AUDIT PASS')
print('equal-weight: 240 rows; PC1: 240 rows; Hansen-J: 288 + 288')
print('main AR: 192/192 empty in both proxy constructions; truly unbounded nonempty: 0')
print('lag reductions: lag1 48/48, lag2 48/48, lag5 42/48, lag10 4/48')
print('primary corrected rho>1: 16/16')
print('primary row incompatibility: 136/288')
print('Hansen-J p<.05: equal 21/288; PC1 28/288')
print('lead placebo F_eff>=10: equal 12/48; PC1 13/48')
