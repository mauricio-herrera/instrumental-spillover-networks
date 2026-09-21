from pathlib import Path
import sys
import numpy as np
import pandas as pd

p = Path(sys.argv[1] if len(sys.argv)>1 else
         'crypto_common_drive_outputs_V3/tables/b5v4_instrument_results__equal_weight_quick.csv')
o = Path(sys.argv[2] if len(sys.argv)>2 else
         'crypto_common_drive_outputs_V3/tables/b5v4_overid_by_asset__equal_weight_quick.csv')

res = pd.read_csv(p)
ov = pd.read_csv(o)

req = {
    'window','kind','q','split','lag','placebo','K','T','F_conv','F_eff_MOP',
    'rho_naive','rho_2sls','ar_lo','ar_hi','ar_empty','ar_frac_bounded',
    'placebo_definition','ar_projection'
}
miss = req - set(res.columns)
if miss:
    raise SystemExit(f'missing columns: {sorted(miss)}')

key = ['window','q','split','lag']
dup = int(res.duplicated(key).sum())
main = res[~res.placebo].copy()
pl = res[res.placebo].copy()

empty = main.ar_empty.fillna(False).astype(bool)
true_unbounded = (~empty) & (~np.isfinite(main.ar_hi))
finite_ar = (~empty) & np.isfinite(main.ar_lo) & np.isfinite(main.ar_hi)

print('results rows:', len(res), 'duplicates on design key:', dup)
print('windows:', res.window.nunique(), sorted(res.window.unique()))
print('q:', sorted(res.q.unique()), 'splits:', sorted(res.split.unique()),
      'lags:', sorted(res.lag.unique()))
print('main rows:', len(main), 'corrected placebo rows:', len(pl))

print('\nAR STATUS (mutually exclusive)')
print('  empty confidence sets:', int(empty.sum()))
print('  truly unbounded upper endpoint:', int(true_unbounded.sum()))
print('  finite projected intervals:', int(finite_ar.sum()))

if finite_ar.any():
    g = main.loc[finite_ar].copy()
    crosses1 = (g.ar_lo <= 1.0) & (g.ar_hi >= 1.0)
    below1 = g.ar_hi < 1.0
    above1 = g.ar_lo > 1.0
    print('  finite intervals crossing rho=1:', int(crosses1.sum()), '/', len(g))
    print('  finite intervals wholly below rho=1:', int(below1.sum()), '/', len(g))
    print('  finite intervals wholly above rho=1:', int(above1.sum()), '/', len(g))

print('\nFIRST STAGE')
print('  main F_eff finite:', int(np.isfinite(main.F_eff_MOP).sum()), '/', len(main))
print('  primary S1/lag1 median F_eff:',
      float(main[(main.split=="S1") & (main.lag==1)].F_eff_MOP.median()))
print('  placebo F_eff finite:', int(np.isfinite(pl.F_eff_MOP).sum()), '/', len(pl))
if len(pl):
    print('  placebo median F_eff:', float(pl.F_eff_MOP.median()))
    print('  placebo F_eff>=10:', int((pl.F_eff_MOP>=10).sum()), '/', len(pl))
    print('  placebo by split:')
    print(pl.groupby('split').F_eff_MOP.agg(['median','min','max']).round(3).to_string())

print('\nAR STATUS BY WINDOW')
tab = main.assign(
    empty=empty,
    true_unbounded=true_unbounded,
    finite_ar=finite_ar
).groupby('window')[['empty','true_unbounded','finite_ar']].sum().astype(int)
print(tab.to_string())

print('\nPRIMARY S1 / LAG 1')
cols = ['window','kind','q','F_eff_MOP','rho_naive','rho_2sls',
        'ar_lo','ar_hi','ar_empty','conn_naive','conn_2sls']
print(main[(main.split=='S1') & (main.lag==1)][cols].round(4).to_string(index=False))

print('\nOVERIDENTIFICATION')
if 'p' in ov:
    print('  Hansen-J rejections at 5%:', int((ov.p<0.05).sum()), '/', len(ov))
    print('  Hansen-J rejections at 10%:', int((ov.p<0.10).sum()), '/', len(ov))
    print('  minimum p:', float(ov.p.min()))

print('\nAR projection methods:', sorted(main.ar_projection.dropna().astype(str).unique()))

if dup:
    raise SystemExit('FAIL: duplicate design keys')
if not np.isfinite(main.F_eff_MOP).all():
    raise SystemExit('FAIL: non-finite primary F_eff')
if not np.isfinite(pl.F_eff_MOP).all():
    raise SystemExit('FAIL: non-finite corrected-placebo F_eff')
if len(pl) and not pl.placebo_definition.astype(str).str.contains('lead_innovation_F').all():
    raise SystemExit('FAIL: placebo rows do not use corrected lead-innovation definition')
if len(main) and main.ar_projection.astype(str).str.contains('not_applicable').any():
    raise SystemExit('FAIL: primary rows missing AR projection')

print('\nOUTPUT AUDIT V3 PASS')
