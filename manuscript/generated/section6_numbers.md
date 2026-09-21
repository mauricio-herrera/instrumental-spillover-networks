# Section 6 final numerical audit

## Frozen design
- Equal-weight rows: 240 = 192 main + 48 corrected lead-innovation placebo.
- Equal-weight Hansen-J rows: 288.
- PC1 rows: 240; PC1 Hansen-J rows: 288.
- Structural row diagnostic: 288 = 16 primary window/q designs x 18 assets.
- Windows: 3 stress + 5 control; q in {0.95, 0.97}; splits S1/S2/S3; main lags 1/2/5/10.

## Instrument strength
- Main equal-weight designs with F_eff >= 10: 192/192.
- Main median F_eff: 1107.5; range 406.1-5766.5.
- Primary S1/lag1 median F_eff: 933.4; range 406.1-4404.8.
- Primary control: median F_eff 598.0 (range 406.1-1241.0).
- Primary shock: median F_eff 1783.1 (range 1397.2-4404.8).

## Lag profile of point correction
- lag 1: rho_2SLS < rho_naive in 48/48; mean delta-rho 0.187; median 0.175; mean rho_2SLS 1.492.
- lag 2: rho_2SLS < rho_naive in 48/48; mean delta-rho 0.107; median 0.090; mean rho_2SLS 1.572.
- lag 5: rho_2SLS < rho_naive in 42/48; mean delta-rho 0.014; median 0.013; mean rho_2SLS 1.665.
- lag 10: rho_2SLS < rho_naive in 4/48; mean delta-rho -0.033; median -0.036; mean rho_2SLS 1.712.
- Primary S1/lag1: reduction in 16/16; mean delta-rho 0.192; mean relative reduction 11.4%.
- Primary S1/lag1 with rho_2SLS < 1: 0/16.

## Robust AR / structural compatibility
- Empty joint AR sets: 192/192 main designs.
- Truly unbounded (nonempty) AR upper endpoints: 0/192.
- Finite nonempty AR intervals: 0/192.
- Rowwise analytic HAC-AR loading sets bounded in all rows: 192/192 designs.
- Primary diagnostic incompatible rows: 136/288 (47.2%).
- control: incompatible rows 58/180 (32.2%); median required-radius ratio 0.714.
- shock: incompatible rows 78/108 (72.2%); median required-radius ratio 1.471.
- Most frequent incompatible assets across 16 primary designs: DOGEUSDT 11/16, BTCUSDT 11/16, ETHUSDT 11/16, BNBUSDT 10/16, TRXUSDT 9/16, NEARUSDT 9/16.

## Exclusion / placebo diagnostics
- Equal-weight lead-innovation placebo: median F_eff 5.82; F_eff >= 10 in 12/48 (25.0%).
- Placebo/main-lag1 F_eff ratio: median 0.63%; maximum 2.77%.
- Equal-weight Hansen J rejections: 21/288 (7.3%) at 5%; 48/288 (16.7%) at 10%.

## PC1 robustness
- Absolute difference in rho_2SLS, PC1 vs equal-weight: median 0.0010, mean 0.0017, maximum 0.0088.
- PC1 empty joint AR sets: 192/192.
- PC1 lead placebo F_eff >= 10: 13/48; median 5.82.
- PC1 Hansen J rejections: 28/288 (9.7%) at 5%; 47/288 (16.3%) at 10%.

## Interpretation guardrails
- The pre-registered strength scenario is A (strong instruments); this does NOT validate exclusion.
- Empty is not unbounded. The full-network robust set is structurally incompatible with B>=0, rather than weakly identified.
- Do not claim a uniform rho reduction over lags 1-10: it holds in 48/48 at lags 1 and 2, 42/48 at lag 5, and only 4/48 at lag 10.
- Do not report response connectedness in the empirical primary table: all primary corrected radii exceed one, so the stable response matrix is not defined there.
- Treat 2SLS radii as point diagnostics under the maintained equation system, not as a validated structural cascade-risk estimate once the joint cone-restricted confidence set is empty.
- PC1 is a robustness check only; equal-weight remains primary.