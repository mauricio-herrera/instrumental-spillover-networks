# V6 single-proxy cross-fitted audit used in the JBES manuscript

This directory contains only the 71-design cross-fitted audit used in Section 6 of the JBES manuscript. Earlier exploratory scaling and identified-bound tables are intentionally excluded from this submission package because the final manuscript does not use those claims.

Frozen summary regenerated from `v6_crossfit_audit.csv`:

- completed cross-fitted designs: 71;
- positive time-adjusted spectral deflation: 71/71;
- circular-shift p < 0.05 for deflation: 66/71;
- frozen-direction held-out projection p < 0.05 (`crossfit_projection_p`): 55/71;
- held-out Brier improvement beyond the flexible time baseline: 41/71;
- held-out log-loss improvement beyond the flexible time baseline: 41/71;
- median time-adjusted delta-rho: 0.08622;
- interquartile range: 0.03593--0.10799;
- median constant-baseline naive rho: 0.90025;
- median flexible-time rho: 0.82606.

These results motivate the two-basket IV analysis but do not establish causal identification by themselves.
