# Final scientific and reproducibility audit

## Empirical files

All final empirical statements were regenerated directly from the five frozen CSVs in `reproducibility/frozen_results/`.

The frozen-output audit verifies:
- 240 equal-weight rows and 240 PC1 rows;
- 288 Hansen-J rows for each proxy method;
- 192/192 main AR sets empty for each proxy method;
- zero truly unbounded nonempty main sets;
- lag-1 reduction 48/48, lag-2 48/48, lag-5 42/48, lag-10 4/48;
- all 16 primary corrected radii remain above one;
- 136/288 incompatible row-designs in the primary structural diagnostic;
- Hansen-J rejections at 5%: 21/288 equal-weight, 28/288 PC1;
- corrected lead-placebo F_eff >= 10: 12/48 equal-weight, 13/48 PC1.

## Interpretation safeguards

1. **Empty is not unbounded.** NaN endpoints associated with `ar_empty=True` are never counted as infinite confidence intervals.
2. **First-stage strength is not exclusion validity.** The manuscript reports lead-innovation and Hansen diagnostics separately.
3. **No horizon-invariant correction claim.** The point reduction is robust at 1–2 minutes, attenuates at 5 minutes, and reverses at 10 minutes.
4. **No connectedness claim in the primary empirical table.** All primary corrected radii exceed one, so a stable all-generation response matrix is not defined.
5. **No post-hoc relaxation of B>=0.** The empty cone-restricted sets are retained as a specification-compatibility finding.
6. **PC1 is sensitivity, not a new primary design.** Its spectral-radius results are nearly identical to equal weighting.

## Code audit

- Python syntax compilation passes for the final scripts.
- `audit_fast_ar_projection.py` reports 10,000 feasible synthetic draws, zero containment violations, and `FAST AR PROJECTION GEOMETRY AUDIT PASS`.
- The container used for this packaging step does not have `cvxpy` installed, so the original `prop_connect_check.py` could not be re-executed here; it remains in the package and had previously been frozen as PASS. The final empirical outputs were produced in the user's environment with CVXPY/CLARABEL.

## Manuscript audit

- Abstract: 199 words.
- PDF: 25 pages, double-spaced, opens cleanly.
- Tables 1--4 were visually inspected after final compilation; all fit within the page without clipping.
- No pending empirical placeholders remain; the response-matrix notation in the introduction was cross-checked against Proposition 2.1.
- Application corrected to 3 stress and 5 control windows.
- Recent JBES references verified from Taylor & Francis metadata.
