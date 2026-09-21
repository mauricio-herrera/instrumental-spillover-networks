"""Proposition (rank-one contamination of the response matrix): Sherman-Morrison identity, entrywise dominance,
connectedness inflation, ranking non-preservation. Prints PASS/FAIL."""
import numpy as np
from cascade_cs import connectedness, spectral_radius
rng=np.random.default_rng(3); K=6
B0=rng.uniform(0,1,(K,K))*(rng.random((K,K))<.6); B0*=0.55/spectral_radius(B0); c=rng.uniform(.3,.8,K); g=rng.uniform(0,.3,K)
A=B0+np.outer(c,g); R0=np.linalg.inv(np.eye(K)-B0); RA=np.linalg.inv(np.eye(K)-A); s=g@R0@c
ok=np.allclose(RA, R0+np.outer(R0@c,g@R0)/(1-s)) and (RA>=R0-1e-12).all() and s<1
c0=connectedness(B0); cA=connectedness(A)
print(f"s={s:.3f}<1 | Sherman-Morrison exact & entrywise dominance: {ok} | total connectedness structural {c0['total']:.1f}% naive {cA['total']:.1f}%")
print("ranking preserved:", np.array_equal(np.argsort(c0['systemic']),np.argsort(cA['systemic'])), "| rho:", round(spectral_radius(B0),3), "->", round(spectral_radius(A),3))
print("PROP CONNECT:", "PASS" if ok and cA['total']>c0['total'] else "FAIL")
