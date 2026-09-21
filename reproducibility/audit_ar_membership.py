"""10,000-replication audit of the MEMBERSHIP event {B0 in S_AR} of the Anderson-Rubin confidence set (Theorem AR):
  - per row: true loading c_z in the analytic AR set at level alpha2/K  (exact t pivot)
  - A0 in the row-wise ellipsoid with F-radius K*F_{K,T-K-1}(1-alpha1/K) and the max residual variance over the AR box
No convex programs needed. Designs: K=5 strong/moderate/weak; K=10 moderate; violation corr(e1,e2)=0.3; t(5) errors."""
import numpy as np, sys, time
from scipy.stats import f as fdist
from scipy.linalg import solve, cholesky
from cascade_cs import ols, anderson_rubin_analytic, ar_pieces_to_dboxes, spectral_radius, effective_F
rng=np.random.default_rng(0)
def make_B0(K): B=rng.uniform(0,1,(K,K))*(rng.random((K,K))<.7); np.fill_diagonal(B,.05); return .7*B/spectral_radius(B)
def run(label,K,T,nf2,tau,reps,rho_e=0.0,errors='gauss',alpha=0.05):
    B0=make_B0(K); c=rng.uniform(.6,1.4,K); a1=a2=alpha/2; qF=K*fdist.ppf(1-a1/K,K,T-K-1); cov=0; Fs=[]; t0=time.time()
    for r in range(reps):
        A_=.6*rng.standard_normal((K,K)); H=rng.standard_normal((T,K))@cholesky(np.eye(K)+A_@A_.T,lower=False); Hc=H-H.mean(0)
        g0=.25*rng.standard_normal(K); v=rng.standard_normal(T); v-=v.mean(); v-=Hc@solve(Hc.T@Hc,Hc.T@v); v*=np.sqrt(nf2)/np.linalg.norm(v); f=Hc@g0+v
        e=rng.standard_normal((T,2))@cholesky(np.array([[1,rho_e],[rho_e,1]]),lower=False)*tau; m1=f+e[:,0]; m2=f+e[:,1]
        eps=rng.standard_normal((T,K)) if errors=='gauss' else rng.standard_t(5,(T,K))/np.sqrt(5/3)
        Y=Hc@B0.T+np.outer(f,c)+eps; Xn=np.column_stack([np.ones(T),Hc]); HtH=Hc.T@Hc
        gam=solve(HtH,Hc.T@(f-f.mean())); A0=B0+np.outer(c,gam); m1hat=solve(HtH,Hc.T@(m1-m1.mean()))
        ok=True; qmax=0.0
        for z in range(K):
            kind,pieces=anderson_rubin_analytic(Y[:,z],Hc,m1,m2,a2/K)
            if not any(lo<=c[z]<=hi for lo,hi in pieces): ok=False; break
            for lo,hi in ar_pieces_to_dboxes(kind,pieces):
                for c0 in (lo, hi if hi is not None else lo+3):   # residual variance is convex in c0 -> max at box ends (unbounded: cap)
                    yy=Y[:,z]-c0*m1; rr=yy-Xn@ols(Xn,yy); qmax=max(qmax, rr@rr/(T-K-1))
        if ok:
            for z in range(K):
                Ahat=ols(Xn,Y[:,z])[1:K+1]; dv=Ahat-A0[z]
                if dv@HtH@dv/qmax > qF: ok=False; break
        cov+=ok
        if r<200: Fs.append(effective_F(m1,m2,Hc))
    print(f"{label:34s} K={K} T={T} F~{np.median(Fs):8.1f} | membership coverage {cov/reps:.4f} (MC se {np.sqrt(cov/reps*(1-cov/reps)/reps):.4f})  [{time.time()-t0:.0f}s]",flush=True)
R=int(sys.argv[1]) if len(sys.argv)>1 else 10000
if len(sys.argv)>2 and sys.argv[2]=="B":
    run("K=10 moderate",10,2000,60.,.5,R); run("VIOLATION corr(e1,e2)=0.3, moderate",5,2000,60.,.5,R,rho_e=0.3); run("t(5) errors, moderate",5,2000,60.,.5,R,errors='t5')
else:
    run("strong (nf2=1200,tau=.5)",5,2000,1200.,.5,R); run("moderate (nf2=60,tau=.5)",5,2000,60.,.5,R); run("weak (nf2=8,tau=.7)",5,2000,8.,.7,R)
