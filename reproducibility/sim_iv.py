"""Paper 2 Monte Carlo: unobserved market factor, two out-of-panel proxies m1=f+e1 (regressor), m2=f+e2 (instrument).
Estimators of rho(B0): naive (constant baseline), OLS with proxy m1, 2SLS (m2 instruments m1).
Inference: 2SLS-Wald delta interval vs AR-projected cone-restricted interval (certified outer bounds):
   S_AR = { A - d m1hat^T >= 0 : A in E_A(q_A), d in AR-box ∩ R+ }   with A = naive H-coefficients (exact Gaussian),
   m1hat = projection coefficient of m1 on H, and per-row AR set for c/theta1 inverted on a grid.
Honest accounting: empty sets and grid-capped AR sets are handled explicitly; empties count as failures."""
import numpy as np, cvxpy as cp, sys, time
from scipy.stats import chi2, norm, t as tdist
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize
from cascade_cs import ols, perron, spectral_radius, two_stage_ls, anderson_rubin_analytic, ar_pieces_to_dboxes
from scipy.stats import f as fdist
rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 0)

def make_B0(K,target=0.70):
    B=rng.uniform(0,1,(K,K))*(rng.random((K,K))<0.7); np.fill_diagonal(B,0.05); return target*B/spectral_radius(B)

def outer_interval(Ahat,SigA,gamma,dbox,qA,K,nx=14):
    Lw=np.linalg.inv(cholesky(SigA,lower=True)); A=cp.Variable((K,K)); d=cp.Variable(K)
    B=A-cp.reshape(d,(K,1),order='C')@np.reshape(gamma,(1,K))
    cons=[cp.sum_squares(Lw@(A[z]-Ahat[z]))<=qA for z in range(K)]+[B>=0,d>=dbox[0]]+[d[z]<=dbox[1][z] for z in range(K) if dbox[1][z] is not None]
    fe=cp.Problem(cp.Minimize(0),cons); fe.solve(solver='CLARABEL')
    if fe.status in('infeasible','infeasible_inaccurate'): return None
    _,u,v=perron(np.maximum(Ahat,0)); xs=[v/v.sum(),np.ones(K)/K]+[np.abs(rng.standard_normal(K)) for _ in range(nx)]
    U=np.inf; L=-np.inf
    for x in xs:
        x=x/x.sum()
        vals=[]
        for i in range(K):
            pr=cp.Problem(cp.Maximize((B@x)[i]/x[i]),cons); val=pr.solve(solver='CLARABEL'); vals.append(np.inf if pr.status=='unbounded' else val)
        U=min(U,max(vals))
        L=max(L,min(cp.Problem(cp.Minimize((B@x)[i]/x[i]),cons).solve(solver='CLARABEL') for i in range(K)))
    return max(L,0.0),U

def run(label,T,nf2,tau,reps,K,B0,c,sigma,alpha=0.05):
    a1=a2=alpha/2; out=dict(bias_naive=[],bias_ols=[],bias_2sls=[],F=[],cov_wald=0,cov_ar=0,empty=0,unb=0,w_ar=[]); t0=time.time()
    for r in range(reps):
        A_=0.6*rng.standard_normal((K,K)); SH=np.eye(K)+A_@A_.T; H=rng.standard_normal((T,K))@cholesky(SH,lower=False); Hc=H-H.mean(0)
        g0=0.25*rng.standard_normal(K); v=rng.standard_normal(T); v-=v.mean(); v-=Hc@solve(Hc.T@Hc,Hc.T@v); v*=np.sqrt(nf2)/np.linalg.norm(v)
        f=Hc@g0+v; m1=f+tau*rng.standard_normal(T); m2=f+tau*rng.standard_normal(T)
        Y=Hc@B0.T+np.outer(f,c)+sigma*rng.standard_normal((T,K)); r0=spectral_radius(B0)
        Xn=np.column_stack([np.ones(T),Hc]); X1=np.column_stack([np.ones(T),Hc,m1])
        An=np.vstack([ols(Xn,Y[:,z])[1:K+1] for z in range(K)]); Bo=np.vstack([ols(X1,Y[:,z])[1:K+1] for z in range(K)])
        B2,c2,F=two_stage_ls(Y,Hc,m1,m2)
        out['bias_naive'].append(spectral_radius(np.maximum(An,0))-r0); out['bias_ols'].append(spectral_radius(np.maximum(Bo,0))-r0)
        out['bias_2sls'].append(spectral_radius(np.maximum(B2,0))-r0); out['F'].append(F)
        # 2SLS-Wald (delta) interval at cone-projected 2SLS point
        Z=np.column_stack([np.ones(T),Hc,m2]); Xh=Z@ols(Z,X1); rh,u,vv=perron(np.maximum(B2,0)); XhtXhi=np.linalg.inv(Xh.T@Xh)
        var=0.0
        for z in range(K):
            res=Y[:,z]-X1@ols(Xh,Y[:,z]); s2=res@res/(T-K-2); Sig=s2*XhtXhi[1:K+1,1:K+1]; var+=u[z]**2*(vv@Sig@vv)/(u@vv)**2
        w=1.96*np.sqrt(var); out['cov_wald']+= (rh-w<=r0<=rh+w)
        # AR-projected set: per-row AR for c/theta1 (theta=1 here), box ∩ R+, capped by cone bound; A-ellipsoid with max residual var over box
        m1hat=solve(Hc.T@Hc,Hc.T@(m1-m1.mean())); lo=np.zeros(K); hi=[None]*K; qmax=0.0; unb=False; empty=False
        HtHi=np.linalg.inv(Hc.T@Hc)
        for z in range(K):
            kind,pieces=anderson_rubin_analytic(Y[:,z],Hc,m1,m2,a2/K)          # exact pivot, level alpha2/K per row
            boxes=ar_pieces_to_dboxes(kind,pieces)
            if not boxes: empty=True; break
            # conservative envelope of the union of boxes: [min lo, max hi] (complement -> unbounded above)
            lo[z]=min(b[0] for b in boxes); hi[z]=None if any(b[1] is None for b in boxes) else max(b[1] for b in boxes)
            if hi[z] is None: unb=True
            for c0 in (lo[z], hi[z] if hi[z] is not None else lo[z]+3):
                yy=Y[:,z]-c0*m1; rr=yy-Xn@ols(Xn,yy); qmax=max(qmax, rr@rr/(T-K-1))
        if empty: out['empty']+=1; continue
        out['unb']+=unb
        SigA=qmax*HtHi; qA=K*fdist.ppf(1-a1/K,K,T-K-1)                          # F radius (estimated variance)
        res=outer_interval(An,SigA,m1hat,(lo,hi),qA,K)
        if res is None: out['empty']+=1; continue
        L,U=res; out['cov_ar']+=(L-1e-6<=r0<=U+1e-6); out['w_ar'].append(U-L if np.isfinite(U) else np.nan)
    n=reps
    print(f"{label:22s} F={np.mean(out['F']):7.1f} | bias rho: naive {np.mean(out['bias_naive']):+.3f} ols-proxy {np.mean(out['bias_ols']):+.3f} 2sls {np.mean(out['bias_2sls']):+.3f} (sd {np.std(out['bias_2sls']):.2f})"
          f" | cov: plug-in 2SLS delta {out['cov_wald']/n:.3f}  AR-proj {out['cov_ar']/n:.3f} (empty {out['empty']}, AR-unbounded {out['unb']}) width {np.nanmean(out['w_ar']) if out['w_ar'] else float('nan'):.2f} (inf-hi {int(np.sum(np.isnan(out['w_ar'])))})  [{time.time()-t0:.0f}s]",flush=True)

K=5; B0=make_B0(K); c=rng.uniform(0.6,1.4,K); sigma=1.0; T=2000
reg=sys.argv[1]
cfg={'strong':(1200.0,0.5),'moderate':(60.0,0.5),'weak':(8.0,0.7)}[reg]
run(reg,T,cfg[0],cfg[1],50,K,B0,c,sigma)
