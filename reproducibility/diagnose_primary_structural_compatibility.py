"""
Diagnose the 16 primary JBES designs (8 windows x 2 q; split S1, lag 1)
row-by-row.  This does not change the confirmatory specification.
It identifies which assets make the joint cone-restricted AR confidence set empty.

Output:
  crypto_common_drive_outputs_V3/tables/primary_ar_empty_row_diagnostic.csv
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.linalg import cholesky
from scipy.stats import f as fdist

if os.path.exists("pipeline_v4.py"):
    exec(open("pipeline_v4.py").read())
else:
    exec(open("harness_loader_v4.py").read())

if os.environ.get("CRYPTO_RAW_BINANCE"):
    RAW = Path(os.environ["CRYPTO_RAW_BINANCE"]).expanduser()
    RUN_DOWNLOAD = False

exec(open("cascade_cs.py").read())

PANEL_SYMBOLS = list(SYMBOLS_FULL)
EXTERNAL = ["XLMUSDT","EOSUSDT","ALGOUSDT","AAVEUSDT",
            "THETAUSDT","VETUSDT","ENJUSDT","MANAUSDT"]
S1 = (["XLMUSDT","EOSUSDT","ALGOUSDT","AAVEUSDT"],
      ["THETAUSDT","VETUSDT","ENJUSDT","MANAUSDT"])
DESIGN = dict(event_type="volume_burst", bin="1min", lags=60, half_life=10)
Q_GRID = [0.95,0.97]
HAC_LAGS = 12
ALPHA = 0.05
TIME_KNOTS = 8

def basket_scores(symbols, window):
    if isinstance(symbols,str):
        symbols=[symbols]
    panel_ext=load_window_panel(symbols,window,bin_size=DESIGN["bin"])
    present=[s for s in symbols if s in set(panel_ext["symbol"])]
    if len(present)<1:
        raise RuntimeError(f"basket {symbols}: no data")
    return continuous_activity_score(panel_to_wide_features(panel_ext,present))

def basket_proxy(act):
    Z=(act-act.mean(0))/(act.std(0)+1e-12)
    return pd.Series(Z.mean(1),index=act.index)

def shift_aligned(m_series,index,lag):
    m=m_series.reindex(index).ffill().fillna(0.0).to_numpy()
    if lag>0: return np.concatenate([np.zeros(lag),m[:-lag]])
    if lag<0: return np.concatenate([m[-lag:],np.zeros(-lag)])
    return m

def time_basis(T,n_knots=8):
    from scipy.interpolate import BSpline
    t=np.linspace(0,1,T)
    knots=np.concatenate([[0]*4,np.linspace(0,1,n_knots+2)[1:-1],[1]*4])
    B=np.column_stack([BSpline(knots,np.eye(n_knots+4)[j],3)(t)
                       for j in range(n_knots+4)])
    B=B[:,1:-1]
    return B-B.mean(0)

def row_geometry(Y,H,m1,m2,assets,alpha=0.05,hac_lags=12):
    T,K=Y.shape
    a1=a2=alpha/2.0
    Hc=H-H.mean(0)
    Xn=np.column_stack([np.ones(T),Hc])
    df=T-Xn.shape[1]

    HtHi=np.linalg.pinv(Hc.T@Hc)
    Ahat=np.vstack([ols(Xn,Y[:,z])[1:K+1] for z in range(K)])
    gamma=(HtHi@Hc.T@(m1-m1.mean()))[:K]

    Vhist=HtHi[:K,:K]
    Vhist=0.5*(Vhist+Vhist.T)
    qA=K*fdist.ppf(1-a1/K,K,df)
    target_norm=np.sqrt(qA)

    out=[]
    for z in range(K):
        kind,pieces=anderson_rubin_analytic(
            Y[:,z],Hc,m1,m2,alpha=a2/K,hac_lags=hac_lags)
        boxes=ar_pieces_to_dboxes(kind,pieces)

        base=dict(asset=assets[z],row=z,ar_kind=kind,
                  n_positive_boxes=len(boxes),qA=qA,
                  target_norm=target_norm)

        if not boxes:
            out.append({**base,"status":"AR_POSITIVE_EMPTY",
                        "required_radius_ratio":np.inf})
            continue

        if any(hi is None for _,hi in boxes):
            out.append({**base,"status":"AR_UNBOUNDED",
                        "required_radius_ratio":np.nan})
            continue

        vals=[]
        for blo,bhi in boxes:
            for c0 in (float(blo),float(bhi)):
                yy=Y[:,z]-c0*m1
                rr=yy-Xn@ols(Xn,yy)
                vals.append(float(rr@rr/df))
        sbar2=max(vals)
        Sig=max(sbar2,1e-14)*Vhist
        Sig=0.5*(Sig+Sig.T)+1e-12*np.eye(K)
        Lw=np.linalg.inv(cholesky(Sig,lower=True))

        best=None
        for box_id,(blo,bhi) in enumerate(boxes):
            A=cp.Variable(K)
            d=cp.Variable()
            t=cp.Variable(nonneg=True)
            cons=[
                cp.norm(Lw@(A-Ahat[z]),2)<=t,
                A-d*gamma>=0,
                d>=float(blo),
                d<=float(bhi),
            ]
            prob=cp.Problem(cp.Minimize(t),cons)
            try:
                prob.solve(solver="CLARABEL")
            except Exception:
                continue
            if prob.status not in ("optimal","optimal_inaccurate"):
                continue
            cand=(float(t.value),float(blo),float(bhi),float(d.value))
            if best is None or cand[0]<best[0]:
                best=cand

        if best is None:
            out.append({**base,"status":"SOLVER_NO_FEASIBLE_BOX",
                        "required_radius_ratio":np.inf})
            continue

        min_norm,blo,bhi,dstar=best
        ratio=(min_norm/target_norm)**2
        status=("FEASIBLE_AT_STATED_RADIUS"
                if ratio<=1.0005 else "ELLIPSOID_CONE_CONFLICT")
        out.append({**base,"status":status,
                    "d_lo":blo,"d_hi":bhi,"d_star":dstar,
                    "min_norm":min_norm,
                    "required_radius_ratio":ratio})
    return pd.DataFrame(out)

rows=[]
for w in WINDOWS_FULL:
    print("\nWINDOW",w["name"],flush=True)
    panel=load_window_panel(PANEL_SYMBOLS,w,bin_size=DESIGN["bin"])
    feats=panel_to_wide_features(panel,PANEL_SYMBOLS)
    acts={s:basket_scores(s,w) for s in EXTERNAL}
    def proxy_for(symbols):
        return basket_proxy(pd.concat([acts[s] for s in symbols],axis=1))
    mA_s=proxy_for(S1[0]); mB_s=proxy_for(S1[1])

    for q in Q_GRID:
        Y_df,_=build_events_from_features(feats,event_type=DESIGN["event_type"],q=q)
        keep=[s for s in Y_df.columns if Y_df[s].sum()>=MIN_EVENT_COUNT_PER_ASSET]
        Y_df=Y_df[keep]
        dummy=pd.DataFrame(0.0,index=Y_df.index,columns=keep)
        Y,H,_,idx=effective_design(Y_df,dummy,DESIGN["lags"],DESIGN["half_life"])
        T,K=Y.shape
        H=np.column_stack([H,time_basis(T,TIME_KNOTS)])
        mA=shift_aligned(mA_s,Y_df.index,1)[-T:]
        mB=shift_aligned(mB_s,Y_df.index,1)[-T:]

        d=row_geometry(Y,H,mA,mB,keep,alpha=ALPHA,hac_lags=HAC_LAGS)
        d.insert(0,"q",q)
        d.insert(0,"kind",w["kind"])
        d.insert(0,"window",w["name"])
        rows.append(d)

        bad=d[d.status!="FEASIBLE_AT_STATED_RADIUS"]
        print(f"  q={q:.2f}: incompatible rows {len(bad)}/{K}; "
              f"assets={bad.asset.tolist()}",flush=True)

ans=pd.concat(rows,ignore_index=True)
out=Path("crypto_common_drive_outputs_V3/tables/primary_ar_empty_row_diagnostic.csv")
ans.to_csv(out,index=False)

print("\nINCOMPATIBLE-ASSET COUNTS ACROSS 16 PRIMARY DESIGNS")
bad=ans[ans.status!="FEASIBLE_AT_STATED_RADIUS"]
tab=bad.groupby("asset").size().sort_values(ascending=False)
print(tab.to_string())

print("\nBY WINDOW")
print(bad.groupby(["window","q"]).size().to_string())

print("\nWROTE",out)
