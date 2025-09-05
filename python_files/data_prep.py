import pandas as pd
from typing import Dict, List, Tuple
import numpy as np


# ----- helper #1 :   β̂  and  F̂  --------------------------------
def fit_beta(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1) strip rows that contain any non-finite value
    row_mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    Xc, Yc   = X[row_mask], Y[row_mask]

    beta, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)               # (d_x, d_y)
    resid    = Yc - Xc @ beta                                    # (n, d_y)
    F_hat    = np.diag(resid.var(axis=0, ddof=1))              # (d_y, d_y)
    return beta, F_hat


# ----- helper #2 :   one-step mean & cov ------------------------
def forecast_one_step(x_i, W_i, beta, F_hat):
    y_hat = x_i @ beta                                         # (d_y,)
    V_hat = beta.T @ W_i @ beta + F_hat                        # (d_y, d_y)
    return y_hat, V_hat

# --- helper --------------------------------------------------------
def is_spd(mat: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Return True if `mat` is symmetric-positive-definite.
    • First check symmetry (fast) – avoids false negatives from tiny asymmetry
    • Then try a Cholesky factorisation.  If it succeeds, SPD.
    • `tol` lets you ignore round-off noise on symmetry test.
    """
    if not np.allclose(mat, mat.T, atol=tol, rtol=0):
        return False
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False

def factor_df_prep(data_path: str):
    df = pd.read_csv(data_path)
    df = df.drop(columns=['Unnamed: 0'])

    df['label'] = df.groupby('ticker')['close_price'].shift(-1).gt(df['close_price']).astype(int) * 2 - 1

    df['ret'] = df.groupby('ticker')['close_price'].pct_change().shift(-1).fillna(0)

    df['ret_excess'] = df['ret']

    return df

def estimate_returns_covariance(df):

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    # -------------------------------------------------------------
    # 1.  Build monthly factor (X_df) and asset-return (Y_df) tables

    Y_df = (df
            .pivot(index='date', columns='ticker', values='ret_excess') # ret_excess can be used or ret
            .sort_index())

    X_df = (df[['date'] + factor_cols]
            .drop_duplicates('date')
            .set_index('date')
            .sort_index())
    # create a label_df
    label_df = (df
                .pivot(index='date', columns='ticker', values='label')
                .sort_index())

    # ------------------------------------------------------------------
    # 1.  Find ticker columns with *any* NaNs in Y_df
    # ------------------------------------------------------------------
    bad_tickers = Y_df.columns[Y_df.isna().any()]

    print(f"🗑  Dropping {len(bad_tickers)} tickers with missing returns:")
    print(", ".join(map(str, bad_tickers)))

    # ------------------------------------------------------------------
    # 2.  Drop them from Y_df  (axis=1 ⇒ columns)
    # ------------------------------------------------------------------
    Y_df = Y_df.drop(columns=bad_tickers)

    # ------------------------------------------------------------------
    # 3.  OPTIONAL: keep your other pivot tables in sync
    #     (uncomment if you have X_df, label_df, etc. with same columns)
    # ------------------------------------------------------------------
    # X_df      = X_df.drop(columns=bad_tickers, errors="ignore")
    label_df  = label_df.drop(columns=bad_tickers, errors="ignore")
    # Sigma_fore = Sigma_fore[np.ix_(good_mask, good_mask)]  # inside loop

    # ------------------------------------------------------------------
    # 4.  Verify — there should be zero NaNs left
    # ------------------------------------------------------------------
    assert not Y_df.isna().any().any(), "Still NaNs lurking in Y_df!"
    print("✅  Y_df is now NaN-free and has", Y_df.shape[1], "tickers.")

    # 1) make sure every index really *is* a DatetimeIndex
    X_df.index      = pd.to_datetime(X_df.index)
    Y_df.index      = pd.to_datetime(Y_df.index)
    label_df.index  = pd.to_datetime(label_df.index)
    df['date'] = pd.to_datetime(df['date'])

    # Every row is month-end already ⇒ just iterate over the index
    month_ends = X_df.index.sort_values()

    lookback = 12          # e.g. use the past 12 months
    month_ends = month_ends[lookback:]

    def make_spd(M, eps=1e-6):
        M = (M + M.T) * 0.5  # enforce symmetry
        jitter = eps
        I = np.eye(M.shape[0])
        for _ in range(10):
            try:
                # This will error if M is not SPD
                np.linalg.cholesky(M)
                return M
            except np.linalg.LinAlgError:
                M = M + jitter * I
                jitter *= 10
        raise RuntimeError("Unable to make SPD matrix")

    results = []
    tickers = Y_df.columns.unique()
    for me_date in month_ends:

        # ---------- 1) pick the estimation window ------------
        win_mask  = (X_df.index <= me_date)                     & \
                    (X_df.index >  me_date - pd.offsets.MonthEnd(lookback))

        X_window  = X_df.loc[win_mask]
        Y_window  = Y_df.loc[win_mask]

        if len(X_window) < 2:
            continue   # still not enough data – skip

        # ---------- 2)  β̂ , F̂  from the window --------------
        beta_hat, F_hat = fit_beta(X_window.values, Y_window.values)

        # ---------- 3)  Σ̂ (covariance)  ----------------------
        # but most people just use a sample/Exp-Wtd cov here:
        W_hat = np.cov(X_window.values.T, ddof=1)
        #W_hat = dcc_garch_cov(X_window.values)
        
        x_today      = X_df.loc[me_date].values
        mu_fore, Sigma_fore = forecast_one_step(x_today, W_hat, beta_hat, F_hat)
        #  └─ make sure your forecast function is set up for “+1 month”,
        #     not “+1 trading day”.

        W = 12
        R_hist = (Y_df.loc[Y_df.index <= me_date]
                    .tail(W)         # 12 rows = 12 months
                    .T.values)       # (n_assets × W)
                
        # 5) features ----------------------------------------
        feature_list = ['exp_ret', 'vol','mom', 'pe_exi', 'de_ratio', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        monthly = (df
                .set_index(['date','ticker'])
                .loc[me_date]
                .reindex(tickers))
        X_feat = monthly[feature_list].values

        # Standardize X_feat now
        mean = X_feat.mean(axis=0, keepdims=True)
        std = X_feat.std(axis=0, keepdims=True) + 1e-8  # to avoid division by zero
        X_feat = (X_feat - mean) / std
        
        # 6) labels ------------------------------------------
        y_vec = label_df.loc[me_date, tickers].values

        # 7) get row of Y_df for the current month-end
        real_returns = Y_df.loc[me_date, tickers].values

        # 8) get the real covariance matrix for the current month-end
        real_sigma = np.cov(Y_df.loc[Y_df.index <= me_date].tail(W).T, ddof=1)
        real_sigma = make_spd(real_sigma)
        # 9) store snapshot ----------------------------------
        results.append(dict(date         = me_date,
                            X_feat       = X_feat,
                            returns_hist = R_hist,
                            y            = y_vec.astype(np.double),
                            mu_fore      = mu_fore.astype(np.double),
                            Sigma_fore   = Sigma_fore.astype(np.double),
                            beta         = beta_hat,
                            F_hat        = F_hat,
                            W_hat        = W_hat,
                            real_returns = real_returns.astype(np.double),
                            real_sigma   = real_sigma.astype(np.double)))
        

    # --- scan the results list ----------------------------------------
    bad = []                 # collect (index, date) for matrices that fail
    for i, snap in enumerate(results):
        if not is_spd(snap["Sigma_fore"]):
            bad.append((i, snap["date"]))
        if not is_spd(snap["real_sigma"]):
            bad.append((i, snap["date"]))
    # --- quick report --------------------------------------------------
    total = len(results)
    print(f"Σ̂ SPD check: {total - len(bad)}/{total} pass, {len(bad)} fail")

    if bad:
        print("First few failures:")
        for idx, d in bad[:10]:
            print(f"  #{idx:3d}  {d}")
    
    return results
