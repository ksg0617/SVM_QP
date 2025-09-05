import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import cvxpy as cp
import torch

def solve_mvo(mask, mu, Sigma, return_goal):
    # mask: length-n (upper bounds, or 1 for selected assets)
    import cvxpy as cp
    n = len(mu)
    w = cp.Variable(n)
    constraints = [
        cp.sum(w) == 1,
        mu @ w >= return_goal,
        w >= 0,
        w <= mask
    ]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
    try:
        prob.solve(solver=cp.OSQP)
        if w.value is not None:
            return np.array(w.value).flatten()
    except Exception as e:
        pass
    return None

def train_svm(C_svm, train_snaps):
    # 1. Pre-train SVM classifier once, on all historical training data
    X_train = np.vstack([s["X_feat"] for s in train_snaps])
    print("X_train shape:", X_train.shape)
    y_train = np.hstack([s["y"] for s in train_snaps])
    print("y_train shape:", y_train.shape)
    svm_clf = SVC(kernel='linear', C=C_svm)
    svm_clf.fit(X_train, y_train)
    return svm_clf

def compute_stats(snaps, model, return_goal, train_snaps, C_svm_init):
    
    r_nn, r_svm = [], []             # realized monthly portfolio returns

    svm_clf = train_svm(C_svm_init, train_snaps)

    for snap in snaps:
        X = snap["X_feat"]; y = snap["y"]
        mu = snap["mu_fore"]; Sigma = snap["Sigma_fore"]
        real_mu = snap["real_returns"]; real_sigma = snap["real_sigma"]

        # Check if SVM has more than one class
        if len(np.unique(y)) < 2:
            print("Skipping snapshot: SVM has only one class")
            continue
        
        # check if mu_fore is feasible
        if mu.max() < return_goal:
            print(f"Skipping snapshot: return_goal={return_goal} not feasible (max mu = {mu.max():.4f})")
            continue
        
        # ---- NN Efficient Frontier ----
        X_t = torch.tensor(X).double(); y_t = torch.tensor(y).double()
        mu_t = torch.tensor(mu).double(); S_t = torch.tensor(Sigma).double()

        with torch.no_grad():
            w_nn, mask, _, _, _ = model(X_t, y_t, mu_t, S_t, return_goal=float(return_goal))
        w_nn = np.asarray(w_nn.cpu().numpy())

        r_nn.append(float(np.dot(real_mu, w_nn)))


        # ---- SVM then MVO ----
        scores = svm_clf.decision_function(X)
        sel = scores > 0
        if sel.sum() < 2:
            print("Skipping snapshot: not enough selected assets for SVM")
            continue  # skip this snapshot

        w_svm = solve_mvo(np.ones(sel.sum()), mu[sel], Sigma[np.ix_(sel, sel)], return_goal)

        if w_svm is None:
            continue
        r_svm.append(float(np.dot(real_mu[sel], w_svm)))


    def series_metrics(r, exvar=None):
        r = np.asarray(r, dtype=float)
        if r.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        mean_ret = r.mean()
        vol = r.std(ddof=1) if r.size > 1 else 0.0
        sharpe = mean_ret / (vol + 1e-12)
        avg_exvar = np.mean(exvar) if exvar is not None and len(exvar) == len(r) else np.nan
        return mean_ret, vol, sharpe, avg_exvar
    

    mean_ret_nn, vol_nn, sharpe_nn, avg_exvar_nn = series_metrics(r_nn)
    mean_ret_svm, vol_svm, sharpe_svm, avg_exvar_svm = series_metrics(r_svm)

    # Print a concise summary
    print(f"NN:  mean r = {mean_ret_nn:.4f}, vol = {vol_nn:.4f}, Sharpe = {sharpe_nn:.4f}, avg ex-post var = {avg_exvar_nn:.4f}")
    print(f"SVM: mean r = {mean_ret_svm:.4f}, vol = {vol_svm:.4f}, Sharpe = {sharpe_svm:.4f}, avg ex-post var = {avg_exvar_svm:.4f}")

    return mean_ret_nn, vol_nn, sharpe_nn, mean_ret_svm, vol_svm, sharpe_svm, r_nn, r_svm
    #return mean_ret_nn, vol_nn, sharpe_nn, mean_ret_svm, vol_svm, sharpe_svm