from data_prep import *
from train import *
from analysis import *
import numpy as np
import pandas as pd

def _max_mu_list(snaps):
    vals = []
    for s in snaps:
        mu = np.asarray(s.get("mu_fore", []), dtype=float)
        if mu.size and np.isfinite(mu).any():
            vals.append(np.nanmax(mu))
    return np.array(vals, dtype=float)

def _coverage(snaps, goal):
    feas_flags = []
    for s in snaps:
        mu = np.asarray(s.get("mu_fore", []), dtype=float)
        if mu.size and np.isfinite(mu).any():
            feas_flags.append(np.nanmax(mu) >= goal)
    return float(np.mean(feas_flags)) if feas_flags else 0.0

def split_no_crisis_for_goals(snapshots, n_val=12, n_test=11):
    snaps = sorted(snapshots, key=lambda s: s["date"])[:-1]
    train_snaps  = snaps[:-(n_val + n_test)]
    val_snaps    = snaps[-(n_val + n_test):-n_test]
    test_snaps   = snaps[-n_test:]
    return train_snaps, val_snaps, test_snaps


def _tag_goal(goal: float, round_bp: int = 5) -> str:
    # e.g. 0.0100 -> "g_1_00" for 1.00%
    step = round_bp / 10000.0
    g = np.round(goal / step) * step
    return f"g_{int(g*100):d}_{int((g*100 - int(g*100))*100):02d}"

grid_case_map = {
(0.1, 0.1, 0.0): "U"
}

def main_2024():
    # 0) Data + snapshots
    #df = factor_df_prep("feature_data/data.csv") # non-crisis
    df = factor_df_prep("feature_data/crisis_data.csv")

    snapshots = estimate_returns_covariance(df)

    # 2) Build & report a 100% train-feasible goal grid (also prints feasibility)
    #goals = np.array([0.016]) for non-crisis
    goals = np.array([0.006])   # for crisis
    rows = []

    # 3) Sweep (C_svm_init, tau, lambda_hinge) × goals
    for (C_svm_init, tau_init, lambda_hinge_init), grid_case in grid_case_map.items():
        for goal in goals:
            goal_tag = _tag_goal(goal)

            # 4) Train the NN for this (grid_case, goal)
            model, tr_snaps_m, va_snaps_m, te_snaps_m = train_crisis(
                snapshots, C_svm_init, tau_init, lambda_hinge_init,
                return_goal=goal, grid_case=f"{grid_case}_{goal_tag}"
            )

            

            # 5) Evaluate (time-series metrics) — SVM baseline trained on tr_snaps_m inside compute_stats
            tr_metrics = compute_stats(tr_snaps_m, model, goal, tr_snaps_m, C_svm_init)
            va_metrics = compute_stats(va_snaps_m, model, goal, tr_snaps_m, C_svm_init)
            te_metrics = compute_stats(te_snaps_m, model, goal, tr_snaps_m, C_svm_init)

            def add_row(split_name, metrics):
                mean_ret_nn, vol_nn, sharpe_nn, mean_ret_svm, vol_svm, sharpe_svm, r_nn, r_svm = metrics
                rows.append({
                    "grid_case": grid_case,
                    "C_svm_init": C_svm_init,
                    "tau": tau_init,
                    "lambda_hinge": lambda_hinge_init,
                    "goal_monthly": goal,
                    "split": split_name,
                    # NN metrics (monthly)
                    "nn_mean_ret": mean_ret_nn,
                    "nn_vol": vol_nn,
                    "nn_sharpe": sharpe_nn,
                    "nn_r_series": r_nn,
                    # Baseline SVM+MVO metrics (monthly)
                    "svm_mean_ret": mean_ret_svm,
                    "svm_vol": vol_svm,
                    "svm_sharpe": sharpe_svm,
                    "svm_r_series": r_svm
                })

            add_row("train", tr_metrics)
            add_row("val",   va_metrics)
            add_row("test",  te_metrics)

    # 6) Save tidy results
    results_df = pd.DataFrame(rows)
    results_df.sort_values(["goal_monthly", "grid_case", "split"], inplace=True)
    results_df.to_csv("svm_qp_results.csv", index=False)
    print("\nResults saved to svm_qp_results.csv")
    print(results_df.head(10))

main_2024()