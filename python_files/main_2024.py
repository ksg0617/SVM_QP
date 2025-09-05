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

def goals_100pct_feasible_with_report(train_snaps,
                                      val_snaps,
                                      n_goals=12,
                                      cap=0.02,        # cap at 2% monthly
                                      buffer_bp=1,     # subtract 1 bp for slack
                                      round_bp=5,      # 0.05% steps
                                      min_goal=0.001,  # ≥0.010% monthly
                                      train_cov_tol=1.0):
    """
    Returns an array of monthly return goals (decimals) that are feasible for
    *every* training month. Prints feasibility % for train & val for each goal.
    """
    max_mu_train = _max_mu_list(train_snaps)
    if max_mu_train.size == 0:
        raise ValueError("No training snapshots with mu_fore found.")

    # 100% train-feasible upper bound with a small safety buffer
    ub = min(float(np.min(max_mu_train)), cap) - buffer_bp / 10000.0
    if ub <= 0:
        raise ValueError(
            f"Upper bound for 100% train feasibility is non-positive ({ub:.6f}). "
            "No positive goal is feasible in all training months."
        )
    lb = max(min_goal, 0.25 * ub)   # start at 25% of ub (or min_goal)

    # raw grid → rounded grid
    raw = np.linspace(lb, ub, num=n_goals)
    step = round_bp / 10000.0
    goals = np.round(raw / step) * step
    goals = np.unique(goals[goals > 0])

    # Print coverage for all candidate goals
    print("\nCandidate monthly goals and feasibility:")
    for g in goals:
        tr = _coverage(train_snaps, g)
        va = _coverage(val_snaps,   g)
        print(f"  goal = {g:.4%} | train feas = {tr*100:5.1f}% | val feas = {va*100:5.1f}%")

    # Enforce 100% train feasibility (tolerate tiny float noise if desired)
    selected = []
    print("\nSelected goals (100% train-feasible):")
    for g in goals:
        tr = _coverage(train_snaps, g)
        if tr >= train_cov_tol:   # ==1.0 is fine; use 0.9999 if paranoid
            va = _coverage(val_snaps, g)
            selected.append(g)
            print(f"  goal = {g:.4%} | train feas = {tr*100:5.1f}% | val feas = {va*100:5.1f}%")

    if len(selected) == 0:
        raise ValueError("No goals remain after enforcing 100% train feasibility. "
                         "Reduce n_goals, lower min_goal, increase round granularity, "
                         "or inspect your forecasts.")

    return np.array(selected, dtype=float)

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


# create a map of the C_svm_init, tau, lambda_hinge to the grid case
grid_case_map = {
    (0.01, 0.01, 0.0): "A",
    #(0.01, 0.01, 0.01): "B",
    #(0.01, 0.01, 0.1): "C",
    #(0.01, 0.01, 1.0): "D",
    (0.01, 0.05, 0.0): "E",
    #(0.01, 0.05, 0.01): "F",
    #(0.01, 0.05, 0.1): "G",
    #(0.01, 0.05, 1.0): "H",
    (0.01, 0.1, 0.0): "I",
    #(0.01, 0.1, 0.01): "J",
    #(0.01, 0.1, 0.1): "K",
    #(0.01, 0.1, 1.0): "L",
    (0.1, 0.01, 0.0): "M",
    #(0.1, 0.01, 0.01): "N",
    #(0.1, 0.01, 0.1): "O",
    #(0.1, 0.01, 1.0): "P",
    (0.1, 0.05, 0.0): "Q",
    #(0.1, 0.05, 0.01): "R",
    #(0.1, 0.05, 0.1): "S",
    #(0.1, 0.05, 1.0): "T",
    (0.1, 0.1, 0.0): "U",
    #(0.1, 0.1, 0.01): "V",
    #(0.1, 0.1, 0.1): "W",
    #(0.1, 0.1, 1.0): "X",
    (1.0, 0.01, 0.0): "Y",
    #(1.0, 0.01, 0.01): "Z",
    #(1.0, 0.01, 0.1): "AA",
    #(1.0, 0.01, 1.0): "AB",
    (1.0, 0.05, 0.0): "AC",
    #(1.0, 0.05, 0.01): "AD",
    #(1.0, 0.05, 0.1): "AE",
    #(1.0, 0.05, 1.0): "AF",
    (1.0, 0.1, 0.0): "AG",
    #(1.0, 0.1, 0.01): "AH",
    #(1.0, 0.1, 0.1): "AI",
    #(1.0, 0.1, 1.0): "AJ"
}

# grid_case_map = {
#     (0.01, 0.01, 0.0): "A",
#     (0.01, 0.01, 0.01): "B",
#     (0.01, 0.01, 0.1): "C",
#     (0.01, 0.01, 1.0): "D",
#     (0.01, 0.05, 0.0): "E",
#     (0.01, 0.05, 0.01): "F",
#     (0.01, 0.05, 0.1): "G",
#     (0.01, 0.05, 1.0): "H",
#     (0.01, 0.1, 0.0): "I",
#     (0.01, 0.1, 0.01): "J",
#     (0.01, 0.1, 0.1): "K",
#     (0.01, 0.1, 1.0): "L",
#     (0.1, 0.01, 0.0): "M",
#     (0.1, 0.01, 0.01): "N",
#     (0.1, 0.01, 0.1): "O",
#     (0.1, 0.01, 1.0): "P",
#     (0.1, 0.05, 0.0): "Q",
#     (0.1, 0.05, 0.01): "R",
#     (0.1, 0.05, 0.1): "S",
#     (0.1, 0.05, 1.0): "T",
#     (0.1, 0.1, 0.0): "U",
#     (0.1, 0.1, 0.01): "V",
#     (0.1, 0.1, 0.1): "W",
#     (0.1, 0.1, 1.0): "X",
#     (1.0, 0.01, 0.0): "Y",
#     (1.0, 0.01, 0.01): "Z",
#     (1.0, 0.01, 0.1): "AA",
#     (1.0, 0.01, 1.0): "AB",
#     (1.0, 0.05, 0.0): "AC",
#     (1.0, 0.05, 0.01): "AD",
#     (1.0, 0.05, 0.1): "AE",
#     (1.0, 0.05, 1.0): "AF",
#     (1.0, 0.1, 0.0): "AG",
#     (1.0, 0.1, 0.01): "AH",
#     (1.0, 0.1, 0.1): "AI",
#     (1.0, 0.1, 1.0): "AJ"
# }


# grid_case_map = {
#     (0.01, 0.01, 0.001): "A",
#     (0.01, 0.05, 0.001): "E",
#     (0.01, 0.1, 0.001): "I",
#     (0.1, 0.01, 0.001): "M",
#     (0.1, 0.05, 0.001): "Q",
#     (0.1, 0.1, 0.001): "U",
#     (1.0, 0.01, 0.001): "Y",
#     (1.0, 0.05, 0.001): "AC",
#     (1.0, 0.1, 0.001): "AG"
# }


def main_2024():
    # 0) Data + snapshots
    df = factor_df_prep("feature_data/data.csv")
    snapshots = estimate_returns_covariance(df)

    # 1) Chronological split (non-crisis)
    train_snaps, val_snaps, test_snaps = split_no_crisis_for_goals(snapshots)

    # 2) Build & report a 100% train-feasible goal grid (also prints feasibility)
    goals = goals_100pct_feasible_with_report(
        train_snaps, val_snaps,
        n_goals=12,
        cap=0.02,        # 2% monthly cap
        buffer_bp=1,     # 1 bp slack from min(max mu)
        round_bp=5,      # 0.05% increments
        min_goal=0.001,  # ≥0.10% monthly
        train_cov_tol=1.0
    )
    print("\nSelected goals:", [f"{g:.2%}" for g in goals])

    rows = []

    # 3) Sweep (C_svm_init, tau, lambda_hinge) × goals
    for (C_svm_init, tau_init, lambda_hinge_init), grid_case in grid_case_map.items():
        for goal in goals:
            goal_tag = _tag_goal(goal)

            # (Optional) echo feasibility again next to this run
            tr_cov = _coverage(train_snaps, goal)
            va_cov = _coverage(val_snaps,   goal)
            print(f"\n=== Run {grid_case}-{goal_tag} | goal={goal:.2%} "
                  f"| train feas={tr_cov*100:.1f}% | val feas={va_cov*100:.1f}% ===")

            # 4) Train the NN for this (grid_case, goal)
            model, tr_snaps_m, va_snaps_m, te_snaps_m = train_no_crisis(
                snapshots, C_svm_init, tau_init, lambda_hinge_init,
                return_goal=goal, grid_case=f"{grid_case}_{goal_tag}"
            )

            # 5) Evaluate (time-series metrics) — SVM baseline trained on tr_snaps_m inside compute_stats
            tr_metrics = compute_stats(tr_snaps_m, model, goal, tr_snaps_m, C_svm_init)
            va_metrics = compute_stats(va_snaps_m, model, goal, tr_snaps_m, C_svm_init)
            te_metrics = compute_stats(te_snaps_m, model, goal, tr_snaps_m, C_svm_init)

            def add_row(split_name, metrics):
                mean_ret_nn, vol_nn, sharpe_nn, mean_ret_svm, vol_svm, sharpe_svm,_,_ = metrics
                rows.append({
                    "grid_case": grid_case,
                    "C_svm_init": C_svm_init,
                    "tau": tau_init,
                    "lambda_hinge": lambda_hinge_init,
                    "goal_monthly": goal,
                    "split": split_name,
                    # feasibility of the goal on the selection split (echoed for convenience)
                    "train_feas_pct": tr_cov * 100.0,
                    "val_feas_pct": va_cov * 100.0,
                    # NN metrics (monthly)
                    "nn_mean_ret": mean_ret_nn,
                    "nn_vol": vol_nn,
                    "nn_sharpe": sharpe_nn,
                    # Baseline SVM+MVO metrics (monthly)
                    "svm_mean_ret": mean_ret_svm,
                    "svm_vol": vol_svm,
                    "svm_sharpe": sharpe_svm,
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