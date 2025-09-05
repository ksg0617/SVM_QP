from data_prep import *
from train import *
from analysis import *


def test():
    # create a map of the C_svm_init, tau, lambda_hinge to the grid case
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
    grid_case_map = {
        (0.01, 0.01, 0.0): "A",
        (20.0, 0.01, 0.0): "B"}
    
    df = factor_df_prep("feature_data/data.csv")
    snapshots_all = estimate_returns_covariance(df)

    train_nn_ret_list, train_nn_var_list, train_nn_sharpe_list = [], [], []
    train_svm_ret_list, train_svm_var_list, train_svm_sharpe_list = [], [], []
    val_nn_ret_list, val_nn_var_list, val_nn_sharpe_list = [], [], []
    val_svm_ret_list, val_svm_var_list, val_svm_sharpe_list = [], [], []
    test_nn_ret_list, test_nn_var_list, test_nn_sharpe_list = [], [], []
    test_svm_ret_list, test_svm_var_list, test_svm_sharpe_list = [], [], []
    
    # iterate over the grid cases
    for (C_svm_init, tau_init, lambda_hinge_init), grid_case in grid_case_map.items():
        print(f"Running grid case: {grid_case} with C_svm_init={C_svm_init}, tau_init={tau_init}, lambda_hinge_init={lambda_hinge_init}")
        
        # Train the model and get the snapshots
        return_goal = 0.035
        grid_case = grid_case_map[(C_svm_init, tau_init, lambda_hinge_init)]
        print(f"Grid case: {grid_case}")
        snapshots = sorted(snapshots_all, key=lambda s: s["date"])   # ensure sorted
        # drop the last 1 snapshot, it is incomplete
        snapshots= snapshots[:-1]  # drop the last snapshot, it is incomplete

        #val_fraction = 0.2
        #n_total      = len(snapshots)
        #n_val        = int(n_total * val_fraction)
        n_val = 12  # e.g. second last 18 months for validation
        n_test = 11  # e.g. last 12 months for testing
        train_snaps  = snapshots[:-(n_val + n_test)]          # earlier dates
        val_snaps    = snapshots[-(n_val + n_test):-n_test]          # second last 12 months
        test_snaps   = snapshots[-n_test:]     

        svm_clf = train_svm(C_svm_init, train_snaps)

        rets_svm, vars_svm, sharpe_svm_list = [], [], []

        for snap in test_snaps:
            X, y, mu, Sigma, real_ret, real_sigma = snap["X_feat"], snap["y"], snap["mu_fore"], snap["Sigma_fore"], snap["real_returns"], snap["real_sigma"]
            # Check if SVM has more than one class
            if len(np.unique(y)) < 2:
                continue
            
            # check if mu_fore is feasible
            if mu.max() < return_goal:
                print(f"Skipping snapshot: return_goal={return_goal} not feasible (max mu = {mu.max():.4f})")
                continue
        
            mu_np = np.asarray(mu)
            Sigma_np = np.asarray(Sigma)

            real_mu_np = np.asarray(real_ret)
            real_sigma_np = np.asarray(real_sigma)

            # ---- SVM then MVO ----
            scores = svm_clf.decision_function(X)
            selected = scores > 0
            if selected.sum() < 2:
                continue  # skip this snapshot

            mu_sel = mu_np[selected]
            Sigma_sel = Sigma_np[np.ix_(selected, selected)]
            real_mu_sel = real_mu_np[selected]
            real_sigma_sel = real_sigma_np[np.ix_(selected, selected)]

            w_svm = solve_mvo(np.ones(len(mu_sel)), mu_sel, Sigma_sel, return_goal)
            if w_svm is None:
                continue

            ret_svm = real_mu_sel @ w_svm
            var_svm = w_svm @ real_sigma_sel @ w_svm
            sharpe_svm = ret_svm / (np.sqrt(var_svm) + 1e-8)

            rets_svm.append(ret_svm)
            vars_svm.append(var_svm)
            sharpe_svm_list.append(sharpe_svm)
        mean_ret_svm = np.mean(rets_svm)
        mean_var_svm = np.mean(vars_svm)
        mean_sharpe_svm = np.mean(sharpe_svm_list)
        print(f"SVM:  Mean Return = {mean_ret_svm:.4f}, Variance = {mean_var_svm:.4f}, Sharpe (avg monthly) = {mean_sharpe_svm:.4f}")
        
        test_svm_ret_list.append(mean_ret_svm)
        print(test_svm_ret_list)

test()
