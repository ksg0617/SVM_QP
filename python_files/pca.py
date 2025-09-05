import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Suppose you trained two models
from train import train_no_crisis
from data_prep import *

def split_no_crisis_for_goals(snapshots, n_val=12, n_test=11):
    snaps = sorted(snapshots, key=lambda s: s["date"])[:-1]
    train_snaps  = snaps[:-(n_val + n_test)]
    val_snaps    = snaps[-(n_val + n_test):-n_test]
    test_snaps   = snaps[-n_test:]
    return train_snaps, val_snaps, test_snaps


def get_support_vectors(model, X_feat, y, mu, Sigma, return_goal, tol=1e-6):
    with torch.no_grad():
        X_t = torch.as_tensor(X_feat, dtype=torch.double)
        y_t = torch.as_tensor(y, dtype=torch.double)
        mu_t = torch.as_tensor(mu, dtype=torch.double)
        Sigma_t = torch.as_tensor(Sigma, dtype=torch.double)

        _, mask, _, C_svm, alpha = model(X_t, y_t, mu_t, Sigma_t, return_goal)
        alpha = alpha.cpu().numpy()
        C_val = C_svm.item()

        support_idx = (alpha > tol) & (alpha < C_val - tol)
    return support_idx

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_hyperplane_side_by_side(model_low, model_high, snapshot,
                                 return_goal_low, return_goal_high,
                                 save_path="hyperplane_side_by_side.png"):
    """
    Visualize hyperplanes learned under two different return goals
    as side-by-side PCA 2D plots.
    """
    # --- 1) Unpack snapshot
    X_feat = snapshot["X_feat"]
    y      = snapshot["y"]
    mu     = snapshot["mu_fore"]
    Sigma  = snapshot["Sigma_fore"]

    X_feat = X_feat.numpy() if torch.is_tensor(X_feat) else X_feat
    y = y.numpy() if torch.is_tensor(y) else y

    # --- 2) PCA projection to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_feat)

    # --- 3) Meshgrid for decision boundary
    x_min, x_max = X_2d[:, 0].min()-0.5, X_2d[:, 0].max()+0.5
    y_min, y_max = X_2d[:, 1].min()-0.5, X_2d[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_back = pca.inverse_transform(grid_points)

    def get_grid_scores(model, return_goal):
        with torch.no_grad():
            # Embed training features for w_svm
            X_t = torch.as_tensor(X_feat, dtype=torch.double)
            y_t = torch.as_tensor(y, dtype=torch.double)
            mu_t = torch.as_tensor(mu, dtype=torch.double)
            Sigma_t = torch.as_tensor(Sigma, dtype=torch.double)

            # Forward pass (gives mask)
            _, mask, _, _, alpha = model(X_t, y_t, mu_t, Sigma_t, return_goal)
            alpha_y = alpha

            # Compute w_svm direction
            w_svm = model.embed(X_t).t().mv(alpha_y)

            # Scores for grid
            Xg = torch.as_tensor(grid_back, dtype=torch.double)
            Xg_emb = model.embed(Xg)
            scores = Xg_emb @ w_svm
        return scores.view(xx.shape).numpy()

    Z_low  = get_grid_scores(model_low,  return_goal_low)
    Z_high = get_grid_scores(model_high, return_goal_high)

    # --- 4) Side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Left: low goal
    axes[0].scatter(X_2d[:,0], X_2d[:,1], c=y, cmap="bwr", edgecolor="k", alpha=0.7)
    axes[0].contour(xx, yy, Z_low, levels=[0], colors="blue", linewidths=2)
    axes[0].set_title(f"Low goal = {return_goal_low:.2%}")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].grid(True)

    # Right: high goal
    axes[1].scatter(X_2d[:,0], X_2d[:,1], c=y, cmap="bwr", edgecolor="k", alpha=0.7)
    axes[1].contour(xx, yy, Z_high, levels=[0], colors="red", linewidths=2)
    axes[1].set_title(f"High goal = {return_goal_high:.2%}")
    axes[1].set_xlabel("PCA 1")
    axes[1].grid(True)

    plt.suptitle("Hyperplane shifts under different return goals", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved side-by-side hyperplane plot to {save_path}")


def plot_hyperplane_shift(model_low, model_high, snapshot,
                          return_goal_low, return_goal_high,
                          save_path="hyperplane_shift.png", tol=1e-3):
    """
    Compare hyperplanes learned under two different return goals by projecting features into 2D
    and plotting the separating boundaries. Highlights *support vectors* only.
    """

    # --- Unpack snapshot ---
    X_feat = snapshot["X_feat"]
    y      = snapshot["y"]
    mu     = snapshot["mu_fore"]
    Sigma  = snapshot["Sigma_fore"]

    X_feat = X_feat.numpy() if torch.is_tensor(X_feat) else X_feat
    y      = y.numpy() if torch.is_tensor(y) else y

    # --- PCA projection to 2D for visualization ---
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_feat)

    # --- Helper: extract support vectors from alphas ---
    def get_support_vectors(model, return_goal):
        with torch.no_grad():
            X_t     = torch.as_tensor(X_feat, dtype=torch.double)
            y_t     = torch.as_tensor(y, dtype=torch.double)
            mu_t    = torch.as_tensor(mu, dtype=torch.double)
            Sigma_t = torch.as_tensor(Sigma, dtype=torch.double)

            _, mask, _, C_svm, alpha = model(X_t, y_t, mu_t, Sigma_t, return_goal)
            alpha = alpha.cpu().numpy()
            C_val = C_svm.item()
            return (alpha > tol) & (alpha < C_val - tol)

    support_low  = get_support_vectors(model_low,  return_goal_low)
    support_high = get_support_vectors(model_high, return_goal_high)

    # --- Meshgrid for decision boundary in PCA plane ---
    x_min, x_max = X_2d[:,0].min()-0.5, X_2d[:,0].max()+0.5
    y_min, y_max = X_2d[:,1].min()-0.5, X_2d[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_back   = pca.inverse_transform(grid_points)

    def get_grid_scores(model, return_goal):
        with torch.no_grad():
            Xg = torch.as_tensor(grid_back, dtype=torch.double)
            Xg_emb = model.embed(Xg)

            X_t = torch.as_tensor(X_feat, dtype=torch.double)
            y_t = torch.as_tensor(y, dtype=torch.double)
            mu_t = torch.as_tensor(mu, dtype=torch.double)
            Sigma_t = torch.as_tensor(Sigma, dtype=torch.double)

            _, mask, _, _, alpha = model(X_t, y_t, mu_t, Sigma_t, return_goal)
            alpha_y = (alpha * y_t).view(-1)  # true α·y
            w_svm   = model.embed(X_t).t().mv(alpha_y)
            scores  = Xg_emb @ w_svm
        return scores.view(xx.shape).numpy()

    Z_low  = get_grid_scores(model_low,  return_goal_low)
    Z_high = get_grid_scores(model_high, return_goal_high)

    # --- Plot ---
    plt.figure(figsize=(8,6))

    # faint background cloud of all assets
    plt.scatter(X_2d[:,0], X_2d[:,1], c="lightgray", alpha=0.2, s=20, label="All assets")

    # support vectors (low goal)
    plt.scatter(X_2d[support_low,0], X_2d[support_low,1],
                edgecolors='blue', facecolors='none', s=120, linewidths=2,
                label="Support Vectors (Low goal)")

    # support vectors (high goal)
    plt.scatter(X_2d[support_high,0], X_2d[support_high,1],
                marker='*', color='gold', s=150,
                label="Support Vectors (High goal)")

    # boundaries
    plt.contour(xx, yy, Z_low,  levels=[0], colors="blue", linestyles="--", linewidths=2, label="Low goal boundary")
    plt.contour(xx, yy, Z_high, levels=[0], colors="red",  linestyles="-", linewidths=2, label="High goal boundary")

    plt.title(f"Shift of Hyperplane with Different Return Goals\nLow={return_goal_low:.2%}, High={return_goal_high:.2%}")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # --- Numerical summary ---
    overlap = np.sum(support_low & support_high)
    union   = np.sum(support_low | support_high)
    jaccard = overlap / union if union > 0 else 0.0

    print(f"Saved hyperplane shift plot to {save_path}")
    print(f"Support vectors (low goal):  {support_low.sum()}")
    print(f"Support vectors (high goal): {support_high.sum()}")
    print(f"Overlap: {overlap}  |  Jaccard similarity: {jaccard:.2%}")



df = factor_df_prep("feature_data/data.csv")
snapshots = estimate_returns_covariance(df)

train_snaps, val_snaps, test_snaps = split_no_crisis_for_goals(snapshots)

model_low, _, _, _  = train_no_crisis(snapshots, 0.1, 0.01, 0, return_goal=0.005, grid_case="low")
model_high, _, _, _ = train_no_crisis(snapshots, 0.1, 0.01, 0, return_goal=0.02, grid_case="high")

# Pick a snapshot to visualize
snapshot = test_snaps[1]  # arbitrary month

# # Plot hyperplane shift
# plot_hyperplane_shift(model_low, model_high, snapshot,
#                       return_goal_low=0.005,
#                       return_goal_high=0.02,
#                       save_path="hyperplane_shift.png")

plot_hyperplane_side_by_side(
    model_low, model_high, snapshot,
    return_goal_low=0.005,
    return_goal_high=0.02,
    save_path="hyperplane_side_by_side.png"
)
