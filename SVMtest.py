import numpy as np
import pandas as pd

# ------------------------------ config ---------------------------------
N_ASSETS   = 20          # number of assets
N_FEATURES = 10          # feature dimension
DAYS       = 252         # trading days in sample
WINDOW     = 21          # rebalance frequency  (≈ one month)
SEED       = 0

rng = np.random.default_rng(SEED)

# 1) Static asset features (e.g. fundamentals, factors) -----------------
features = pd.DataFrame(
    rng.standard_normal(size=(N_ASSETS, N_FEATURES)),
    columns=[f"f{j+1}" for j in range(N_FEATURES)],
    index=[f"A{i+1}" for i in range(N_ASSETS)]
)

# 2) Simulate daily log-returns with an AR(1) structure ------------------
phi   = rng.uniform(0.05, 0.95, size=N_ASSETS)   # persistence per asset
sigma = rng.uniform(0.01, 0.03, size=N_ASSETS)   # daily vol per asset

rets = np.zeros((DAYS, N_ASSETS))
eps  = rng.standard_normal(size=(DAYS, N_ASSETS)) * sigma       # shocks

for t in range(1, DAYS):
    rets[t] = phi * rets[t-1] + eps[t]

returns = pd.DataFrame(
    rets,
    columns=features.index,         # asset names match feature index
    index=[f"day{t+1}" for t in range(DAYS)]
)

# 3) Rebalance windows: compute Σ_t and μ_t for each 21-day block -------
period_ends = range(WINDOW, DAYS + 1, WINDOW)    # 21, 42, …, 252
mu_list, cov_list = [], []

for end in period_ends:
    block = returns.iloc[end-WINDOW:end]         # last 21 days
    mu_list.append(block.mean().to_numpy())      # shape (20,)
    cov_list.append(np.cov(block.T, ddof=1))     # shape (20, 20)

# --------------------------- sanity checks -----------------------------
print("Feature matrix shape   :", features.shape)      # (20, 10)
print("Returns matrix shape   :", returns.shape)       # (252, 20)
print("Rebalance periods      :", len(cov_list))       # 12
print("Covariance shape       :", cov_list[0].shape)   # (20, 20)
