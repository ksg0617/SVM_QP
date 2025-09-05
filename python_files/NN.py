import torch
import torch.nn as nn
from qpth.qp import QPFunction
from data_prep import *
from torch.utils.data import Dataset, DataLoader
import numpy as np

def sanitise_spd(M: torch.Tensor,
                 name: str = "Q",
                 eps: float = 1e-6) -> torch.Tensor:
    """
    1. reports NaN/Inf, duplicate rows, condition number, min/max eigenvalues
    2. drops exact duplicate rows/cols (they kill Cholesky)
    3. adds minimal jitter to make SPD
    """

    # 0) NaN / Inf check -------------------------------------------------
    if torch.isnan(M).any() or torch.isinf(M).any():
        raise ValueError(f"[{name}] contains NaN or Inf – cannot factorise")

    # 1) deduplicate rows/cols ------------------------------------------
    #    (X rows that are numerically identical give zero-variance directions)
    # NOTE: this assumes you constructed M as  y yᵀ ⊙ X Xᵀ
    #       If that is not the case just delete the dedup block.
    with torch.no_grad():
        diag = torch.diag(M)
        mask_nonzero = diag > 0      # rows with all-zero features have 0 on diag
        if mask_nonzero.sum() < len(mask_nonzero):
            M = M[mask_nonzero][:, mask_nonzero]

    # 2) minimal jitter --------------------------------------------------
    eigvals = torch.linalg.eigvalsh(M)
    λmin    = eigvals.min().item()
    λmax    = eigvals.max().item()
    cond    = λmax / max(λmin, eps)
    #_report("λ_min",  λmin)
    #_report("λ_max",  λmax)
    #_report("cond",   cond)

    jitter  = max(eps, -λmin + eps)
    M       = M + jitter * torch.eye(M.size(0), device=M.device, dtype=M.dtype)

    # final safety: Cholesky must succeed now
    try:
        torch.linalg.cholesky(M)
    except RuntimeError as e:
        raise RuntimeError(f"[{name}] still not SPD even after jitter = {jitter}") from e

    return M

class EndToEndSVM_MVO_Sigmoid(nn.Module):
    def __init__(self,
                 in_features: int,
                 C_svm_init: float,
                 eps: float,
                 tau_init: float,
                 lambda_hinge_init: float
            ):
        """
        in_features : number of raw features per asset
        C_svm_init       : SVM dual box-constraint
        eps         : jitter to ensure all Q-matrices are SPD
        tau_init         : sigmoid temperature for soft gating
        """
        super().__init__()
        # 1)  make the learnable projection W (d × d, no bias)
        self.embed = nn.Linear(in_features, in_features, bias=False).double()

        # 2)  start it as an identity matrix instead of tiny random numbers
        nn.init.eye_(self.embed.weight)          # ← this line
        with torch.no_grad():
            self.embed.weight += 0.01 * torch.randn_like(self.embed.weight)
            
        self.log_C = nn.Parameter(torch.log(torch.tensor([C_svm_init], dtype=torch.double)))
        # self.log_tau = nn.Parameter(torch.log(torch.tensor([tau_init], dtype=torch.double)))
        self.tau = tau_init
        self.eps = eps
        # self.log_lambda_hinge = nn.Parameter(torch.log(torch.tensor([lambda_hinge_init], dtype=torch.double)))
        self.lambda_hinge = lambda_hinge_init 

    def forward(self,
                X_feat: torch.Tensor,     # (n,d)
                y: torch.Tensor,          # ±1
                mu_est: torch.Tensor,     # (n,)  
                Sigma_est: torch.Tensor,  # (n,n) 
                return_goal: float
               ) -> torch.Tensor:
        n, d = X_feat.shape
        # A. raw input
        #_check("X_feat", X_feat)


        # 1) feature embedding
        Xp = self.embed(X_feat.double())
        #_check("Xp", Xp)
        y  = y.view(-1).double()           # now y is float64 ±1

        # 2) SVM dual QP
        K      = Xp @ Xp.t()                             # (n, n)
        #_check("K", K)

        yy    = y.unsqueeze(1) * y.unsqueeze(0)  # (n,n) float64
        Q_svm  = sanitise_spd((y[:,None] * y[None,:]) * K, name="Q_svm", eps=self.eps)
        #_check("Q_svm", Q_svm)
        p_svm  = -torch.ones(n, device=Xp.device, dtype=Xp.dtype)

        G_svm  = torch.cat([
            -torch.eye(n, device=Xp.device, dtype=Xp.dtype),
             torch.eye(n, device=Xp.device, dtype=Xp.dtype)
        ], dim=0)                                        # (2n, n)

        C_svm = torch.exp(self.log_C)

        h_svm  = torch.cat([
            torch.zeros(n, device=Xp.device, dtype=Xp.dtype),
            C_svm * torch.ones(n, device=Xp.device, dtype=Xp.dtype)
        ], dim=0)                                        # (2n,)

        A_svm  = y.view(1,-1)#.to(Xp)                     # (1, n)
        b_svm  = torch.zeros(1, device=Xp.device, dtype=Xp.dtype)
        # ---------------------------------------------------------------
        # handle single-class case (all +1  *or*  all –1)
        # ---------------------------------------------------------------
        if (y == y[0]).all():              # every label identical
            A_svm = torch.empty(0, n, device=Xp.device, dtype=Xp.dtype)  # shape (0, n)
            b_svm = torch.empty(0,       device=Xp.device, dtype=Xp.dtype)  # shape (0,)
            print("Warning: all labels identical, no SVM hyperplane constructed.")
        else:
            A_svm = y.view(1, -1)                                           # (1, n)
            b_svm = torch.zeros(1, device=Xp.device, dtype=Xp.dtype)        # (1,)


        alpha      = QPFunction(verbose=False, maxIter=100)(
                    Q_svm, p_svm, G_svm, h_svm, A_svm, b_svm, 
                 )                                  # (n,)
        alpha = torch.clamp(alpha, min=-1e-6, max=C_svm.item() + 1e-6)        

        # after solving for alpha
        # construct w_svm properly:
        # make sure alphas is a 1-D tensor of length 
        alpha = alpha.view(-1)               

        # ensure y is double or double to match alphas dtype
        y = y.to(alpha.dtype)            

        # elementwise product alpha_i * y_i
        alpha_y = alpha * y                
        w_svm = Xp.t().mv(alpha_y)                   # or torch.matmul(X.t(), alpha_y)

        scores = Xp @ w_svm                 # (n_assets,)

        # diagnostic
        # with torch.no_grad():
        #     print("‖w_svm‖₂       :", w_svm.norm().item())
        #     print("‖alpha‖₁       :", alpha.abs().sum().item())
        #     print("scores min/max :", scores.min().item(), scores.max().item())
            
        hinge = torch.clamp(1.0 - y * scores, min=0.0).mean()
        # Dual SVM loss: -1ᵗα + ½ αᵗQα
        #hinge = -alpha.sum() + 0.5 * alpha @ (Q_svm @ alpha)

        # 4) differentiable sigmoid gate
        tau = self.tau
        mask = torch.sigmoid(scores / tau)
        #print("Mask mean value: ", mask.mean())


        # 5) MVO QP *over all assets* with w_i ≤ mask_i
        #    compute moments for every asset
        # ---------- 2) MVO QP using *forecast* μ, Σ ----------
        mu     = mu_est                             # (n,)
        Sigma  = Sigma_est

        P_mvo  = Sigma
        q_mvo  = torch.zeros(n, device=Sigma.device, dtype=Sigma.dtype)

        # box constraints: 0 ≤ w ≤ mask
        G_box  = torch.cat([
            -torch.eye(n, device=Sigma.device, dtype=Sigma.dtype),  # -w ≤ 0
             torch.eye(n, device=Sigma.device, dtype=Sigma.dtype)   #  w ≤ mask
        ], dim=0)                                                   # (2n, n)
        h_box  = torch.cat([
            torch.zeros(n, device=Sigma.device, dtype=Sigma.dtype),
            mask
        ], dim=0)                                                   # (2n,)

        # ------------------ NEW inequality: μᵀw ≥ return_goal -------------
        G_ret = -mu.unsqueeze(0)                                     # (1, n)
        h_ret = -torch.tensor([return_goal],
                            device=Sigma.device, dtype=Sigma.dtype)

        # concat all inequalities
        G_ineq = torch.cat([G_box, G_ret], dim=0)                    # (2n+1, n)
        h_ineq = torch.cat([h_box, h_ret], dim=0)                    # (2n+1,)

        # equality: sum(w)=1
        A_eq = torch.ones(1, n, device=Sigma.device, dtype=Sigma.dtype)  # (1, n)                                             
        b_eq = torch.tensor([1.0], device=Sigma.device, dtype=Sigma.dtype)

        w_opt  = QPFunction(verbose=False, maxIter=100)(
                    P_mvo, q_mvo, G_ineq, h_ineq, A_eq, b_eq
                 )
        # After computing w_opt
        if torch.isnan(w_opt).any() or torch.isinf(w_opt).any():
            print("Warning: NaN or Inf in QP solution — skipping this snapshot.")
            raise ValueError("Invalid QP solution")
                                       
        # 2. Check return target vs feasible region
        #print("mask_min/max:", mask.min().item(), mask.max().item())
        #print("mu_min/max:",   mu.min().item(),   mu.max().item(), "goal:", return_goal)
        
        return w_opt.view(-1), mask, hinge, C_svm, alpha
    


# ---------- 1.  tiny helper --------------------------------------------------
def to_tensor(x, *, dtype=torch.float64):
    """
    NumPy → torch, replace NaN/Inf with finite numbers
    (you can swap 'nan=0.0' for any imputation of your choice).
    """
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.as_tensor(x, dtype=dtype)


# ---------- 2.  custom Dataset ----------------------------------------------
class SnapshotDataset(Dataset):
    """
    Each item is one time-snapshot:
        X_feat, y, mu_fore, Sigma_fore, return_goal
    Shapes:
        X_feat      (n, d)   – features
        y           (n,)     – ±1 labels
        mu_fore     (n,)
        Sigma_fore  (n, n)
    """
    def __init__(self, results, return_goal):
        self.data = []

        for snap in results:
            # normalize the X_feat
            X_feat = snap["X_feat"]  # (n, d)
            #X_feat = (X_feat - X_feat.mean(axis=0)) / (X_feat.std(axis=0) + 1e-8)
            X  = to_tensor(X_feat)      # (n, d)
            y  = to_tensor(snap["y"]).view(-1)  # (n,)
            mu = to_tensor(snap["mu_fore"])     # (n,)
            S  = to_tensor(snap["Sigma_fore"])  # (n, n)
            real_mu = to_tensor(snap["real_returns"])
            real_sigma = to_tensor(snap["real_sigma"])

            # ---------- basic sanity: drop rows that are still all-zero ------
            # (happens if the original had only NaNs)
            keep = (X.abs().sum(dim=1) > 0)
            if keep.sum() < 2:                  # need ≥2 points for an SVM
                continue                         # skip this snapshot

            X, y, mu, real_mu = X[keep], y[keep], mu[keep], real_mu[keep]
            real_sigma = real_sigma[keep][:, keep]  # (n, n)
            S        = S[keep][:, keep]

            self.data.append((X, y, mu, S, real_mu, real_sigma, float(return_goal)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]          # batch = tuple of 5 tensors