import sys
import math
from typing import Tuple
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def make_fake_data(n: int = 20, d: int = 10, T: int = 252) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (features, returns) tensors."""
    g = torch.Generator().manual_seed(0)
    features = torch.randn(n, d, generator=g)
    # AR(1) Gaussian returns
    phi = 0.1 + 0.8*torch.rand(n, generator=g)
    eps = 0.02*torch.randn(T, n, generator=g)
    rets = torch.zeros(T, n)
    for t in range(1, T):
        rets[t] = phi*rets[t-1] + eps[t]
    return features.double(), rets.double()

class SVM_MVO_Net(nn.Module):
    """End-to-end network with two differentiable QP layers."""
    def __init__(self, n: int, d: int, C: float = 1.0, tau: float = 1e-2, r_goal: float = 0.0):
        super().__init__()
        self.tau = tau; self.C = float(C); self.r_goal = float(r_goal)

        # learnable (linear) feature map
        self.feat = nn.Linear(d, n, bias=False, dtype=torch.double)

        # constant ±1 label pattern → DPP friendly
        y_np = ((torch.arange(n)%2)*2 - 1).double().numpy()
        self.register_buffer('y', torch.from_numpy(y_np))

        # build constant matrices for canonical QP: Gα≤h, Aα=0
        G_np = torch.vstack([torch.eye(n), -torch.eye(n)]).double().numpy()
        h_np = torch.hstack([torch.full((n,), self.C), torch.zeros(n)]).double().numpy()
        A_np = y_np[None, :]
        b_np = 0.0
        self.register_buffer('G', torch.from_numpy(G_np))
        self.register_buffer('h', torch.from_numpy(h_np))
        self.register_buffer('A', torch.from_numpy(A_np))
        self.register_buffer('b', torch.tensor(b_np, dtype=torch.double))

        # --- SVM QP layer ---
        a = cp.Variable(n)
        Qp = cp.Parameter((n, n), PSD=True)
        obj = cp.Minimize(0.5*cp.quad_form(a, Qp) - cp.sum(a))
        svm_prob = cp.Problem(obj, [cp.Constant(G_np)@a <= cp.Constant(h_np),
                                    cp.Constant(A_np)@a == cp.Constant(b_np)])
        self.svm_layer = CvxpyLayer(svm_prob, [Qp], [a])

        # --- Markowitz QP layer ---
        Σp, μp, mp = (cp.Parameter((n, n), PSD=True), cp.Parameter(n), cp.Parameter(n))
        w = cp.Variable(n)
        mvo_prob = cp.Problem(cp.Minimize(0.5*cp.quad_form(w, Σp)),
                            [cp.sum(w) == 1,
                            μp@w >= self.r_goal,
                            w >= 0, w <= mp])
        self.mvo_layer = CvxpyLayer(mvo_prob, [Σp, μp, mp], [w])

    # ------------------------------------------------------------------
    def forward(self, feat_mat: torch.Tensor, Σ: torch.Tensor, μ: torch.Tensor):
        n = feat_mat.size(0)
        y = self.y.to(feat_mat)

        # kernel Gram matrix
        φ = self.feat(feat_mat)
        K = φ @ φ.T
        Q = (y[:, None]*K)*y[None, :]

        # solve SVM dual
        a_star, = self.svm_layer(Q)

        # functional margin & soft mask
        margins = (a_star*y) @ K
        sv_idx = ((a_star > 1e-4) & (a_star < self.C-1e-4)).double().argmax().item()
        b = y[sv_idx] - margins[sv_idx]
        m = torch.sigmoid((margins + b)/self.tau)

        # Markowitz optimisation
        w_star, = self.mvo_layer(Σ, μ, m)
        loss = 0.5*w_star @ Σ @ w_star
        return loss, (a_star, m, w_star)

def main():
    n, d = 20, 10
    feat, rets = make_fake_data(n, d, 252)
    Σ = torch.eye(n, dtype=torch.double)*0.05   # toy covariance
    μ = torch.zeros(n, dtype=torch.double)      # zero mean
    
    model = SVM_MVO_Net(n, d)
    loss, (a, m, w) = model(feat, Σ, μ)
    print(f"loss = {loss.item():.6f}")
    print(f"non‑zero α: {(a>1e-6).sum().item()}/{n}")
    print(f"mask m range: {m.min():.3f}–{m.max():.3f}")
    print(f"weights sum: {w.sum().item():.3f}")

main()