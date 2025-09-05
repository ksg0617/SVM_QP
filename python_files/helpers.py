#######################################################################
# 0. imports & helper  ###############################################
#######################################################################
import torch
from qpth.qp import QPFunction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------
# EXTRA ➋   classify assets by SVM sign -------------------------------
# ---------------------------------------------------------------------
def classify_assets(model, X, y, C, names=None):
    """returns two Python lists: invest, not_invest"""
    alpha, w_svm, _ = solve_svm(model, X, y, C)
    scores = model.embed(X.double()) @ w_svm            # (n,)
    invest = (scores > 0).nonzero(as_tuple=True)[0]     # long side
    avoid  = (scores <= 0).nonzero(as_tuple=True)[0]    # short / 0
    if names is not None:
        invest = [names[i] for i in invest.cpu().numpy()]
        avoid  = [names[i] for i in avoid.cpu().numpy()]
    else:
        invest = invest.cpu().tolist()
        avoid  = avoid.cpu().tolist()
    return invest, avoid

def make_spd(M, eps=1e-6):
    """Add minimal diagonal jitter until Cholesky succeeds."""
    I = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
    jitter = eps
    for _ in range(10):
        try:
            torch.linalg.cholesky(M)
            return M
        except RuntimeError:
            M = (M + M.t()) * 0.5 + jitter * I
            jitter *= 10
    raise RuntimeError("Unable to make SPD matrix")

def solve_svm(model, X_feat, y, C):
    """
    Runs *just* the SVM part of EndToEndSVM_MVO_Sigmoid and
    returns (alpha, w_svm, support_index_tensor)
    """
    Xp = model.embed(X_feat.double())            # (n,d)
    y  = y.view(-1).double()                     # (n,)

    K      = Xp @ Xp.t()
    Q_svm  = (y[:,None] * y[None,:]) * K
    Q_svm  = make_spd(Q_svm)

    n      = Xp.size(0)
    p_svm  = -torch.ones(n, dtype=Xp.dtype, device=Xp.device)
    G_svm  = torch.cat([-torch.eye(n, dtype=Xp.dtype, device=Xp.device),
                         torch.eye(n, dtype=Xp.dtype,  device=Xp.device)], 0)
    h_svm  = torch.cat([torch.zeros(n, dtype=Xp.dtype, device=Xp.device),
                        C*torch.ones(n, dtype=Xp.dtype, device=Xp.device)], 0)

    if (y == y[0]).all():              # single-class edge case
        A_svm = torch.empty(0, n, dtype=Xp.dtype, device=Xp.device)
        b_svm = torch.empty(0,    dtype=Xp.dtype, device=Xp.device)
    else:
        A_svm = y.unsqueeze(0)
        b_svm = torch.zeros(1, dtype=Xp.dtype, device=Xp.device)

    alpha = QPFunction(verbose=False)(Q_svm, p_svm, G_svm, h_svm, A_svm, b_svm)
    alpha = torch.clamp(alpha, min=0.0, max=C.item()).view(-1)

    w_svm = Xp.t().mv(alpha * y)       # weight vector in embedded space
    sv    = (alpha > 1e-6)             # boolean mask of support vectors
    #sv = (alpha > 1e-6)
    return alpha, w_svm, sv.nonzero(as_tuple=True)[0]

