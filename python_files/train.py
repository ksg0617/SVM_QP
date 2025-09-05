import torch
from NN import *
from helpers import *
import pathlib
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def plot_mask_histogram(flat_mask, epoch, return_goal, year, grid_case):
    # ---- PLOT MASK HISTOGRAM ------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(flat_mask, bins=20, range=(0, 1), color='blue', edgecolor='black')
    plt.title(f"Mask Value Distribution — Epoch {epoch}")
    plt.xlabel("mask value")
    plt.ylabel("count")
    plt.grid(True)
    plt.tight_layout()
    # save the plot to a path
    plot_dir = pathlib.Path("plots") / f"{year}/{return_goal*100}percent/{grid_case}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"mask_histogram_epoch_{epoch}.png")
    plt.close()  # Close the plot to free memory

def plot_loss_curves(loss_hist, val_hist, return_goal, year, grid_case):
    # ---- PLOT LOSS CURVES ---------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(loss_hist, label="train")
    plt.plot(val_hist,  label="val")
    plt.xlabel("Epoch"); plt.ylabel("Mean loss")
    plt.legend()
    plt.grid(True)
    #save the plot to a path
    plot_dir = pathlib.Path("plots") / f"{year}/{return_goal*100}percent/{grid_case}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / "loss_curves.png")
    plt.close()  # Close the plot to free memory

def train_no_crisis(results, C_svm_init, tau_init, lambda_hinge_init, return_goal, grid_case):
    torch.manual_seed(99)
    snapshots = sorted(results, key=lambda s: s["date"])   # ensure sorted
    # drop the last 1 snapshot, it is incomplete
    snapshots = snapshots[:-1]  # drop the last snapshot, it is incomplete

    #val_fraction = 0.2
    #n_total      = len(snapshots)
    #n_val        = int(n_total * val_fraction)
    n_val = 12  # e.g. second last 18 months for validation
    n_test = 11  # e.g. last 12 months for testing
    train_snaps  = snapshots[:-(n_val + n_test)]          # earlier dates
    val_snaps    = snapshots[-(n_val + n_test):-n_test]          # second last 12 months
    test_snaps   = snapshots[-n_test:]               # last 12 months

    # For storing masks each epoch
    epoch_mask_values = []

    #######################################################################
    # 1. create model & *save* the initial weights  #######################
    #######################################################################
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = EndToEndSVM_MVO_Sigmoid(in_features=10, C_svm_init=C_svm_init,
                                    eps=1e-6, tau_init=tau_init, lambda_hinge_init=lambda_hinge_init).to(device)


    #######################################################################
    # 2. … your usual training loop here …
    #######################################################################
    # (use early-stopping or fixed epochs – whatever you prefer)
    # after training 'model' contains the *trained* embed weights
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # 0.  choose a unique, file-system-safe tag for this goal
    # --------------------------------------------------------------------
    goal_tag = f"{return_goal:.3f}".replace(".", "_")   # e.g. 0.15 -> "0_150"

    # you might also want a dedicated sub-folder:

    ckpt_dir = pathlib.Path("checkpoints") / f"goal_{goal_tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "best_model.pt"     # goal-specific file

    train_set = SnapshotDataset(train_snaps, return_goal=return_goal)
    val_set   = SnapshotDataset(val_snaps,   return_goal=return_goal)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False)

    n_features = results[0]["X_feat"].shape[1]

    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)

    patience   = 10         # stop if no progress for 10 epochs
    min_delta  = 1e-6      # what counts as “progress”
    #min_delta  = 1e-6      # what counts as “progress”
    max_epochs = 500      # hard cap (safety)
    best_val   = math.inf
    wait       = 0
    loss_hist, val_hist = [], []


    for epoch in range(1, max_epochs+1):
        # ---- TRAIN --------------------------------------------------------
        print("-----------------------------------------Epoch: ", epoch, "----------------------------------------")
        model.train()
        train_loss, n_batches = 0.0, 0
        for X, y, mu, Sigma, real_mu, real_sigma, goal in train_loader:
            if torch.unique(y).numel() < 2:        # ← all +1 or all –1
                continue                           # skip this batch entirely

            if mu.max().item() < goal:
                print(f"Skipping snapshot: return_goal={goal} not feasible (max mu = {mu.max().item():.4f})")
                continue
            X, y, mu, Sigma, real_mu, real_sigma = (t.squeeze(0).to(device) for t in (X, y, mu, Sigma, real_mu, real_sigma))

            try:
                w, mask, hinge, C_svm, alpha = model(X, y, mu, Sigma, goal)
            except ValueError as e:
                print(f"Skipping snapshot due to QP failure: {e}")
                continue
            # Collect masks for histogram
            epoch_mask_values.append(mask.detach().cpu().numpy())
            
            #print(f"#assets selected (mask > 0.5): {(mask > 0.5).sum().item()} / {mask.numel()}")

            var_loss = torch.dot(w, real_sigma @ w)
            #lambda_hinge = torch.exp(model.log_lambda_hinge)
            lambda_hinge = lambda_hinge_init
            loss  = var_loss + lambda_hinge * hinge
            
            #mu_p = torch.dot(mu, w)
            #var_p = torch.dot(w, Sigma @ w)
            #sharpe = mu_p / (var_p.sqrt() + 1e-8)
            #lambda_hinge = lambda_hinge_init 
            #loss = -sharpe + lambda_hinge * hinge
            #loss = -(mu_p - 0.5 * var_p) + lambda_hinge * hinge

            optim.zero_grad()
            loss.backward()
            # put this inside the training loop, **after** loss.backward()
            gnorm = model.embed.weight.grad.norm()
            #print(f"grad‖embed‖ = {gnorm.item():.3e}")   # should not be 0
            optim.step()

            train_loss += loss.item();  n_batches += 1

        train_loss /= n_batches
        loss_hist.append(train_loss)

        # ---- VALIDATE -----------------------------------------------------
        model.eval();  val_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for X, y, mu, Sigma, real_mu, real_sigma, goal in val_loader:
                if torch.unique(y).numel() < 2:        # ← all +1 or all –1
                    continue                           # skip this batch entirely
                if mu.max().item() < goal:
                    print(f"Skipping snapshot: return_goal={goal} not feasible (max mu = {mu.max().item():.4f})")
                    continue
                X, y, mu, Sigma, real_mu, real_sigma = (t.squeeze(0).to(device) for t in (X, y, mu, Sigma, real_mu, real_sigma))
                try:
                    w, mask, hinge, C_svm, alpha = model(X, y, mu, Sigma, goal)
                except ValueError as e:
                    print(f"Skipping snapshot due to QP failure: {e}")
                    continue
                var_val_loss = torch.dot(w, real_sigma @ w)
                #lambda_hinge = torch.exp(model.log_lambda_hinge)
                lambda_hinge = lambda_hinge_init
                loss = var_val_loss + lambda_hinge * hinge

                #mu_p = torch.dot(mu, w)
                #var_p = torch.dot(w, Sigma @ w)
                #sharpe = mu_p / (var_p.sqrt() + 1e-8)
                #lambda_hinge = lambda_hinge_init
                #loss = -sharpe + lambda_hinge * hinge
                #loss = -(mu_p - 0.5 * var_p) + lambda_hinge * hinge
                
                val_loss += loss.item(); n_batches += 1
        val_loss /= n_batches
        val_hist.append(val_loss)

        print(f"epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f}")



        # ---- EARLY-STOPPING LOGIC ----------------------------------------
        if val_loss < best_val - min_delta:
            best_val = val_loss
            wait     = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ checkpoint saved to {ckpt_path}")
            # Clear for next epoch
            epoch_mask_values = []
            
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop: no val improvement in {patience} epochs")
                flat_mask = np.concatenate(epoch_mask_values)
                break

    # ---- PLOT MASK HISTOGRAM ------------------------------------------

    plot_mask_histogram(flat_mask, epoch, return_goal, "2024", grid_case)
    plot_loss_curves(loss_hist, val_hist, return_goal, "2024", grid_case)

    return model, train_snaps, val_snaps, test_snaps



def train_crisis(results, C_svm_init, tau_init, lambda_hinge_init, return_goal, grid_case):
    torch.manual_seed(99)
    snapshots = sorted(results, key=lambda s: s["date"])   # ensure sorted
    # drop the last 1 snapshot, it is incomplete
    snapshots = snapshots[:-1]  # drop the last snapshot, it is incomplete

    #val_fraction = 0.2
    #n_total      = len(snapshots)
    #n_val        = int(n_total * val_fraction)
    n_val = 12  # e.g. second last 18 months for validation
    n_test = 23  # e.g. last 12 months for testing
    train_snaps  = snapshots[:-(n_val + n_test)]          # earlier dates
    val_snaps    = snapshots[-(n_val + n_test):-n_test]          # second last 12 months
    test_snaps   = snapshots[-n_test:]               # last 12 months

    # For storing masks each epoch
    epoch_mask_values = []

    #######################################################################
    # 1. create model & *save* the initial weights  #######################
    #######################################################################
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = EndToEndSVM_MVO_Sigmoid(in_features=10, C_svm_init=C_svm_init,
                                    eps=1e-6, tau_init=tau_init, lambda_hinge_init=lambda_hinge_init).to(device)


    #######################################################################
    # 2. … your usual training loop here …
    #######################################################################
    # (use early-stopping or fixed epochs – whatever you prefer)
    # after training 'model' contains the *trained* embed weights
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # 0.  choose a unique, file-system-safe tag for this goal
    # --------------------------------------------------------------------
    goal_tag = f"{return_goal:.3f}".replace(".", "_")   # e.g. 0.15 -> "0_150"

    # you might also want a dedicated sub-folder:

    ckpt_dir = pathlib.Path("checkpoints") / f"goal_{goal_tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "best_model.pt"     # goal-specific file

    train_set = SnapshotDataset(train_snaps, return_goal=return_goal)
    val_set   = SnapshotDataset(val_snaps,   return_goal=return_goal)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False)

    n_features = results[0]["X_feat"].shape[1]

    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)

    patience   = 10         # stop if no progress for 10 epochs
    min_delta  = 1e-6      # what counts as “progress”
    #min_delta  = 1e-6      # what counts as “progress”
    max_epochs = 500      # hard cap (safety)
    best_val   = math.inf
    wait       = 0
    loss_hist, val_hist = [], []


    for epoch in range(1, max_epochs+1):
        # ---- TRAIN --------------------------------------------------------
        print("-----------------------------------------Epoch: ", epoch, "----------------------------------------")
        model.train()
        train_loss, n_batches = 0.0, 0
        for X, y, mu, Sigma, real_mu, real_sigma, goal in train_loader:
            if torch.unique(y).numel() < 2:        # ← all +1 or all –1
                continue                           # skip this batch entirely

            if mu.max().item() < goal:
                print(f"Skipping snapshot: return_goal={goal} not feasible (max mu = {mu.max().item():.4f})")
                continue
            X, y, mu, Sigma, real_mu, real_sigma = (t.squeeze(0).to(device) for t in (X, y, mu, Sigma, real_mu, real_sigma))

            try:
                w, mask, hinge, C_svm, alpha = model(X, y, mu, Sigma, goal)
            except ValueError as e:
                print(f"Skipping snapshot due to QP failure: {e}")
                continue
            # Collect masks for histogram
            epoch_mask_values.append(mask.detach().cpu().numpy())
            
            #print(f"#assets selected (mask > 0.5): {(mask > 0.5).sum().item()} / {mask.numel()}")

            var_loss = torch.dot(w, real_sigma @ w)
            #lambda_hinge = torch.exp(model.log_lambda_hinge)
            lambda_hinge = lambda_hinge_init
            loss  = var_loss + lambda_hinge * hinge
            
            #mu_p = torch.dot(mu, w)
            #var_p = torch.dot(w, Sigma @ w)
            #sharpe = mu_p / (var_p.sqrt() + 1e-8)
            #lambda_hinge = lambda_hinge_init 
            #loss = -sharpe + lambda_hinge * hinge
            #loss = -(mu_p - 0.5 * var_p) + lambda_hinge * hinge

            optim.zero_grad()
            loss.backward()
            # put this inside the training loop, **after** loss.backward()
            gnorm = model.embed.weight.grad.norm()
            #print(f"grad‖embed‖ = {gnorm.item():.3e}")   # should not be 0
            optim.step()

            train_loss += loss.item();  n_batches += 1

        train_loss /= n_batches
        loss_hist.append(train_loss)

        # ---- VALIDATE -----------------------------------------------------
        model.eval();  val_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for X, y, mu, Sigma, real_mu, real_sigma, goal in val_loader:
                if torch.unique(y).numel() < 2:        # ← all +1 or all –1
                    continue                           # skip this batch entirely
                if mu.max().item() < goal:
                    print(f"Skipping snapshot: return_goal={goal} not feasible (max mu = {mu.max().item():.4f})")
                    continue
                X, y, mu, Sigma, real_mu, real_sigma = (t.squeeze(0).to(device) for t in (X, y, mu, Sigma, real_mu, real_sigma))
                try:
                    w, mask, hinge, C_svm, alpha = model(X, y, mu, Sigma, goal)
                except ValueError as e:
                    print(f"Skipping snapshot due to QP failure: {e}")
                    continue
                var_val_loss = torch.dot(w, real_sigma @ w)
                #lambda_hinge = torch.exp(model.log_lambda_hinge)
                lambda_hinge = lambda_hinge_init
                loss = var_val_loss + lambda_hinge * hinge

                #mu_p = torch.dot(mu, w)
                #var_p = torch.dot(w, Sigma @ w)
                #sharpe = mu_p / (var_p.sqrt() + 1e-8)
                #lambda_hinge = lambda_hinge_init
                #loss = -sharpe + lambda_hinge * hinge
                #loss = -(mu_p - 0.5 * var_p) + lambda_hinge * hinge
                
                val_loss += loss.item(); n_batches += 1
        val_loss /= n_batches
        val_hist.append(val_loss)

        print(f"epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f}")



        # ---- EARLY-STOPPING LOGIC ----------------------------------------
        if val_loss < best_val - min_delta:
            best_val = val_loss
            wait     = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ checkpoint saved to {ckpt_path}")
            # Clear for next epoch
            epoch_mask_values = []
            
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop: no val improvement in {patience} epochs")
                flat_mask = np.concatenate(epoch_mask_values)
                break

    # ---- PLOT MASK HISTOGRAM ------------------------------------------

    plot_mask_histogram(flat_mask, epoch, return_goal, "2008", grid_case)
    plot_loss_curves(loss_hist, val_hist, return_goal, "2008", grid_case)

    return model, train_snaps, val_snaps, test_snaps
