import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
# 1. CONFIG
###############################################################################
DATA_PATH = './output/unscaled_processed_dataset.csv'
TARGET_COL = "15. How would you rate your overall academic performance (GPA or grades) in the past semester?"
YEAR_COL  = "1. What is your year of study?"  # numeric for optional domain edges

SEEDS = [42, 100, 2023]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3-class classification: (0,1)->0, 2->1, (3,4)->2
# Basic training hyperparams
LR           = 0.001
DROPOUT      = 0.2
WEIGHT_DECAY = 5e-4
EPOCHS       = 150
PATIENCE     = 30

# Common parameter search
HIDDEN_DIMS     = [128, 256]
TOP_K_LIST      = [10, 15]
USE_DOMAIN_EDGES= [False, True]
NUM_LAYERS_LIST = [2, 3]
MODEL_LIST      = ["mlp", "gcn", "sage"]  # we'll handle MLP vs GCN vs SAGE in one unified loop

###############################################################################
# 2. CLASS MERGING
###############################################################################
def merge_classes(y_vals):
    """
    Merge:
       (0,1) => 0 = 'Below Average'
       (2)   => 1 = 'Average'
       (3,4) => 2 = 'High'
    """
    new_y = []
    for v in y_vals:
        if v in [0, 1]:
            new_y.append(0)
        elif v == 2:
            new_y.append(1)
        else:  # v in (3,4)
            new_y.append(2)
    return np.array(new_y, dtype=int)

###############################################################################
# 3. MLP BASELINE
###############################################################################
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_mlp_baseline(X, y, train_mask, val_mask, test_mask,
                       hidden_dim=64, dropout=0.2, lr=0.001,
                       weight_decay=5e-4, epochs=150, patience=30):
    device = X.device
    in_dim = X.size(1)
    out_dim = len(torch.unique(y))

    mlp = MLP(in_dim, hidden_dim, out_dim, dropout).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step():
        mlp.train()
        optimizer.zero_grad()
        logits = mlp(X[train_mask])
        loss = F.cross_entropy(logits, y[train_mask])
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(mask):
        mlp.eval()
        logits = mlp(X[mask])
        preds = logits.argmax(dim=-1)
        correct = (preds == y[mask]).sum().item()
        total = int(mask.sum())
        return correct / total if total > 0 else 0.0

    best_val = 0.0
    best_test = 0.0
    no_improve = 0

    for ep in range(1, epochs+1):
        loss_val = train_step()
        train_acc = evaluate(train_mask)
        val_acc   = evaluate(val_mask)
        test_acc  = evaluate(test_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStop - MLP] ep={ep}")
                break

        if ep % 10 == 0:
            print(f"[MLP] epoch={ep}, Loss={loss_val:.4f}, "
                  f"TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, TestAcc={test_acc:.4f}")

    return best_val, best_test

###############################################################################
# 4. GCN / GraphSAGE
###############################################################################
class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.dropout_val = dropout
        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(GCNConv(in_channels, hidden_channels))
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        # Last layer
        self.layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_val, training=self.training)
        return x

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.dropout_val = dropout
        self.layers = nn.ModuleList()
        # First
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        # Middle
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        # Last
        self.layers.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.layers):
            # ignoring edge_attr for standard SAGE
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_val, training=self.training)
        return x

###############################################################################
# 5. Build Graph
###############################################################################
def build_topk_graph(x_scaled, top_k=10):
    sim_matrix = cosine_similarity(x_scaled)  # shape [N,N]
    N = sim_matrix.shape[0]
    edges = []
    weights = []
    for i in range(N):
        row = sim_matrix[i]
        sorted_idx = np.argsort(-row)
        count = 0
        for idx in sorted_idx:
            if idx == i:  # skip self-loop
                continue
            edges.append([i, idx])
            weights.append(row[idx])
            count += 1
            if count >= top_k:
                break
    edge_index = np.array(edges).T  # [2, E]
    edge_wts   = np.array(weights)
    return edge_index, edge_wts

def add_domain_edges(edge_index, edge_wts, year_values, weight=1.0):
    """
    If year_values is not None, fully connect nodes with same year
    at 'weight'. Returns new edge_index, edge_wts concatenated.
    """
    if year_values is None:
        return edge_index, edge_wts

    from collections import defaultdict
    year_dict = defaultdict(list)
    for i, yv in enumerate(year_values):
        year_dict[yv].append(i)

    dom_edges = []
    dom_wts   = []
    for yv, lst in year_dict.items():
        if len(lst) < 2:
            continue
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                ni = lst[i]
                nj = lst[j]
                dom_edges.append([ni, nj])
                dom_edges.append([nj, ni])
                dom_wts.append(weight)
                dom_wts.append(weight)

    all_edges = np.concatenate([edge_index, np.array(dom_edges).T], axis=1)
    all_wts   = np.concatenate([edge_wts, np.array(dom_wts)], axis=0)
    return all_edges, all_wts

def rowwise_normalize(edge_index_t, edge_attr_t, num_nodes):
    sums = torch.zeros(num_nodes, dtype=torch.float, device=edge_attr_t.device)
    src = edge_index_t[0]
    sums.index_add_(0, src, edge_attr_t)
    eps = 1e-12
    new_attr = edge_attr_t / (sums[src] + eps)
    return new_attr

def build_graph_data(x_scaled, y_arr, top_k, year_values=None, domain_edge_weight=0.5):
    edge_i, edge_w = build_topk_graph(x_scaled, top_k)
    edge_i, edge_w = add_domain_edges(edge_i, edge_w, year_values, weight=domain_edge_weight)

    x_torch = torch.tensor(x_scaled, dtype=torch.float)
    y_torch = torch.tensor(y_arr, dtype=torch.long)

    edge_index_t = torch.tensor(edge_i, dtype=torch.long)
    edge_wts_t   = torch.tensor(edge_w, dtype=torch.float)
    # rowwise norm
    num_nodes = x_torch.size(0)
    edge_wts_t = rowwise_normalize(edge_index_t, edge_wts_t, num_nodes)

    data = Data(
        x=x_torch,
        y=y_torch,
        edge_index=edge_index_t,
        edge_attr=edge_wts_t
    )
    return data

###############################################################################
# 6. Train GNN
###############################################################################
def train_gnn(data, model_type="gcn", hidden_dim=64, num_layers=2, dropout=0.2,
              lr=0.001, weight_decay=5e-4, epochs=150, patience=30):

    in_channels = data.num_features
    out_channels = len(torch.unique(data.y))

    # Build model
    if model_type == "gcn":
        model = GCNModel(in_channels, hidden_dim, out_channels, num_layers, dropout)
    else:
        model = GraphSAGEModel(in_channels, hidden_dim, out_channels, num_layers, dropout)

    model = model.to(DEVICE)
    data = data.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(mask):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_attr)
        preds = out.argmax(dim=-1)
        correct = (preds[mask] == data.y[mask]).sum().item()
        total = int(mask.sum())
        return correct / total if total > 0 else 0.0

    best_val = 0.0
    best_test= 0.0
    no_improve=0

    for ep in range(1, epochs+1):
        loss_val = train_step()
        train_acc = evaluate(data.train_mask)
        val_acc   = evaluate(data.val_mask)
        test_acc  = evaluate(data.test_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_test= test_acc
            no_improve=0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStop - {model_type.upper()}] ep={ep}")
                break

        if ep % 10 == 0:
            print(f"[{model_type.upper()}] ep={ep}, Loss={loss_val:.4f}, "
                  f"TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, TestAcc={test_acc:.4f}")

    return best_val, best_test

###############################################################################
# 7. MAIN
###############################################################################
def main():
    # For final plotting
    results_log = []

    # 1) Load & Merge classes
    df = pd.read_csv(DATA_PATH).select_dtypes(include=[np.number])
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target '{TARGET_COL}' in columns: {df.columns.tolist()}")

    y_merged = merge_classes(df[TARGET_COL].values)
    df[TARGET_COL] = y_merged

    # Possibly retrieve year info for domain edges
    if YEAR_COL in df.columns:
        year_values_full = df[YEAR_COL].values
    else:
        print(f"WARNING: No '{YEAR_COL}' column found; can't add domain edges.")
        year_values_full = None

    # 2) Split X,y
    X_df = df.drop(columns=[TARGET_COL], errors='ignore')
    y_df = df[TARGET_COL]
    X_full = X_df.values
    y_full = y_df.values

    for seed in SEEDS:
        print("\n=====================================")
        print(f"            SEED = {seed}")
        print("=====================================")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Train/Val/Test split
        idx_all = np.arange(len(X_full))
        tr_idx, te_idx = train_test_split(idx_all, test_size=0.2, random_state=seed)
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.25, random_state=seed)
        # => 60% train, 20% val, 20% test

        X_train = X_full[tr_idx]
        y_train = y_full[tr_idx]

        # SMOTE
        sm = SMOTE(random_state=seed)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"SMOTE from {len(X_train)} to {len(X_res)} examples for training")

        X_val  = X_full[va_idx]
        y_val  = y_full[va_idx]
        X_test = X_full[te_idx]
        y_test = y_full[te_idx]

        # Re-index for final arrays
        new_tr_idx = np.arange(len(X_res))
        val_off = len(X_res)
        new_va_idx = np.arange(val_off, val_off+len(X_val))
        test_off = val_off + len(X_val)
        new_te_idx = np.arange(test_off, test_off+len(X_test))

        X_all = np.concatenate([X_res, X_val, X_test], axis=0)
        y_all = np.concatenate([y_res, y_val, y_test], axis=0)

        # Domain edges updated for oversampled training
        if year_values_full is not None:
            yr_train = year_values_full[tr_idx]
            yr_res, _ = sm.fit_resample(yr_train.reshape(-1,1), y_train)
            yr_val = year_values_full[va_idx]
            yr_tst = year_values_full[te_idx]
            year_all = np.concatenate([yr_res.flatten(), yr_val, yr_tst], axis=0)
        else:
            year_all = None

        # Scale
        scaler = StandardScaler()
        X_all_sc = scaler.fit_transform(X_all)

        # Build masks
        fullN = len(X_all_sc)
        train_mask = torch.zeros(fullN, dtype=torch.bool)
        val_mask   = torch.zeros(fullN, dtype=torch.bool)
        test_mask  = torch.zeros(fullN, dtype=torch.bool)

        train_mask[new_tr_idx] = True
        val_mask[new_va_idx]   = True
        test_mask[new_te_idx]  = True

        # Convert to torch
        X_torch = torch.tensor(X_all_sc, dtype=torch.float, device=DEVICE)
        y_torch = torch.tensor(y_all, dtype=torch.long, device=DEVICE)

        # 3) Loop Over Model Types & Config
        for hd in HIDDEN_DIMS:
            for tk in TOP_K_LIST:
                for dom_flag in USE_DOMAIN_EDGES:
                    for nl in NUM_LAYERS_LIST:
                        for model_type in MODEL_LIST:
                            # We'll define a paramSet descriptor for plotting
                            param_set_str = (f"hd={hd}, top_k={tk}, "
                                             f"dom={dom_flag}, nl={nl}")

                            print(f"\n=== MODEL={model_type.upper()}, {param_set_str} ===")

                            if model_type == "mlp":
                                # MLP doesn't use top_k, domain edges, or num_layers
                                # Just train MLP with hidden_dim=hd
                                mlp_val, mlp_test = train_mlp_baseline(
                                    X_torch, y_torch,
                                    train_mask, val_mask, test_mask,
                                    hidden_dim=hd,
                                    dropout=DROPOUT,
                                    lr=LR,
                                    weight_decay=WEIGHT_DECAY,
                                    epochs=EPOCHS,
                                    patience=PATIENCE
                                )
                                print(f"[MLP-Result] Val={mlp_val:.4f}, Test={mlp_test:.4f}")

                                results_log.append({
                                    "Seed": seed,
                                    "ModelType": "MLP",
                                    "ParamSet": param_set_str,
                                    "ValAccuracy": mlp_val,
                                    "TestAccuracy": mlp_test
                                })

                            else:
                                # GNN path
                                data = build_graph_data(
                                    X_all_sc,
                                    y_all,
                                    top_k=tk,
                                    year_values=year_all if dom_flag else None,
                                    domain_edge_weight=0.5  # can tune
                                )
                                data.train_mask = train_mask
                                data.val_mask   = val_mask
                                data.test_mask  = test_mask

                                best_val, best_test = train_gnn(
                                    data,
                                    model_type=model_type,
                                    hidden_dim=hd,
                                    num_layers=nl,
                                    dropout=DROPOUT,
                                    lr=LR,
                                    weight_decay=WEIGHT_DECAY,
                                    epochs=EPOCHS,
                                    patience=PATIENCE
                                )
                                print(f"[{model_type.upper()}-Result] Val={best_val:.4f}, Test={best_test:.4f}")

                                results_log.append({
                                    "Seed": seed,
                                    "ModelType": model_type.upper(),
                                    "ParamSet": param_set_str,
                                    "ValAccuracy": best_val,
                                    "TestAccuracy": best_test
                                })

    # ------------------ PLOTTING SECTION ------------------ #
    df_results = pd.DataFrame(results_log)

    # If you want to average across seeds, group by (ParamSet, ModelType)
    grouped = df_results.groupby(["ParamSet", "ModelType"], as_index=False).mean(numeric_only=True)

    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(
        data=grouped,
        x="ParamSet",
        y="TestAccuracy",
        hue="ModelType"
    )
    plt.title("Comparison of Test Accuracy Across Models & Configurations (Averaged over Seeds)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir,"model_comparison.png"), dpi = 300)
    print("\nSaved plot to '/output/model_comparison.png'")

    mean_accuracies = df_results.groupby("ModelType")[["TestAccuracy"]].mean().reset_index()

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.barplot(
        data=mean_accuracies,
        x="ModelType",
        y="TestAccuracy",
        palette="deep"
    )
    for index, row in mean_accuracies.iterrows():
        plt.text(index, row.TestAccuracy + 0.005, f"{row.TestAccuracy:.4f}", 
                     color='black', ha="center")
    plt.title("Mean Test Accuracy per Model")
    plt.xlabel("Model Type")
    plt.ylabel("Mean Test Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"mean_test_accuracy_per_model.png"), dpi=300)
    print("Saved plot to '/output/mean_test_accuracy_per_model.png'")

if __name__ == "__main__":
    main()
