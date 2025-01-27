import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt

###############################################################################
# CONFIG
###############################################################################
DATA_PATH    = './output/unscaled_processed_dataset.csv'
TARGET_COL   = "15. How would you rate your overall academic performance (GPA or grades) in the past semester?"
SEEDS        = [42, 100, 2023]
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Merge classes: (0,1)->0, 2->1, (3,4)->2 => 3 classes total
TOP_K        = 15
HIDDEN_DIM   = 256
NUM_LAYERS   = 3
DROPOUT      = 0.2
LR           = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS       = 150
PATIENCE     = 30

# Always drop "Timestamp"
TIMESTAMP_COL = "Timestamp"
# We'll toggle dropping "4. On average, how many hours of sleep do you get on a typical day?"
HOURS_COL     = "4. On average, how many hours of sleep do you get on a typical day?"

###############################################################################
# 1. CLASS MERGING
###############################################################################
def merge_classes(y_vals):
    """
    (0,1)->0, 2->1, (3,4)->2
    """
    new_y = []
    for v in y_vals:
        if v in [0,1]:
            new_y.append(0)
        elif v == 2:
            new_y.append(1)
        else:  # 3 or 4
            new_y.append(2)
    return np.array(new_y, dtype=int)

###############################################################################
# 2. Build top-k adjacency
###############################################################################
def build_topk_graph(x_scaled, k=10):
    sim_matrix = cosine_similarity(x_scaled)
    N = sim_matrix.shape[0]
    edges, wts = [], []
    for i in range(N):
        row = sim_matrix[i]
        sorted_idx = np.argsort(-row)  # descending
        count=0
        for idx in sorted_idx:
            if idx == i:
                continue
            edges.append([i, idx])
            wts.append(row[idx])
            count+=1
            if count>= k:
                break
    edge_index = np.array(edges).T  # shape [2, E]
    edge_wts   = np.array(wts)
    return edge_index, edge_wts

def rowwise_normalize(edge_index_t, edge_attr_t, num_nodes):
    sums = torch.zeros(num_nodes, dtype=torch.float, device=edge_attr_t.device)
    src  = edge_index_t[0]
    sums.index_add_(0, src, edge_attr_t)
    eps  = 1e-12
    new_attr = edge_attr_t / (sums[src] + eps)
    return new_attr

###############################################################################
# 3. GraphSAGE Model
###############################################################################
class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super().__init__()
        self.dropout_val = dropout
        self.layers = nn.ModuleList()
        # first layer
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        # middle layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        # last layer
        self.layers.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)  # SAGEConv typically ignores edge_attr by default
            if i < len(self.layers)-1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_val, training=self.training)
        return x

###############################################################################
# 4. Train GraphSAGE
###############################################################################
def train_sage(data, hidden_dim=256, num_layers=3, dropout=0.2,
               lr=0.001, weight_decay=5e-4, epochs=150, patience=30):
    in_channels  = data.num_features
    out_channels = len(torch.unique(data.y))

    model= GraphSAGEModel(in_channels, hidden_dim, out_channels, num_layers, dropout).to(DEVICE)
    data = data.to(DEVICE)
    optimizer= torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step():
        model.train()
        optimizer.zero_grad()
        out= model(data.x, data.edge_index, data.edge_attr)
        loss= F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(mask):
        model.eval()
        out= model(data.x, data.edge_index, data.edge_attr)
        preds= out.argmax(dim=-1)
        correct= (preds[mask]== data.y[mask]).sum().item()
        total= int(mask.sum())
        return correct/ total if total>0 else 0.0

    best_val= 0.0
    best_test=0.0
    no_improve=0

    for ep in range(1, epochs+1):
        loss_val= train_step()
        val_acc= evaluate(data.val_mask)
        test_acc= evaluate(data.test_mask)

        if val_acc> best_val:
            best_val= val_acc
            best_test= test_acc
            no_improve=0
        else:
            no_improve+=1
            if no_improve>= patience:
                print(f"[EarlyStop] epoch={ep}")
                break

        if ep%10==0:
            train_acc= evaluate(data.train_mask)
            print(f"[SAGE] ep={ep}, Loss={loss_val:.4f}, "
                  f"TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, TestAcc={test_acc:.4f}")

    return best_val, best_test

###############################################################################
# 5. MLP Baseline
###############################################################################
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp_baseline(X, y, train_mask, val_mask, test_mask,
                       hidden_dim=256, dropout=0.2, lr=0.001,
                       weight_decay=5e-4, epochs=150, patience=30):
    device= X.device
    in_dim= X.size(1)
    out_dim= len(torch.unique(y))

    mlp= MLP(in_dim, hidden_dim, out_dim, dropout).to(device)
    optimizer= torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step():
        mlp.train()
        optimizer.zero_grad()
        logits= mlp(X[train_mask])
        loss= F.cross_entropy(logits, y[train_mask])
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(mask):
        mlp.eval()
        out= mlp(X[mask])
        preds= out.argmax(dim=-1)
        correct= (preds== y[mask]).sum().item()
        total= int(mask.sum())
        return correct/ total if total>0 else 0.0

    best_val= 0.0
    best_test=0.0
    no_improve=0

    for ep in range(1, epochs+1):
        loss_val= train_step()
        val_acc= evaluate(val_mask)
        test_acc= evaluate(test_mask)

        if val_acc> best_val:
            best_val= val_acc
            best_test= test_acc
            no_improve=0
        else:
            no_improve+=1
            if no_improve>= patience:
                print(f"[EarlyStop - MLP] epoch={ep}")
                break

        if ep%10==0:
            tr_acc= evaluate(train_mask)
            print(f"[MLP] ep={ep}, Loss={loss_val:.4f}, "
                  f"TrainAcc={tr_acc:.4f}, ValAcc={val_acc:.4f}, TestAcc={test_acc:.4f}")

    return best_val, best_test

###############################################################################
# 6. RUN EXPERIMENTS: Keep vs Drop Hours-of-Sleep Column
###############################################################################
def run_experiment(drop_hours=False):
    """
    Returns (mlp_mean, mlp_std), (sage_mean, sage_std)
    across seeds, toggling whether to drop "4. On average, how many hours..."
    """
    mlp_test_scores  = []
    sage_test_scores = []

    for seed in SEEDS:
        print(f"\n=== SEED={seed}, drop_hours={drop_hours} ===")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 1) Load numeric data
        df = pd.read_csv(DATA_PATH)
        df = df.select_dtypes(include=[np.number])  # numeric only

        # Always drop Timestamp if present
        if TIMESTAMP_COL in df.columns:
            df.drop(columns=[TIMESTAMP_COL], inplace=True, errors='ignore')

        # Optionally drop the hours-of-sleep column
        if drop_hours and (HOURS_COL in df.columns):
            df.drop(columns=[HOURS_COL], inplace=True, errors='ignore')

        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing target '{TARGET_COL}' in columns: {df.columns.tolist()}")

        # Merge classes => 3
        df[TARGET_COL] = merge_classes(df[TARGET_COL].values)

        # Build X,y
        X_df = df.drop(columns=[TARGET_COL], errors='ignore')
        y_df = df[TARGET_COL]
        X_full= X_df.values
        y_full= y_df.values

        # 2) Train/Val/Test
        idx_all= np.arange(len(X_full))
        tr_idx, te_idx= train_test_split(idx_all, test_size=0.2, random_state=seed)
        tr_idx, va_idx= train_test_split(tr_idx, test_size=0.25, random_state=seed)
        # => 60/20/20

        X_train= X_full[tr_idx]
        y_train= y_full[tr_idx]

        # SMOTE
        sm= SMOTE(random_state=seed)
        X_res, y_res= sm.fit_resample(X_train, y_train)
        print(f"[Seed={seed}] SMOTE from {len(X_train)} => {len(X_res)}. drop_hours={drop_hours}")

        X_val=  X_full[va_idx]
        y_val=  y_full[va_idx]
        X_test= X_full[te_idx]
        y_test= y_full[te_idx]

        # Re-map new indices
        new_tr_idx= np.arange(len(X_res))
        val_off= len(X_res)
        new_va_idx= np.arange(val_off, val_off+ len(X_val))
        test_off= val_off+ len(X_val)
        new_te_idx= np.arange(test_off, test_off+ len(X_test))

        X_all= np.concatenate([X_res, X_val, X_test], axis=0)
        y_all= np.concatenate([y_res, y_val, y_test], axis=0)

        # 3) Scale
        scaler= StandardScaler()
        X_all_sc= scaler.fit_transform(X_all)

        # Masks
        fullN= len(X_all_sc)
        train_mask= torch.zeros(fullN, dtype=torch.bool)
        val_mask=   torch.zeros(fullN, dtype=torch.bool)
        test_mask=  torch.zeros(fullN, dtype=torch.bool)
        train_mask[new_tr_idx] = True
        val_mask[new_va_idx]   = True
        test_mask[new_te_idx]  = True

        # 4) MLP Baseline
        X_torch= torch.tensor(X_all_sc, dtype=torch.float, device=DEVICE)
        y_torch= torch.tensor(y_all, dtype=torch.long, device=DEVICE)

        mlp_val, mlp_test= train_mlp_baseline(
            X_torch, y_torch, train_mask, val_mask, test_mask,
            hidden_dim= HIDDEN_DIM,
            dropout= DROPOUT,
            lr= LR,
            weight_decay= WEIGHT_DECAY,
            epochs= EPOCHS,
            patience= PATIENCE
        )
        mlp_test_scores.append(mlp_test)
        print(f"[Seed={seed}] MLP => bestVal={mlp_val:.4f}, test={mlp_test:.4f}")

        # 5) GraphSAGE
        edge_i, edge_w= build_topk_graph(X_all_sc, k=TOP_K)
        ei_t= torch.tensor(edge_i, dtype=torch.long)
        ew_t= torch.tensor(edge_w, dtype=torch.float)
        num_nodes= X_torch.size(0)
        ew_t= rowwise_normalize(ei_t, ew_t, num_nodes)

        data= Data(
            x= X_torch.cpu(),
            y= y_torch.cpu(),
            edge_index= ei_t.cpu(),
            edge_attr= ew_t.cpu()
        )
        data.train_mask= train_mask.cpu()
        data.val_mask=   val_mask.cpu()
        data.test_mask=  test_mask.cpu()

        sage_val, sage_test= train_sage(
            data,
            hidden_dim= HIDDEN_DIM,
            num_layers= NUM_LAYERS,
            dropout= DROPOUT,
            lr= LR,
            weight_decay= WEIGHT_DECAY,
            epochs= EPOCHS,
            patience= PATIENCE
        )
        sage_test_scores.append(sage_test)
        print(f"[Seed={seed}] SAGE => bestVal={sage_val:.4f}, test={sage_test:.4f}")

    # Final average
    mlp_mean  = np.mean(mlp_test_scores)
    mlp_std   = np.std(mlp_test_scores)
    sage_mean = np.mean(sage_test_scores)
    sage_std  = np.std(sage_test_scores)
    return (mlp_mean, mlp_std), (sage_mean, sage_std)

###############################################################################
# MAIN
###############################################################################
def main():
    # Experiment A: KEEP hours-of-sleep column
    print("\n========== EXPERIMENT A: Keep Hours-of-Sleep ===========\n")
    mlp_resA, sage_resA = run_experiment(drop_hours=False)
    mlp_meanA, mlp_stdA = mlp_resA
    sage_meanA, sage_stdA= sage_resA

    # Experiment B: DROP hours-of-sleep column
    print("\n========== EXPERIMENT C: Drop Hours-of-Sleep ===========\n")
    mlp_resB, sage_resB = run_experiment(drop_hours=True)
    mlp_meanB, mlp_stdB = mlp_resB
    sage_meanB, sage_stdB= sage_resB

    # Compare final results
    headers = ["Experiment", "Model", "Test Accuracy (Mean)", "Standard Deviation"]
    rows = [
        ["A (Keep Hours of Sleep)", "MLP", f"{mlp_meanA:.4f}", f"± {mlp_stdA:.4f}"],
        ["A (Keep Hours of Sleep)", "SAGE", f"{sage_meanA:.4f}", f"± {sage_stdA:.4f}"],
        ["C (Drop Hours of Sleep)", "MLP", f"{mlp_meanB:.4f}", f"± {mlp_stdB:.4f}"],
        ["C (Drop Hours of Sleep)", "SAGE", f"{sage_meanB:.4f}", f"± {sage_stdB:.4f}"],
    ]

    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("tight")
    ax.axis("off")

    header_colors = ["#4CAF50"] * len(headers)
    row_colors = ["#f9f9f9", "#ffffff"]

    # Add the table
    table = ax.table(
        cellText=[headers] + rows,
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3, 0.4, 0.4]
    )
    # Style the header row 
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor(header_colors[i % len(header_colors)])
        cell.set_text_props(weight="bold", color="white")
        cell.set_edgecolor("black")

    # Style the data rows
    for row_idx, row in enumerate(rows, start=1):
        for col_idx in range(len(headers)):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(row_colors[row_idx % len(row_colors)])
            cell.set_edgecolor("black")

    # Adjust font size and column widths
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(headers))))

    # Save the figure
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "drop_hours_of_sleep_experiment.svg")
    plt.savefig(output_file, bbox_inches="tight")

    output_file

if __name__=="__main__":
    main()
