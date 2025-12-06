import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/content/drive/MyDrive/CS7643 Project/OMSCS_7643_F2025_Project")
RUN_DIR = PROJECT_ROOT / "output/hyper_runs"
SUMMARY_CSV = PROJECT_ROOT / "output/hyper_runs/summary.csv"

df = pd.read_csv(SUMMARY_CSV)


# ===========================================================
# Load metrics.json for run
# ===========================================================
def load_metrics(run_id):
    p = RUN_DIR / f"run_{run_id:04d}" / "metrics.json"
    with open(p, "r") as f:
        j = json.load(f)

    baseline = j["linear_regression_baseline"]
    base_test = baseline["regression_metrics"]
    base_hold = baseline["holdout_metrics"]["regression_metrics"]

    mtest = j["model_predictions"]["subset_metrics"]["test"]["regression_metrics"]
    mhold = j["model_predictions"]["subset_metrics"]["holdout"]["regression_metrics"]

    return {
        # baseline
        "bl_da_hold": base_hold["directional_accuracy"],
        "bl_rmse_hold": base_hold["rmse"],

        # model
        "da_test": mtest["directional_accuracy"],
        "da_hold": mhold["directional_accuracy"],
        "rmse_test": mtest["rmse"],
        "rmse_hold": mhold["rmse"],
        "ic_test": mtest["pearson_ic"],
        "ic_hold": mhold["pearson_ic"],
        "sr_test": j["model_predictions"]["subset_metrics"]["test"]["strategy_metrics"]["sharpe"],
        "sr_hold": j["model_predictions"]["subset_metrics"]["holdout"]["strategy_metrics"]["sharpe"],
    }


# ===========================================================
# STAGE 1: IC + Sharpe Filtering
# ===========================================================
stage1 = df[
    (df["IC_test"] > 0) &
    (df["IC_holdout"] > 0) &
    (df["Sharpe_test"] > 0) &
    (df["Sharpe_holdout"] > 0)
].copy()

print(f"Stage 1: {len(stage1)} runs passed IC/SR filtering.\n")


# ===========================================================
# STAGE 2: DA filtering (holdout >0.5 AND > baseline)
# ===========================================================
stage2_list = []
for _, row in stage1.iterrows():

    run_id = int(row["run_id"])
    m = load_metrics(run_id)

    cond1 = m["da_hold"] > 0.5
    cond2 = m["da_hold"] > m["bl_da_hold"]

    if cond1 and cond2:
        stage2_list.append(row)

stage2 = pd.DataFrame(stage2_list)
print(f"Stage 2: {len(stage2)} runs remain after DA filtering.\n")

print("=== Candidates Remaining Per Model ===")
print(stage2["model"].value_counts(), "\n")


# ===========================================================
# STAGE 3: Select best run per model based on score
# ===========================================================
best_models = {}

for model in stage2["model"].unique():
    sub = stage2[stage2["model"] == model]
    best = sub.sort_values("score", ascending=False).iloc[0]
    best_models[model] = best

print("=== Final Best Models (Stage 3) ===")
for m, r in best_models.items():
    print(f"{m}: run_id={int(r['run_id'])}, score={r['score']:.4f}")
print("\n")


# ===========================================================
# Save metrics of best runs
# ===========================================================
best_details = {}

for model, row in best_models.items():
    run_id = int(row["run_id"])
    m = load_metrics(run_id)
    best_details[model] = m

# ===========================================================
# STAGE 3: Normalize best_models to {model: run_id}
# ===========================================================
best_models = {m: int(r["run_id"]) for m, r in best_models.items()}

print("=== Final Best Models (Stage 3) ===")
for m, run_id in best_models.items():
    print(f"{m}: run_id={run_id}")
print("\n")

# # ===========================================================
# # (2) PLOT: Overfitting Plot (Test vs Holdout IC & DA)
# # ===========================================================
# plt.figure(figsize=(12, 8))

# i = 1
# for model, row in best_models.items():
#     run_id = int(row["run_id"])
#     m = load_metrics(run_id)

#     plt.subplot(2, 2, i)
#     plt.title(f"{model.upper()} (run {run_id})")

#     # IC comparison
#     plt.plot(["Test IC", "Holdout IC"], [m["ic_test"], m["ic_hold"]],
#              marker="o", label="IC")

#     # DA comparison
#     plt.plot(["Test DA", "Holdout DA"], [m["da_test"], m["da_hold"]],
#              marker="o", label="DA")

#     plt.ylim(0, 1)
#     plt.legend()
#     i += 1

# plt.tight_layout()
# plt.savefig("best_models_overfitting.png", dpi=300)
# plt.show()


# # ===========================================================
# # (3) TABLE: Model Comparison Table
# # ===========================================================
# table_rows = []

# for model, row in best_models.items():
#     run_id = int(row["run_id"])
#     m = load_metrics(run_id)
#     bl = m["bl_da_hold"]

#     table_rows.append([
#         model,
#         row["run_id"],
#         m["sr_test"], m["sr_hold"],
#         m["ic_test"], m["ic_hold"],
#         m["rmse_test"], m["rmse_hold"],
#         m["da_test"], m["da_hold"],
#         m["bl_da_hold"]
#     ])

# table = pd.DataFrame(table_rows, columns=[
#     "Model", "Run",
#     "SR Test", "SR Holdout",
#     "IC Test", "IC Holdout",
#     "RMSE Test", "RMSE Holdout",
#     "DA Test", "DA Holdout",
#     "Baseline DA Holdout"
# ])

# print("\n=== Final Comparison Table ===")
# print(table.to_string(index=False))

# table.to_csv("comparison_table.csv", index=False)
import re
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ----------------------------------------------------
# Utility: parse train and val losses from log.txt
# ----------------------------------------------------
def parse_losses(log_path):
    train_losses = []
    val_losses = []

    if not log_path.exists():
        return train_losses, val_losses

    with open(log_path, "r") as f:
        for line in f:
            # Match: Train Loss: x.xxx | Val Loss: y.yyy
            m = re.search(r"Train Loss:\s*([\d\.e-]+)\s*\|\s*Val Loss:\s*([\d\.e-]+)", line)
            if m:
                train_losses.append(float(m.group(1)))
                val_losses.append(float(m.group(2)))

    return train_losses, val_losses


# -------------------------------------
# Best model runs determined from Stage 3
# -------------------------------------
# best_models = {
#     "MLP": 5,
#     "LSTM": 47,
#     "GRU": 96,
#     "Transformer": 156
# }
# best_models = { "mlp": row, "lstm": row, ... }

BASE = Path("/content/drive/MyDrive/CS7643 Project/OMSCS_7643_F2025_Project")
HYPER_RUNS = BASE / "output" / "hyper_runs"

# Create a folder for figures if not exists
FIG_DIR = BASE / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------
# Plot learning curves
# -------------------------------------
plt.figure(figsize=(12, 10))

for i, (model_name, run_id) in enumerate(best_models.items(), 1):

    log_path = HYPER_RUNS / f"run_{run_id:04d}" / "log.txt"
    train_losses, val_losses = parse_losses(log_path)

    plt.subplot(2, 2, i)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"{model_name} Best Model", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()

plt.tight_layout()

# ----------------------------------------------------
# 🔥 SAVE THE FIGURE
# ----------------------------------------------------
save_path = FIG_DIR / "learning_curves_4models.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Learning curve figure saved to:\n{save_path}")

# ===========================================================
# Utility: Extract key hyperparameters for each model
# ===========================================================
KEY_PARAMS = {
    "mlp": ["lookback_mode", "lookback", "lr", "hidden_dim", "dropout"],
    "lstm": ["lookback_mode", "lookback", "seq_len", "lr", "hidden_dim", 
             "lstm_proj_dim", "num_layers"],
    "gru": ["lookback_mode", "lookback", "seq_len", "lr", "hidden_dim",
            "num_layers"],
    "transformer": ["lookback_mode", "lookback", "seq_len", "lr",
                    "tf_d_model", "tf_heads", "tf_layers", "tf_ff_dim"],
}

def load_config(run_id):
    p = RUN_DIR / f"run_{run_id:04d}" / "config.json"
    with open(p, "r") as f:
        return json.load(f)

# ===========================================================
# Build Two Tables:
#  1) Performance table
#  2) Hyperparameter table
# ===========================================================

perf_rows = []
hyper_rows = []

for model, run_id in best_models.items():

    run_id = int(run_id)   
    m = load_metrics(run_id)
    cfg = load_config(run_id)

    # ================ Performance Table ================
    perf_rows.append([
        model,
        run_id,
        m["rmse_test"], m["rmse_hold"],
        m["ic_test"], m["ic_hold"],
        m["sr_test"], m["sr_hold"],
        m["da_test"], m["da_hold"],
        m["bl_da_hold"],
        abs(m["ic_test"] - m["ic_hold"]),
        abs(m["sr_test"] - m["sr_hold"]),
    ])

    # ================ Hyperparameter Table ================
    keys = KEY_PARAMS[model.lower()]
    selected_params = {k: cfg.get(k, None) for k in keys}

    hyper_rows.append([model, run_id, selected_params])

# Convert to DataFrames
perf_table = pd.DataFrame(perf_rows, columns=[
    "Model", "Run",
    "RMSE Test", "RMSE Holdout",
    "IC Test", "IC Holdout",
    "SR Test", "SR Holdout",
    "DA Test", "DA Holdout",
    "Baseline Holdout DA",
    "IC Gap", "SR Gap"
])

hyper_table = pd.DataFrame(hyper_rows, columns=[
    "Model", "Run", "Key Hyperparameters"
])

# Save tables
perf_path = PROJECT_ROOT / "output" / "figures" / "final_model_performance.csv"
hyper_path = PROJECT_ROOT / "output" / "figures" / "final_model_hyperparams.csv"

perf_table.to_csv(perf_path, index=False)
hyper_table.to_csv(hyper_path, index=False)

print("\n=== Final Performance Table ===")
print(perf_table.to_string(index=False))

print("\n=== Final Hyperparameter Table ===")
print(hyper_table.to_string(index=False))

print("\nTables saved to:")
print(perf_path)
print(hyper_path)
