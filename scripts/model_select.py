import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import re
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

    sm = j["model_predictions"]["subset_metrics"]

    # Train
    train_reg = sm["train"]["regression_metrics"]
    train_str = sm["train"]["strategy_metrics"]

    # Test
    test_reg = sm["test"]["regression_metrics"]
    test_str = sm["test"]["strategy_metrics"]

    # Holdout
    hold_reg = sm["holdout"]["regression_metrics"]
    hold_str = sm["holdout"]["strategy_metrics"]

    # Baseline
    baseline = j["linear_regression_baseline"]
    base_reg_test = baseline["regression_metrics"]
    base_reg_hold = baseline["holdout_metrics"]["regression_metrics"]
    base_sr_test = baseline["strategy_metrics"]["sharpe"]
    base_sr_hold = baseline["holdout_metrics"]["strategy_metrics"]["sharpe"]

    return {

        # ---- Train ----
        "ic_train": train_reg["pearson_ic"],
        "sr_train": train_str["sharpe"],
        "rmse_train": train_reg["rmse"],   

        # ---- Test ----
        "ic_test": test_reg["pearson_ic"],
        "sr_test": test_str["sharpe"],
        "rmse_test": test_reg["rmse"],     
        "da_test": test_reg["directional_accuracy"],

        # ---- Holdout ----
        "ic_hold": hold_reg["pearson_ic"],
        "sr_hold": hold_str["sharpe"],
        "rmse_hold": hold_reg["rmse"],     
        "da_hold": hold_reg["directional_accuracy"],

        # -------- Baseline --------
        "bl_ic_test": base_reg_test["pearson_ic"],
        "bl_ic_hold": base_reg_hold["pearson_ic"],
        "bl_sr_test": base_sr_test,
        "bl_sr_hold": base_sr_hold,
        "bl_rmse_test": base_reg_test["rmse"],
        "bl_rmse_hold": base_reg_hold["rmse"],
        "bl_da_test": base_reg_test["directional_accuracy"],
        "bl_da_hold": base_reg_hold["directional_accuracy"],
    }
## Separately load summary for plots
summary = df.copy()

summary["ic_train"] = summary["run_id"].apply(lambda rid: load_metrics(int(rid))["ic_train"])
summary["ic_test"]  = summary["run_id"].apply(lambda rid: load_metrics(int(rid))["ic_test"])

summary["sr_train"] = summary["run_id"].apply(lambda rid: load_metrics(int(rid))["sr_train"])
summary["sr_test"]  = summary["run_id"].apply(lambda rid: load_metrics(int(rid))["sr_test"])

summary["avg_ic"]   = (summary["ic_train"] + summary["ic_test"]) / 2
summary["avg_sr"]   = (summary["sr_train"] + summary["sr_test"]) / 2

print("Summary with IC/SR added:")
print(summary.head())
# ===========================================================
# STAGE 1: Filter runs with positive train/test IC & SR
# ===========================================================
stage1_list = []

for _, row in df.iterrows():
    run_id = int(row["run_id"])
    m = load_metrics(run_id)

    if (
        m["ic_train"] > 0 and
        m["sr_train"] > 0 and
        m["ic_test"]  > 0 and
        m["sr_test"]  > 0
    ):
        stage1_list.append(row)

stage1 = pd.DataFrame(stage1_list)
print(f"Stage 1: {len(stage1)} runs passed train/test IC/SR filtering.\n")
print("=== Remaining runs per model after Stage 1 ===")
print(stage1["model"].value_counts(), "\n")

# ===========================================================
# STAGE 2: Select best run per model based on avg(IC_train, IC_test)
# ===========================================================

def ranking_score(run_id):
    m = load_metrics(run_id)
    return (m["ic_train"] + m["ic_test"]) / 2


best_models = {}

for model in stage1["model"].unique():
    sub = stage1[stage1["model"] == model].copy()

    sub["ranking_score"] = sub["run_id"].apply(lambda rid: ranking_score(int(rid)))

    best = sub.sort_values("ranking_score", ascending=False).iloc[0]
    best_models[model] = best

print("=== Final Best Models (Stage 2) ===")
for m, r in best_models.items():
    print(f"{m}: run_id={int(r['run_id'])}, avg_IC={r['ranking_score']:.5f}")
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
# Normalize best_models to {model: run_id}
# ===========================================================
best_models = {m: int(r["run_id"]) for m, r in best_models.items()}

print("=== Final Best Model IDs ===")
for m, run_id in best_models.items():
    print(f"{m}: run_id={run_id}")
print("\n")

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
# Best model runs determined from Stage 2
# -------------------------------------

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
# SAVE THE FIGURE
# ----------------------------------------------------
save_path = FIG_DIR / "learning_curves_4models.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Learning curve figure saved to:\n{save_path}")

# Utility: Extract key hyperparameters for each model
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

# Build Two Tables:
#  1) Performance table
#  2) Hyperparameter table

perf_rows = []
hyper_rows = []

for model, run_id in best_models.items():

    run_id = int(run_id)   
    m = load_metrics(run_id)
    cfg = load_config(run_id)

    # Performance Table 
    perf_rows.append([
        model, run_id,

        # -------- Train --------
        m["rmse_train"], m["ic_train"], m["sr_train"],

        # -------- Test --------
        m["rmse_test"], m["ic_test"], m["sr_test"], m["da_test"],

        # -------- Holdout --------
        m["rmse_hold"], m["ic_hold"], m["sr_hold"], m["da_hold"],

        # -------- Baseline --------
        m["bl_rmse_test"], m["bl_rmse_hold"],
        m["bl_ic_test"],   m["bl_ic_hold"],
        m["bl_sr_test"],   m["bl_sr_hold"],
        m["bl_da_test"],   m["bl_da_hold"],

        # -------- Generalization Gaps --------
        abs(m["ic_test"] - m["ic_hold"]),
        abs(m["sr_test"] - m["sr_hold"]),
    ])



    # Hyperparameter Table 
    keys = KEY_PARAMS[model.lower()]
    selected_params = {k: cfg.get(k, None) for k in keys}

    hyper_rows.append([model, run_id, selected_params])

# Convert to DataFrames
perf_table = pd.DataFrame(perf_rows, columns=[
    "Model", "Run",

    # Train
    "RMSE Train", "IC Train", "SR Train",

    # Test
    "RMSE Test", "IC Test", "SR Test", "DA Test",

    # Holdout
    "RMSE Holdout", "IC Holdout", "SR Holdout", "DA Holdout",

    # Baseline
    "BL RMSE Test", "BL RMSE Holdout",
    "BL IC Test",   "BL IC Holdout",
    "BL SR Test",   "BL SR Holdout",
    "BL DA Test",   "BL DA Holdout",

    # Gaps
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

def plot_all_models_ic_sr(summary, model_list):

    fig, axes = plt.subplots(2, len(model_list), figsize=(4*len(model_list), 10))

    # ----------------------
    # Row 1: IC scatter
    # ----------------------
    for col, model_name in enumerate(model_list):
        sub = summary[summary["model"] == model_name]

        ax_ic = axes[0, col]
        ax_ic.scatter(sub["ic_train"], sub["ic_test"], alpha=0.7)

        lim_ic = [
            min(sub["ic_train"].min(), sub["ic_test"].min()),
            max(sub["ic_train"].max(), sub["ic_test"].max())
        ]
        ax_ic.plot(lim_ic, lim_ic, "k--")

        ax_ic.set_title(f"{model_name} – IC", fontsize=12)
        ax_ic.set_xlabel("Train IC")
        ax_ic.set_ylabel("Test IC")
        ax_ic.grid(True)
        ax_ic.set_box_aspect(1)
    # ----------------------
    # Row 2: Sharpe scatter
    # ----------------------
    for col, model_name in enumerate(model_list):
        sub = summary[summary["model"] == model_name]

        ax_sr = axes[1, col]
        ax_sr.scatter(sub["sr_train"], sub["sr_test"], alpha=0.7)

        lim_sr = [
            min(sub["sr_train"].min(), sub["sr_test"].min()),
            max(sub["sr_train"].max(), sub["sr_test"].max())
        ]
        ax_sr.plot(lim_sr, lim_sr, "k--")

        ax_sr.set_title(f"{model_name} – Sharpe", fontsize=12)
        ax_sr.set_xlabel("Train Sharpe")
        ax_sr.set_ylabel("Test Sharpe")
        ax_sr.grid(True)
        ax_sr.set_box_aspect(1)
    plt.tight_layout()
    
    out_path = FIG_DIR / "ic_sr_scatter_2x4.png"
    plt.savefig(out_path, dpi=300)
    print("Saved:", out_path)

    plt.show()


plot_all_models_ic_sr(summary, ["mlp", "lstm", "gru", "transformer"])


def plot_global_ic_sr(summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # -------------------------
    # Left Panel: IC
    # -------------------------
    ax_ic = axes[0]
    ax_ic.scatter(summary["ic_train"], summary["ic_test"], alpha=0.6)

    lim_ic = [
        min(summary["ic_train"].min(), summary["ic_test"].min()),
        max(summary["ic_train"].max(), summary["ic_test"].max())
    ]
    ax_ic.plot(lim_ic, lim_ic, "k--")
    ax_ic.set_title("Train vs Test IC (All Models)", fontsize=13)
    ax_ic.set_xlabel("Train IC")
    ax_ic.set_ylabel("Test IC")
    ax_ic.grid(True)
    ax_ic.set_box_aspect(1)

    # -------------------------
    # Right Panel: Sharpe
    # -------------------------
    ax_sr = axes[1]
    ax_sr.scatter(summary["sr_train"], summary["sr_test"], alpha=0.6)

    lim_sr = [
        min(summary["sr_train"].min(), summary["sr_test"].min()),
        max(summary["sr_train"].max(), summary["sr_test"].max())
    ]
    ax_sr.plot(lim_sr, lim_sr, "k--")

    ax_sr.set_title("Train vs Test Sharpe (All Models)", fontsize=13)
    ax_sr.set_xlabel("Train Sharpe")
    ax_sr.set_ylabel("Test Sharpe")
    ax_sr.grid(True)
    ax_sr.set_box_aspect(1)

    plt.tight_layout()

    out_path = FIG_DIR / "ic_sr_scatter_global.png"
    plt.savefig(out_path, dpi=300)
    print("Saved:", out_path)

    plt.show()
plot_global_ic_sr(summary)


