"""
Hyperparameter Search Framework
Supports:
  - Resume on interruption
  - IC maximization (Sharpe > 0 required)
  - Compatible with train.py, predict.py, evaluate.py
  - Callable API (HyperSearch class) and CLI
"""

import os
import json
import time
import random
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
#   0. GLOBAL PATH CONFIG
# ============================================================

# Change this to your actual project path:
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/CS7643 Project/OMSCS_7643_F2025_Project/"

SCRIPTS_DIR = Path(GOOGLE_DRIVE_PATH) / "scripts"
OUTPUT_DIR = Path(GOOGLE_DRIVE_PATH) / "output"
HYPER_DIR = OUTPUT_DIR / "hyper_runs"

TRAIN_PY = SCRIPTS_DIR / "train.py"
PREDICT_PY = SCRIPTS_DIR / "predict.py"
EVAL_PY = SCRIPTS_DIR / "evaluate.py"

HYPER_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================
#   1. UTILITY FUNCTIONS
# ============================================================

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_log(path, text):
    with open(path, "a") as f:
        f.write(text + "\n")


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def next_run_id():
    """
    Scan hyper_runs/ and determine next run_xxxx number.
    If interrupted, the next run id continues sequentially.
    """
    HYPER_DIR.mkdir(exist_ok=True)
    runs = sorted([d for d in HYPER_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    if len(runs) == 0:
        return 1
    
    last = runs[-1].name.replace("run_", "")
    return int(last) + 1


# ============================================================
#   2. SEARCH SPACE DEFINITION (FULL, QUICK, CUSTOM)
# ============================================================

def get_search_space(mode="full"):
    """
    Returns a dictionary of parameter lists for sweeping.
    mode = "full" / "quick" / "custom"
    """

    if mode == "quick":
        # Light search for debugging
        return {
            "model": ["mlp", "lstm", "transformer"],
            "seq_len": [1, 4],
            "batch_size": [64],
            "lr": [1e-4],
            "epochs": [5],          # very small for quick test
            "weight_decay": [0.0],
            "scheduler": [None],
            "ema_decay": [None],
            "hidden_dim": [128],
            "lstm_proj_dim": [64],
            "tf_d_model": [128],
            "tf_heads": [4],
            "tf_layers": [2],
            "tf_ff_dim": [256],
            "tf_pool": ["attention"],
            "tf_dropout": [0.1],
        }

    # ======================================================
    # FULL SEARCH SPACE (recommended for IC optimization)
    # ======================================================
    if mode == "full":
        return {

            # Feature CHOICE
            "lookback_mode": ["mean", "max", "volume", "exp_decay", "attn"],
            "horizon": [1, 3],
            "lookback": [6, 12, 24, 48],

            # =======================================
            #  Model choice
            # =======================================
            "model": ["mlp", "lstm", "gru", "transformer"],

            "seq_len": [1, 4, 8, 12],

            # =======================================
            #  Optimization
            # =======================================
            "batch_size": [32, 64, 128],

            # Learning rate 
            "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],

            # epochs
            "epochs": [15, 30],

            # weight decay
            "weight_decay": [0.0, 1e-5, 1e-4],

            # warmup + cosine scheduler
            "scheduler": [None, "cosine_warmup"],
            "warmup_pct": [0.05, 0.1],

            # grad clip
            "grad_clip": [None, 1.0, 5.0],

            "ema_decay": [None],

            # =======================================
            #  MLP
            # =======================================
            "hidden_dim": [128, 256, 512, 768],
            "dropout": [0.1, 0.2],

            # =======================================
            #  LSTM
            # =======================================
            "num_layers": [1, 2, 3],
            "lstm_proj_dim": [32, 64, 128, 256],
            "lstm_use_layernorm": [0, 1],
            "lstm_use_attention": [0, 1],

            # =======================================
            #  GRU
            # =======================================

            # =======================================
            #  Transformer
            # =======================================
            "tf_d_model": [128, 256, 384, 512],
            "tf_heads": [2, 4, 8],
            "tf_layers": [2, 4],
            "tf_ff_dim": [256, 512, 1024],
            "tf_pool": ["attention", "cls"],
            "tf_dropout": [0.1, 0.2],
            "tf_learnable_pos": [0, 1],
            "tf_use_cls_token": [0, 1],
            "tf_embed_scale": [0.01, 0.1, 1.0]
          

        }

    # ======================================================
    # CUSTOM (user-defined later)
    # ======================================================
    if mode == "custom":
        return {}

    raise ValueError(f"Unknown search mode: {mode}")

# ============================================================
#   3. RUN EXECUTOR (train → predict → evaluate)
# ============================================================

class SingleRunExecutor:
    def __init__(self, run_dir: Path, config: dict):
        self.run_dir = run_dir
        self.config = config
        self.log_path = run_dir / "log.txt"

    def rebuild_features(self):
      cfg = self.config
      cmd = [
          "python", str(SCRIPTS_DIR / "build_features.py"),
          "--horizon", str(cfg["horizon"]),
          "--lookback", str(cfg["lookback"]),
          "--lookback-mode", cfg["lookback_mode"],
      ]

      write_log(self.log_path, f"Rebuilding features with: {cmd}")
      subprocess.run(cmd, check=True)


    # --------------------------------------------------------
    # Build train.py command line args
    # --------------------------------------------------------
    def build_train_cmd(self):
        cfg = self.config
        cmd = ["python", str(TRAIN_PY)]

        # Required
        cmd += ["--model", cfg["model"]]
        cmd += ["--seq_len", str(cfg["seq_len"])]
        cmd += ["--batch_size", str(cfg["batch_size"])]
        cmd += ["--lr", str(cfg["lr"])]
        cmd += ["--epochs", str(cfg["epochs"])]
        cmd += ["--weight_decay", str(cfg["weight_decay"])]

        # Optional
        if cfg["scheduler"] is not None:
            cmd += ["--scheduler", cfg["scheduler"]]
        if cfg["ema_decay"] is not None:
            cmd += ["--ema_decay", str(cfg["ema_decay"])]
        if cfg["warmup_pct"] is not None:
            cmd += ["--warmup_pct", str(cfg["warmup_pct"])]

        # Model-specific
        cmd += ["--hidden_dim", str(cfg["hidden_dim"])]
        cmd += ["--lstm_proj_dim", str(cfg["lstm_proj_dim"])]
        cmd += ["--num_layers", str(cfg.get("num_layers", 2))]

        # Transformer-specific
        cmd += ["--tf_d_model", str(cfg["tf_d_model"])]
        cmd += ["--tf_heads", str(cfg["tf_heads"])]
        cmd += ["--tf_layers", str(cfg["tf_layers"])]
        cmd += ["--tf_ff_dim", str(cfg["tf_ff_dim"])]
        cmd += ["--tf_pool", cfg["tf_pool"]]
        cmd += ["--tf_dropout", str(cfg["tf_dropout"])]

        return cmd

    # --------------------------------------------------------
    # Build prediction command
    # --------------------------------------------------------
    def build_predict_cmd(self):
        cfg = self.config
        out_csv = self.run_dir / "predictions.csv"
        cmd = [
            "python", str(PREDICT_PY),
            "--model", cfg["model"],
            "--seq_len", str(cfg["seq_len"]),
            "--batch_size", str(cfg["batch_size"]),
            "--cutoff_date", cfg.get("cutoff_date", "2024-10-01")
        ]
        return cmd, out_csv

    # --------------------------------------------------------
    # Build evaluation command
    # --------------------------------------------------------
    def build_eval_cmd(self, pred_csv):
        cmd = [
            "python", str(EVAL_PY),
            "--predictions", str(pred_csv)
        ]
        return cmd

    # --------------------------------------------------------
    # Helper: run command and log output
    # --------------------------------------------------------
    def run_cmd(self, cmd):
        write_log(self.log_path, f"\n[{timestamp()}] Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        for line in process.stdout:
            write_log(self.log_path, line.rstrip())

        process.wait()
        return process.returncode

    # --------------------------------------------------------
    # Execute one full run (train → predict → evaluate)
    # --------------------------------------------------------
    def execute(self):
        write_log(self.log_path, f"=== RUN START {timestamp()} ===")
        # STEP 0: rebuild features
        self.rebuild_features()
        print('New features built')
        # Save config
        save_json(self.config, self.run_dir / "config.json")

        # ===========================
        # 1. TRAIN
        # ===========================
        t0 = time.time()
        train_cmd = self.build_train_cmd()
        rc = self.run_cmd(train_cmd)
        if rc != 0:
            write_log(self.log_path, "ERROR: train.py failed")
            return None
        train_time = time.time() - t0

        # ===========================
        # 2. PREDICT
        # ===========================
        predict_cmd, pred_csv = self.build_predict_cmd()
        rc = self.run_cmd(predict_cmd)
        if rc != 0:
            write_log(self.log_path, "ERROR: predict.py failed")
            return None

        # predictions save inside output/predictions/predictions_{model}.csv
        pred_source = Path(GOOGLE_DRIVE_PATH) / "output" / "predictions" / f"predictions_{self.config['model']}.csv"
        if pred_source.exists():
            shutil.copy(pred_source, pred_csv)
        else:
            write_log(self.log_path, "ERROR: Prediction file missing.")
            return None

        # ===========================
        # 3. EVALUATE
        # ===========================
        eval_cmd = self.build_eval_cmd(pred_csv)
        rc = self.run_cmd(eval_cmd)
        if rc != 0:
            write_log(self.log_path, "ERROR: evaluate.py failed")
            return None

        # Evaluation output always saved to:
        #   output/reports/performance_report.json
        report_path = Path(GOOGLE_DRIVE_PATH) / "output" / "reports" / "performance_report.json"
        if not report_path.exists():
            write_log(self.log_path, "ERROR: evaluate output not found.")
            return None

        metrics = load_json(report_path)
        save_json(metrics, self.run_dir / "metrics.json")

        # ===========================
        # Extract scores
        # ===========================
        model_pred = metrics.get("model_predictions", {})
        subsets = model_pred.get("subset_metrics", {})
        holdout = subsets.get("holdout", {})

        reg = holdout.get("regression_metrics", {})
        strat = holdout.get("strategy_metrics", {})

        IC = reg.get("pearson_ic", None)
        sharpe = strat.get("sharpe", None)

        write_log(self.log_path, f"IC = {IC}, Sharpe = {sharpe}")

        return {
            "IC": IC,
            "sharpe": sharpe,
            "train_time": train_time,
            "config": self.config,
            "run_dir": self.run_dir,
            "metrics": metrics
        }
# ============================================================
#   4. SCORING, LOGGING & VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt


def compute_score(IC, sharpe):
    """
    Main objective:
        Maximize IC
    Constraint:
        Sharpe must be > 0 to avoid useless/no-trade models
    """
    if IC is None or sharpe is None:
        return -9999

    if sharpe <= 0:
        return -9999

    return IC   # maximize IC


def save_global_best(best, path):
    """
    Save best_run.json structure:
    {
        "best_score": ...,
        "best_IC": ...,
        "best_sharpe": ...,
        "best_run_dir": ...,
        "config": {...}
    }
    """
    info = {
        "best_score": best["score"],
        "best_IC": best["IC"],
        "best_sharpe": best["sharpe"],
        "best_run_dir": str(best["run_dir"]),
        "config": best["config"]
    }
    save_json(info, path)


def plot_metric_curve(values, ylabel, save_path):
    if len(values) == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(values, marker="o")
    plt.xlabel("Run ID")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def update_summary_csv(all_results, save_path):
    """
    Save summary of all runs as CSV:
      run_id | IC | sharpe | score | model | seq_len | lr | ...
    """
    rows = []
    for r in all_results:
        row = {
            "run_id": r["run_id"],
            "IC": r["IC"],
            "sharpe": r["sharpe"],
            "score": r["score"]
        }
        # add config parameters
        for k, v in r["config"].items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
# ============================================================
#   5. HYPERSEARCH MAIN CLASS
# ============================================================

class HyperSearch:
    def __init__(
        self,
        max_runs=50,
        search_mode="full",
        custom_space=None
    ):
        self.max_runs = max_runs

        # load search space
        if search_mode == "custom" and custom_space is not None:
            self.space = custom_space
        else:
            self.space = get_search_space(search_mode)

        self.results = []
        self.best = {
            "score": -9999,
            "IC": None,
            "sharpe": None,
            "run_dir": None,
            "config": None,
        }

        self.best_path = HYPER_DIR / "best_run.json"
        self.summary_csv = HYPER_DIR / "summary.csv"

        self.IC_history = []
        self.sharpe_history = []
        self.score_history = []


    # --------------------------------------------------------
    # Sample a random configuration
    # --------------------------------------------------------
    def sample_config(self):
        cfg = {}
        for k, values in self.space.items():
            if len(values) == 0:
                continue
            cfg[k] = random.choice(values)

        if cfg["model"] == "mlp":
            cfg["seq_len"] = 1

        # default cutoff_date for predict
        cfg["cutoff_date"] = "2024-10-01"
        return cfg


    # --------------------------------------------------------
    # Run a single hyperparameter configuration
    # --------------------------------------------------------
    def run_once(self, run_id):
        run_dir = HYPER_DIR / f"run_{run_id:04d}"
        run_dir.mkdir(exist_ok=True)

        cfg = self.sample_config()
        executor = SingleRunExecutor(run_dir, cfg)

        result = executor.execute()
        if result is None:
            return None

        IC = result["IC"]
        sharpe = result["sharpe"]
        score = compute_score(IC, sharpe)

        # record
        self.IC_history.append(IC)
        self.sharpe_history.append(sharpe)
        self.score_history.append(score)

        # update best
        if score > self.best["score"]:
            self.best.update({
                "score": score,
                "IC": IC,
                "sharpe": sharpe,
                "run_dir": result["run_dir"],
                "config": result["config"]
            })
            save_global_best(self.best, self.best_path)

        # save to results list
        self.results.append({
            "run_id": run_id,
            "IC": IC,
            "sharpe": sharpe,
            "score": score,
            "config": result["config"],
            "run_dir": str(result["run_dir"])
        })

        update_summary_csv(self.results, self.summary_csv)

        # write plots
        plot_metric_curve(self.IC_history, "IC", HYPER_DIR / "IC_history.png")
        plot_metric_curve(self.sharpe_history, "Sharpe", HYPER_DIR / "sharpe_history.png")
        plot_metric_curve(self.score_history, "Score", HYPER_DIR / "score_history.png")

        return score


    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    def run(self):
        print("===== HyperSearch Starting =====")
        print(f"Output: {HYPER_DIR}")
        print(f"Max runs: {self.max_runs}")
        print(f"Search space keys: {list(self.space.keys())}")

        start_id = next_run_id()

        for i in range(start_id, start_id + self.max_runs):
            print(f"\n=== RUN {i} / {start_id + self.max_runs - 1} ===")
            score = self.run_once(i)
            print(f"Run {i} finished, score={score}")

        print("\n===== HyperSearch Completed =====")
        print(f"Best score = {self.best['score']}")
        print(f"Best IC    = {self.best['IC']}")
        print(f"Best Sharpe= {self.best['sharpe']}")
        print(f"Best run at: {self.best['run_dir']}")


# ============================================================
#   6. CLI ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-runs", type=int, default=50)
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "quick", "custom"])

    args = parser.parse_args()

    hs = HyperSearch(
        max_runs=args.max_runs,
        search_mode=args.mode,
    )
    hs.run()
