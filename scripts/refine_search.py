"""One-at-a-time hyperparameter refinement with fixed training budget.

This script:
  1) Sweeps individual hyperparameters around a fixed base config.
  2) Runs train → predict → evaluate for each config.
  3) Logs test/holdout IC (Spearman) and Sharpe.
  4) Plots IC/Sharpe vs. the swept value.
  5) Picks the best test IC per model family and saves a summary CSV.

Assumptions:
  - Features are already built for lookback=24 (fixed here).
  - Train/predict/evaluate scripts live under ./scripts and write to ./output.
  - predict.py writes output/predictions/predictions_<model>.csv (we copy it per run).
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
OUTPUT = PROJECT_ROOT / "output"
RUNS_DIR = OUTPUT / "refine_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed training/eval defaults
BASE = {
    "epochs": 30,
    "batch_size": 128,
    "warmup_pct": 0.05,
    "weight_decay": 0.0,
    "lookback": 24,
    "horizon": 1,
    "cutoff_date": "2024-10-01",
    "pretest_fraction": 0.2,
    "signal_percentile": 50.0,
    "scheduler": "cosine_warmup",
}


def run_cmd(cmd: List[str], workdir: Path):
    subprocess.run(cmd, cwd=workdir, check=True)


def build_features(lookback_mode: str, lookback: int, horizon: int):
    args = [
        "python",
        str(SCRIPTS / "build_features.py"),
        "--lookback-mode",
        lookback_mode,
        "--lookback",
        str(lookback),
        "--horizon",
        str(horizon),
    ]
    run_cmd(args, PROJECT_ROOT)


def plot_seq_ic(df_all: pd.DataFrame, out_path: Path):
    df = df_all[(df_all["experiment"] == "seq_len") & (df_all["model"] == "transformer")].copy()
    if df.empty or "seq_len" not in df.columns:
        return
    df = df.sort_values("seq_len")
    df["avg_ic"] = (df["train_ic"] + df["test_ic"]) / 2.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["seq_len"], df["train_ic"], marker="o", label="train IC")
    ax.plot(df["seq_len"], df["test_ic"], marker="o", label="test IC")
    ax.plot(df["seq_len"], df["avg_ic"], marker="o", label="avg(train,test) IC")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Spearman IC")
    ax.set_title("Transformer: seq_len tuning (IC)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_capacity_ic_grid(df_all: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    families = [("mlp", axes[0, 0]), ("gru", axes[0, 1]), ("lstm", axes[1, 0]), ("transformer", axes[1, 1])]
    plotted = False
    for name, ax in families:
        df = df_all[(df_all["experiment"] == "capacity") & (df_all["model"] == name)].copy()
        idx_col = "tf_d_model" if name == "transformer" else "hidden_dim"
        if df.empty or idx_col not in df.columns:
            ax.set_title(f"{name.upper()} (missing)")
            ax.axis("off")
            continue
        df = df.sort_values(idx_col)
        df["avg_ic"] = (df["train_ic"] + df["test_ic"]) / 2.0
        ax.plot(df[idx_col], df["train_ic"], marker="o", label="train IC")
        ax.plot(df[idx_col], df["test_ic"], marker="o", label="test IC")
        ax.plot(df[idx_col], df["avg_ic"], marker="o", label="avg(train,test) IC")
        if "holdout_ic" in df.columns:
            ax.plot(df[idx_col], df["holdout_ic"], marker="o", label="holdout IC")
        ax.set_title(f"{name.upper()} capacity")
        ax.set_xlabel(idx_col)
        ax.grid(True, linestyle="--", alpha=0.4)
        plotted = True
    axes[0, 0].set_ylabel("Spearman IC")
    axes[1, 0].set_ylabel("Spearman IC")
    if plotted:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved → {out_path}")


def train(config: Dict, run_tag: str):
    args = [
        "python",
        str(SCRIPTS / "train.py"),
        "--model",
        config["model"],
        "--seq_len",
        str(config["seq_len"]),
        "--batch_size",
        str(BASE["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--epochs",
        str(BASE["epochs"]),
        "--weight_decay",
        str(BASE["weight_decay"]),
        "--scheduler",
        BASE["scheduler"],
        "--warmup_pct",
        str(BASE["warmup_pct"]),
        "--train_end_date",
        BASE["cutoff_date"],
    ]

    # Model-specific args
    if config["model"] in {"mlp", "lstm", "gru"}:
        args += [
            "--hidden_dim",
            str(config.get("hidden_dim", 256)),
            "--dropout",
            str(config.get("dropout", 0.1)),
            "--num_layers",
            str(config.get("num_layers", 2)),
        ]
        if config["model"] == "lstm":
            args += [
                "--lstm_proj_dim",
                str(config.get("lstm_proj_dim", 128)),
                "--lstm_use_layernorm",
                str(config.get("lstm_use_layernorm", 1)),
                "--lstm_use_attention",
                str(config.get("lstm_use_attention", 1)),
            ]
    if config["model"] == "transformer":
        args += [
            "--tf_d_model",
            str(config.get("tf_d_model", 256)),
            "--tf_heads",
            str(config.get("tf_heads", 4)),
            "--tf_layers",
            str(config.get("tf_layers", 2)),
            "--tf_ff_dim",
            str(config.get("tf_ff_dim", 512)),
            "--tf_pool",
            config.get("tf_pool", "attention"),
            "--tf_dropout",
            str(config.get("tf_dropout", 0.1)),
            "--tf_learnable_pos",
            str(int(config.get("tf_learnable_pos", 1))),
            "--tf_use_cls_token",
            str(int(config.get("tf_use_cls_token", 1))),
            "--tf_embed_scale",
            str(config.get("tf_embed_scale", 0.1)),
        ]

    run_cmd(args, PROJECT_ROOT)


def predict(config: Dict):
    args = [
        "python",
        str(SCRIPTS / "predict.py"),
        "--model",
        config["model"],
        "--seq_len",
        str(config["seq_len"]),
        "--batch_size",
        str(BASE["batch_size"]),
        "--cutoff_date",
        BASE["cutoff_date"],
        "--pretest_fraction",
        "0.8",
    ]
    run_cmd(args, PROJECT_ROOT)
    pred_file = OUTPUT / "predictions" / f"predictions_{config['model']}.csv"
    return pred_file


def evaluate(pred_path: Path, report_name: str):
    args = [
        "python",
        str(SCRIPTS / "evaluate.py"),
        "--predictions",
        str(pred_path),
        "--cutoff-date",
        BASE["cutoff_date"],
        "--pretest-fraction",
        str(BASE["pretest_fraction"]),
        "--signal-percentile",
        str(BASE["signal_percentile"]),
        "--include-holdout",
        "--report-name",
        report_name,
    ]
    run_cmd(args, PROJECT_ROOT)
    return OUTPUT / "reports" / report_name


def extract_metrics(report_path: Path) -> Dict:
    with report_path.open("r") as f:
        rep = json.load(f)
    mp = rep.get("model_predictions", {})
    out = {}
    for subset in ["train", "test", "holdout"]:
        # Evaluate reports store subset metrics under model_predictions["subset_metrics"]
        subset_block = None
        if "subset_metrics" in mp:
            subset_block = mp["subset_metrics"].get(subset)
        elif subset in mp:
            subset_block = mp[subset]
        if subset_block:
            reg = subset_block.get("regression_metrics", {})
            strat = subset_block.get("strategy_metrics", {})
            out[f"{subset}_ic"] = reg.get("spearman_ic")
            out[f"{subset}_sharpe"] = strat.get("sharpe")
            out[f"{subset}_mse"] = reg.get("mse")
            out[f"{subset}_rmse"] = reg.get("rmse")
            out[f"{subset}_mae"] = reg.get("mae")
            out[f"{subset}_r2"] = reg.get("r2")
    return out


def sweep(name: str, configs: List[Dict]) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        tag = cfg["tag"]
        run_dir = RUNS_DIR / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        # Rebuild features for the desired pooling/lookback/horizon
        build_features(cfg.get("lookback_mode", "mean"), BASE["lookback"], BASE["horizon"])

        train(cfg, tag)
        pred_path = predict(cfg)
        # Copy predictions to isolate runs
        local_pred = run_dir / "predictions.csv"
        shutil.copy(pred_path, local_pred)
        report_name = f"report_{tag}.json"
        report_path = evaluate(local_pred, report_name)
        shutil.copy(report_path, run_dir / report_name)

        metrics = extract_metrics(report_path)
        # Flatten metrics into the row
        flat_metrics = {f"{k}": v for k, v in metrics.items()}
        row = {"experiment": name, **cfg, **flat_metrics}
        rows.append(row)
        # Persist the config+metrics for traceability
        with (run_dir / "config.json").open("w") as f:
            json.dump(cfg, f, indent=2)
        with (run_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        # Also persist the full report for auditing
        shutil.copy(report_path, run_dir / "report.json")
    return pd.DataFrame(rows)


def plot_metric(df: pd.DataFrame, param: str, metric: str, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = df[param]
    ax.plot(x, df[metric], marker="o")
    ax.set_xlabel(param)
    ax.set_ylabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="One-at-a-time refinement sweeps.")
    parser.add_argument("--model", type=str, default="transformer", choices=["mlp", "lstm", "gru", "transformer"])
    parser.add_argument("--analyze-only", action="store_true", help="Skip training; aggregate cached run metrics and plot.")
    args = parser.parse_args()

    base_lr = 1e-4 if args.model != "transformer" else 1e-5
    base_cfg = {
        "model": args.model,
        "seq_len": 12 if args.model != "mlp" else 1,
        "lr": base_lr,
        "lookback_mode": "attn",
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
        "lstm_proj_dim": 128,
        "lstm_use_layernorm": 1,
        "lstm_use_attention": 1,
        "tf_d_model": 256,
        "tf_heads": 4,
        "tf_layers": 2,
        "tf_ff_dim": 512,
        "tf_pool": "attention",
        "tf_dropout": 0.1,
        "tf_learnable_pos": 1,
        "tf_use_cls_token": 1,
        "tf_embed_scale": 0.1,
    }

    # Experiment 1: seq_len sweep (transformer only; mlp fixed at 1)
    seq_vals = [6, 12, 24, 48, 120, 360, 720] if args.model == "transformer" else ([1] if args.model == "mlp" else [])
    seq_configs = []
    for s in seq_vals:
        cfg = base_cfg.copy()
        cfg["tag"] = f"seq_{args.model}_{s}"
        cfg["seq_len"] = s
        cfg["lookback_mode"] = base_cfg.get("lookback_mode", "attn")
        seq_configs.append(cfg)

    # Experiment 2: capacity sweep (run across all models)
    cap_configs = []
    if args.model == "transformer":
        for d in [64, 128, 256, 384, 512]:
            cfg = base_cfg.copy()
            cfg["tag"] = f"cap_tf_{d}"
            cfg["tf_d_model"] = d
            cfg["lookback_mode"] = base_cfg.get("lookback_mode", "attn")
            cap_configs.append(cfg)
    else:
        for h in [32, 64, 128, 256, 512]:
            cfg = base_cfg.copy()
            cfg["tag"] = f"cap_{args.model}_{h}"
            cfg["hidden_dim"] = h
            cfg["lookback_mode"] = base_cfg.get("lookback_mode", "attn")
            cap_configs.append(cfg)

    sweeps = []
    if seq_configs:
        sweeps.append(("seq_len", seq_configs, "seq_len"))
    sweeps.append(("capacity", cap_configs, "hidden_dim" if args.model != "transformer" else "tf_d_model"))

    def analyze_cached():
        rows = []
        for run_dir in RUNS_DIR.iterdir():
            if not run_dir.is_dir():
                continue
            cfg_path = run_dir / "config.json"
            # Report filename matches saved name: report_<tag>.json
            report_files = list(run_dir.glob("report_*.json"))
            if not cfg_path.exists() or not report_files:
                continue
            report_path = report_files[0]
            cfg = json.loads(cfg_path.read_text())
            metrics = extract_metrics(report_path)
            flat_metrics = {f"{k}": v for k, v in metrics.items()}
            row = {"tag": cfg.get("tag", run_dir.name), "model": cfg.get("model", args.model), **cfg, **flat_metrics}
            # Infer experiment type from tag prefix
            if row["tag"].startswith("pool_"):
                row["experiment"] = "pooling"
            elif row["tag"].startswith("seq_"):
                row["experiment"] = "seq_len"
            elif row["tag"].startswith("cap_"):
                row["experiment"] = "capacity"
            rows.append(row)
        if not rows:
            print("No cached runs found in refine_runs/.")
            return
        df_all = pd.DataFrame(rows)

        for exp_name, grp in df_all.groupby("experiment"):
            csv_path = RUNS_DIR / f"{exp_name}_{args.model}_cached.csv"
            grp.to_csv(csv_path, index=False)
            print(f"Saved cached {exp_name} metrics → {csv_path}")

        # Plots from cached runs
        plot_seq_ic(df_all, RUNS_DIR / "plot_seq_len_ic_transformer_cached.png")
        plot_capacity_ic_grid(df_all, RUNS_DIR / "plot_capacity_ic_grid_cached.png")

        # Best per model family from cached runs
        if "test_ic" in df_all.columns:
            df_all["test_ic"] = pd.to_numeric(df_all["test_ic"])
            best = df_all.loc[df_all["test_ic"].idxmax()].copy()
            summary_cols = [
                "experiment",
                "tag",
                "model",
                "seq_len",
                "lookback_mode",
                "hidden_dim",
                "tf_d_model",
                "test_ic",
                "test_sharpe",
                "holdout_ic",
                "holdout_sharpe",
            ]
            summary = pd.DataFrame([best.get(c) for c in summary_cols]).T
            summary.columns = summary_cols
            summary_path = RUNS_DIR / f"best_{args.model}_cached.csv"
            summary.to_csv(summary_path, index=False)
            print(f"Best cached test IC for {args.model} → {summary_path}")

    if args.analyze_only:
        analyze_cached()
        return

    all_results = []

    for name, cfgs, param in sweeps:
        df = sweep(name, cfgs)
        csv_path = RUNS_DIR / f"{name}_{args.model}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {name} results → {csv_path}")

        # # Plot test/holdout IC & Sharpe if present
        # if "test_ic" in df.columns and "holdout_ic" in df.columns:
        #     plot_metric(df, param, "test_ic", f"{name} test IC", RUNS_DIR / f"{name}_{args.model}_test_ic.png")
        #     plot_metric(df, param, "holdout_ic", f"{name} holdout IC", RUNS_DIR / f"{name}_{args.model}_holdout_ic.png")
        # if "test_sharpe" in df.columns and "holdout_sharpe" in df.columns:
        #     plot_metric(df, param, "test_sharpe", f"{name} test Sharpe", RUNS_DIR / f"{name}_{args.model}_test_sharpe.png")
        #     plot_metric(df, param, "holdout_sharpe", f"{name} holdout Sharpe", RUNS_DIR / f"{name}_{args.model}_holdout_sharpe.png")

        all_results.append(df)

    # # Pick best test IC per model family (within runs just executed)
    # full = pd.concat(all_results, ignore_index=True)
    # full["test_ic"] = pd.to_numeric(full.get("test_ic"))
    # best = full.loc[full["test_ic"].idxmax()].copy()
    # summary_cols = [
    #     "experiment",
    #     "tag",
    #     "model",
    #     "seq_len",
    #     "lookback_mode",
    #     "hidden_dim",
    #     "tf_d_model",
    #     "test_ic",
    #     "test_sharpe",
    #     "holdout_ic",
    #     "holdout_sharpe",
    # ]
    # summary = best[summary_cols]
    # summary_df = pd.DataFrame([summary])
    # summary_path = RUNS_DIR / f"best_{args.model}.csv"
    # summary_df.to_csv(summary_path, index=False)
    # print(f"Best test IC for {args.model} → {summary_path}")


def custom_plot(metric='rmse', model=None, exp='seq_len'):
    import pandas as pd
    RUNS_DIR = '/Users/niniliu/Documents/GitHub/OMSCS_7643_F2025_Project/output/refine_runs/'
    df = pd.read_csv(RUNS_DIR + f"{exp}_transformer_cached.csv")
    if model is not None:
        df = df[df['model'] == model]
        index_col = 'hidden_dim' if model != 'transformer' else 'tf_d_model'
    else:
        index_col = exp
    plot_df = df[[index_col, f"train_{metric}", f"test_{metric}", f"holdout_{metric}"]].set_index(index_col).sort_index()
    plot_df.plot(title=f"{exp} training curve" + f" ({model})" if model else '')


if __name__ == "__main__":
    main()
