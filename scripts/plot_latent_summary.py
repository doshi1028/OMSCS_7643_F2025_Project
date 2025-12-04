"""Plot PCA variance explained and transformer PC importance from latent_summary_*.json."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ANALYSIS_DIR = Path("output/analysis")


def load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {path}")
    with path.open("r") as f:
        return json.load(f)


def plot_variance(symbol: str, variance: list, out_dir: Path) -> Path:
    pcs = range(1, len(variance) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pcs, variance, color="#1f77b4")
    ax.set_title(f"{symbol} PCA Variance Explained")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Ratio")
    ax.set_xticks(pcs)
    fig.tight_layout()
    out_path = out_dir / f"{symbol}_pca_variance.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_importance(symbol: str, importance: dict, out_dir: Path) -> Path:
    if not importance:
        raise ValueError("No transformer importance available in summary.")
    pcs, vals = zip(*importance.items())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pcs, vals, color="#ff7f0e")
    ax.set_title(f"{symbol} Transformer PC Importance")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Gradient Magnitude (avg)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out_path = out_dir / f"{symbol}_pc_importance.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot latent summaries (variance and importance).")
    parser.add_argument("--symbol", type=str, default="BTC")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Path to latent_summary_<symbol>.json (default: output/analysis/latent_summary_<SYMBOL>.json)",
    )
    args = parser.parse_args()

    summary_path = args.summary_path
    if summary_path is None:
        summary_path = ANALYSIS_DIR / f"latent_summary_{args.symbol}.json"

    summary = load_summary(summary_path)
    out_dir = summary_path.parent

    var_path = plot_variance(args.symbol, summary.get("pca_variance_ratio", []), out_dir)
    print(f"Saved PCA variance plot → {var_path}")

    tf_result = summary.get("transformer_analysis", {})
    importance = tf_result.get("importance", {})
    if importance:
        imp_path = plot_importance(args.symbol, importance, out_dir)
        print(f"Saved transformer importance plot → {imp_path}")
    else:
        print("No transformer importance found; skipped importance plot.")


if __name__ == "__main__":
    main()
