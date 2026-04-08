"""
Reward visualization utilities for training runs.

Generates reward curves from CSV logs with trend lines and statistics.
"""

import argparse
import csv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_rewards(csv_path: Path, out_path: Path = None, title: str = "Training Reward Curve"):
    """
    Plot reward curves from a CSV log.

    CSV format: episode, reward, [optional extra columns]

    Args:
        csv_path: Path to reward log CSV
        out_path: Output path for plot (defaults to csv_path with .png extension)
        title: Plot title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    episodes, rewards = [], []
    extra_cols = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # Detect column names
        ep_col = "episode" if "episode" in headers else headers[0]
        reward_col = next((h for h in headers if "reward" in h.lower() and "total" in h.lower()),
                         headers[1] if len(headers) > 1 else "reward")

        for row in reader:
            episodes.append(int(row[ep_col]))
            rewards.append(float(row[reward_col]))

            # Track additional metrics
            for col in headers:
                if col not in (ep_col, reward_col):
                    if col not in extra_cols:
                        extra_cols[col] = []
                    try:
                        extra_cols[col].append(float(row[col]))
                    except (ValueError, KeyError):
                        pass

    if not episodes:
        logger.warning("No episodes to plot")
        return

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    # Rolling average
    window = min(10, len(episodes))
    def rolling_avg(vals):
        return [sum(vals[max(0,i-window):i+1]) / min(i+1, window) for i in range(len(vals))]

    rolling = rolling_avg(rewards)

    # Main reward plot
    axes.plot(episodes, rewards, alpha=0.25, color="blue", marker="o", markersize=3, label="Per episode")
    axes.plot(episodes, rolling, color="blue", linewidth=2.5, label=f"Rolling avg ({window})")

    # Trend line
    z = np.polyfit(episodes, rewards, 1)
    trend = np.poly1d(z)
    axes.plot(episodes, trend(episodes), color="red", linewidth=1.5, linestyle="--",
             label=f"Trend ({'↑' if z[0] > 0 else '↓'} {abs(z[0]):.3f}/ep)")

    axes.set_ylabel("Reward")
    axes.set_xlabel("Episode")
    axes.set_title(title)
    axes.legend()
    axes.grid(True, alpha=0.3)
    axes.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Annotate stats
    axes.text(0.02, 0.02,
             f"Episodes: {len(episodes)} | Final avg: {rolling[-1]:.2f} | Best: {max(rewards):.2f}",
             transform=axes.transAxes, fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    save_path = out_path or csv_path.with_suffix(".png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Reward plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training rewards from CSV")
    parser.add_argument("csv_path", type=str, help="Path to reward log CSV")
    parser.add_argument("--output", "-o", type=str, help="Output plot path")
    parser.add_argument("--title", "-t", type=str, default="Training Reward Curve", help="Plot title")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_path = Path(args.output) if args.output else None

    logging.basicConfig(level=logging.INFO)
    plot_rewards(csv_path, out_path, args.title)


if __name__ == "__main__":
    main()
