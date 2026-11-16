#!/usr/bin/env python3
"""Generate the charts for the blog post from experiments/runs.json
and experiments/temperature.json.

Palette: the sanoj.dev site palette (dark background, yellow + blue
accents, Josefin Sans). Every chart is saved as a PNG under visuals/
with the raw numbers as a JSON file next to it.
Usage: PYTHONPATH=. python scripts/make_visuals.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent.parent
FONT = HERE.parent / "blog" / "kit" / "JosefinSans-VF.ttf"
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = "Josefin Sans"

BG = (19 / 255, 19 / 255, 19 / 255)
INK = (235 / 255, 232 / 255, 225 / 255)
YELLOW = (242 / 255, 193 / 255, 78 / 255)
BLUE = (201 / 255, 220 / 255, 238 / 255)
MUTED = (151 / 255, 148 / 255, 141 / 255)
GRID = (60 / 255, 60 / 255, 60 / 255)

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "savefig.facecolor": BG,
    }
)


def style_ax(ax, title):
    """Shared axes styling: title, light spines, no top/right."""
    ax.set_title(title, color=INK, fontsize=15, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def save_json(path: Path, payload) -> None:
    """Write the raw numbers behind a chart."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def loss_chart(runs_dir: Path, out: Path) -> None:
    """Train/val loss curves for the main run, with checkpoint marks."""
    runs = json.loads((runs_dir / "runs.json").read_text())["runs"]
    main = next(r for r in runs if r["run_id"] == "main")
    history = main["loss_history"]
    steps = [h["step"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, train, color=BLUE, linewidth=2, label="train loss")
    ax.plot(steps, val, color=YELLOW, linewidth=2, label="val loss")
    for step in (500, 1000, 1500):
        ax.axvline(step, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("training steps")
    ax.set_ylabel("loss (lower is better)")
    ax.legend(frameon=False, loc="upper right")
    style_ax(ax, "Training the tiny GPT: loss falls as it reads jokes")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    save_json(
        out.with_suffix(".json"),
        {
            "steps": steps,
            "train_loss": train,
            "val_loss": val,
            "checkpoints": [500, 1000, 1500],
        },
    )


def ablations_chart(runs_dir: Path, out: Path) -> None:
    """Val loss + complete-joke rate across the three 500-step runs."""
    runs = json.loads((runs_dir / "runs.json").read_text())["runs"]
    by_id = {r["run_id"]: r for r in runs}
    labels = [
        "full data\n(15k jokes)",
        "1k jokes",
        "small model\n(64-dim, 2 layers)",
    ]
    keys = ["main-early", "data-1k", "small"]
    val_loss = [by_id[k]["val_loss"] for k in keys]
    complete = [by_id[k]["quality"]["complete_pct"] for k in keys]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    bars = ax1.bar(labels, val_loss, color=[YELLOW, BLUE, MUTED], width=0.6)
    for bar, value in zip(bars, val_loss):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}",
            ha="center",
            color=INK,
            fontsize=11,
        )
    ax1.set_ylabel("val loss (lower is better)")
    style_ax(ax1, "Less data / smaller model = worse loss")
    bars = ax2.bar(labels, complete, color=[YELLOW, BLUE, MUTED], width=0.6)
    for bar, value in zip(bars, complete):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.0f}%",
            ha="center",
            color=INK,
            fontsize=11,
        )
    ax2.set_ylabel("% of samples that are complete jokes")
    ax2.set_ylim(0, 100)
    style_ax(ax2, "…but almost none finish a joke")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    save_json(
        out.with_suffix(".json"),
        {
            "labels": labels,
            "val_loss": val_loss,
            "complete_pct": complete,
        },
    )


def temperature_chart(runs_dir: Path, out: Path) -> None:
    """Quality signals vs sampling temperature."""
    sweep = json.loads((runs_dir / "temperature.json").read_text())
    temps = [row["temperature"] for row in sweep["rows"]]
    complete = [row["quality"]["complete_pct"] for row in sweep["rows"]]
    duplicate = [row["quality"]["duplicate_pct"] for row in sweep["rows"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        temps,
        complete,
        color=YELLOW,
        linewidth=2,
        marker="o",
        label="complete jokes (%)",
    )
    ax.plot(
        temps,
        duplicate,
        color=BLUE,
        linewidth=2,
        marker="o",
        label="duplicates (%)",
    )
    for temp, value in zip(temps, complete):
        ax.text(temp, value + 1.5, f"{value:.0f}%", ha="center", color=INK)
    for temp, value in zip(temps, duplicate):
        ax.text(temp, 1.2, f"{value:.0f}%", ha="center", color=INK)
    ax.set_xlabel("sampling temperature")
    ax.set_ylabel("percent of 50 samples")
    ax.set_xticks(temps)
    ax.set_ylim(0, 30)
    ax.legend(frameon=False, loc="upper left")
    style_ax(ax, "Temperature: no setting rescues the jokes")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    save_json(
        out.with_suffix(".json"),
        {
            "temperature": temps,
            "complete_pct": complete,
            "duplicate_pct": duplicate,
        },
    )


def architecture_chart(out: Path, params: int) -> None:
    """A clean diagram of the tiny transformer."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def box(x, y, w, h, text, sub, color=YELLOW):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor=BG,
            edgecolor=color,
            linewidth=1.6,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2 + 0.18,
            text,
            ha="center",
            va="center",
            color=INK,
            fontsize=13,
        )
        ax.text(
            x + w / 2,
            y + h / 2 - 0.32,
            sub,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )

    def arrow(x0, y0, x1, y1):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                color=MUTED,
                linewidth=1.4,
            )
        )

    box(0.4, 4.2, 2.4, 0.8, "Input", "joke characters", BLUE)
    arrow(2.8, 4.6, 3.4, 4.6)
    box(3.4, 4.2, 2.6, 0.8, "Embedding", "each char = 128 numbers")
    arrow(6.0, 4.6, 6.6, 4.6)
    box(6.6, 4.2, 3.0, 0.8, "Position", "+ where it sits", BLUE)
    ax.plot([8.1, 8.1], [4.2, 2.8], color=MUTED, linewidth=1.4)
    arrow(8.1, 2.8, 7.4, 2.8)
    box(
        2.6,
        2.4,
        4.8,
        0.8,
        "Transformer block × 4",
        "attention + feed-forward",
        YELLOW,
    )
    arrow(5.0, 2.4, 5.0, 1.7)
    box(2.6, 0.9, 4.8, 0.8, "Output head", "next character", BLUE)
    ax.text(
        5.0,
        0.35,
        f"{params:,} parameters · reads 128 characters "
        "at a time · predicts the next one",
        ha="center",
        color=MUTED,
        fontsize=11,
    )
    ax.set_title(
        "jokeGPT: a tiny transformer, drawn to scale of ideas",
        color=INK,
        fontsize=15,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    save_json(out.with_suffix(".json"), {"params": params})


def main():
    """Build all four charts (skips gracefully if runs are missing)."""
    runs_dir = HERE / "experiments"
    visuals = HERE / "visuals"
    visuals.mkdir(exist_ok=True)
    if not (runs_dir / "runs.json").exists():
        print("no experiments/runs.json yet — run the suite first")
        return
    runs = json.loads((runs_dir / "runs.json").read_text())["runs"]
    main_run = next(r for r in runs if r["run_id"] == "main")
    loss_chart(runs_dir, visuals / "loss.png")
    print("wrote visuals/loss.png")
    ablations_chart(runs_dir, visuals / "ablations.png")
    print("wrote visuals/ablations.png")
    if (runs_dir / "temperature.json").exists():
        temperature_chart(runs_dir, visuals / "temperature.png")
        print("wrote visuals/temperature.png")
    architecture_chart(visuals / "architecture.png", main_run["params"])
    print("wrote visuals/architecture.png")


if __name__ == "__main__":
    main()
