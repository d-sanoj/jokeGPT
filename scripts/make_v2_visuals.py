#!/usr/bin/env python3
"""Charts for the v2 upgrade sequence, from experiments/runs-v2.json.

One chart per step: left panel is the good%/complete% progress across
v1 and every v2 step so far; right panel is that step's loss curve.
Palette: the sanoj.dev site palette (dark background, yellow + blue
accents, Josefin Sans). Every chart is saved as a PNG under visuals/
with the raw numbers as a JSON file next to it.
Usage: PYTHONPATH=. python scripts/make_v2_visuals.py [run_id]
With no run_id, charts the latest run in runs-v2.json.
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

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
    ax.set_title(title, color=INK, fontsize=14, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def save_json(path: Path, payload) -> None:
    """Write the raw numbers behind a chart."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_steps():
    """v1 baseline + every judged v2 run, in order."""
    steps = []
    v1_path = HERE / "experiments" / "runs.json"
    if v1_path.exists():
        runs = json.loads(v1_path.read_text())["runs"]
        main = next(r for r in runs if r["run_id"] == "main")
        steps.append(
            {
                "label": "v1\n(char)",
                "complete_pct": main["quality"]["complete_pct"],
                # v1: 0 of 50 samples was a joke (FINDINGS.md).
                "good_pct": 0.0,
            }
        )
    v2_path = HERE / "experiments" / "runs-v2.json"
    if v2_path.exists():
        for run in json.loads(v2_path.read_text())["runs"]:
            if run.get("good_pct") is None:
                continue
            steps.append(
                {
                    "label": run["run_id"].replace("-", "\n"),
                    "complete_pct": run["quality"]["complete_pct"],
                    "good_pct": run["good_pct"],
                }
            )
    return steps


def step_chart(run_id: str) -> None:
    """Progress bars + loss curve for one v2 run."""
    v2 = json.loads((HERE / "experiments" / "runs-v2.json").read_text())
    run = next(r for r in v2["runs"] if r["run_id"] == run_id)
    steps = load_steps()
    labels = [s["label"] for s in steps]
    complete = [s["complete_pct"] for s in steps]
    good = [s["good_pct"] for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(labels))
    width = 0.36
    bars = ax1.bar(
        [i - width / 2 for i in x],
        complete,
        width,
        color=YELLOW,
        label="complete (%)",
    )
    for bar, value in zip(bars, complete):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            color=INK,
            fontsize=11,
        )
    bars = ax1.bar(
        [i + width / 2 for i in x],
        good,
        width,
        color=BLUE,
        label="good (%)",
    )
    for bar, value in zip(bars, good):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            color=INK,
            fontsize=11,
        )
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("% of 50 samples")
    ax1.set_ylim(0, 100)
    ax1.legend(frameon=False, loc="upper left")
    style_ax(ax1, "Good jokes: still 0%")

    history = run["loss_history"]
    hist_steps = [h["step"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]
    ax2.plot(hist_steps, train, color=BLUE, linewidth=2, label="train loss")
    ax2.plot(hist_steps, val, color=YELLOW, linewidth=2, label="val loss")
    for step in sorted({500, 1000, run["steps"]}):
        if step <= run["steps"]:
            ax2.axvline(step, color=MUTED, linewidth=0.8, linestyle="--")
    ax2.set_xlabel("training steps")
    ax2.set_ylabel("loss (lower is better)")
    ax2.legend(frameon=False, loc="upper right")
    style_ax(ax2, f"Step {run.get('step', '?')}: loss curve")

    out = HERE / "visuals" / f"v2-{run_id}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    save_json(
        out.with_suffix(".json"),
        {
            "steps": labels,
            "complete_pct": complete,
            "good_pct": good,
            "loss_steps": hist_steps,
            "train_loss": train,
            "val_loss": val,
        },
    )
    print(f"wrote visuals/v2-{run_id}.png")


def main():
    """Chart one run (default: the latest in runs-v2.json)."""
    v2_path = HERE / "experiments" / "runs-v2.json"
    if not v2_path.exists():
        print("no experiments/runs-v2.json yet — run a v2 step first")
        return
    runs = json.loads(v2_path.read_text())["runs"]
    run_id = sys.argv[1] if len(sys.argv) > 1 else runs[-1]["run_id"]
    if not any(r["run_id"] == run_id for r in runs):
        print(f"no v2 run {run_id!r} in runs-v2.json")
        return
    (HERE / "visuals").mkdir(exist_ok=True)
    step_chart(run_id)


if __name__ == "__main__":
    main()
