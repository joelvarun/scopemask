"""Render docs/before_after.png from docs/results.json.

Three panels, one measure each, no second y axis anywhere:

left    tokens generated per valid program, baseline versus scopemask, at three
        generator strengths
middle  valid-program rate against program length, which is the reason rejection
        sampling loses ground as programs get longer
right   mask construction cost per decoded token, counted in automaton character
        steps so the comparison does not depend on the host language
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedddb"
NEWLINE = chr(10)

BASELINE = "#2a78d6"
AFTER = "#eb6834"

BASELINE_LABEL = "baseline: grammar plus lexicon, reject and resample"
AFTER_LABEL = "scopemask: scope and types in the mask, one pass"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    ax.tick_params(which="minor", length=0)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)


def panel_tokens(ax, levels):
    labels = []
    before = []
    after = []
    for level in levels:
        rate = 100.0 * level["baseline"]["valid_rate"]
        labels.append(level["label"] + NEWLINE + "(" + format(rate, ".0f")
                      + " percent valid)")
        before.append(level["baseline"]["tokens_per_valid"])
        after.append(level["scopemask"]["tokens_per_valid"])

    x = range(len(labels))
    width = 0.38
    gap = 0.012
    ax.bar([i - width / 2 - gap for i in x], before, width, color=BASELINE,
           label=BASELINE_LABEL, zorder=3)
    ax.bar([i + width / 2 + gap for i in x], after, width, color=AFTER,
           label=AFTER_LABEL, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(10, max(before) * 4)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - width / 2 - gap, b * 1.09, format(b, ".0f"), ha="center",
                va="bottom", fontsize=9, color=INK)
        ax.text(i + width / 2 + gap, a * 1.09, format(a, ".0f"), ha="center",
                va="bottom", fontsize=9, color=INK)
        ax.text(i, max(b, a) * 2.0, format(b / a, ".1f") + "x fewer",
                ha="center", va="bottom", fontsize=9, color=INK_MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("tokens generated per valid program (log scale)")
    ax.set_xlabel("generator strength")
    ax.set_title("Cost of one valid program", fontsize=11, loc="left", pad=26)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: format(v, "g")))
    ax.yaxis.set_minor_formatter(NullFormatter())
    _style(ax)


def panel_compounding(ax, level):
    """Per-position accuracy versus per-program validity, same sampled programs."""
    baseline = level["baseline"]
    after = level["scopemask"]
    measures = ["identifier use positions" + NEWLINE + "that resolve",
                "whole programs" + NEWLINE + "that resolve"]
    before = [100.0 * (1.0 - baseline["use_error_rate"]),
              100.0 * baseline["valid_rate"]]
    tight = [100.0 * (1.0 - after["use_error_rate"]),
             100.0 * after["valid_rate"]]

    x = range(len(measures))
    width = 0.38
    gap = 0.012
    ax.bar([i - width / 2 - gap for i in x], before, width, color=BASELINE,
           label=BASELINE_LABEL, zorder=3)
    ax.bar([i + width / 2 + gap for i in x], tight, width, color=AFTER,
           label=AFTER_LABEL, zorder=3)
    for i, (b, a) in enumerate(zip(before, tight)):
        ax.text(i - width / 2 - gap, b + 2.5, format(b, ".1f"), ha="center",
                va="bottom", fontsize=9, color=INK)
        ax.text(i + width / 2 + gap, a + 2.5, format(a, ".1f"), ha="center",
                va="bottom", fontsize=9, color=INK)
        ax.text(i, max(b, a) + 14, format(a - b, ".0f") + " point gap",
                ha="center", va="bottom", fontsize=9, color=INK_MUTED)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(list(x))
    ax.set_xticklabels(measures, fontsize=9)
    ax.set_ylabel("percent, " + level["label"] + " generator")
    ax.set_title("A small per-position gap compounds", fontsize=11, loc="left",
                 pad=26)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    _style(ax)


def panel_masks(ax, timing):
    names = ["naive per-token" + NEWLINE + "automaton replay",
             "scopemask" + NEWLINE + "factorized masks"]
    values = [timing["naive_automaton_steps"], timing["fast_automaton_steps"]]
    notes = [format(timing["naive_us"] / 1000.0, ".1f") + " ms per token",
             format(timing["fast_us"], ".0f") + " us per token"]
    ax.barh([1, 0], values, 0.44, color=[BASELINE, AFTER], zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1, max(values) * 12)
    for y, value, note in zip((1, 0), values, notes):
        ax.text(value * 1.25, y, format(value, ".0f") + " steps", va="center",
                ha="left", fontsize=9, color=INK)
        ax.text(value * 1.25, y - 0.22, note, va="center", ha="left",
                fontsize=8, color=INK_MUTED)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("automaton character steps per mask (log scale)")
    ax.set_title("Cost of building one mask, " + format(
        timing["step_reduction"], ".0f") + "x fewer steps",
        fontsize=11, loc="left", pad=26)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: format(v, "g")))
    ax.xaxis.set_minor_formatter(NullFormatter())
    _style(ax)


def build(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))
    fig.patch.set_facecolor(SURFACE)
    levels = results["levels"]
    panel_tokens(axes[0], levels)
    panel_compounding(axes[1], levels[1])
    panel_masks(axes[2], results["mask_timing"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, labelcolor=INK_MUTED, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "scopemask: folding scope and coarse types into the decoder token mask",
        fontsize=13, color=INK, x=0.008, ha="left", y=0.975)
    fig.text(0.008, 0.915,
             "MiniPy, vocabulary " + str(results["composition"]["vocab_size"])
             + " entries, static namespace " + str(results["static_functions"])
             + " functions, order-" + str(results["lm_order"])
             + " n-gram generator; each baseline bar collects "
             + str(levels[0]["baseline"]["valid"])
             + " valid programs, each scopemask bar samples "
             + str(levels[0]["scopemask"]["attempts"]),
             fontsize=9.5, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.885))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    return out_path


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "docs", "results.json")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "docs", "before_after.png")
    with open(src, encoding="utf-8") as handle:
        results = json.load(handle)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("wrote " + build(results, out))


if __name__ == "__main__":
    main()
