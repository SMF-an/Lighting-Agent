from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np


COLORS = ["#9BC26B", "#F4D35E"]

# OUTPUT_PATH = Path("../../report/imgs/mean_saturation.png")
# Y_LABEL = "Mean Saturation"
# LABELS = ["w/o color loss", "Ours"]
# VALUES = [0.3642321659780229, 0.46262124630331136]
# STD = [0.15630304775941098, 0.1649089973790532]

OUTPUT_PATH = Path("../../report/imgs/high_freq.png")
Y_LABEL = "High Freq (×10^-3)"
LABELS = ["w/o high-freq loss", "Ours"]
# Original values are very small; display scaled by 1e3 (×10^-3)
SCALE = 1000.0
VALUES = [1.3304126293967045e-05, 1.5009617200689718e-05]
STD = [1.3636983630066964e-05, 2.3489364424292238e-05]


def build_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    fig, ax = plt.subplots(figsize=(3.15, 3.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(VALUES))
    width = 0.72

    # Apply scaling for display so values are readable
    scaled_values = [v * SCALE for v in VALUES]
    scaled_std = [s * SCALE for s in STD]

    bars = ax.bar(
        x,
        scaled_values,
        width=width,
        color=COLORS,
        edgecolor="#2d2d2d",
        linewidth=0.9,
        zorder=3,
    )

    # Draw soft rectangular shadows behind each bar (slightly offset)
    shadow_dx = 0.06  # horizontal offset in data units
    max_std = max(scaled_std) if scaled_std else 0.0
    # Compute y limits based on scaled values and std. Keep lower bound near zero to avoid "stuck-to-ground" look
    y_min = min(0.0, min(scaled_values) - max_std * 0.6)
    # If lower bound is too negative relative to scale, clamp it to a small negative margin
    if y_min < -0.1 * max(1.0, max(scaled_values)):
        y_min = -0.1 * max(1.0, max(scaled_values))
    y_max = max(scaled_values) + max_std * 1.2

    for bar, h in zip(bars, scaled_values):
        xi = bar.get_x()
        w = bar.get_width()
        shadow_dy = y_min * 0.5
        shadow = patches.Rectangle(
            (xi + shadow_dx, shadow_dy),
            w,
            max(0.0, h - shadow_dy),
            color="black",
            alpha=0.12,
            zorder=1,
            linewidth=0,
        )
        ax.add_patch(shadow)

    ax.set_ylabel(Y_LABEL, fontsize=12, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(LABELS))
    ax.set_xlabel("")

    # Use scaled y-limits and sensible tick formatting (no percent labels)
    ax.set_ylim(y_min, y_max)
    # Use MaxNLocator to pick a reasonable number of tick locations and avoid overlap
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    for label in ax.get_yticklabels():
        label.set_fontsize(10)

    ax.grid(axis="y", color="#e5e5e5", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    # Place a single legend for the figure above the axes to avoid overlap
    handles = bars
    fig.legend(
        handles,
        LABELS,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        fontsize=11,
        handlelength=1.5,
        columnspacing=1.8,
    )

    # Add vertical error bars (std) displayed as vertical lines with caps using scaled values
    x_centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    ax.errorbar(
        x_centers,
        scaled_values,
        yerr=scaled_std,
        fmt="none",
        ecolor="#2d2d2d",
        elinewidth=1.6,
        capsize=6,
        capthick=1.4,
        zorder=4,
    )

    # Reserve space at the top for the external legend and avoid label overlap
    fig.subplots_adjust(top=0.86)
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_plot()
    print(f"Saved figure to {OUTPUT_PATH}")
