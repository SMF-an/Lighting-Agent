from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = Path("../../report/imgs/result.png")
LABELS = ["Pretrain", "Lora", "Full"]
COLORS = ["#F3D46B", "#9BC26B", "#E58A8A"]

METRICS = [
    {
        "name": "Clip Score",
        "values": [0.276803778950125, 0.28453736437950283, 0.2890748723875731],
        "std": [0.02297477503777884, 0.017350406522212272, 0.018156464696540118],
    },
    {
        "name": "Scene Leak Score",
        "values": [0.23057443741708994, 0.21978320449125022, 0.22596518660429865],
        "std": [0.023668606611158112, 0.008676662679075613, 0.010243669229100612],
    },
    {
        "name": "Prompt Consistency",
        "values": [0.766427468508482, 0.9199289493262768, 0.9171055108308792],
        "std": [0.06740802482769102, 0.014147850104084012, 0.016271438482112568],
    },
    {
        "name": "Hue Js",
        "values": [0.33145685071027875, 0.3323309109910549, 0.40167805731231243],
        "std": [0.13548838211867614, 0.09156953947221787, 0.1105740363494209],
    },
    {
        "name": "Kid",
        "values": [0.21635655641555787, 0.09507142305374146, 0.0722844386100769],
        "std": [0.00437228551189838, 0.0036772419375255493, 0.003093132795103475],
    },
    {
        "name": "High Freq",
        "display_name": "High Freq (×10^-3)",
        "display_scale": 1000.0,
        "values": [0.004365411208077319, 9.673712828561776e-07, 7.939266147616308e-06],
        "std": [0.010205872818825904, 1.1944151292776638e-06, 7.2096590869283394e-06],
    },
]


def build_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 4.4), dpi=200)
    fig.patch.set_facecolor("white")
    axes = np.asarray(axes).ravel()

    x = np.arange(len(LABELS))
    width = 0.72
    legend_handles = None

    for ax, metric in zip(axes, METRICS):
        ax.set_facecolor("white")
        scale = float(metric.get("display_scale", 1.0))
        values = [value * scale for value in metric["values"]]
        std = [value * scale for value in metric["std"]]
        metric_name = metric.get("display_name", metric["name"])

        bars = ax.bar(
            x,
            values,
            width=width,
            color=COLORS,
            edgecolor="#2d2d2d",
            linewidth=0.9,
            zorder=3,
        )

        if legend_handles is None:
            legend_handles = bars

        shadow_dx = 0.06
        shadow_dy = -0.01
        for bar in bars:
            xi = bar.get_x()
            w = bar.get_width()
            h = bar.get_height()
            shadow = patches.Rectangle(
                (xi + shadow_dx, shadow_dy),
                w,
                max(0.0, h - shadow_dy),
                color="black",
                alpha=0.16,
                zorder=1,
                linewidth=0,
            )
            ax.add_patch(shadow)

        upper = max(v + s for v, s in zip(values, std))
        y_max = upper * 1.18 if upper > 0 else 1.0
        y_min = min(0.0, min(v - s for v, s in zip(values, std)) * 1.15)
        ax.set_ylim(y_min, y_max)

        ax.set_xticks(x)
        ax.set_xticklabels([""] * len(LABELS))
        ax.set_xlabel(metric_name, fontsize=12, labelpad=10)

        ax.grid(axis="y", color="#e5e5e5", linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)

        ax.errorbar(
            x,
            values,
            yerr=std,
            fmt="none",
            ecolor="#2d2d2d",
            elinewidth=1.6,
            capsize=6,
            capthick=1.4,
            zorder=4,
        )

    for ax in axes[len(METRICS):]:
        ax.axis("off")

    fig.legend(
        legend_handles,
        LABELS,
        frameon=True,
        fancybox=True,
        edgecolor="#2d2d2d",
        loc="upper left",
        bbox_to_anchor=(0.06, 0.99),
        ncol=3,
        fontsize=13,
        handlelength=1.9,
        columnspacing=1.8,
        borderpad=0.35,
    )

    plt.tight_layout(rect=[0.01, 0.0, 1.0, 0.92])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_plot()
    print(f"Saved figure to {OUTPUT_PATH}")