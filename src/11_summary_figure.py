"""Isochrone age summary figures.

Reads out/isochrone_summary.csv and writes two PNGs:
  out/isochrone_summary.png        wide version (report)
  out/isochrone_summary_slide.png  compact version with larger text (slides)
Bar = plausible range (sensitivity envelope), dot = baseline age,
whiskers = bootstrap 16-84 percentile, red x = extreme A_V case,
diamond = Praesepe at [Fe/H]=+0.15. Unconstrained clusters are hatched.
Paths are relative to this file, so it runs from any directory.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

df = pd.read_csv(OUT / "isochrone_summary.csv")
order = ["pleiades", "praesepe", "m67"]  # youngest at the top
names = {"pleiades": "Pleiades", "praesepe": "Praesepe", "m67": "M67"}
df = df.set_index("cluster").loc[order]


def make(path, figsize, fs, legend_below, dpi=200):
    plt.rcParams.update({"font.size": fs})
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_xlim(50, 16000)
    for i, (c, r) in enumerate(df.iterrows()):
        constrained = bool(r["age_constrained"])
        ax.barh(
            i, r["age_max_myr"] - r["age_min_myr"], left=r["age_min_myr"], height=0.5,
            color="#4c78a8" if constrained else "#d9d9d9",
            edgecolor="#4c78a8" if constrained else "#888888",
            hatch=None if constrained else "///", alpha=0.55 if constrained else 1.0,
            linewidth=1.0, zorder=1,
        )
        ax.errorbar(
            r["best_age_myr"], i,
            xerr=[[r["best_age_myr"] - r["boot_p16_myr"]], [r["boot_p84_myr"] - r["best_age_myr"]]],
            fmt="o", color="black", markerfacecolor="black" if constrained else "white",
            capsize=3, zorder=3,
        )
        ax.plot(r["age_extreme_av_myr"], i, "x", color="#d62728", markersize=8, mew=2, zorder=4)
        if pd.notna(r.get("age_alt_feh015_myr")):
            ax.plot(r["age_alt_feh015_myr"], i + 0.32, "D", color="#2ca02c", markersize=6, zorder=4)
        label = f"{r['best_age_myr']:.0f} Myr"
        if not constrained:
            label += "\n(not constrained)"
        ax.text(r["age_max_myr"] * 1.15, i, label, va="center", fontsize=fs - 1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[c] for c in order])
    ax.invert_yaxis()
    ax.set_xlabel("Age (Myr)")
    ax.grid(axis="x", which="both", alpha=0.25)
    handles = [
        plt.Line2D([], [], color="#4c78a8", lw=8, alpha=0.55, label="plausible range"),
        plt.Line2D([], [], marker="o", color="black", lw=1, label="baseline, bootstrap 16-84%"),
        plt.Line2D([], [], marker="x", color="#d62728", lw=0, mew=2, label="extreme A_V = 0.3"),
        plt.Line2D([], [], marker="D", color="#2ca02c", lw=0, label="Praesepe [Fe/H]=+0.15"),
    ]
    if legend_below:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.3),
                  ncol=2, fontsize=fs - 1, frameon=False)
    else:
        ax.set_title("Isochrone ages: systematic range (bar), bootstrap (whiskers)", fontsize=fs + 1)
        ax.legend(handles=handles, loc="upper right", fontsize=fs - 1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("saved", path)


make(OUT / "isochrone_summary.png", (9, 3.6), 10, legend_below=False)
make(OUT / "isochrone_summary_slide.png", (6.4, 3.3), 12, legend_below=True)