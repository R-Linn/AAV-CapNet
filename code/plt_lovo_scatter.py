# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = ["Arial"]
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 11
os.makedirs("./figures", exist_ok=True)

if __name__ == "__main__":
    csv_path = "./leave_one_variant_out_results/lovo_sample_level_predictions.csv"
    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True,
                                   gridspec_kw={"width_ratios": [0.50, 0.15]})
    fig.subplots_adjust(wspace=0.07, right=0.82)
    variants = sorted(df["variant"].unique())
    cmap = plt.cm.tab20
    for idx, var in enumerate(variants):
        sub = df[df["variant"] == var]
        ax1.scatter(
            sub["y_true_reg"],
            sub["y_pred_reg"],
            color=cmap(idx),
            alpha=0.7,
            s=35
        )
        ax2.scatter(
            sub["y_true_reg"],
            sub["y_pred_reg"],
            label=var,
            color=cmap(idx),
            alpha=0.7,
            s=35
        )
    ax1.set_xlim(-0.05, 0.45)
    ax2.set_xlim(0.90, 1.05)
    ax1.set_ylim(-0.1, 0.20)
    ax2.set_ylim(-0.1, 0.20)

    ax1.plot(ax1.get_xlim(), ax1.get_xlim(), "k--", lw=1.5, label="Ideal Prediction (y=x)")
    ax2.plot(ax2.get_xlim(), ax2.get_xlim(), "k--", lw=1.5)

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d*3, +d*3), (-d, +d), **kwargs)
    ax2.plot((-d*3, +d*3), (1 - d, 1 + d), **kwargs)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.tick_params(axis='y', left=True, labelleft=True)
    ax2.tick_params(axis='y', left=False, labelleft=False)

    fmt_two_dec = FuncFormatter(lambda val, pos: f"{val:.2f}")
    ax1.xaxis.set_major_formatter(fmt_two_dec)
    ax2.xaxis.set_major_formatter(fmt_two_dec)

    fig.text(0.39, 0.01, "True Normalized Infectivity",
             ha='center', fontweight="bold", fontsize = 12)
    fig.text(0.28, 0.7, "y = x",
                 ha='center', fontweight="bold", fontsize = 12)
    ax1.set_ylabel("Predicted Normalized Infectivity", 
                   fontweight="bold", x=0.1, fontsize = 12)
    fig.suptitle("LOVO Cross-Validation: Predicted vs. True Infectivity",
                 fontweight="bold", y=0.92, x=0.39, fontsize = 16)
    ax2.legend(bbox_to_anchor=(1, 1.015), loc="upper left", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 0.8, 0.96])
    save_file = "./figures/LOVO scatter.pdf"
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()