"""Figures written for every run: training curves (curves.png) and solution / error (solution.png)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_run(run_dir, P, arrays):
    h = pd.read_csv(os.path.join(run_dir, "history.csv"))
    v = h.dropna(subset=["relL2"]) if "relL2" in h else h.iloc[0:0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    ax[0].semilogy(h["epoch"], h["loss"], color="#2a78d6", linewidth=1.5, label="total loss")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); _style(ax[0])
    for f, c in zip(P.fields, ["#eb6834", "#1baf7a"]):
        if f"relL2_{f}" in v:
            ax[1].semilogy(v["epoch"], v[f"relL2_{f}"], color=c, linewidth=1.5, label=f"relative L² ({f})")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Relative L² error"); _style(ax[1]); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "curves.png")); plt.close(fig)

    X, pred, ex = arrays["coords"], arrays["pred"], arrays["exact"]
    if X.shape[1] == 1:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
        for i, f in enumerate(P.fields):
            ax[0].plot(X[:, 0], ex[:, i], color=INK, linewidth=1.5, label=f"exact {f}")
            ax[0].plot(X[:, 0], pred[:, i], "--", color="#eb6834" if i == 0 else "#1baf7a", linewidth=1.5,
                       label=f"predicted {f}")
            ax[1].semilogy(X[:, 0], np.abs(pred[:, i] - ex[:, i]) + 1e-16, linewidth=1.2, label=f)
        ax[0].legend(frameon=False); ax[0].set_xlabel(P.coords[0]); _style(ax[0])
        ax[1].set_xlabel(P.coords[0]); ax[1].set_ylabel("|error|"); _style(ax[1])
    else:
        n1, n2 = arrays["grid_shape"]
        A, B = X[:, 0].reshape(n1, n2), X[:, 1].reshape(n1, n2)
        E, Pr = ex[:, 0].reshape(n1, n2), pred[:, 0].reshape(n1, n2)
        ext = [B.min(), B.max(), A.min(), A.max()]
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), dpi=150)
        for a, data, title in zip(ax, [E, Pr, np.abs(Pr - E)], ["exact", "predicted", "|error|"]):
            im = a.imshow(data, origin="lower", extent=ext, aspect="auto", cmap="viridis")
            a.set_title(title, fontsize=10); a.set_xlabel(P.coords[1]); a.set_ylabel(P.coords[0])
            fig.colorbar(im, ax=a)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "solution.png")); plt.close(fig)
