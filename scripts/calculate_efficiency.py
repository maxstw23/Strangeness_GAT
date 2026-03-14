"""
calculate_efficiency.py — Fit kaon detection efficiency from ROOT histograms.

Reads hSelPtEta (reconstructed) and hSelPtEtaMc (MC truth) from cen8 and cen9
ROOT files, computes efficiency = reco/mc, fits a 2D parametric function, and
produces QA plots.

Usage:
    venv/bin/python scripts/calculate_efficiency.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import curve_fit
import uproot
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data"
PLOTS_DIR = "plots/QA"
os.makedirs(PLOTS_DIR, exist_ok=True)

PARTICLES = {
    "Kp": ("kplus",  ["cen8.kplus.root",  "cen9.kplus.root"]),
    "Km": ("kminus", ["cen8.kminus.root", "cen9.kminus.root"]),
}

PT_MIN, PT_MAX = 0.15, 2.0   # GeV/c
ETA_MAX = 1.5                # |eta| cut

# Slices for QA plots
ETA_SLICES  = [-1.0, -0.5, 0.0, 0.5, 1.0]   # fixed eta for pT slices
PT_SLICES   = [0.3, 0.5, 0.8, 1.2, 1.8]     # fixed pT for eta slices
ETA_WINDOW  = 0.05   # half-width for averaging around slice
PT_WINDOW   = 0.05   # half-width for averaging around slice

# Initial fit parameters and bounds
P0_INIT = [0.78, 0.16, 2.2, 0.04, -0.006, 1.56, 3.5]
P_LOWER = [0.0,  1e-3, 0.1, -5.0, -5.0,   0.1,  0.1]
P_UPPER = [5.0,  1.0,  10., 5.0,   5.0,   5.0,  20.]

# ---------------------------------------------------------------------------
# 2D efficiency function
# ---------------------------------------------------------------------------
def fcn_2D(x_y, p0, p1, p2, p3, p4, p5, p6):
    x, y = x_y  # x=pT, y=eta
    return (
        (p0 + p3 * x + p4 * x * x)
        * np.exp(-np.power(p1 / x, p2))
        * np.exp(-np.power(np.power(y / p5, 2), p6 / 2.0))
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_combined(files, hist_name):
    """Sum a 2D histogram across multiple ROOT files."""
    total = None
    axes = None
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        with uproot.open(path) as f:
            h = f[hist_name]
            vals = h.values()
            if total is None:
                total = vals.copy().astype(float)
                edges0 = h.axes[0].edges()
                edges1 = h.axes[1].edges()
                axes = (edges0, edges1)
            else:
                total += vals
    return total, axes


def bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def fit_efficiency(pt_centers, eta_centers, eff2d, mask2d):
    """Fit the 2D efficiency map. Returns popt, pcov."""
    pt_grid, eta_grid = np.meshgrid(pt_centers, eta_centers, indexing="ij")
    pt_flat   = pt_grid[mask2d]
    eta_flat  = eta_grid[mask2d]
    eff_flat  = eff2d[mask2d]

    popt, pcov = curve_fit(
        fcn_2D,
        (pt_flat, eta_flat),
        eff_flat,
        p0=P0_INIT,
        bounds=(P_LOWER, P_UPPER),
        maxfev=50000,
    )
    return popt, pcov, pt_flat, eta_flat, eff_flat


# ---------------------------------------------------------------------------
# QA Plots
# ---------------------------------------------------------------------------
def plot_qa(label, pt_centers, eta_centers, eff2d, mask2d, popt, tag):
    """Three figures: 3D surface, pT slices, eta slices."""

    pt_grid, eta_grid = np.meshgrid(pt_centers, eta_centers, indexing="ij")
    fit2d = fcn_2D((pt_grid, eta_grid), *popt)
    residuals = np.where(mask2d, eff2d - fit2d, np.nan)

    # ---- Figure 1: 2D heatmaps (data / fit / residual) -------------------
    fig, axes_2d = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Efficiency QA — {label}")

    eff_plot = np.where(mask2d, eff2d, np.nan)
    fit_plot = np.where(mask2d, fit2d, np.nan)

    for ax, data, title, cmap in zip(
        axes_2d,
        [eff_plot, fit_plot, residuals],
        ["Data ε(pT,η)", "Fit ε(pT,η)", "Residual (data−fit)"],
        ["viridis", "viridis", "RdBu_r"],
    ):
        im = ax.pcolormesh(
            pt_centers, eta_centers, data.T,
            cmap=cmap,
            vmin=(0 if "Residual" not in title else -0.05),
            vmax=(1 if "Residual" not in title else 0.05),
        )
        ax.set_xlabel("pT (GeV/c)")
        ax.set_ylabel("η")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"efficiency_qa_{tag}_heatmap.png"), dpi=120)
    plt.close(fig)

    # ---- Figure 2: pT slices at fixed η ----------------------------------
    ncols = len(ETA_SLICES)
    fig2, axes2 = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)
    fig2.suptitle(f"pT slices — {label}")

    pt_dense = np.linspace(PT_MIN, PT_MAX, 300)

    for ax, eta_val in zip(axes2, ETA_SLICES):
        # Select eta bins within window
        eta_sel = np.abs(eta_centers - eta_val) <= ETA_WINDOW
        if not eta_sel.any():
            ax.set_visible(False)
            continue
        # Average efficiency over selected eta bins (for each pT bin)
        eff_slice = np.nanmean(
            np.where(mask2d[:, eta_sel], eff2d[:, eta_sel], np.nan), axis=1
        )
        pt_valid = ~np.isnan(eff_slice)
        ax.errorbar(
            pt_centers[pt_valid], eff_slice[pt_valid],
            fmt="o", ms=3, label="data", color="steelblue"
        )
        fit_curve = fcn_2D((pt_dense, np.full_like(pt_dense, eta_val)), *popt)
        ax.plot(pt_dense, fit_curve, "r-", label="fit")
        ax.set_xlabel("pT (GeV/c)")
        ax.set_title(f"η = {eta_val:.1f}")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    axes2[0].set_ylabel("ε")
    plt.tight_layout()
    fig2.savefig(os.path.join(PLOTS_DIR, f"efficiency_qa_{tag}_pt_slices.png"), dpi=120)
    plt.close(fig2)

    # ---- Figure 3: η slices at fixed pT ----------------------------------
    ncols = len(PT_SLICES)
    fig3, axes3 = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)
    fig3.suptitle(f"η slices — {label}")

    eta_dense = np.linspace(-ETA_MAX, ETA_MAX, 300)

    for ax, pt_val in zip(axes3, PT_SLICES):
        pt_sel = np.abs(pt_centers - pt_val) <= PT_WINDOW
        if not pt_sel.any():
            ax.set_visible(False)
            continue
        eff_slice = np.nanmean(
            np.where(mask2d[pt_sel, :], eff2d[pt_sel, :], np.nan), axis=0
        )
        eta_valid = ~np.isnan(eff_slice)
        ax.errorbar(
            eta_centers[eta_valid], eff_slice[eta_valid],
            fmt="o", ms=3, label="data", color="steelblue"
        )
        fit_curve = fcn_2D((np.full_like(eta_dense, pt_val), eta_dense), *popt)
        ax.plot(eta_dense, fit_curve, "r-", label="fit")
        ax.set_xlabel("η")
        ax.set_title(f"pT = {pt_val:.1f} GeV/c")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    axes3[0].set_ylabel("ε")
    plt.tight_layout()
    fig3.savefig(os.path.join(PLOTS_DIR, f"efficiency_qa_{tag}_eta_slices.png"), dpi=120)
    plt.close(fig3)

    print(f"  Saved plots for {label}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_particle(name, tag, files):
    print(f"\n{'='*60}")
    print(f"  Particle: {name} ({tag})")
    print(f"{'='*60}")

    sel, axes   = load_combined(files, "hSelPtEta")
    mc,  _      = load_combined(files, "hSelPtEtaMc")

    pt_edges, eta_edges = axes
    pt_centers  = bin_centers(pt_edges)
    eta_centers = bin_centers(eta_edges)

    # Kinematic mask
    pt_ok   = (pt_centers  >= PT_MIN) & (pt_centers  <= PT_MAX)
    eta_ok  = (np.abs(eta_centers) <= ETA_MAX)
    mc_ok   = mc > 0

    # 2D mask: shape (n_pt, n_eta)
    pt_mask, eta_mask = np.meshgrid(pt_ok, eta_ok, indexing="ij")
    mask = pt_mask & eta_mask & mc_ok

    # Restrict arrays to masked region bins
    eff = np.where(mask, np.divide(sel, mc, where=mc_ok), np.nan)
    eff = np.clip(eff, 0.0, 1.0)

    # Exclude zeros (no data)
    mask_fit = mask & (eff > 0.0) & ~np.isnan(eff)

    n_points = mask_fit.sum()
    print(f"  Fit points: {n_points}")

    popt, pcov, pt_f, eta_f, eff_f = fit_efficiency(pt_centers, eta_centers, eff, mask_fit)
    perr = np.sqrt(np.diag(pcov))

    param_names = ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]
    print("\n  Fitted parameters:")
    for pname, val, err in zip(param_names, popt, perr):
        print(f"    {pname} = {val:.7g} ± {err:.3g}")

    # Residuals summary
    fit_vals = fcn_2D((pt_f, eta_f), *popt)
    resid    = eff_f - fit_vals
    print(f"\n  Residual RMS : {np.sqrt(np.mean(resid**2)):.4f}")
    print(f"  Max |residual|: {np.max(np.abs(resid)):.4f}")

    # Print copy-pasteable config snippet
    print(f"\n  # config.py snippet for '{name}':")
    print(f"  '{name}': {{")
    for pname, val in zip(param_names, popt):
        print(f"      '{pname}': [{val:.7g}],  # × 9 centrality bins")
    print("  },")

    plot_qa(f"{name} (cen 8+9)", pt_centers, eta_centers, eff, mask_fit, popt, tag)

    return popt


if __name__ == "__main__":
    results = {}
    for name, (tag, files) in PARTICLES.items():
        results[name] = process_particle(name, tag, files)

    print("\n" + "="*60)
    print("  SUMMARY — paste into config.py EFFICIENCY_PARAMS")
    print("="*60)
    param_names = ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]
    print("EFFICIENCY_PARAMS = {")
    for name, popt in results.items():
        print(f"    '{name}': {{")
        for pname, val in zip(param_names, popt):
            print(f"        '{pname}': [{val:.7g}] * 9,")
        print("    },")
    print("}")
