"""
How to choose the Heaps fitting window from the data, instead of guessing.

Two diagnostics:

1. LOCAL SLOPE. beta is the slope of log V against log N. Measure that slope
   in a rolling window and plot it against N. If the corpus obeys a single
   power law, the local slope settles onto a flat line, and where it settles
   is where the fit should start.

2. N_min SWEEP. Refit above every candidate starting point and record beta,
   R^2, and the worst residual. If a genuine power-law region exists, beta
   goes flat as N_min moves through it.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CURVE_CSV = "out/heaps-anime/heaps_curve.csv"
OUT_DIR = "out/heaps-anime"

# Half-width of the rolling window, in log10 units of N.
WINDOW = 0.35


def load_curve(path):
    grid = []
    mean = []

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            grid.append(float(row["N"]))
            mean.append(float(row["V_mean"]))

    return np.array(grid), np.array(mean)


def local_slope(grid, mean, window=WINDOW):
    """d(log V)/d(log N) measured over a rolling window in log space."""
    log_n = np.log10(grid)
    log_v = np.log10(mean)

    centres = []
    slopes = []

    for index in range(len(log_n)):
        low = log_n[index] - window
        high = log_n[index] + window
        mask = (log_n >= low) & (log_n <= high)

        if mask.sum() < 6:
            continue

        slope, _ = np.polyfit(log_n[mask], log_v[mask], 1)
        centres.append(grid[index])
        slopes.append(slope)

    return np.array(centres), np.array(slopes)


def sweep_nmin(grid, mean, candidates):
    """Refit above each candidate start point."""
    log_n = np.log10(grid)
    log_v = np.log10(mean)

    rows = []

    for n_min in candidates:
        mask = grid >= n_min

        if mask.sum() < 10:
            continue

        beta, log_k = np.polyfit(log_n[mask], log_v[mask], 1)
        predicted = log_k + beta * log_n[mask]
        residuals = log_v[mask] - predicted

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((log_v[mask] - log_v[mask].mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot

        rows.append({
            "n_min": int(n_min),
            "beta": round(float(beta), 4),
            "K": round(float(10 ** log_k), 3),
            "r_squared": round(r_squared, 6),
            "max_abs_residual_dex": round(float(np.max(np.abs(residuals))), 5),
            "points": int(mask.sum()),
        })

    return rows


def plot_diagnostics(centres, slopes, rows, path):
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 9))

    top.semilogx(centres, slopes, linewidth=1.8, color="#2f6f8f")
    top.axhline(1.0, linestyle=":", color="#888888", linewidth=1.2)
    top.text(centres[0] * 1.2, 1.005, "slope = 1: every token is a new word",
             fontsize=8, color="#666666")

    top.set_ylabel("local slope  d(log V) / d(log N)")
    top.set_xlabel("N — tokens seen (log scale)")
    top.set_title("Local slope never goes flat\n"
                  "so no single beta describes the whole corpus",
                  fontsize=12, pad=10)
    top.grid(True, which="major", linewidth=0.3, alpha=0.35)
    top.set_ylim(0, 1.05)

    n_mins = [row["n_min"] for row in rows]
    betas = [row["beta"] for row in rows]
    residuals = [row["max_abs_residual_dex"] for row in rows]

    bottom.semilogx(n_mins, betas, linewidth=1.8, color="#c1440e",
                    label="fitted beta above N_min")
    bottom.set_xlabel("N_min — where the fit starts (log scale)")
    bottom.set_ylabel("fitted beta", color="#c1440e")
    bottom.tick_params(axis="y", labelcolor="#c1440e")
    bottom.grid(True, which="major", linewidth=0.3, alpha=0.35)

    twin = bottom.twinx()
    twin.semilogx(n_mins, residuals, linewidth=1.4, color="#3f8c5a",
                  linestyle="--", label="worst residual (dex)")
    twin.set_ylabel("worst residual, log10 units", color="#3f8c5a")
    twin.tick_params(axis="y", labelcolor="#3f8c5a")

    bottom.set_title("Fitted beta keeps sliding as N_min moves\n"
                     "there is no stable plateau to lock onto",
                     fontsize=12, pad=10)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    grid, mean = load_curve(CURVE_CSV)

    centres, slopes = local_slope(grid, mean)

    print("local slope at selected N:")
    for target in [1e3, 1e4, 1e5, 3e5, 1e6, 3e6]:
        if target > centres[-1]:
            continue

        value = float(np.interp(target, centres, slopes))
        print(f"  N = {int(target):>9,}   local slope = {value:.3f}")

    candidates = np.unique(np.logspace(2, 6.2, 60).astype(int))
    rows = sweep_nmin(grid, mean, candidates)

    print()
    print(f"{'N_min':>10}{'beta':>9}{'K':>10}{'R2':>10}"
          f"{'worst resid':>13}{'points':>8}")
    print("-" * 60)

    for row in rows[::6]:
        print(f"{row['n_min']:>10,}{row['beta']:>9.4f}{row['K']:>10.2f}"
              f"{row['r_squared']:>10.5f}{row['max_abs_residual_dex']:>13.4f}"
              f"{row['points']:>8}")

    with open(os.path.join(OUT_DIR, "window_selection.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_diagnostics(centres, slopes, rows,
                     os.path.join(OUT_DIR, "heaps_window_selection.png"))

    print()
    print("wrote heaps_window_selection.png and window_selection.csv")


if __name__ == "__main__":
    main()
