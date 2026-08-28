"""Combined visualization of a refined candidate and its symmetric piston."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from refinement import symmetric_values
from result_io import load_refined_results


def normalized(values):
    values = np.asarray(values, dtype=float)
    stroke = float(np.ptp(values))
    return np.zeros_like(values) if stroke == 0.0 else (values - np.min(values)) / stroke


def circular_mask(theta, start, end):
    return ((np.asarray(theta) - start) % 360.0) <= (end - start) % 360.0 + 1e-10


def precompression_ratio(solution):
    """Read the single metric, with compatibility for older refined NPZ."""
    metrics = solution["physical_metrics"]
    if "precompression_ratio" in metrics:
        return float(metrics["precompression_ratio"])
    a3 = float(solution["score_components"]["a3_angle"]) % 360.0
    legacy_key = (
        "precompression_1_to_2_ratio" if a3 > 180.0
        else "precompression_2_to_1_ratio"
    )
    return float(metrics[legacy_key])


def list_refined_solutions(filename="refined_results.npz"):
    """Print E then F candidates, ranked by decreasing final score."""
    loaded = load_refined_results(filename)
    try:
        metadata = loaded["metadata"]
        for family in ("E", "F"):
            solutions = sorted(
                metadata[family],
                key=lambda solution: solution["final_score"],
                reverse=True,
            )
            print(f"\nSOLUTIONS {family}")
            print("=" * 104)
            if not solutions:
                print("Aucune solution.")
                continue
            for rank, solution in enumerate(solutions, start=1):
                geometry = solution["final_geometry"]
                support = solution["support"]
                metrics = solution["physical_metrics"]
                if family == "E":
                    position = f"CDE={support['cde_angle_deg']:.2f}°"
                else:
                    position = (
                        f"F=({support['local_x']:.2f}, "
                        f"{support['local_y']:.2f}) mm"
                    )
                print(
                    f"{rank:3d} | score={solution['final_score']:.9f} | "
                    f"AB/BC/CD={geometry['crank']:.2f}/"
                    f"{geometry['coupler']:.2f}/{geometry['rocker']:.2f} mm | "
                    f"{position} | course={metrics['useful_stroke']:.2f} mm | "
                    f"précomp.={100.0 * precompression_ratio(solution):.2f}%"
                )
    finally:
        loaded["data"].close()


def plot_refined_solution(filename="refined_results.npz", family="E", index=0):
    family = family.upper()
    if family not in ("E", "F"):
        raise ValueError("family doit être E ou F")
    loaded = load_refined_results(filename)
    try:
        solutions = loaded["metadata"][family]
        if not 0 <= index < len(solutions):
            raise IndexError(f"Solution {family} #{index} inexistante")
        solution = solutions[index]
        prefix = f"{family}_{index}"
        theta = np.asarray(loaded["data"][f"{prefix}_theta"], dtype=float).copy()
        x = np.asarray(loaded["data"][f"{prefix}_x"], dtype=float).copy()
    finally:
        loaded["data"].close()

    metrics = solution["physical_metrics"]
    components = solution["score_components"]
    piston1 = normalized(x)
    piston2 = normalized(symmetric_values(theta, x))
    plateau1 = circular_mask(
        theta, metrics["real_plateau_start_deg"], metrics["real_plateau_end_deg"]
    )
    plateau2 = circular_mask(
        theta, metrics["symmetric_plateau_start_deg"],
        metrics["symmetric_plateau_end_deg"],
    )

    figure, axis = plt.subplots(figsize=(14, 7.5))
    axis.fill_between(
        theta, 0.0, 1.0, where=~(plateau1 | plateau2), step="mid",
        transform=axis.get_xaxis_transform(), color="tab:orange", alpha=0.15,
        label="Phases d’échange (aucun piston au plateau)",
    )
    axis.plot(theta, piston1, color="tab:blue", linewidth=1.8, label="Piston 1 — f(θ)")
    axis.plot(theta, piston2, color="tab:red", linewidth=1.6, label="Piston 2 — f(-θ mod 360)")

    for label, key, style in (
        ("A1", "a1_angle", "--"), ("A2", "a2_angle", "--"),
        ("A3", "a3_angle", ":"),
    ):
        angle = float(components[key]) % 360.0
        axis.axvline(angle, color="black", linestyle=style, linewidth=1.0)
        axis.annotate(
            f"{label} {angle:.1f}°", (angle, 1.0), xycoords=("data", "axes fraction"),
            xytext=(3, -4), textcoords="offset points", rotation=90,
            va="top", fontsize=8,
        )

    for label, key, color, style in (
        ("P1 début", "real_plateau_start_deg", "tab:blue", "-"),
        ("P1 fin", "real_plateau_end_deg", "tab:blue", "-"),
        ("P2 début", "symmetric_plateau_start_deg", "tab:red", "-."),
        ("P2 fin", "symmetric_plateau_end_deg", "tab:red", "-."),
    ):
        angle = float(metrics[key]) % 360.0
        axis.axvline(angle, color=color, linestyle=style, linewidth=1.0, alpha=0.75)
        axis.annotate(
            f"{label} {angle:.2f}°", (angle, 0.0),
            xycoords=("data", "axes fraction"), xytext=(3, 5),
            textcoords="offset points", rotation=90, va="bottom",
            fontsize=7.5, color=color,
        )

    geometry = solution["final_geometry"]
    support = solution["support"]
    support_text = (
        f"CDE={support['cde_angle_deg']:.4f}°" if family == "E" else
        f"F=({support['local_x']:.4f}, {support['local_y']:.4f}) mm"
    )
    text = (
        f"Score : {solution['initial_score']:.9f} → {solution['final_score']:.9f}\n"
        f"AD/AB/BC/CD : {geometry['ground']:.4f} / {geometry['crank']:.4f} / "
        f"{geometry['coupler']:.4f} / {geometry['rocker']:.4f} mm\n"
        f"Support : {support_text}\n"
        f"Course : {metrics['useful_stroke']:.4f} mm | "
        f"plateau : {metrics['real_plateau_width_deg']:.3f}°\n"
        f"Échanges 1→2 / 2→1 : {metrics['exchange_1_to_2_deg']:.3f}° / "
        f"{metrics['exchange_2_to_1_deg']:.3f}° | précompression : "
        f"{100 * precompression_ratio(solution):.2f}%"
    )
    figure.text(
        0.5, 0.025, text, va="bottom", ha="center", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    axis.set(
        title=f"Solution raffinée {family} #{index} — pistons symétriques",
        xlabel="Angle de manivelle θ (°)", ylabel="Position X normalisée",
        xlim=(0.0, 360.0), ylim=(-0.03, 1.03),
    )
    axis.set_xticks(np.arange(0.0, 361.0, 45.0))
    axis.grid(True, alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.205),
        ncol=3, frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.30, 1.0, 1.0))
    return figure


def main():
    parser = argparse.ArgumentParser(description="Visualiseur DADA niveau 2")
    parser.add_argument("filename", nargs="?")
    parser.add_argument("family", nargs="?")
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.filename is None or (args.family is None and args.index is None):
        list_refined_solutions(args.filename or "refined_results.npz")
        return
    if args.family is None or args.index is None:
        parser.error("le tracé requiert : fichier famille index")
    plot_refined_solution(args.filename, args.family, args.index)
    plt.show()


if __name__ == "__main__":
    main()


__all__ = ["list_refined_solutions", "plot_refined_solution", "precompression_ratio"]
