
# =========================
# File: visualizer.py
# =========================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from result_io import load_results


def list_solutions(filename="results.npz"):
    """
    Print the saved E/F solutions and return their metadata.
    """

    results = load_results(filename)
    metadata = results["metadata"]

    for kind in ("E", "F"):

        print()
        print("=" * 60)
        print(f"SOLUTIONS {kind}")
        print("=" * 60)

        for i, solution in enumerate(
            metadata[kind],
            start=1,
        ):

            mechanism = solution["mechanism"]

            support = solution["support"]

            if kind == "E":
                # E est défini dans le repère lié au culbuteur CD.
                # L'angle CDE est donc directement l'angle de
                # la grille du demi-cercle.
                angle_cde = np.degrees(
                    np.arctan2(
                        support["local_y"],
                        support["local_x"],
                    )
                )

                support_info = (
                    f"CDE={angle_cde:.0f}°"
                )

            else:
                support_info = (
                    f"F=("
                    f"{support['local_x']:.1f}, "
                    f"{support['local_y']:.1f})"
                )

            print(
                f"{i:3d} | "
                f"score={solution['score']:.6f} | "
                f""
                f"crank={mechanism['crank']:.0f} "
                f"coupler={mechanism['coupler']:.0f} "
                f"rocker={mechanism['rocker']:.0f} | "
                f"{support_info}"
            )

    return results


def plot_solution(
    kind,
    index,
    filename="results.npz",
):
    """
    Display one saved solution.

    kind:
        "E" or "F"

    index:
        zero-based solution index.
    """

    results = load_results(filename)

    metadata = results["metadata"]

    kind = kind.upper()

    if kind not in ("E", "F"):
        raise ValueError("kind doit être E ou F")

    solutions = metadata[kind]

    if not 0 <= index < len(solutions):
        raise IndexError(
            f"Solution {index} inexistante pour {kind}"
        )

    solution = solutions[index]

    data = results["data"]

    prefix = f"{kind}_{index}"

    theta = data[f"{prefix}_theta"]
    x = data[f"{prefix}_x"]
    y = data[f"{prefix}_y"]

    score_components = solution["score_components"]

    # --------------------------------------------------------
    # Projection X normalisée
    # --------------------------------------------------------

    xmin = np.min(x)
    xmax = np.max(x)

    if xmax > xmin:
        x_normalized = (
            (x - xmin)
            / (xmax - xmin)
        )
    else:
        x_normalized = np.zeros_like(x)

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, (ax_mechanism, ax_position) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    # --------------------------------------------------------
    # Trajectoire du support
    # --------------------------------------------------------

    ax_mechanism.plot(
        x,
        y,
        linewidth=1.0,
    )

    ax_mechanism.plot(
        x[0],
        y[0],
        marker="o",
    )

    ax_mechanism.set_aspect(
        "equal",
        adjustable="box",
    )

    ax_mechanism.set_xlabel("X")
    ax_mechanism.set_ylabel("Y")
    ax_mechanism.set_title(
        f"Trajectoire du support {kind}"
    )
    ax_mechanism.grid(True)

    # --------------------------------------------------------
    # Position normalisée
    # --------------------------------------------------------

    ax_position.plot(
        theta,
        x_normalized,
    )

    markers = (
        ("A1", "plateau_start_angle"),
        ("A2", "plateau_end_angle"),
        ("A3", "a3_angle"),
    )

    for label, key in markers:

        angle = score_components.get(key)

        if angle is None:
            continue

        ax_position.axvline(
            angle,
            linestyle="--",
            linewidth=1.0,
        )

        ax_position.annotate(
            f"{label} = {angle:.0f}°",
            xy=(angle, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, 5),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            ha="left",
        )

    ax_position.set_xlim(0.0, 360.0)
    ax_position.set_xticks(
        np.arange(0.0, 361.0, 45.0)
    )
    ax_position.set_ylim(0.0, 1.0)
    ax_position.margins(x=0.01)

    ax_position.set_xlabel(
        "Angle de manivelle (°)"
    )

    ax_position.set_ylabel(
        "Projection X normalisée"
    )

    ax_position.set_title(
        f"Déplacement — {kind}"
    )

    ax_position.grid(True)

    fig.tight_layout()

    return fig


def show_solution(
    kind,
    index,
    filename="results.npz",
):
    fig = plot_solution(
        kind,
        index,
        filename,
    )

    plt.show()


def main():
    """
    Command-line visualizer.

    Examples:

        python visualizer.py
        python visualizer.py E 0
        python visualizer.py F 1
    """

    import sys

    if len(sys.argv) == 1:
        list_solutions()
        return

    if len(sys.argv) != 3:
        print(
            "Usage: python visualizer.py [E|F] [index]"
        )
        return

    kind = sys.argv[1]
    index = int(sys.argv[2])

    show_solution(
        kind,
        index,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "list_solutions",
    "plot_solution",
    "show_solution",
]
