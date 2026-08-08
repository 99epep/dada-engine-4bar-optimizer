# =========================
# File: main.py
# =========================

from config import SolverConfig
from optimizer import (
    optimize_with_statistics,
    print_statistics,
)
from result_io import save_results


def print_solutions(title, solutions):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    if not solutions:
        print("Aucune solution.")
        return

    for i, solution in enumerate(
        solutions,
        start=1,
    ):

        components = getattr(
            solution,
            "score_components",
            {},
        )

        acceleration = components.get(
            "mean_abs_acceleration",
            float("nan"),
        )

        print(
            f"{i:3d} | "
            f"accel={acceleration:.8f} | "
            f"score={solution.score:.6f} | "
            f"support={solution.support.kind.value}"
        )


def main() -> None:

    config = SolverConfig()

    solutions_E, solutions_F, stats = (
        optimize_with_statistics(config)
    )

    print_statistics(stats)

    output_file = save_results(
        solutions_E,
        solutions_F,
        stats,
        "results.npz",
    )

    print()
    print(f"Résultats sauvegardés : {output_file}")

    print_solutions(
        "MEILLEURES SOLUTIONS E",
        solutions_E,
    )

    print_solutions(
        "MEILLEURES SOLUTIONS F",
        solutions_F,
    )


if __name__ == "__main__":
    main()
