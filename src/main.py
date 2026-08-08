# =========================
# File: main.py
# =========================

from config import SolverConfig
from optimizer import optimize_with_statistics, print_statistics


def main() -> None:
    config = SolverConfig()

    solutions, stats = optimize_with_statistics(config)

    print_statistics(stats)

    print()
    print("=" * 60)
    print(f"{len(solutions)} solution(s) retenue(s)")
    print("=" * 60)

    for i, solution in enumerate(solutions, start=1):

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
            f"support={solution.support.kind.value}"
        )


if __name__ == "__main__":
    main()
