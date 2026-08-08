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
        components = getattr(solution, "score_components", {})

        print(
            f"{i:3d} | "
            f"score={solution.score:.6f} | "
            f"support={solution.support.kind.value} | "
            f"fast={components.get('fast', 0.0):.3f} | "
            f"slow={components.get('slow', 0.0):.3f} | "
            f"center={components.get('center', 0.0):.3f} | "
            f"static={components.get('static', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
