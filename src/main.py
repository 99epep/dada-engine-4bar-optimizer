# =========================
# File: main.py
# =========================

from config import SolverConfig
from optimizer import optimize


def main() -> None:
    config = SolverConfig()

    solutions = optimize(config)

    print()
    print("=" * 60)
    print(f"{len(solutions)} solution(s) retenue(s)")
    print("=" * 60)

    for i, solution in enumerate(solutions, start=1):
        print(
            f"{i:3d} | "
            f"score={solution.score:.6f} | "
            f"support={solution.support.kind.value}"
        )


if __name__ == "__main__":
    main()