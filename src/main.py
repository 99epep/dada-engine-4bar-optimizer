"""Run the two optimization levels for one support family."""

import argparse
from pathlib import Path

import numpy as np

from config import SolverConfig
from optimizer import optimize_family_with_statistics, print_statistics
from refine import print_summary
from refinement import refine_candidate, deduplicate_refined_solutions
from result_io import save_results, save_refined_results


def run_family(
    family,
    config,
    level1_output=None,
    output=None,
    limit_refinement=None,
):
    family = family.upper()
    level1_output = Path(level1_output or f"results_{family}.npz")
    output = Path(output or f"refined_results_{family}.npz")

    print(f"RECHERCHE GLOBALE — FAMILLE {family}", flush=True)
    solutions, statistics = optimize_family_with_statistics(config, family)
    print_statistics(statistics)
    save_results(
        solutions if family == "E" else [],
        solutions if family == "F" else [],
        statistics,
        level1_output,
    )
    print(f"Résultats du niveau 1 : {level1_output}")

    entries = solutions if limit_refinement is None else solutions[:limit_refinement]
    rng = np.random.default_rng(config.random_seed)
    refined = []
    for source_index, solution in enumerate(entries):
        # Reuse the exact serialized structure expected by refinement without
        # reopening the file just written.
        entry = {
            "score": solution.score,
            "mechanism": {
                "ground": solution.mechanism.ground,
                "crank": solution.mechanism.crank,
                "coupler": solution.mechanism.coupler,
                "rocker": solution.mechanism.rocker,
            },
            "support": {
                "kind": family,
                "local_x": solution.support.local_x,
                "local_y": solution.support.local_y,
            },
            "score_components": dict(solution.score_components),
        }
        print(
            f"[{family}] raffinement {source_index + 1}/{len(entries)}",
            flush=True,
        )
        refined.append(
            refine_candidate(entry, family, source_index, config, rng)
        )

    refined = deduplicate_refined_solutions(refined, family, config)
    save_refined_results(
        refined if family == "E" else [],
        refined if family == "F" else [],
        output,
    )
    print_summary(family, refined)
    print(f"\nRésultats finaux : {output}")
    return refined, statistics


def main():
    parser = argparse.ArgumentParser(
        description="Recherche globale et raffinement DADA pour E ou F"
    )
    parser.add_argument("family", choices=("E", "F", "e", "f"))
    parser.add_argument("--level1-output")
    parser.add_argument("--output")
    parser.add_argument("--limit-refinement", type=int)
    parser.add_argument(
        "--low-def", action="store_true",
        help="utiliser la grille rapide de validation",
    )
    args = parser.parse_args()
    run_family(
        args.family,
        SolverConfig(high_definition=not args.low_def),
        args.level1_output,
        args.output,
        args.limit_refinement,
    )


if __name__ == "__main__":
    main()
