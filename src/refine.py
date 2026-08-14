"""Command-line entry point for DADA optimization level 2."""

import argparse

import numpy as np

from config import SolverConfig
from refinement import refine_candidate
from result_io import load_results, save_refined_results


def print_summary(family, solutions):
    print(f"\n{'=' * 76}\nSOLUTIONS {family} RAFFINÉES\n{'=' * 76}")
    if not solutions:
        print("Aucune solution.")
        return
    for rank, solution in enumerate(solutions, 1):
        mechanism = solution.mechanism
        metrics = solution.physical_metrics
        support = (
            f"CDE={solution.cde_angle_deg:.6f}°" if family == "E" else
            f"F=({solution.support.local_x:.6f}, {solution.support.local_y:.6f}) mm"
        )
        print(
            f"{rank:3d} | source={solution.source_index + 1:3d} | "
            f"score={solution.initial_score:.9f} -> {solution.final_score:.9f} | "
            f"AD/AB/BC/CD={mechanism.ground:.4f}/{mechanism.crank:.4f}/"
            f"{mechanism.coupler:.4f}/{mechanism.rocker:.4f} | {support} | "
            f"course={metrics['useful_stroke']:.4f} mm | "
            f"plateau={metrics['real_plateau_width_deg']:.3f}° | "
            f"échanges={metrics['exchange_1_to_2_deg']:.3f}/"
            f"{metrics['exchange_2_to_1_deg']:.3f}° | "
            f"précomp.={100 * metrics['precompression_1_to_2_ratio']:.2f}/"
            f"{100 * metrics['precompression_2_to_1_ratio']:.2f}%"
        )


def run(input_filename="results.npz", output_filename="refined_results.npz", limit_e=None, limit_f=None):
    config = SolverConfig()
    loaded = load_results(input_filename)
    try:
        metadata = loaded["metadata"]
    finally:
        loaded["data"].close()
    rng = np.random.default_rng(config.random_seed)
    refined = {}
    for family, limit in (("E", limit_e), ("F", limit_f)):
        entries = metadata[family] if limit is None else metadata[family][:limit]
        solutions = []
        for source_index, entry in enumerate(entries):
            print(
                f"[{family}] candidat source {source_index + 1}/{len(entries)}",
                flush=True,
            )
            solutions.append(
                refine_candidate(entry, family, source_index, config, rng)
            )
        refined[family] = sorted(
            solutions, key=lambda solution: solution.final_score, reverse=True
        )
    output = save_refined_results(refined["E"], refined["F"], output_filename)
    print_summary("E", refined["E"])
    print_summary("F", refined["F"])
    print(f"\nRésultats raffinés sauvegardés : {output}")
    return refined


def main():
    parser = argparse.ArgumentParser(description="Raffinement DADA niveau 2")
    parser.add_argument("--input", default="results.npz")
    parser.add_argument("--output", default="refined_results.npz")
    parser.add_argument("--limit-e", type=int, default=None)
    parser.add_argument("--limit-f", type=int, default=None)
    args = parser.parse_args()
    run(args.input, args.output, args.limit_e, args.limit_f)


if __name__ == "__main__":
    main()
