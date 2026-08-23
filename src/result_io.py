
# =========================
# File: result_io.py
# =========================

from pathlib import Path
import json

import numpy as np


def save_results(
    solutions_E,
    solutions_F,
    statistics,
    filename="results.npz",
):
    """
    Save retained E/F solutions and search statistics.

    The complete kinematic curves are stored so that the
    visualizer never needs to rerun the solver.
    """

    filename = Path(filename)

    data = {
        "solutions_E": _serialize_solutions(solutions_E),
        "solutions_F": _serialize_solutions(solutions_F),
        "statistics": {
            "mechanisms_tested": statistics.mechanisms_tested,
            "supports_generated": statistics.supports_generated,
            "filter1_rejected": statistics.filter1_rejected,
            "filter2_rejected": statistics.filter2_rejected,
            "accepted": statistics.accepted,
        },
    }

    # JSON is used inside the NPZ for the structural information.
    # Arrays remain native NumPy arrays.
    arrays = {}

    metadata = {
        "statistics": data["statistics"],
        "E": [],
        "F": [],
    }

    for kind in ("E", "F"):
        solutions = data[f"solutions_{kind}"]

        for i, solution in enumerate(solutions):

            prefix = f"{kind}_{i}"

            metadata[kind].append({
                "score": solution["score"],
                "scoring_quantity": (
                    "rocker_angle_deg" if kind == "E" else "support_x_mm"
                ),
                "mechanism": solution["mechanism"],
                "support": solution["support"],
                "score_components": solution["score_components"],
            })

            arrays[f"{prefix}_theta"] = solution["theta"]
            arrays[f"{prefix}_x"] = solution["x"]
            arrays[f"{prefix}_y"] = solution["y"]
            arrays[f"{prefix}_displacement"] = (
                solution["displacement"]
            )
            arrays[f"{prefix}_velocity"] = (
                solution["velocity"]
            )

    arrays["metadata"] = np.array(
        json.dumps(metadata),
        dtype=object,
    )

    np.savez_compressed(
        filename,
        **arrays,
    )

    return filename


def load_results(filename="results.npz"):
    """
    Load saved E/F solutions without rerunning the solver.
    """

    filename = Path(filename)

    data = np.load(
        filename,
        allow_pickle=True,
    )

    metadata = json.loads(
        str(data["metadata"].item())
    )

    return {
        "metadata": metadata,
        "data": data,
    }


def save_refined_results(solutions_E, solutions_F, filename="refined_results.npz"):
    """Save level-2 results in a format separate from ``results.npz``."""
    filename = Path(filename)
    metadata = {
        "format": "dada-refined-results-v2",
        "E": [],
        "F": [],
    }
    arrays = {}
    for family, solutions in (("E", solutions_E), ("F", solutions_F)):
        for final_index, solution in enumerate(solutions):
            prefix = f"{family}_{final_index}"
            metadata[family].append(solution.to_metadata(final_index))
            for name in ("theta", "x", "y", "displacement", "velocity"):
                arrays[f"{prefix}_{name}"] = np.asarray(
                    getattr(solution.curve, name)
                )
    arrays["metadata"] = np.array(json.dumps(metadata), dtype=object)
    np.savez_compressed(filename, **arrays)
    return filename


def load_refined_results(filename="refined_results.npz"):
    """Load and validate a level-2 result archive."""
    data = np.load(Path(filename), allow_pickle=True)
    metadata = json.loads(str(data["metadata"].item()))
    if metadata.get("format") != "dada-refined-results-v2":
        data.close()
        raise ValueError("Unsupported refined result format")
    return {"metadata": metadata, "data": data}


def _serialize_solutions(solutions):
    result = []

    for solution in solutions:

        mechanism = solution.mechanism
        support = solution.support
        curve = solution.curve

        result.append({
            "score": solution.score,

            "mechanism": {
                "ground": mechanism.ground,
                "crank": mechanism.crank,
                "coupler": mechanism.coupler,
                "rocker": mechanism.rocker,
            },

            "support": {
                "kind": support.kind.value,
                "local_x": support.local_x,
                "local_y": support.local_y,
            },

            "score_components": dict(
                solution.score_components
            ),

            "theta": curve.theta,
            "x": curve.x,
            "y": curve.y,
            "displacement": curve.displacement,
            "velocity": curve.velocity,
        })

    return result


__all__ = [
    "save_results",
    "load_results",
    "save_refined_results",
    "load_refined_results",
]
