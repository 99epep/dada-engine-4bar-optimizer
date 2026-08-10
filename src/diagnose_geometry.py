import sys

import numpy as np

from config import SolverConfig
from models import Mechanism
from kinematics import solve, build_candidate_curve
from support_search import generate_candidates
from objective import (
    _compute_metrics,
    _find_plateau_candidates,
    _filter_2,
    _mean_abs_acceleration,
)


def diagnose_supports(mechanism, config):

    kinematics = solve(mechanism, config)

    if not kinematics.valid:
        print("Cinématique invalide.")
        return

    supports = generate_candidates(
        mechanism,
        config,
    )

    print(f"Supports générés : {len(supports)}")

    results = {
        "E": {
            "f1": [],
            "accepted": [],
        },
        "F": {
            "f1": [],
            "accepted": [],
        },
    }

    for support in supports:

        curve = build_candidate_curve(
            kinematics,
            support,
        )

        metrics = _compute_metrics(
            curve,
            config,
        )

        plateaus = _find_plateau_candidates(
            metrics,
            config,
        )

        if not plateaus:
            continue

        kind = support.kind.value

        for plateau in plateaus:

            candidate = dict(metrics)
            candidate.update(plateau)

            results[kind]["f1"].append(
                (
                    support,
                    candidate,
                )
            )

            ok, reason = _filter_2(
                candidate,
                config,
            )

            if not ok:
                continue

            acceleration = _mean_abs_acceleration(
                candidate,
                config,
            )

            candidate["mean_abs_acceleration"] = acceleration

            results[kind]["accepted"].append(
                (
                    support,
                    candidate,
                )
            )

    for kind in ("E", "F"):

        print()
        print("=" * 60)
        print(f"SUPPORT {kind}")
        print("=" * 60)

        f1 = results[kind]["f1"]
        accepted = results[kind]["accepted"]

        print(f"Passant filtre #1 : {len(f1)}")
        print(f"Passant filtre #2 : {len(accepted)}")

        if f1:

            best_f1 = min(
                f1,
                key=lambda item: (
                    item[1]["plateau_amplitude"],
                ),
            )

            support, candidate = best_f1

            print()
            print("MEILLEUR CANDIDAT APRÈS FILTRE #1")
            print(
                f"support local = "
                f"({support.local_x:.3f}, "
                f"{support.local_y:.3f})"
            )
            print(
                f"A1 = "
                f"{candidate['plateau_start_angle']:.2f}°"
            )
            print(
                f"A2 = "
                f"{candidate['plateau_end_angle']:.2f}°"
            )
            print(
                f"centre = "
                f"{candidate['plateau_center']:.2f}°"
            )
            print(
                f"largeur = "
                f"{candidate['plateau_width']:.2f}°"
            )
            print(
                f"amplitude = "
                f"{candidate['plateau_amplitude']:.5f}"
            )

        if accepted:

            best = min(
                accepted,
                key=lambda item: (
                    item[1]["mean_abs_acceleration"],
                ),
            )

            support, candidate = best

            print()
            print("MEILLEUR CANDIDAT ACCEPTÉ")
            print(
                f"support local = "
                f"({support.local_x:.3f}, "
                f"{support.local_y:.3f})"
            )
            print(
                f"A1 = "
                f"{candidate['plateau_start_angle']:.2f}°"
            )
            print(
                f"A2 = "
                f"{candidate['plateau_end_angle']:.2f}°"
            )
            print(
                f"centre = "
                f"{candidate['plateau_center']:.2f}°"
            )
            print(
                f"A3 = "
                f"{candidate['a3_angle']:.2f}°"
            )
            print(
                f"bissectrice = "
                f"{candidate['bisector_angle']:.2f}°"
            )
            print(
                f"accélération moyenne = "
                f"{candidate['mean_abs_acceleration']:.8f}"
            )


def main():

    config = SolverConfig()

    mechanism = Mechanism(
        ground=100.0,
        crank=55.0,
        coupler=100.0,
        rocker=110.0,
    )

    print("=" * 60)
    print("DIAGNOSTIC DE LA GÉOMÉTRIE DE RÉFÉRENCE")
    print("=" * 60)
    print()
    print("AD = 100")
    print("AB = 55")
    print("BC = 100")
    print("CD = 110")

    diagnose_supports(
        mechanism,
        config,
    )


if __name__ == "__main__":
    main()
