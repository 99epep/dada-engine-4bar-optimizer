"""Create a GIF of a refined DADA mechanism and its opposed twin."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon
import numpy as np

from config import SolverConfig
from kinematics import build_candidate_curve, solve
from models import Mechanism, Support, SupportKind
from result_io import load_refined_results


# ============================================================================
# DEFAULT ANIMATION SETTINGS — edit here to give every GIF the same format.
# ============================================================================

DEFAULT_FPS = 25
DEFAULT_ANGLE_STEP_DEG = 4.0
DEFAULT_FIGURE_SIZE = (5.0, 6.18)  # 500 x 618 px; two areas of 309 px
DEFAULT_DPI = 100
DEFAULT_PISTON_ROD_LENGTH = 300.0
DEFAULT_CYLINDER_1_WIDTH = 140.0
DEFAULT_CYLINDER_2_WIDTH = 70.0
DEFAULT_SHOW_TRACES = False
DEFAULT_SHOW_SYMMETRIC = True

COLOR_FIXED = "#111111"
COLOR_FIRST_CARRIER = "#49bff2"   # bleu ciel
COLOR_FIRST_OTHER = "#0837b3"     # bleu marine
COLOR_SECOND_CARRIER = "#ff2020"  # rouge vif
COLOR_SECOND_OTHER = "#c0001c"    # bordeaux
COLOR_CYLINDER = "#808080"        # gris moyen
COLOR_PISTON = "#404040"          # gris foncé
COLOR_GRAPH_GRID = "#b0b0b0"

CYLINDER_LINE_WIDTH = 1.8
PISTON_LINE_WIDTH = 4.0


def _normalized(values):
    values = np.asarray(values, dtype=float)
    stroke = float(np.ptp(values))
    return np.zeros_like(values) if stroke == 0.0 else (values - np.min(values)) / stroke


def _circular_mask(theta, start, end):
    return ((theta - start) % 360.0) <= (end - start) % 360.0 + 1e-10


def _slider_positions(points, axis_y, rod_length, branch=1.0):
    vertical = points[:, 1] - axis_y
    remaining = rod_length * rod_length - vertical * vertical
    if np.min(remaining) < -1e-9:
        required = float(np.max(np.abs(vertical)))
        raise ValueError(
            f"Bielle de piston trop courte: {rod_length:g} mm; "
            f"minimum requis {required:g} mm"
        )
    return points[:, 0] + branch * np.sqrt(np.maximum(remaining, 0.0))


def _support_from_metadata(solution):
    support = solution["support"]
    return Support(
        SupportKind(solution["family"]),
        float(support["local_x"]),
        float(support["local_y"]),
    )


def _piston_installation(
    points, axis_y, rod_length, support_x, empty_is_maximum, width,
    mechanism_center_x, line_clearance,
):
    """Choose the slider branch with the empty piston farthest from the linkage."""
    empty_index = int(np.argmax(support_x) if empty_is_maximum else np.argmin(support_x))
    full_index = int(np.argmin(support_x) if empty_is_maximum else np.argmax(support_x))
    choices = []
    for branch in (1.0, -1.0):
        piston_x = _slider_positions(points, axis_y, rod_length, branch)
        empty_x = float(piston_x[empty_index])
        full_x = float(piston_x[full_index])
        empty_distance = abs(empty_x - mechanism_center_x)
        full_distance = abs(full_x - mechanism_center_x)
        choices.append((empty_distance > full_distance, empty_distance, piston_x))
    valid = [choice for choice in choices if choice[0]]
    _, _, piston_x = max(valid or choices, key=lambda choice: choice[1])
    empty_x = float(piston_x[empty_index])
    full_x = float(piston_x[full_index])
    outward = 1.0 if empty_x >= mechanism_center_x else -1.0
    # Leave room for the visible half-widths of both strokes. Matplotlib line
    # widths are expressed in points, so a small width-relative clearance is
    # deliberately used in mechanism coordinates.
    clearance = min(float(line_clearance), 0.45 * width)
    back_x = empty_x + outward * clearance
    open_x = full_x - outward * clearance
    return piston_x, {
        "open_x": open_x,
        "back_x": back_x,
        "bottom": axis_y - 0.5 * width,
        "length": abs(back_x - open_x),
        "width": width,
        "empty_x": empty_x,
        "outward": outward,
        "piston_half_height": 0.5 * width - clearance,
    }


def _load_motion(filename, family, index, angle_step):
    loaded = load_refined_results(filename)
    try:
        solution = loaded["metadata"][family][index]
    finally:
        loaded["data"].close()
    geometry = solution["final_geometry"]
    mechanism = Mechanism(**geometry)
    support = _support_from_metadata(solution)
    config = replace(SolverConfig(), angle_step_deg=angle_step)
    kinematic = solve(mechanism, config)
    if not kinematic.valid:
        raise ValueError("La géométrie raffinée n'est plus cinématiquement valide")
    curve = build_candidate_curve(kinematic, support)

    # Remove 360°: the GIF must end one regular angular step before 0°.
    n = len(kinematic.theta) - 1
    indices = np.arange(n)
    reverse = (-indices) % n
    motion = {
        "theta": kinematic.theta[:-1],
        "A1": kinematic.A[:-1],
        "B1": kinematic.B[:-1],
        "C1": kinematic.C[:-1],
        "D1": kinematic.D[:-1],
        "P1": np.column_stack((curve.x[:-1], curve.y[:-1])),
    }
    # Reflection about AD of the mechanism evaluated at -theta. B is shared.
    for target, source in (("A2", "A1"), ("B2", "B1"), ("C2", "C1"), ("D2", "D1"), ("P2", "P1")):
        values = motion[source][reverse].copy()
        values[:, 1] *= -1.0
        motion[target] = values
    return solution, mechanism, support, motion


def create_animation(
    filename,
    family,
    index,
    output=None,
    fps=DEFAULT_FPS,
    angle_step=DEFAULT_ANGLE_STEP_DEG,
    piston_rod_length=DEFAULT_PISTON_ROD_LENGTH,
    cylinder_1_width=DEFAULT_CYLINDER_1_WIDTH,
    cylinder_2_width=DEFAULT_CYLINDER_2_WIDTH,
    show_traces=DEFAULT_SHOW_TRACES,
    show_symmetric=DEFAULT_SHOW_SYMMETRIC,
):
    family = family.upper()
    if family not in ("E", "F"):
        raise ValueError("family doit être E ou F")
    solution, _, _, motion = _load_motion(filename, family, index, angle_step)
    theta = motion["theta"]
    point1, point2 = motion["P1"], motion["P2"]
    axis_y1 = 0.5 * (float(np.min(point1[:, 1])) + float(np.max(point1[:, 1])))
    axis_y2 = 0.5 * (float(np.min(point2[:, 1])) + float(np.max(point2[:, 1])))
    metrics = solution["physical_metrics"]
    empty_is_maximum = bool(metrics["empty_plateau_is_maximum"])
    mechanism_center_x = 0.5 * float(motion["D1"][0, 0])
    # Stroke widths are fixed in screen points. A shared absolute clearance
    # therefore protects the narrower cylinder proportionally more.
    line_clearance = 0.04 * max(cylinder_1_width, cylinder_2_width)
    piston_x1, cylinder1 = _piston_installation(
        point1, axis_y1, piston_rod_length, point1[:, 0], empty_is_maximum,
        cylinder_1_width, mechanism_center_x, line_clearance,
    )
    piston_x2, cylinder2 = _piston_installation(
        point2, axis_y2, piston_rod_length, point2[:, 0], empty_is_maximum,
        cylinder_2_width, mechanism_center_x, line_clearance,
    )

    figure, (mechanism_axis, graph_axis) = plt.subplots(
        2, 1, figsize=DEFAULT_FIGURE_SIZE, dpi=DEFAULT_DPI,
        gridspec_kw={"height_ratios": (1.0, 1.0)},
    )
    mechanism_axis.set_aspect("equal", adjustable="box")
    mechanism_axis.set_axis_off()

    cylinder_artists = []
    for cylinder in (cylinder1, cylinder2):
        top = cylinder["bottom"] + cylinder["width"]
        bottom = cylinder["bottom"]
        for x_values, y_values in (
            ((cylinder["open_x"], cylinder["back_x"]), (top, top)),
            ((cylinder["back_x"], cylinder["back_x"]), (bottom, top)),
            ((cylinder["back_x"], cylinder["open_x"]), (bottom, bottom)),
        ):
            artist, = mechanism_axis.plot(
                x_values, y_values, color=COLOR_CYLINDER,
                lw=CYLINDER_LINE_WIDTH,
            )
            cylinder_artists.append(artist)

    crank_circle = Circle(
        tuple(motion["A1"][0]), radius=float(np.linalg.norm(
            motion["B1"][0] - motion["A1"][0]
        )), fill=False, edgecolor=COLOR_FIXED, linewidth=0.8,
    )
    mechanism_axis.add_patch(crank_circle)
    crank, = mechanism_axis.plot([], [], color=COLOR_FIXED, lw=2.4)
    bc1, = mechanism_axis.plot([], [], lw=2.2)
    cd1, = mechanism_axis.plot([], [], lw=2.2)
    bc2, = mechanism_axis.plot([], [], lw=2.2)
    cd2, = mechanism_axis.plot([], [], lw=2.2)
    carrier1, other1 = (
        (cd1, bc1) if family == "E" else (bc1, cd1)
    )
    carrier2, other2 = (
        (cd2, bc2) if family == "E" else (bc2, cd2)
    )
    carrier1.set_color(COLOR_FIRST_CARRIER)
    other1.set_color(COLOR_FIRST_OTHER)
    carrier2.set_color(COLOR_SECOND_CARRIER)
    other2.set_color(COLOR_SECOND_OTHER)
    triangle1 = Polygon(
        np.zeros((3, 2)), closed=True, fill=False,
        edgecolor=COLOR_FIRST_CARRIER, lw=1.4,
    )
    triangle2 = Polygon(
        np.zeros((3, 2)), closed=True, fill=False,
        edgecolor=COLOR_SECOND_CARRIER, lw=1.4,
    )
    mechanism_axis.add_patch(triangle1)
    mechanism_axis.add_patch(triangle2)
    piston_rod1, = mechanism_axis.plot([], [], color=COLOR_PISTON, lw=1.5)
    piston_rod2, = mechanism_axis.plot([], [], color=COLOR_PISTON, lw=1.5)
    piston1, = mechanism_axis.plot(
        [], [], color=COLOR_PISTON, lw=PISTON_LINE_WIDTH,
        solid_capstyle="butt",
    )
    piston2, = mechanism_axis.plot(
        [], [], color=COLOR_PISTON, lw=PISTON_LINE_WIDTH,
        solid_capstyle="butt",
    )
    for artist in (
        *cylinder_artists[3:], bc2, cd2, triangle2, piston_rod2, piston2,
    ):
        artist.set_visible(show_symmetric)
    if show_traces:
        mechanism_axis.plot(point1[:, 0], point1[:, 1], color=COLOR_FIRST_CARRIER, alpha=0.2)
        mechanism_axis.plot(point2[:, 0], point2[:, 1], color=COLOR_SECOND_CARRIER, alpha=0.2)

    all_x = np.concatenate((
        motion["A1"][:, 0], motion["B1"][:, 0], motion["C1"][:, 0],
        point1[:, 0], point2[:, 0], piston_x1, piston_x2,
        np.array([cylinder1["open_x"], cylinder1["back_x"],
                  cylinder2["open_x"], cylinder2["back_x"]]),
    ))
    all_y = np.concatenate((
        motion["C1"][:, 1], motion["C2"][:, 1], point1[:, 1], point2[:, 1],
        np.array([cylinder1["bottom"], cylinder1["bottom"] + cylinder1["width"],
                  cylinder2["bottom"], cylinder2["bottom"] + cylinder2["width"]]),
    ))
    margin = 0.015 * max(float(np.ptp(all_x)), float(np.ptp(all_y)))
    mechanism_axis.set_xlim(float(np.min(all_x)) - margin, float(np.max(all_x)) + margin)
    mechanism_axis.set_ylim(float(np.min(all_y)) - margin, float(np.max(all_y)) + margin)

    x1 = _normalized(point1[:, 0])
    x2 = _normalized(point2[:, 0])
    graph_theta = np.r_[theta, 360.0]
    graph_x1 = np.r_[x1, x1[0]]
    graph_x2 = np.r_[x2, x2[0]]
    for value in np.arange(0.0, 1.01, 0.25):
        graph_axis.axhline(
            value, color=COLOR_GRAPH_GRID, lw=0.55, zorder=0,
        )
    for value in np.arange(0.0, 361.0, 90.0):
        graph_axis.vlines(
            value, 0.0, 1.0, color=COLOR_GRAPH_GRID, lw=0.55, zorder=0,
        )
    graph_axis.plot(
        graph_theta, graph_x1, color=COLOR_FIRST_CARRIER, lw=1.5,
    )
    if show_symmetric:
        graph_axis.plot(
            graph_theta, graph_x2, color=COLOR_SECOND_CARRIER, lw=1.5,
        )
    angle_line, = graph_axis.plot(
        (0.0, 0.0), (0.0, 1.0), color=COLOR_FIXED, lw=1.2,
    )
    graph_axis.set(xlim=(0.0, 360.0), ylim=(-0.04, 1.04))
    graph_axis.set_axis_off()

    def update(frame):
        a = motion["A1"][frame]
        b = motion["B1"][frame]
        crank.set_data((a[0], b[0]), (a[1], b[1]))
        for artist, start, end in (
            (bc1, motion["B1"][frame], motion["C1"][frame]),
            (cd1, motion["C1"][frame], motion["D1"][frame]),
            (bc2, motion["B2"][frame], motion["C2"][frame]),
            (cd2, motion["C2"][frame], motion["D2"][frame]),
        ):
            artist.set_data((start[0], end[0]), (start[1], end[1]))
        if family == "E":
            triangle1.set_xy(np.vstack((motion["C1"][frame], motion["D1"][frame], point1[frame])))
            triangle2.set_xy(np.vstack((motion["C2"][frame], motion["D2"][frame], point2[frame])))
        else:
            triangle1.set_xy(np.vstack((motion["B1"][frame], motion["C1"][frame], point1[frame])))
            triangle2.set_xy(np.vstack((motion["B2"][frame], motion["C2"][frame], point2[frame])))
        piston_rod1.set_data((point1[frame, 0], piston_x1[frame]), (point1[frame, 1], axis_y1))
        piston_rod2.set_data((point2[frame, 0], piston_x2[frame]), (point2[frame, 1], axis_y2))
        piston1.set_data((piston_x1[frame], piston_x1[frame]), (
            axis_y1 - cylinder1["piston_half_height"],
            axis_y1 + cylinder1["piston_half_height"],
        ))
        piston2.set_data((piston_x2[frame], piston_x2[frame]), (
            axis_y2 - cylinder2["piston_half_height"],
            axis_y2 + cylinder2["piston_half_height"],
        ))
        angle_line.set_xdata([theta[frame], theta[frame]])
        return (
            crank, bc1, cd1, bc2, cd2, triangle1, triangle2,
            piston_rod1, piston_rod2, piston1, piston2, angle_line,
        )

    figure.subplots_adjust(
        left=0.0, right=1.0, bottom=0.0, top=1.0, hspace=0.0,
    )
    animation = FuncAnimation(
        figure, update, frames=len(theta), interval=1000.0 / fps, blit=False,
    )
    if output is None:
        output = f"mechanism-{family}{index}.gif"
    animation.save(output, writer=PillowWriter(fps=fps), dpi=DEFAULT_DPI)
    plt.close(figure)
    return Path(output)


def main():
    parser = argparse.ArgumentParser(description="Animation GIF du mécanisme DADA")
    parser.add_argument("filename")
    parser.add_argument("family", choices=("E", "F", "e", "f"))
    parser.add_argument("index", type=int)
    parser.add_argument("--output")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--angle-step", type=float, default=DEFAULT_ANGLE_STEP_DEG)
    parser.add_argument("--piston-rod-length", type=float, default=DEFAULT_PISTON_ROD_LENGTH)
    parser.add_argument("--cylinder-1-width", type=float, default=DEFAULT_CYLINDER_1_WIDTH)
    parser.add_argument("--cylinder-2-width", type=float, default=DEFAULT_CYLINDER_2_WIDTH)
    parser.add_argument("--show-traces", action="store_true", default=DEFAULT_SHOW_TRACES)
    parser.add_argument("--hide-symmetric", action="store_false", dest="show_symmetric", default=DEFAULT_SHOW_SYMMETRIC)
    args = parser.parse_args()
    output = create_animation(
        args.filename, args.family, args.index, args.output, args.fps,
        args.angle_step, args.piston_rod_length, args.cylinder_1_width,
        args.cylinder_2_width, args.show_traces, args.show_symmetric,
    )
    print(f"GIF sauvegardé : {output}")


if __name__ == "__main__":
    main()
