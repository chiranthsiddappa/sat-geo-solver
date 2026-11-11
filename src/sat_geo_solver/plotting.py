"""
Plotting and visualization utilities for satellite geolocation.
"""

from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pymap3d as pm

from sat_geo_solver.models import ErrorEllipse, GeolocationResult


def plot_error_ellipse(
    result: GeolocationResult,
    ax: Optional[plt.Axes] = None,
    true_pos: Optional[Sequence[float]] = None,
    ref_pos: Optional[Sequence[float]] = None,
    show_axes: bool = True,
    title: str = "Emitter Location & Error Ellipse (95% Confidence)",
) -> plt.Axes:
    """
    Plot the estimated emitter location and its 2D horizontal error ellipse.

    :param result: GeolocationResult returned by GeolocationSolver
    :param ax: Matplotlib axes (optional)
    :param true_pos: Optional ground truth position [lat, lon, alt]
    :param ref_pos: Optional reference station position [lat, lon, alt]
    :param show_axes: Whether to draw the semi-major and semi-minor axis vectors
    :param title: Plot title
    :return: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ee = result.error_ellipse

    # Generate ellipse boundary in local ENU (meters)
    east_m, north_m = ee.to_enu_points(n_points=200)

    # Plot ellipse in local coordinates centered at (0, 0)
    ax.plot(east_m, north_m, 'b-', lw=2, label=f"Error Ellipse ({ee.confidence*100:.0f}%)")
    ax.fill(east_m, north_m, 'b', alpha=0.15)
    ax.plot(0, 0, 'b*', markersize=14, label=f"Estimated Emitter ({result.lat:.4f}°, {result.lon:.4f}°)")

    # Plot True Position if provided
    if true_pos is not None:
        e_true, n_true, _ = pm.geodetic2enu(
            true_pos[0], true_pos[1], true_pos[2] if len(true_pos) > 2 else 0.0,
            result.lat, result.lon, result.alt,
            ell=pm.Ellipsoid.from_name(result.ellipsoid) if isinstance(result.ellipsoid, str) else result.ellipsoid
        )
        ax.plot(e_true, n_true, 'r^', markersize=12, label=f"True Emitter ({true_pos[0]:.4f}°, {true_pos[1]:.4f}°)")

    # Draw semi-major and semi-minor axes
    if show_axes:
        angle_rad = np.radians(ee.angle_deg)
        major_e = ee.semi_major_m * np.cos(angle_rad)
        major_n = ee.semi_major_m * np.sin(angle_rad)
        minor_e = -ee.semi_minor_m * np.sin(angle_rad)
        minor_n = ee.semi_minor_m * np.cos(angle_rad)

        ax.plot([-major_e, major_e], [-major_n, major_n], 'k--', lw=1.5,
                label=f"Semi-Major ({ee.semi_major_m:.1f} m, Az={ee.azimuth_deg:.1f}°)")
        ax.plot([-minor_e, minor_e], [-minor_n, minor_n], 'g--', lw=1.5,
                label=f"Semi-Minor ({ee.semi_minor_m:.1f} m)")

    ax.set_xlabel("East Offset (m)", fontsize=11)
    ax.set_ylabel("North Offset (m)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal')
    ax.legend(loc='best', fontsize=9)

    return ax


def plot_convergence_history(
    result: GeolocationResult,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the step size and residual norm reduction across solver iterations.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    iters = [step.iteration for step in result.history]
    steps = [step.step_m for step in result.history]
    res_norms = [step.residual_norm for step in result.history]

    ax.semilogy(iters[1:], steps[1:], 'o-', color='tab:blue', lw=2, label="Step Norm (m)")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Step Size (m)", color='tab:blue', fontsize=11)
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax.twinx()
    ax2.semilogy(iters, res_norms, 's--', color='tab:red', lw=2, label="Residual RMS")
    ax2.set_ylabel("Residual RMS", color='tab:red', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax.set_title(f"Solver Convergence Path ({result.iterations} iterations)", fontweight='bold')

    return ax


def plot_fidelity_sensitivity(
    two_sat,
    target_pos: Sequence[float],
    ref_pos: Sequence[float],
    uplink: float,
    translation_frequency: float,
    sigma_t_range: Sequence[float],
    sigma_f_range: Sequence[float],
    altitude: float = 0.0,
    confidence: float = 0.95,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Analyze and plot how time fidelity (sigma_t in seconds) and frequency fidelity (sigma_f in Hz)
    affect the resulting horizontal error ellipse semi-major and semi-minor axes in meters.
    """
    from sat_geo_solver.solver import GeolocationSolver

    dto_tar = float(two_sat.dto(target_pos))
    dto_ref = float(two_sat.dto(ref_pos))
    tdoa = dto_tar - dto_ref

    dfo_tar = float(two_sat.dfo(target_pos, uplink, translation_frequency))
    dfo_ref = float(two_sat.dfo(ref_pos, uplink, translation_frequency))
    fdoa = dfo_tar - dfo_ref

    solver = GeolocationSolver(max_iterations=1)

    # 1. Sweep sigma_t with fixed baseline sigma_f
    fixed_sigma_f = 0.5  # Hz
    major_t = []
    minor_t = []
    for st in sigma_t_range:
        res = solver.solve_from_twosat(
            two_sat=two_sat,
            tdoa=tdoa,
            fdoa=fdoa,
            uplink=uplink,
            translation_frequency=translation_frequency,
            ref_emitter=ref_pos,
            error_fidelity=(st, fixed_sigma_f),
            altitude=altitude,
            initial_guess=target_pos,
            confidence=confidence,
        )
        major_t.append(res.error_ellipse.semi_major_m)
        minor_t.append(res.error_ellipse.semi_minor_m)

    # 2. Sweep sigma_f with fixed baseline sigma_t
    fixed_sigma_t = 1e-6  # 1 microsecond
    major_f = []
    minor_f = []
    for sf in sigma_f_range:
        res = solver.solve_from_twosat(
            two_sat=two_sat,
            tdoa=tdoa,
            fdoa=fdoa,
            uplink=uplink,
            translation_frequency=translation_frequency,
            ref_emitter=ref_pos,
            error_fidelity=(fixed_sigma_t, sf),
            altitude=altitude,
            initial_guess=target_pos,
            confidence=confidence,
        )
        major_f.append(res.error_ellipse.semi_major_m)
        minor_f.append(res.error_ellipse.semi_minor_m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    st_micro = np.array(sigma_t_range) * 1e6
    ax1.plot(st_micro, major_t, 'b-o', lw=2, label="Semi-Major Axis (m)")
    ax1.plot(st_micro, minor_t, 'g-s', lw=2, label="Semi-Minor Axis (m)")
    ax1.set_xlabel("Time Fidelity $\\sigma_t$ (µs)", fontsize=11)
    ax1.set_ylabel("Error Ellipse Semi-Axes (m)", fontsize=11)
    ax1.set_title(f"Sensitivity to TDOA Fidelity (fixed $\\sigma_f$ = {fixed_sigma_f} Hz)", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(sigma_f_range, major_f, 'b-o', lw=2, label="Semi-Major Axis (m)")
    ax2.plot(sigma_f_range, minor_f, 'g-s', lw=2, label="Semi-Minor Axis (m)")
    ax2.set_xlabel("Frequency Fidelity $\\sigma_f$ (Hz)", fontsize=11)
    ax2.set_ylabel("Error Ellipse Semi-Axes (m)", fontsize=11)
    ax2.set_title(f"Sensitivity to FDOA Fidelity (fixed $\\sigma_t$ = {fixed_sigma_t*1e6:.1f} µs)", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    return fig, (ax1, ax2)
