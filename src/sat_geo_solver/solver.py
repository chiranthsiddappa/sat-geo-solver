"""
Satellite Geolocation Solver for TDOA and FDOA measurements.

Supports:
  - Combined TDOA + FDOA pairs
  - Sequences of TDOA-only measurements (multilateration / time-difference curves)
  - Sequences of FDOA-only measurements (multi-epoch or multi-satellite Doppler curves)
  - Single or multiple reference emitters for transponder and downlink bias calibration
  - State vector error propagation for CAF fidelity (seconds, Hz) and satellite position/velocity errors
  - Multiple WGS and geodetic ellipsoids via pymap3d (WGS-84, GRS-80, IERS-2003, etc.)
  - Constrained altitude surfaces or 3D unconstrained solving
  - Error ellipse parameters (semi-major, semi-minor, azimuth, and quadratic equation constants A, B, C, F)
  - 3D Error ellipsoids and Spherical Error Probable (SEP) metrics
  - Fully exposed iteration parameters (max_iterations, tol, damping, initial_guess)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pymap3d as pm
from skyfield.constants import C as SPEED_OF_LIGHT

from sat_geo_solver.models import (
    ErrorEllipse,
    ErrorEllipsoid,
    GeolocationResult,
    IterationStep,
    Observation,
    SatelliteState,
)
from sat_geo_solver.observe import (
    Observe,
    distance_to,
    doppler,
    dt_to_ts,
    light_seconds,
    range_and_rate,
    relativistic_doppler,
)


def predict_tdoa(
    sat1_pos: np.ndarray,
    sat2_pos: np.ndarray,
    emitter_pos: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute theoretical differential time of arrival (Secondary - Primary) in seconds:
        tau = (||s2 - x|| - ||s1 - x||) / c

    :param sat1_pos: Primary satellite ECEF position in meters [x, y, z]
    :param sat2_pos: Secondary satellite ECEF position in meters [x, y, z]
    :param emitter_pos: Emitter ECEF position in meters [x, y, z]
    :return: (tau_sec, range1_m, range2_m)
    """
    diff1 = sat1_pos - emitter_pos
    diff2 = sat2_pos - emitter_pos
    r1 = float(np.linalg.norm(diff1))
    r2 = float(np.linalg.norm(diff2))
    tau = (r2 - r1) / SPEED_OF_LIGHT
    return tau, r1, r2


def predict_fdoa(
    sat1_pos: np.ndarray,
    sat1_vel: np.ndarray,
    sat2_pos: np.ndarray,
    sat2_vel: np.ndarray,
    emitter_pos: np.ndarray,
    center_freq: float,
) -> Tuple[float, float, float]:
    """
    Compute theoretical differential frequency of arrival (Secondary - Primary) in Hz:
        rr1 = u1 . v1,  rr2 = u2 . v2
        fdoa = - (center_freq / c) * (rr2 - rr1)

    :param sat1_pos: Primary satellite ECEF position in meters [x, y, z]
    :param sat1_vel: Primary satellite ECEF velocity in m/s [vx, vy, vz]
    :param sat2_pos: Secondary satellite ECEF position in meters [x, y, z]
    :param sat2_vel: Secondary satellite ECEF velocity in m/s [vx, vy, vz]
    :param emitter_pos: Emitter ECEF position in meters [x, y, z]
    :param center_freq: Nominal uplink center frequency in Hz
    :return: (fdoa_hz, range_rate1_mps, range_rate2_mps)
    """
    diff1 = sat1_pos - emitter_pos
    diff2 = sat2_pos - emitter_pos
    r1 = np.linalg.norm(diff1)
    r2 = np.linalg.norm(diff2)

    u1 = diff1 / r1
    u2 = diff2 / r2

    # Line of sight range rate (positive receding)
    rr1 = float(np.dot(u1, sat1_vel))
    rr2 = float(np.dot(u2, sat2_vel))

    # Re-use Doppler calculations from observe.py
    f1_rx = doppler(center_freq, rr1)
    f2_rx = doppler(center_freq, rr2)
    fdoa = float(f2_rx - f1_rx)
    return fdoa, rr1, rr2


def gradient_tdoa_ecef(
    sat1_pos: np.ndarray,
    sat2_pos: np.ndarray,
    emitter_pos: np.ndarray,
) -> np.ndarray:
    """
    Analytical gradient of TDOA (Secondary - Primary) with respect to emitter position x:
        grad_x tau = (u1 - u2) / c
    where u_i = (s_i - x) / ||s_i - x||.
    """
    diff1 = sat1_pos - emitter_pos
    diff2 = sat2_pos - emitter_pos
    u1 = diff1 / np.linalg.norm(diff1)
    u2 = diff2 / np.linalg.norm(diff2)
    return (u1 - u2) / SPEED_OF_LIGHT


def gradient_fdoa_ecef(
    sat1_pos: np.ndarray,
    sat1_vel: np.ndarray,
    sat2_pos: np.ndarray,
    sat2_vel: np.ndarray,
    emitter_pos: np.ndarray,
    center_freq: float,
) -> np.ndarray:
    """
    Analytical gradient of FDOA (Secondary - Primary) with respect to emitter position x:
        grad_x fdoa = (f0 / c) * ( v2_perp / r2 - v1_perp / r1 )
    where v_i_perp = v_i - (u_i . v_i) * u_i.
    """
    diff1 = sat1_pos - emitter_pos
    diff2 = sat2_pos - emitter_pos
    r1 = np.linalg.norm(diff1)
    r2 = np.linalg.norm(diff2)

    u1 = diff1 / r1
    u2 = diff2 / r2

    v1_perp = sat1_vel - np.dot(u1, sat1_vel) * u1
    v2_perp = sat2_vel - np.dot(u2, sat2_vel) * u2

    grad = (center_freq / SPEED_OF_LIGHT) * (v2_perp / r2 - v1_perp / r1)
    return grad


def compute_local_enu_basis(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute orthonormal East, North, Up unit vectors in ECEF coordinates at given geodetic coordinates.
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)

    east = np.array([-np.sin(lam), np.cos(lam), 0.0])
    north = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
    up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])

    return east, north, up


class GeolocationSolver:
    """
    Non-linear least squares solver for satellite emitter geolocation using TDOA and/or FDOA.

    :param ellipsoid: Ellipsoid model name or pymap3d.Ellipsoid instance (default: 'wgs84')
    :param max_iterations: Maximum optimization iterations (default: 50)
    :param tol: Convergence tolerance on step norm in meters (default: 1e-4)
    :param damping: Initial Levenberg-Marquardt damping factor (default: 1e-3)
    """

    def __init__(
        self,
        ellipsoid: Any = 'wgs84',
        max_iterations: int = 50,
        tol: float = 1e-4,
        damping: float = 1e-3,
    ):
        if isinstance(ellipsoid, str):
            self.ellipsoid = pm.Ellipsoid.from_name(ellipsoid)
            self.ellipsoid_name = ellipsoid
        elif hasattr(ellipsoid, 'name'):
            self.ellipsoid = ellipsoid
            self.ellipsoid_name = ellipsoid.name
        else:
            self.ellipsoid = ellipsoid
            self.ellipsoid_name = str(ellipsoid)

        self.max_iterations = max_iterations
        self.tol = tol
        self.damping = damping

    def _effective_variance(
        self,
        obs: Observation,
        emitter_ecef: np.ndarray,
    ) -> float:
        """
        Compute total measurement variance by propagating CAF peak fidelity and satellite state errors:
            var_total = var_CAF + var_sat_pos + var_sat_vel + var_ref
        """
        var_total = obs.sigma**2

        # Satellite 1 and 2 state covariances
        diff1 = obs.sat1.pos_m - emitter_ecef
        diff2 = obs.sat2.pos_m - emitter_ecef
        r1 = np.linalg.norm(diff1)
        r2 = np.linalg.norm(diff2)
        u1 = diff1 / r1
        u2 = diff2 / r2

        if obs.kind == 'tdoa':
            # Sensitivity to sat1 and sat2 positions: -u1/c and u2/c
            if obs.sat1.pos_cov is not None:
                var_total += float((u1 @ obs.sat1.pos_cov @ u1) / (SPEED_OF_LIGHT**2))
            if obs.sat2.pos_cov is not None:
                var_total += float((u2 @ obs.sat2.pos_cov @ u2) / (SPEED_OF_LIGHT**2))
        elif obs.kind == 'fdoa':
            f0_over_c = obs.center_freq / SPEED_OF_LIGHT
            v1_perp = obs.sat1.vel_mps - np.dot(u1, obs.sat1.vel_mps) * u1
            v2_perp = obs.sat2.vel_mps - np.dot(u2, obs.sat2.vel_mps) * u2

            # Position error contribution
            if obs.sat1.pos_cov is not None:
                g_pos1 = (f0_over_c / r1) * v1_perp
                var_total += float(g_pos1 @ obs.sat1.pos_cov @ g_pos1)
            if obs.sat2.pos_cov is not None:
                g_pos2 = -(f0_over_c / r2) * v2_perp
                var_total += float(g_pos2 @ obs.sat2.pos_cov @ g_pos2)

            # Velocity error contribution
            if obs.sat1.vel_cov is not None:
                g_vel1 = -f0_over_c * u1
                var_total += float(g_vel1 @ obs.sat1.vel_cov @ g_vel1)
            if obs.sat2.vel_cov is not None:
                g_vel2 = f0_over_c * u2
                var_total += float(g_vel2 @ obs.sat2.vel_cov @ g_vel2)

        if obs.ref_sigma is not None and obs.ref_sigma > 0:
            var_total += obs.ref_sigma**2

        return var_total

    def _calibrate_reference_biases(
        self,
        observations: Sequence[Observation],
        ref_emitters: Optional[Sequence[Any]] = None,
        ref_measurements: Optional[Sequence[Dict[str, float]]] = None,
    ) -> List[Observation]:
        """
        If multiple reference emitters are provided, compute the optimal weighted bias
        and return calibrated observations.
        """
        if not ref_emitters:
            return list(observations)

        # Standardize reference emitters list
        ref_list = []
        for ref in ref_emitters:
            ref_arr = np.asarray(ref, dtype=float)
            ref_list.append(ref_arr)

        calibrated_obs = []
        for obs in observations:
            # If obs already has ref_emitter set and no external list, keep it
            if obs.ref_emitter is not None and not ref_measurements:
                calibrated_obs.append(obs)
                continue

            # Compute theoretical reference values for all references
            theo_refs = []
            weights = []
            biases = []

            for i, ref_ll in enumerate(ref_list):
                alt = ref_ll[2] if len(ref_ll) > 2 else 0.0
                ref_ecef = np.array(pm.geodetic2ecef(ref_ll[0], ref_ll[1], alt, ell=self.ellipsoid))

                if obs.kind == 'tdoa':
                    theo_val, _, _ = predict_tdoa(obs.sat1.pos_m, obs.sat2.pos_m, ref_ecef)
                else:
                    theo_val, _, _ = predict_fdoa(
                        obs.sat1.pos_m, obs.sat1.vel_mps,
                        obs.sat2.pos_m, obs.sat2.vel_mps,
                        ref_ecef, obs.center_freq
                    )

                theo_refs.append(theo_val)

                if ref_measurements and i < len(ref_measurements):
                    meas_val = ref_measurements[i].get(obs.kind)
                    meas_sig = ref_measurements[i].get('sigma', obs.sigma)
                    if meas_val is not None:
                        w = 1.0 / (meas_sig**2)
                        weights.append(w)
                        biases.append(meas_val - theo_val)

            if biases and weights:
                # Weighted average bias
                w_arr = np.array(weights)
                bias_arr = np.array(biases)
                mean_bias = float(np.sum(w_arr * bias_arr) / np.sum(w_arr))
                bias_var = float(1.0 / np.sum(w_arr))

                # Calibrated observation: target_calibrated = target_measured - mean_bias
                new_obs = Observation(
                    kind=obs.kind,
                    value=obs.value - mean_bias,
                    sat1=obs.sat1,
                    sat2=obs.sat2,
                    center_freq=obs.center_freq,
                    sigma=obs.sigma,
                    ref_emitter=None,
                    ref_sigma=np.sqrt(bias_var),
                    time=obs.time,
                )
                calibrated_obs.append(new_obs)
            else:
                # Default: use first reference emitter directly
                new_obs = Observation(
                    kind=obs.kind,
                    value=obs.value,
                    sat1=obs.sat1,
                    sat2=obs.sat2,
                    center_freq=obs.center_freq,
                    sigma=obs.sigma,
                    ref_emitter=ref_list[0],
                    ref_sigma=obs.ref_sigma,
                    time=obs.time,
                )
                calibrated_obs.append(new_obs)

        return calibrated_obs

    def _evaluate_residuals_and_jacobian(
        self,
        observations: Sequence[Observation],
        curr_geodetic: np.ndarray,
        is_3d: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate measurement residuals, Jacobian matrix in local ENU frame, and weight vector.
        """
        lat, lon, alt = curr_geodetic
        emitter_ecef = np.array(pm.geodetic2ecef(lat, lon, alt, ell=self.ellipsoid))

        east, north, up = compute_local_enu_basis(lat, lon)
        if is_3d:
            basis = np.column_stack([east, north, up])  # (3, 3)
            n_dim = 3
        else:
            basis = np.column_stack([east, north])       # (3, 2)
            n_dim = 2

        n_obs = len(observations)
        residuals = np.zeros(n_obs)
        weights = np.zeros(n_obs)
        jacobian = np.zeros((n_obs, n_dim))

        for i, obs in enumerate(observations):
            # 1. Forward model prediction for emitter
            if obs.kind == 'tdoa':
                pred_target, _, _ = predict_tdoa(obs.sat1.pos_m, obs.sat2.pos_m, emitter_ecef)
                grad_ecef = gradient_tdoa_ecef(obs.sat1.pos_m, obs.sat2.pos_m, emitter_ecef)
            else:
                pred_target, _, _ = predict_fdoa(
                    obs.sat1.pos_m, obs.sat1.vel_mps,
                    obs.sat2.pos_m, obs.sat2.vel_mps,
                    emitter_ecef, obs.center_freq,
                )
                grad_ecef = gradient_fdoa_ecef(
                    obs.sat1.pos_m, obs.sat1.vel_mps,
                    obs.sat2.pos_m, obs.sat2.vel_mps,
                    emitter_ecef, obs.center_freq,
                )

            # 2. Reference transmitter offset (if differential)
            if obs.ref_emitter is not None:
                ref_ll = obs.ref_emitter
                ref_alt = ref_ll[2] if len(ref_ll) > 2 else 0.0
                ref_ecef = np.array(pm.geodetic2ecef(ref_ll[0], ref_ll[1], ref_alt, ell=self.ellipsoid))

                if obs.kind == 'tdoa':
                    pred_ref, _, _ = predict_tdoa(obs.sat1.pos_m, obs.sat2.pos_m, ref_ecef)
                else:
                    pred_ref, _, _ = predict_fdoa(
                        obs.sat1.pos_m, obs.sat1.vel_mps,
                        obs.sat2.pos_m, obs.sat2.vel_mps,
                        ref_ecef, obs.center_freq,
                    )
                pred_total = pred_target - pred_ref
            else:
                pred_total = pred_target

            # Residual: measured - predicted
            residuals[i] = obs.value - pred_total

            # Local ENU Jacobian: d(pred)/d(enu) = grad_ecef @ basis
            # Residual is (y - pred), so d(residual)/d(enu) = - J
            jacobian[i, :] = grad_ecef @ basis

            # Measurement variance
            var = self._effective_variance(obs, emitter_ecef)
            weights[i] = 1.0 / var

        return residuals, jacobian, weights

    def _coarse_grid_search_obs(
        self,
        observations: Sequence[Observation],
        altitude: float = 0.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        grid_step: float = 3.0,
    ) -> np.ndarray:
        """
        Scan a 2D geographic grid over candidate coordinates and return the point
        minimizing the weighted residual sum of squares.

        :param observations: List of Observation objects
        :param altitude: Elevation in meters
        :param bounds: (min_lat, max_lat, min_lon, max_lon) in degrees, or None for auto bounds
        :param grid_step: Grid sampling step in degrees
        :return: [best_lat, best_lon, altitude]
        """
        if bounds is not None:
            min_lat, max_lat, min_lon, max_lon = bounds
        else:
            # Automatic bounds based on sub-satellite points
            s1_lla = pm.ecef2geodetic(*observations[0].sat1.pos_m, ell=self.ellipsoid)
            s2_lla = pm.ecef2geodetic(*observations[0].sat2.pos_m, ell=self.ellipsoid)
            mean_lon = (s1_lla[1] + s2_lla[1]) / 2.0
            min_lat, max_lat = -60.0, 60.0
            min_lon, max_lon = mean_lon - 65.0, mean_lon + 65.0

        lats = np.arange(min_lat, max_lat + 0.1, grid_step)
        lons = np.arange(min_lon, max_lon + 0.1, grid_step)

        best_cost = float('inf')
        best_pos = np.array([(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0, altitude])

        for lat in lats:
            for lon in lons:
                pos = np.array([lat, lon, altitude])
                residuals, _, W = self._evaluate_residuals_and_jacobian(observations, pos, is_3d=False)
                cost = float(np.sum(W * residuals**2))
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos

        return best_pos

    def solve(
        self,
        observations: Sequence[Observation],
        initial_guess: Optional[Sequence[float]] = None,
        altitude: Optional[float] = 0.0,
        ref_emitters: Optional[Sequence[Any]] = None,
        ref_measurements: Optional[Sequence[Dict[str, float]]] = None,
        confidence: float = 0.95,
        coarse_grid_search: bool = False,
        coarse_grid_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> GeolocationResult:
        """
        Solve for emitter position from TDOA and/or FDOA observations.

        :param observations: Sequence of Observation objects
        :param initial_guess: Initial geodetic position [lat, lon] or [lat, lon, alt]
        :param altitude: Target altitude in meters (if float, constrained to this altitude;
                         if None, solves for 3D position including altitude)
        :param ref_emitters: Optional sequence of known reference transmitter positions
        :param ref_measurements: Optional sequence of reference measurement dicts
        :param confidence: Probability level for error ellipse / ellipsoid (default: 0.95)
        :param coarse_grid_search: If True, executes a coarse 2D grid search to find the global
                                   basin of attraction before running least squares optimization
        :param coarse_grid_bounds: Optional (min_lat, max_lat, min_lon, max_lon) bounding box for grid search
        :return: GeolocationResult
        """
        if not observations:
            raise ValueError("No observations provided to GeolocationSolver.")

        is_3d = altitude is None
        min_obs = 3 if is_3d else 2
        if len(observations) < min_obs:
            raise ValueError(
                f"At least {min_obs} independent observations are required to solve "
                f"{'in 3D' if is_3d else 'on the ellipsoidal surface'}, got {len(observations)}."
            )

        # Calibrate references if provided
        obs_list = self._calibrate_reference_biases(observations, ref_emitters, ref_measurements)

        # Set initial guess
        if initial_guess is not None and not coarse_grid_search:
            init_arr = np.asarray(initial_guess, dtype=float)
            if len(init_arr) == 2:
                curr_alt = 0.0 if altitude is None else float(altitude)
                curr_pos = np.array([init_arr[0], init_arr[1], curr_alt])
            else:
                curr_pos = init_arr[:3].copy()
                if altitude is not None:
                    curr_pos[2] = float(altitude)
        elif coarse_grid_search:
            curr_alt = 0.0 if altitude is None else float(altitude)
            curr_pos = self._coarse_grid_search_obs(
                obs_list, altitude=curr_alt, bounds=coarse_grid_bounds
            )
        else:
            # Intelligent initial guess: use reference emitter or centroid of sub-satellite points
            if obs_list[0].ref_emitter is not None:
                ref = obs_list[0].ref_emitter
                ref_alt = ref[2] if len(ref) > 2 else (0.0 if altitude is None else float(altitude))
                curr_pos = np.array([ref[0], ref[1], ref_alt])
            else:
                # Average of sub-satellite points
                s1_lla = pm.ecef2geodetic(*obs_list[0].sat1.pos_m, ell=self.ellipsoid)
                s2_lla = pm.ecef2geodetic(*obs_list[0].sat2.pos_m, ell=self.ellipsoid)
                mean_lat = (s1_lla[0] + s2_lla[0]) / 2.0
                mean_lon = (s1_lla[1] + s2_lla[1]) / 2.0
                curr_alt = 0.0 if altitude is None else float(altitude)
                curr_pos = np.array([mean_lat, mean_lon, curr_alt])

        damping = self.damping
        converged = False
        history: List[IterationStep] = []

        residuals, J, W = self._evaluate_residuals_and_jacobian(obs_list, curr_pos, is_3d)
        curr_cost = float(np.sum(W * residuals**2))

        history.append(IterationStep(
            iteration=0,
            lat=float(curr_pos[0]),
            lon=float(curr_pos[1]),
            alt=float(curr_pos[2]),
            step_m=0.0,
            residual_norm=float(np.sqrt(np.mean(residuals**2))),
        ))

        for it in range(1, self.max_iterations + 1):
            # Normal equations with Levenberg-Marquardt damping:
            # (J^T W J + lambda * diag(J^T W J)) delta = J^T W r
            W_mat = np.diag(W)
            H = J.T @ W_mat @ J
            g = J.T @ (W * residuals)

            diag_H = np.diag(np.diag(H))
            if np.all(np.diag(diag_H) == 0):
                diag_H = np.eye(H.shape[0])

            H_damped = H + damping * diag_H

            try:
                delta = np.linalg.solve(H_damped, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H_damped, g, rcond=None)[0]

            step_m = float(np.linalg.norm(delta))

            # Propose updated position
            de = delta[0]
            dn = delta[1]
            du = delta[2] if is_3d else 0.0

            cand_lat, cand_lon, cand_alt = pm.enu2geodetic(
                de, dn, du, curr_pos[0], curr_pos[1], curr_pos[2], ell=self.ellipsoid
            )
            if not is_3d and altitude is not None:
                cand_alt = float(altitude)

            cand_pos = np.array([cand_lat, cand_lon, cand_alt])
            cand_residuals, cand_J, cand_W = self._evaluate_residuals_and_jacobian(obs_list, cand_pos, is_3d)
            cand_cost = float(np.sum(cand_W * cand_residuals**2))

            # Accept or reject step
            if cand_cost < curr_cost:
                curr_pos = cand_pos
                residuals = cand_residuals
                J = cand_J
                W = cand_W
                curr_cost = cand_cost
                damping = max(damping * 0.2, 1e-10)

                history.append(IterationStep(
                    iteration=it,
                    lat=float(curr_pos[0]),
                    lon=float(curr_pos[1]),
                    alt=float(curr_pos[2]),
                    step_m=step_m,
                    residual_norm=float(np.sqrt(np.mean(residuals**2))),
                ))

                if step_m < self.tol:
                    converged = True
                    break
            else:
                damping = min(damping * 5.0, 1e10)
                history.append(IterationStep(
                    iteration=it,
                    lat=float(curr_pos[0]),
                    lon=float(curr_pos[1]),
                    alt=float(curr_pos[2]),
                    step_m=step_m,
                    residual_norm=float(np.sqrt(np.mean(residuals**2))),
                ))
                if damping >= 1e10:
                    break

        # Final covariance computation: Cov_enu = (J^T W J)^-1
        W_mat = np.diag(W)
        H_final = J.T @ W_mat @ J
        try:
            cov_enu = np.linalg.inv(H_final)
        except np.linalg.LinAlgError:
            cov_enu = np.linalg.pinv(H_final)

        # GDOP computation: sqrt(trace((J^T J)^-1))
        JTJ = J.T @ J
        try:
            inv_JTJ = np.linalg.inv(JTJ)
            gdop = float(np.sqrt(np.trace(inv_JTJ)))
        except np.linalg.LinAlgError:
            gdop = float(np.sqrt(np.trace(np.linalg.pinv(JTJ))))

        # Horizontal Error Ellipse
        cov_horiz = cov_enu[0:2, 0:2]
        error_ellipse = ErrorEllipse.from_covariance_2d(cov_horiz, confidence=confidence)

        # 3D Error Ellipsoid
        error_ellipsoid = ErrorEllipsoid.from_covariance_3d(cov_enu, confidence=confidence) if is_3d else None

        final_ecef = np.array(pm.geodetic2ecef(curr_pos[0], curr_pos[1], curr_pos[2], ell=self.ellipsoid))

        msg = "Converged successfully." if converged else "Reached iteration limit or damping maximum."

        return GeolocationResult(
            lat=float(curr_pos[0]),
            lon=float(curr_pos[1]),
            alt=float(curr_pos[2]),
            ecef=final_ecef,
            converged=converged,
            iterations=len(history) - 1,
            residuals=residuals,
            cost=curr_cost,
            cov_enu=cov_enu,
            error_ellipse=error_ellipse,
            error_ellipsoid=error_ellipsoid,
            history=history,
            gdop=gdop,
            message=msg,
            ellipsoid=self.ellipsoid_name,
        )

    def solve_from_twosat(
        self,
        two_sat,
        tdoa: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        fdoa: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        uplink: Optional[float] = None,
        translation_frequency: float = 0.0,
        ref_emitter: Optional[Sequence[Any]] = None,
        error_fidelity: Optional[Union[Tuple[float, float], float, Dict[str, float]]] = None,
        sat1_cov: Optional[Dict[str, Any]] = None,
        sat2_cov: Optional[Dict[str, Any]] = None,
        initial_guess: Optional[Sequence[float]] = None,
        altitude: Optional[float] = 0.0,
        confidence: float = 0.95,
        use_twosat_model: bool = True,
        coarse_grid_search: bool = False,
        coarse_grid_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> GeolocationResult:
        """
        High-level solver integration for a Skyfield TwoSat scenario.

        :param two_sat: TwoSat instance from sat_geo_solver.scenarios
        :param tdoa: Measured TDOA value(s) in seconds (Secondary - Primary)
        :param fdoa: Measured FDOA value(s) in Hz (Secondary - Primary)
        :param uplink: Uplink center frequency in Hz (required if FDOA is provided)
        :param translation_frequency: Satellite transponder translation frequency in Hz
        :param ref_emitter: Known reference transmitter position [lat, lon, alt] or list of positions
        :param error_fidelity: CAF peak fidelity:
                               - tuple: (sigma_tdoa, sigma_fdoa)
                               - scalar: applied to provided measurement type
                               - dict: {'tdoa': sigma_t, 'fdoa': sigma_f}
        :param sat1_cov: Optional dict with 'pos_cov' and/or 'vel_cov' for primary satellite
        :param sat2_cov: Optional dict with 'pos_cov' and/or 'vel_cov' for secondary satellite
        :param initial_guess: Initial guess [lat, lon] or [lat, lon, alt]
        :param altitude: Constrained altitude in meters (default: 0.0; set to None for 3D)
        :param confidence: Probability level for error ellipse / ellipsoid (default: 0.95)
        :param use_twosat_model: If True, evaluates directly via TwoSat forward methods (recommended for TwoSat)
        :param coarse_grid_search: If True, executes a coarse 2D grid search before non-linear solving
        :param coarse_grid_bounds: Optional (min_lat, max_lat, min_lon, max_lon) for grid search
        :return: GeolocationResult
        """
        # Parse error fidelity
        sigma_t = 1e-6  # Default 1 microsecond
        sigma_f = 0.5   # Default 0.5 Hz
        if error_fidelity is not None:
            if isinstance(error_fidelity, tuple) and len(error_fidelity) == 2:
                sigma_t, sigma_f = float(error_fidelity[0]), float(error_fidelity[1])
            elif isinstance(error_fidelity, dict):
                sigma_t = float(error_fidelity.get('tdoa', sigma_t))
                sigma_f = float(error_fidelity.get('fdoa', sigma_f))
            elif isinstance(error_fidelity, (int, float)):
                sigma_t = float(error_fidelity)
                sigma_f = float(error_fidelity)

        # Standardize reference emitters
        ref_list = []
        if ref_emitter is not None:
            ref_arr = np.asarray(ref_emitter, dtype=float)
            if ref_arr.ndim == 1:
                ref_list = [ref_arr]
            elif ref_arr.ndim == 2:
                ref_list = [row for row in ref_arr]

        if use_twosat_model:
            return self._solve_twosat_direct(
                two_sat=two_sat,
                tdoa=tdoa,
                fdoa=fdoa,
                uplink=uplink,
                translation_frequency=translation_frequency,
                ref_list=ref_list,
                sigma_t=sigma_t,
                sigma_f=sigma_f,
                sat1_cov=sat1_cov,
                sat2_cov=sat2_cov,
                initial_guess=initial_guess,
                altitude=altitude,
                confidence=confidence,
                coarse_grid_search=coarse_grid_search,
                coarse_grid_bounds=coarse_grid_bounds,
            )

        # Fallback to ECEF Observation objects
        at = two_sat.at
        epochs = list(at) if isinstance(at, (list, tuple, np.ndarray)) else [at]

        pos1_cov = sat1_cov.get('pos_cov') if sat1_cov else None
        vel1_cov = sat1_cov.get('vel_cov') if sat1_cov else None
        pos2_cov = sat2_cov.get('pos_cov') if sat2_cov else None
        vel2_cov = sat2_cov.get('vel_cov') if sat2_cov else None

        observations: List[Observation] = []

        if tdoa is not None:
            tdoa_arr = np.atleast_1d(tdoa)
            for i, epoch in enumerate(epochs):
                if i >= len(tdoa_arr):
                    break
                val = float(tdoa_arr[i])
                obs1 = Observe(two_sat.primary.sat, epoch)
                obs2 = Observe(two_sat.secondary.sat, epoch)
                sat1_st = SatelliteState.from_observe(obs1, pos_cov=pos1_cov, vel_cov=vel1_cov)
                sat2_st = SatelliteState.from_observe(obs2, pos_cov=pos2_cov, vel_cov=vel2_cov)

                obs = Observation(
                    kind='tdoa',
                    value=val,
                    sat1=sat1_st,
                    sat2=sat2_st,
                    center_freq=0.0,
                    sigma=sigma_t,
                    ref_emitter=ref_list[0] if ref_list else None,
                    time=epoch,
                )
                observations.append(obs)

        if fdoa is not None:
            if uplink is None:
                raise ValueError("uplink frequency must be provided for FDOA calculations.")
            fdoa_arr = np.atleast_1d(fdoa)
            for i, epoch in enumerate(epochs):
                if i >= len(fdoa_arr):
                    break
                val = float(fdoa_arr[i])
                obs1 = Observe(two_sat.primary.sat, epoch)
                obs2 = Observe(two_sat.secondary.sat, epoch)
                sat1_st = SatelliteState.from_observe(obs1, pos_cov=pos1_cov, vel_cov=vel1_cov)
                sat2_st = SatelliteState.from_observe(obs2, pos_cov=pos2_cov, vel_cov=vel2_cov)

                obs = Observation(
                    kind='fdoa',
                    value=val,
                    sat1=sat1_st,
                    sat2=sat2_st,
                    center_freq=uplink,
                    sigma=sigma_f,
                    ref_emitter=ref_list[0] if ref_list else None,
                    time=epoch,
                )
                observations.append(obs)

        return self.solve(
            observations=observations,
            initial_guess=initial_guess,
            altitude=altitude,
            ref_emitters=ref_list,
            confidence=confidence,
            coarse_grid_search=coarse_grid_search,
            coarse_grid_bounds=coarse_grid_bounds,
        )

    def _solve_twosat_direct(
        self,
        two_sat,
        tdoa: Optional[Union[float, Sequence[float], np.ndarray]],
        fdoa: Optional[Union[float, Sequence[float], np.ndarray]],
        uplink: Optional[float],
        translation_frequency: float,
        ref_list: List[np.ndarray],
        sigma_t: float,
        sigma_f: float,
        sat1_cov: Optional[Dict[str, Any]],
        sat2_cov: Optional[Dict[str, Any]],
        initial_guess: Optional[Sequence[float]],
        altitude: Optional[float],
        confidence: float,
        coarse_grid_search: bool = False,
        coarse_grid_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> GeolocationResult:
        """
        Direct non-linear least squares solver utilizing TwoSat forward equations.
        """
        is_3d = altitude is None
        n_dim = 3 if is_3d else 2

        # Check measurement availability
        has_tdoa = tdoa is not None
        has_fdoa = fdoa is not None

        if not has_tdoa and not has_fdoa:
            raise ValueError("At least one of tdoa or fdoa must be provided.")

        tdoa_meas = np.atleast_1d(tdoa) if has_tdoa else np.array([])
        fdoa_meas = np.atleast_1d(fdoa) if has_fdoa else np.array([])

        total_obs = len(tdoa_meas) + len(fdoa_meas)
        min_obs = 3 if is_3d else 2
        if total_obs < min_obs:
            raise ValueError(
                f"At least {min_obs} independent observations are required, got {total_obs}."
            )

        # Precompute reference transmitter values (if provided)
        dto_ref_list = []
        dfo_ref_list = []
        for ref in ref_list:
            if has_tdoa:
                dto_ref_list.append(np.atleast_1d(two_sat.dto(ref)))
            if has_fdoa:
                dfo_ref_list.append(np.atleast_1d(two_sat.dfo(ref, uplink, translation_frequency)))

        # Average reference offset across multiple references if provided
        dto_ref = np.mean(dto_ref_list, axis=0) if dto_ref_list else np.zeros_like(tdoa_meas)
        dfo_ref = np.mean(dfo_ref_list, axis=0) if dfo_ref_list else np.zeros_like(fdoa_meas)

        # Formulate measurement weight vector W
        w_parts = []
        if has_tdoa:
            w_parts.extend([1.0 / (sigma_t**2)] * len(tdoa_meas))
        if has_fdoa:
            w_parts.extend([1.0 / (sigma_f**2)] * len(fdoa_meas))
        W = np.array(w_parts)

        def eval_meas(pos):
            res_parts = []
            if has_tdoa:
                dto_val = np.atleast_1d(two_sat.dto(pos))
                res_parts.append(dto_val - dto_ref - tdoa_meas)
            if has_fdoa:
                dfo_val = np.atleast_1d(two_sat.dfo(pos, uplink, translation_frequency))
                res_parts.append(dfo_val - dfo_ref - fdoa_meas)
            return np.concatenate(res_parts)

        # Set initial guess
        curr_alt = 0.0 if altitude is None else float(altitude)
        if initial_guess is not None and not coarse_grid_search:
            init_arr = np.asarray(initial_guess, dtype=float)
            curr_pos = np.array([init_arr[0], init_arr[1], curr_alt])
        elif coarse_grid_search:
            # 2D grid search
            if coarse_grid_bounds is not None:
                min_lat, max_lat, min_lon, max_lon = coarse_grid_bounds
            else:
                # Default continental bounds (e.g., North America / Americas covering G16/G19)
                min_lat, max_lat = 20.0, 60.0
                min_lon, max_lon = -130.0, -60.0

            grid_step = 4.0
            lats = np.arange(min_lat, max_lat + 0.1, grid_step)
            lons = np.arange(min_lon, max_lon + 0.1, grid_step)

            best_cost = float('inf')
            best_pos = np.array([(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0, curr_alt])

            for lat_cand in lats:
                for lon_cand in lons:
                    cand_pt = np.array([lat_cand, lon_cand, curr_alt])
                    r_cand = eval_meas(cand_pt)
                    cost_cand = float(np.sum(W * r_cand**2))
                    if cost_cand < best_cost:
                        best_cost = cost_cand
                        best_pos = cand_pt

            curr_pos = best_pos
        elif ref_list:
            ref0 = ref_list[0]
            curr_pos = np.array([ref0[0], ref0[1], curr_alt])
        else:
            curr_pos = np.array([39.0, -100.0, curr_alt])

        damping = self.damping
        converged = False
        history: List[IterationStep] = []
        h_step = 1.0  # meter for finite difference

        res = eval_meas(curr_pos)
        curr_cost = float(np.sum(W * res**2))

        history.append(IterationStep(
            iteration=0,
            lat=float(curr_pos[0]),
            lon=float(curr_pos[1]),
            alt=float(curr_pos[2]),
            step_m=0.0,
            residual_norm=float(np.sqrt(np.mean(res**2))),
        ))

        for it in range(1, self.max_iterations + 1):
            # Compute central finite-difference Jacobian in local ENU
            lat_ep, lon_ep, alt_ep = pm.enu2geodetic(h_step, 0, 0, curr_pos[0], curr_pos[1], curr_pos[2], ell=self.ellipsoid)
            lat_em, lon_em, alt_em = pm.enu2geodetic(-h_step, 0, 0, curr_pos[0], curr_pos[1], curr_pos[2], ell=self.ellipsoid)
            lat_np, lon_np, alt_np = pm.enu2geodetic(0, h_step, 0, curr_pos[0], curr_pos[1], curr_pos[2], ell=self.ellipsoid)
            lat_nm, lon_nm, alt_nm = pm.enu2geodetic(0, -h_step, 0, curr_pos[0], curr_pos[1], curr_pos[2], ell=self.ellipsoid)

            if not is_3d:
                alt_ep = alt_em = alt_np = alt_nm = curr_pos[2]

            res_ep = eval_meas([lat_ep, lon_ep, alt_ep])
            res_em = eval_meas([lat_em, lon_em, alt_em])
            res_np = eval_meas([lat_np, lon_np, alt_np])
            res_nm = eval_meas([lat_nm, lon_nm, alt_nm])

            col_e = (res_ep - res_em) / (2.0 * h_step)
            col_n = (res_np - res_nm) / (2.0 * h_step)

            if is_3d:
                lat_up, lon_up, alt_up = pm.enu2geodetic(0, 0, h_step, curr_pos[0], curr_pos[1], curr_pos[2], ell=self.ellipsoid)
                lat_um, lon_um, alt_um = pm.enu2geodetic(0, 0, -h_step, curr_pos[0], curr_pos[1], curr_pos[2], ell=self.ellipsoid)
                res_up = eval_meas([lat_up, lon_up, alt_up])
                res_um = eval_meas([lat_um, lon_um, alt_um])
                col_u = (res_up - res_um) / (2.0 * h_step)
                J = np.column_stack([col_e, col_n, col_u])
            else:
                J = np.column_stack([col_e, col_n])

            # Normal equations
            W_mat = np.diag(W)
            H = J.T @ W_mat @ J
            g = J.T @ (W * res)

            diag_H = np.diag(np.diag(H))
            if np.all(np.diag(diag_H) == 0):
                diag_H = np.eye(H.shape[0])

            H_damped = H + damping * diag_H

            try:
                delta = -np.linalg.solve(H_damped, g)
            except np.linalg.LinAlgError:
                delta = -np.linalg.lstsq(H_damped, g, rcond=None)[0]

            step_m = float(np.linalg.norm(delta))

            de = delta[0]
            dn = delta[1]
            du = delta[2] if is_3d else 0.0

            cand_lat, cand_lon, cand_alt = pm.enu2geodetic(
                de, dn, du, curr_pos[0], curr_pos[1], curr_pos[2], ell=self.ellipsoid
            )
            if not is_3d and altitude is not None:
                cand_alt = float(altitude)

            cand_pos = np.array([cand_lat, cand_lon, cand_alt])
            cand_res = eval_meas(cand_pos)
            cand_cost = float(np.sum(W * cand_res**2))

            if cand_cost < curr_cost:
                curr_pos = cand_pos
                res = cand_res
                curr_cost = cand_cost
                damping = max(damping * 0.2, 1e-10)

                history.append(IterationStep(
                    iteration=it,
                    lat=float(curr_pos[0]),
                    lon=float(curr_pos[1]),
                    alt=float(curr_pos[2]),
                    step_m=step_m,
                    residual_norm=float(np.sqrt(np.mean(res**2))),
                ))

                if step_m < self.tol:
                    converged = True
                    break
            else:
                damping = min(damping * 5.0, 1e10)
                history.append(IterationStep(
                    iteration=it,
                    lat=float(curr_pos[0]),
                    lon=float(curr_pos[1]),
                    alt=float(curr_pos[2]),
                    step_m=step_m,
                    residual_norm=float(np.sqrt(np.mean(res**2))),
                ))
                if damping >= 1e10:
                    break

        # Final covariance and GDOP
        W_mat = np.diag(W)
        H_final = J.T @ W_mat @ J
        try:
            cov_enu = np.linalg.inv(H_final)
        except np.linalg.LinAlgError:
            cov_enu = np.linalg.pinv(H_final)

        JTJ = J.T @ J
        try:
            gdop = float(np.sqrt(np.trace(np.linalg.inv(JTJ))))
        except np.linalg.LinAlgError:
            gdop = float(np.sqrt(np.trace(np.linalg.pinv(JTJ))))

        cov_horiz = cov_enu[0:2, 0:2]
        error_ellipse = ErrorEllipse.from_covariance_2d(cov_horiz, confidence=confidence)
        error_ellipsoid = ErrorEllipsoid.from_covariance_3d(cov_enu, confidence=confidence) if is_3d else None
        final_ecef = np.array(pm.geodetic2ecef(curr_pos[0], curr_pos[1], curr_pos[2], ell=self.ellipsoid))

        msg = "Converged successfully via direct TwoSat." if converged else "Reached iteration limit or damping max."

        return GeolocationResult(
            lat=float(curr_pos[0]),
            lon=float(curr_pos[1]),
            alt=float(curr_pos[2]),
            ecef=final_ecef,
            converged=converged,
            iterations=len(history) - 1,
            residuals=res,
            cost=curr_cost,
            cov_enu=cov_enu,
            error_ellipse=error_ellipse,
            error_ellipsoid=error_ellipsoid,
            history=history,
            gdop=gdop,
            message=msg,
            ellipsoid=self.ellipsoid_name,
        )


def solve_geolocation(
    observations: Sequence[Observation],
    initial_guess: Optional[Sequence[float]] = None,
    altitude: Optional[float] = 0.0,
    ellipsoid: Any = 'wgs84',
    max_iterations: int = 50,
    tol: float = 1e-4,
    confidence: float = 0.95,
    coarse_grid_search: bool = False,
    coarse_grid_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> GeolocationResult:
    """
    Convenience function to solve satellite geolocation from observations.
    """
    solver = GeolocationSolver(
        ellipsoid=ellipsoid,
        max_iterations=max_iterations,
        tol=tol,
    )
    return solver.solve(
        observations=observations,
        initial_guess=initial_guess,
        altitude=altitude,
        confidence=confidence,
        coarse_grid_search=coarse_grid_search,
        coarse_grid_bounds=coarse_grid_bounds,
    )
