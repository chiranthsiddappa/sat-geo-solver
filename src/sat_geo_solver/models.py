"""
Data models and containers for satellite geolocation solving.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pymap3d as pm
from scipy.stats import chi2
from skyfield.api import wgs84
from skyfield.constants import C as SPEED_OF_LIGHT
from skyfield.framelib import itrs
from skyfield.timelib import Time


@dataclass
class SatelliteState:
    """
    Satellite state representation in Earth-Centered, Earth-Fixed (ECEF / ITRS) coordinates.

    :param pos_m: Position [x, y, z] in meters (shape: (3,))
    :param vel_mps: Velocity [vx, vy, vz] in meters/second (shape: (3,))
    :param pos_cov: Optional position error covariance matrix (3, 3) or variance in m^2
    :param vel_cov: Optional velocity error covariance matrix (3, 3) or variance in (m/s)^2
    """
    pos_m: np.ndarray
    vel_mps: np.ndarray
    pos_cov: Optional[np.ndarray] = None
    vel_cov: Optional[np.ndarray] = None

    def __post_init__(self):
        self.pos_m = np.asarray(self.pos_m, dtype=float).reshape(3)
        self.vel_mps = np.asarray(self.vel_mps, dtype=float).reshape(3)

        if self.pos_cov is not None:
            pos_cov_arr = np.asarray(self.pos_cov, dtype=float)
            if pos_cov_arr.ndim == 0 or (pos_cov_arr.ndim == 1 and pos_cov_arr.size == 1):
                self.pos_cov = np.eye(3) * float(pos_cov_arr)
            elif pos_cov_arr.shape == (3,):
                self.pos_cov = np.diag(pos_cov_arr)
            elif pos_cov_arr.shape == (3, 3):
                self.pos_cov = pos_cov_arr
            else:
                raise ValueError(f"Invalid shape for pos_cov: {pos_cov_arr.shape}")

        if self.vel_cov is not None:
            vel_cov_arr = np.asarray(self.vel_cov, dtype=float)
            if vel_cov_arr.ndim == 0 or (vel_cov_arr.ndim == 1 and vel_cov_arr.size == 1):
                self.vel_cov = np.eye(3) * float(vel_cov_arr)
            elif vel_cov_arr.shape == (3,):
                self.vel_cov = np.diag(vel_cov_arr)
            elif vel_cov_arr.shape == (3, 3):
                self.vel_cov = vel_cov_arr
            else:
                raise ValueError(f"Invalid shape for vel_cov: {vel_cov_arr.shape}")

    @classmethod
    def from_skyfield(cls, sat, time_obj: Union[Time, datetime], dt_sec: float = 0.05,
                      pos_cov: Optional[np.ndarray] = None, vel_cov: Optional[np.ndarray] = None) -> "SatelliteState":
        """
        Extract ECEF position and velocity from a Skyfield satellite object.

        Velocity in the rotating ITRS frame is derived via high-accuracy central finite differences:
            v_itrs = (r_itrs(t + dt) - r_itrs(t - dt)) / (2 * dt)
        """
        from sat_geo_solver.observe import __ts__, dt_to_ts

        if isinstance(time_obj, datetime):
            ts_time = dt_to_ts(time_obj)
        else:
            ts_time = time_obj

        # Midpoint position
        pos_mid = sat.at(ts_time).frame_xyz(itrs).m

        # Time offsets for velocity in rotating frame
        dt_days = dt_sec / 86400.0
        ts_minus = __ts__.tt_jd(ts_time.tt - dt_days)
        ts_plus = __ts__.tt_jd(ts_time.tt + dt_days)

        pos_minus = sat.at(ts_minus).frame_xyz(itrs).m
        pos_plus = sat.at(ts_plus).frame_xyz(itrs).m
        vel_itrs = (pos_plus - pos_minus) / (2.0 * dt_sec)

        return cls(pos_m=pos_mid, vel_mps=vel_itrs, pos_cov=pos_cov, vel_cov=vel_cov)

    @classmethod
    def from_observe(cls, obs_obj, dt_sec: float = 0.05,
                     pos_cov: Optional[np.ndarray] = None, vel_cov: Optional[np.ndarray] = None) -> "SatelliteState":
        """
        Create a SatelliteState from an Observe instance (re-using sat and epoch).
        """
        # obs_obj is an instance of Observe
        return cls.from_skyfield(obs_obj.sat, obs_obj.observe_ts, dt_sec=dt_sec, pos_cov=pos_cov, vel_cov=vel_cov)


@dataclass
class Observation:
    """
    Representation of an individual TDOA or FDOA observation.

    :param kind: 'tdoa' (time difference of arrival in seconds) or 'fdoa' (frequency difference of arrival in Hz)
    :param value: Measured differential value (target or differential vs reference)
    :param sat1: State of primary satellite (SatelliteState)
    :param sat2: State of secondary satellite (SatelliteState)
    :param center_freq: Nominal uplink/center frequency in Hz (required for FDOA)
    :param sigma: Measurement uncertainty / CAF fidelity (seconds for TDOA, Hz for FDOA)
    :param ref_emitter: Optional reference emitter position [lat_deg, lon_deg, alt_m]
    :param ref_value: Optional measured reference value (if differential calibration is needed)
    :param ref_sigma: Optional uncertainty of the reference measurement
    :param time: Optional observation timestamp
    """
    kind: str
    value: float
    sat1: SatelliteState
    sat2: SatelliteState
    center_freq: float = 0.0
    sigma: float = 1.0
    ref_emitter: Optional[np.ndarray] = None
    ref_value: Optional[float] = None
    ref_sigma: Optional[float] = None
    time: Optional[Any] = None

    def __post_init__(self):
        self.kind = self.kind.lower()
        if self.kind not in ('tdoa', 'fdoa'):
            raise ValueError(f"kind must be 'tdoa' or 'fdoa', got '{self.kind}'")
        if self.kind == 'fdoa' and self.center_freq <= 0:
            raise ValueError("center_freq must be positive for FDOA observations.")
        if self.sigma <= 0:
            raise ValueError("sigma (fidelity/uncertainty) must be positive.")
        if self.ref_emitter is not None:
            self.ref_emitter = np.asarray(self.ref_emitter, dtype=float)


@dataclass
class ErrorEllipse:
    """
    2D Error Ellipse in the local horizontal East-North plane.

    Corresponds to the confidence region defined by the quadratic form:
        r^T C^-1 r = k^2
    which expands to:
        A * e^2 + B * e * n + C * n^2 = 1 (when normalized by k^2)
        or A * e^2 + B * e * n + C * n^2 + F = 0 with F = -1.

    :param semi_major_m: Semi-major axis length in meters
    :param semi_minor_m: Semi-minor axis length in meters
    :param azimuth_deg: Azimuth of semi-major axis (degrees clockwise from True North, 0..180)
    :param angle_deg: Polar orientation angle of semi-major axis (degrees counterclockwise from East)
    :param quadratic_constants: Dict with keys 'A', 'B', 'C', 'F' for the quadratic form:
                                A * x^2 + B * x * y + C * y^2 + F = 0 (where x=East, y=North)
    :param confidence: Statistical confidence level (e.g. 0.95 for 95%, or 0.3935 for 1-sigma)
    :param cov_enu: 2x2 covariance matrix in local [East, North] in m^2
    """
    semi_major_m: float
    semi_minor_m: float
    azimuth_deg: float
    angle_deg: float
    quadratic_constants: Dict[str, float]
    confidence: float
    cov_enu: np.ndarray

    @classmethod
    def from_covariance_2d(cls, cov_2d: np.ndarray, confidence: float = 0.95) -> "ErrorEllipse":
        """
        Construct an ErrorEllipse from a 2x2 covariance matrix in [East, North].

        :param cov_2d: 2x2 covariance matrix [[var_E, cov_EN], [cov_EN, var_N]] in m^2
        :param confidence: Probability level (default: 0.95 for 95% confidence)
        """
        cov = np.asarray(cov_2d, dtype=float).reshape(2, 2)
        var_e = cov[0, 0]
        var_n = cov[1, 1]
        cov_en = cov[0, 1]

        det = var_e * var_n - cov_en**2
        if det <= 0:
            # Handle near-singular covariance gracefully with minimal floor
            det = max(det, 1e-12)

        # Scale factor k for chi-square with 2 degrees of freedom
        # Prob(X <= k^2) = confidence => k^2 = chi2.ppf(confidence, df=2) = -2 * ln(1 - confidence)
        k_sq = float(chi2.ppf(confidence, df=2))
        k = np.sqrt(k_sq)

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[idx], 1e-12)
        eigvecs = eigvecs[:, idx]

        semi_major = k * np.sqrt(eigvals[0])
        semi_minor = k * np.sqrt(eigvals[1])

        # Principal eigenvector (direction of semi-major axis in [East, North])
        major_vec = eigvecs[:, 0]
        # Angle counterclockwise from East: atan2(North, East)
        angle_rad = np.arctan2(major_vec[1], major_vec[0])
        angle_deg = float(np.degrees(angle_rad) % 360)

        # Azimuth clockwise from North: atan2(East, North)
        azimuth_rad = np.arctan2(major_vec[0], major_vec[1])
        azimuth_deg = float(np.degrees(azimuth_rad) % 360)

        # Quadratic form: [x, y] C^-1 [x, y]^T = k^2
        # C^-1 = (1/det) * [[var_N, -cov_EN], [-cov_EN, var_E]]
        # (var_N * x^2 - 2 * cov_EN * x * y + var_E * y^2) / (k^2 * det) = 1
        A = float(var_n / (k_sq * det))
        B = float(-2.0 * cov_en / (k_sq * det))
        C = float(var_e / (k_sq * det))
        F = -1.0

        return cls(
            semi_major_m=float(semi_major),
            semi_minor_m=float(semi_minor),
            azimuth_deg=azimuth_deg,
            angle_deg=angle_deg,
            quadratic_constants={'A': A, 'B': B, 'C': C, 'F': F},
            confidence=confidence,
            cov_enu=cov,
        )

    def to_enu_points(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate (East, North) coordinate arrays in meters along the ellipse perimeter.

        :param n_points: Number of perimeter samples
        :return: (east_m, north_m) arrays
        """
        theta = np.linspace(0, 2 * np.pi, n_points)
        # Standard ellipse aligned with principal axes
        x_prime = self.semi_major_m * np.cos(theta)
        y_prime = self.semi_minor_m * np.sin(theta)

        # Rotate by angle_deg (counterclockwise from East)
        phi = np.radians(self.angle_deg)
        east = x_prime * np.cos(phi) - y_prime * np.sin(phi)
        north = x_prime * np.sin(phi) + y_prime * np.cos(phi)
        return east, north

    def to_geodetic_points(self, origin_lat: float, origin_lon: float, origin_alt: float = 0.0,
                           ell: Any = 'wgs84', n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate (lat, lon) coordinate arrays in degrees along the ellipse perimeter on the ellipsoid.

        :param origin_lat: Center latitude in degrees
        :param origin_lon: Center longitude in degrees
        :param origin_alt: Altitude in meters
        :param ell: pymap3d Ellipsoid name or instance
        :param n_points: Number of perimeter samples
        :return: (lat_deg, lon_deg) arrays
        """
        east, north = self.to_enu_points(n_points)
        up = np.zeros_like(east)
        lat, lon, _ = pm.enu2geodetic(east, north, up, origin_lat, origin_lon, origin_alt, ell=ell)
        return lat, lon

    @property
    def standard_form(self) -> Dict[str, Any]:
        """
        Return the operational standard (most common) form for describing an error ellipse:
        - semi_major_axis_m: Semi-major axis length (meters)
        - semi_minor_axis_m: Semi-minor axis length (meters)
        - azimuth_deg: Bearing of semi-major axis (degrees clockwise from North, 0..180)
        - orientation_deg: Angle of semi-major axis (degrees counterclockwise from East)
        - confidence_level: Probability contained within the ellipse (e.g. 0.95 for 95%)
        - confidence_k: Expansion multiplier scale factor k (sqrt(chi2(confidence, df=2)))
        - cep_50_m: Circular Error Probable equivalent (approximate radius containing 50% probability)
        """
        k_val = np.sqrt(float(chi2.ppf(self.confidence, df=2)))
        # Standard approximation for CEP: 0.562 * a + 0.438 * b for 1-sigma, or scaled
        # Direct CEP formula approximation: 0.59 * (semi_major / k + semi_minor / k) * sqrt(chi2(0.5, df=2))
        sigma_a = self.semi_major_m / k_val
        sigma_b = self.semi_minor_m / k_val
        cep_50 = float(0.59 * (sigma_a + sigma_b) * np.sqrt(chi2.ppf(0.50, df=2)))

        return {
            "semi_major_axis_m": self.semi_major_m,
            "semi_minor_axis_m": self.semi_minor_m,
            "azimuth_deg": self.azimuth_deg,
            "orientation_deg": self.angle_deg,
            "confidence_level": self.confidence,
            "confidence_k": float(k_val),
            "cep_50_m": cep_50,
            "quadratic_constants": dict(self.quadratic_constants),
        }

    def contains_point(self, east_m: float, north_m: float) -> bool:
        """
        Test whether a local coordinate point (East, North in meters relative to center)
        lies on or inside the confidence error ellipse using the quadratic equation:
            A * e^2 + B * e * n + C * n^2 <= 1.0

        :param east_m: East coordinate in meters from ellipse center
        :param north_m: North coordinate in meters from ellipse center
        :return: True if (east_m, north_m) is inside or on the ellipse boundary
        """
        A = self.quadratic_constants['A']
        B = self.quadratic_constants['B']
        C = self.quadratic_constants['C']
        val = A * (east_m**2) + B * (east_m * north_m) + C * (north_m**2)
        return bool(val <= 1.0 + 1e-9)

    def contains_geodetic(self, lat: float, lon: float, origin_lat: float, origin_lon: float,
                          origin_alt: float = 0.0, ell: Any = 'wgs84') -> bool:
        """
        Test whether a geodetic coordinate (lat, lon) lies inside the error ellipse centered at (origin_lat, origin_lon).
        """
        e, n, _ = pm.geodetic2enu(lat, lon, origin_alt, origin_lat, origin_lon, origin_alt, ell=ell)
        return self.contains_point(float(e), float(n))

    def intersects_bbox(self, min_east: float, max_east: float, min_north: float, max_north: float) -> bool:
        """
        Test whether the error ellipse intersects or overlaps an axis-aligned bounding box (min_east, max_east, min_north, max_north).
        """
        # 1. If ellipse center is inside the bbox -> overlaps
        if min_east <= 0.0 <= max_east and min_north <= 0.0 <= max_north:
            return True

        # 2. If any bbox corner is inside the ellipse -> overlaps
        corners = [
            (min_east, min_north), (max_east, min_north),
            (min_east, max_north), (max_east, max_north)
        ]
        for ce, cn in corners:
            if self.contains_point(ce, cn):
                return True

        # 3. Check ellipse perimeter points against the bbox
        e_pts, n_pts = self.to_enu_points(n_points=120)
        in_e = (e_pts >= min_east) & (e_pts <= max_east)
        in_n = (n_pts >= min_north) & (n_pts <= max_north)
        if np.any(in_e & in_n):
            return True

        # 4. Check if bounding box segment cuts through the ellipse
        # By sampling points along the bbox perimeter
        t_samples = np.linspace(0, 1, 25)
        edges = [
            (min_east + t_samples * (max_east - min_east), np.full_like(t_samples, min_north)),
            (min_east + t_samples * (max_east - min_east), np.full_like(t_samples, max_north)),
            (np.full_like(t_samples, min_east), min_north + t_samples * (max_north - min_north)),
            (np.full_like(t_samples, max_east), min_north + t_samples * (max_north - min_north)),
        ]
        for edge_e, edge_n in edges:
            for pe, pn in zip(edge_e, edge_n):
                if self.contains_point(pe, pn):
                    return True

        return False

    def intersects_zone(self, polygon_enu: Sequence[Tuple[float, float]]) -> bool:
        """
        Test whether the confidence ellipse boundary or interior intersects a planar 2D zone (polygon)
        given as a sequence of (East, North) vertices in meters.

        :param polygon_enu: List or array of (east_m, north_m) vertices defining the zone
        :return: True if the error ellipse intersects or overlaps the zone
        """
        poly = np.asarray(polygon_enu, dtype=float)
        if len(poly) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")

        # Check if ellipse center is inside polygon (ray casting)
        x, y = 0.0, 0.0
        n_vert = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n_vert + 1):
            p2x, p2y = poly[i % n_vert]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        if inside:
            return True

        # Check if any polygon vertex is inside ellipse
        for vx, vy in poly:
            if self.contains_point(vx, vy):
                return True

        # Check if ellipse boundary samples fall inside polygon
        e_pts, n_pts = self.to_enu_points(n_points=120)
        for ep, np_val in zip(e_pts, n_pts):
            # Point in polygon test for perimeter point
            in_poly = False
            p1x, p1y = poly[0]
            for i in range(1, n_vert + 1):
                p2x, p2y = poly[i % n_vert]
                if np_val > min(p1y, p2y):
                    if np_val <= max(p1y, p2y):
                        if ep <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (np_val - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or ep <= xinters:
                                in_poly = not in_poly
                p1x, p1y = p2x, p2y
            if in_poly:
                return True

        return False

    def intersects_geodetic_zone(
        self,
        polygon_latlon: Sequence[Tuple[float, float]],
        origin_lat: float,
        origin_lon: float,
        origin_alt: float = 0.0,
        ell: Any = 'wgs84',
    ) -> bool:
        """
        Test whether the confidence ellipse centered at (origin_lat, origin_lon) intersects a geographic zone (lat, lon vertices).
        """
        poly_enu = []
        for lat, lon in polygon_latlon:
            e, n, _ = pm.geodetic2enu(lat, lon, origin_alt, origin_lat, origin_lon, origin_alt, ell=ell)
            poly_enu.append((float(e), float(n)))
        return self.intersects_zone(poly_enu)


@dataclass
class ErrorEllipsoid:
    """
    3D Error Ellipsoid in the local East-North-Up coordinate frame.

    Defined by the confidence region:
        r^T C^-1 r <= k_3d^2

    :param semi_axes_m: Tuple of semi-axis lengths (a, b, c) in meters sorted descending
    :param eigenvectors: 3x3 matrix where columns are the principal axes in ENU
    :param sep_m: Spherical Error Probable radius in meters (50% spherical probability)
    :param cov_enu: 3x3 covariance matrix in ENU in m^2
    :param quadratic_matrix: 3x3 matrix representing the normalized quadratic form: r^T A r = 1
    :param confidence: Statistical confidence level
    """
    semi_axes_m: Tuple[float, float, float]
    eigenvectors: np.ndarray
    sep_m: float
    cov_enu: np.ndarray
    quadratic_matrix: np.ndarray
    confidence: float

    @classmethod
    def from_covariance_3d(cls, cov_3d: np.ndarray, confidence: float = 0.95) -> "ErrorEllipsoid":
        """
        Construct an ErrorEllipsoid from a 3x3 covariance matrix in ENU coordinates.
        """
        cov = np.asarray(cov_3d, dtype=float).reshape(3, 3)
        k_sq = float(chi2.ppf(confidence, df=3))
        k = np.sqrt(k_sq)

        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[idx], 1e-12)
        eigvecs = eigvecs[:, idx]

        semi_axes = tuple(float(k * np.sqrt(val)) for val in eigvals)

        # Spherical Error Probable (approximate radius containing 50% probability in 3D):
        # Commonly approximated as 0.51 * (sigma_x + sigma_y + sigma_z) or ~1.538 * sigma_mean
        sigmas = np.sqrt(eigvals)
        sep_m = float(0.51 * np.sum(sigmas))

        # Normalized quadratic matrix A = C^-1 / k^2
        cov_inv = np.linalg.pinv(cov)
        quad_mat = cov_inv / k_sq

        return cls(
            semi_axes_m=semi_axes,
            eigenvectors=eigvecs,
            sep_m=sep_m,
            cov_enu=cov,
            quadratic_matrix=quad_mat,
            confidence=confidence,
        )


@dataclass
class IterationStep:
    """Record of a single solver iteration for tracing and visualization."""
    iteration: int
    lat: float
    lon: float
    alt: float
    step_m: float
    residual_norm: float


@dataclass
class GeolocationResult:
    """
    Result returned by the GeolocationSolver.

    :param lat: Estimated geodetic latitude in degrees
    :param lon: Estimated geodetic longitude in degrees
    :param alt: Estimated or constrained altitude in meters
    :param ecef: Estimated ECEF coordinates [x, y, z] in meters
    :param converged: Whether the optimization converged within tolerance
    :param iterations: Number of iterations completed
    :param residuals: Array of final measurement residuals
    :param cost: Weighted sum of squared residuals (chi-square value)
    :param cov_enu: Covariance matrix in local ENU frame (2x2 for horizontal, 3x3 for 3D)
    :param error_ellipse: Horizontal 2D ErrorEllipse instance
    :param error_ellipsoid: 3D ErrorEllipsoid instance (if 3D was solved)
    :param history: List of IterationStep objects documenting the convergence path
    :param gdop: Geometric Dilution of Precision
    :param message: Termination message
    :param ellipsoid: Ellipsoid model name used
    """
    lat: float
    lon: float
    alt: float
    ecef: np.ndarray
    converged: bool
    iterations: int
    residuals: np.ndarray
    cost: float
    cov_enu: np.ndarray
    error_ellipse: ErrorEllipse
    error_ellipsoid: Optional[ErrorEllipsoid]
    history: List[IterationStep]
    gdop: float
    message: str
    ellipsoid: str

    def __repr__(self) -> str:
        status = "CONVERGED" if self.converged else "FAILED"
        return (
            f"GeolocationResult(\n"
            f"  status: {status} in {self.iterations} iterations\n"
            f"  location: lat={self.lat:.6f}°, lon={self.lon:.6f}°, alt={self.alt:.2f} m\n"
            f"  error ellipse (95%): semi_major={self.error_ellipse.semi_major_m:.2f} m, "
            f"semi_minor={self.error_ellipse.semi_minor_m:.2f} m, "
            f"azimuth={self.error_ellipse.azimuth_deg:.1f}°\n"
            f"  quadratic form: {self.error_ellipse.quadratic_constants['A']:.3e}*x^2 + "
            f"{self.error_ellipse.quadratic_constants['B']:.3e}*x*y + "
            f"{self.error_ellipse.quadratic_constants['C']:.3e}*y^2 = 1\n"
            f"  gdop: {self.gdop:.2f}, cost: {self.cost:.3e}, ellipsoid: {self.ellipsoid}\n"
            f")"
        )
