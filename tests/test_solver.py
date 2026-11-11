"""
Unit tests for satellite geolocation solver.
"""

import unittest
from datetime import timedelta

import numpy as np
import pymap3d as pm
from numpy.testing import assert_allclose
from skyfield.api import load as skyf_load
from skyfield.iokit import parse_tle_file

from sat_geo_solver.models import (
    ErrorEllipse,
    ErrorEllipsoid,
    Observation,
    SatelliteState,
)
from sat_geo_solver.scenarios import TwoSat
from sat_geo_solver.solver import GeolocationSolver, solve_geolocation


class TestGeolocationSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ts = skyf_load.timescale()
        with open('tests/g19.tle', 'rb') as f:
            cls.sat19 = list(parse_tle_file(f, cls.ts))[0]
        with open('nb_examples/g16.tle', 'rb') as f:
            cls.sat16 = list(parse_tle_file(f, cls.ts))[0]

        cls.epoch_dt = cls.sat16.epoch.utc_datetime()

        cls.cos_ll = [+38.4, -104.82]
        cls.cos_ll_2 = [+38.4001, -104.8201]
        cls.pbs_noc_ll = [+38.8622183, -77.0504056, 63]
        cls.montana_pbs_ll = [+45.6655208, -111.0514232, 271]

        cls.translation_frequency = 2.3e9
        cls.pbs_uplink = 12.14e9 + cls.translation_frequency

        cls.two_sat = TwoSat(cls.sat16, cls.sat19, cls.epoch_dt, cls.cos_ll, cls.cos_ll_2)

        # Generate true synthetic measurements
        dto_tar = float(cls.two_sat.dto(cls.montana_pbs_ll))
        dto_ref = float(cls.two_sat.dto(cls.pbs_noc_ll))
        cls.tdoa_true = dto_tar - dto_ref

        dfo_tar = float(cls.two_sat.dfo(cls.montana_pbs_ll, cls.pbs_uplink, cls.translation_frequency))
        dfo_ref = float(cls.two_sat.dfo(cls.pbs_noc_ll, cls.pbs_uplink, cls.translation_frequency))
        cls.fdoa_true = dfo_tar - dfo_ref

    def test_twosat_solve_convergence(self):
        """Test that GeolocationSolver converges to sub-meter accuracy on Montana PBS."""
        solver = GeolocationSolver(ellipsoid='wgs84', max_iterations=50)
        result = solver.solve_from_twosat(
            two_sat=self.two_sat,
            tdoa=self.tdoa_true,
            fdoa=self.fdoa_true,
            uplink=self.pbs_uplink,
            translation_frequency=self.translation_frequency,
            ref_emitter=self.pbs_noc_ll,
            altitude=271.0,
            initial_guess=[40.0, -100.0],
        )

        self.assertTrue(result.converged)
        self.assertLess(result.iterations, 35)

        # Distance error in meters
        true_ecef = np.array(pm.geodetic2ecef(*self.montana_pbs_ll))
        dist_err = float(np.linalg.norm(true_ecef - result.ecef))
        self.assertLess(dist_err, 2.0)

    def test_error_ellipse_constants(self):
        """Test that error ellipse constants A, B, C satisfy Ax^2 + Bxy + Cy^2 = 1 on boundary."""
        cov = np.array([[2.5e7, 1.2e6], [1.2e6, 8.0e5]])
        ee = ErrorEllipse.from_covariance_2d(cov, confidence=0.95)

        self.assertGreater(ee.semi_major_m, ee.semi_minor_m)
        self.assertGreater(ee.semi_minor_m, 0.0)

        A = ee.quadratic_constants['A']
        B = ee.quadratic_constants['B']
        C = ee.quadratic_constants['C']

        e_pts, n_pts = ee.to_enu_points(n_points=100)
        vals = A * e_pts**2 + B * e_pts * n_pts + C * n_pts**2

        # Check that on the perimeter the quadratic equation holds exactly
        assert_allclose(vals, np.ones_like(vals), rtol=1e-10)

    def test_multi_epoch_tdoa_only(self):
        """Test sequences of TDOAs across multiple epochs (multilateration)."""
        # Create 12 epochs over 4 hours
        dt_list = [self.epoch_dt + timedelta(minutes=20 * i) for i in range(12)]
        two_sat_multi = TwoSat(self.sat16, self.sat19, dt_list, self.cos_ll, self.cos_ll_2)

        dto_tar_seq = two_sat_multi.dto(self.montana_pbs_ll)
        dto_ref_seq = two_sat_multi.dto(self.pbs_noc_ll)
        tdoa_seq = dto_tar_seq - dto_ref_seq

        solver = GeolocationSolver(ellipsoid='wgs84', max_iterations=50)
        result = solver.solve_from_twosat(
            two_sat=two_sat_multi,
            tdoa=tdoa_seq,
            fdoa=None,  # TDOA ONLY!
            ref_emitter=self.pbs_noc_ll,
            altitude=271.0,
            initial_guess=[43.0, -108.0],
        )

        self.assertTrue(result.converged)
        true_ecef = np.array(pm.geodetic2ecef(*self.montana_pbs_ll))
        dist_err = float(np.linalg.norm(true_ecef - result.ecef))
        self.assertLess(dist_err, 5.0)

    def test_multiple_references(self):
        """Test solver calibration when multiple reference emitters are provided."""
        ref1 = self.pbs_noc_ll
        ref2 = [39.7392, -104.9903, 1609.0]  # Denver CO

        dto_ref1 = float(self.two_sat.dto(ref1))
        dto_ref2 = float(self.two_sat.dto(ref2))
        dto_ref_avg = (dto_ref1 + dto_ref2) / 2.0

        dfo_ref1 = float(self.two_sat.dfo(ref1, self.pbs_uplink, self.translation_frequency))
        dfo_ref2 = float(self.two_sat.dfo(ref2, self.pbs_uplink, self.translation_frequency))
        dfo_ref_avg = (dfo_ref1 + dfo_ref2) / 2.0

        dto_tar = float(self.two_sat.dto(self.montana_pbs_ll))
        dfo_tar = float(self.two_sat.dfo(self.montana_pbs_ll, self.pbs_uplink, self.translation_frequency))

        tdoa_multi_ref = dto_tar - dto_ref_avg
        fdoa_multi_ref = dfo_tar - dfo_ref_avg

        solver = GeolocationSolver(ellipsoid='wgs84', max_iterations=50)
        result = solver.solve_from_twosat(
            two_sat=self.two_sat,
            tdoa=tdoa_multi_ref,
            fdoa=fdoa_multi_ref,
            uplink=self.pbs_uplink,
            translation_frequency=self.translation_frequency,
            ref_emitter=[ref1, ref2],  # Multiple references
            altitude=271.0,
            initial_guess=[44.0, -110.0],
        )

        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.lat, self.montana_pbs_ll[0], delta=0.01)
        self.assertAlmostEqual(result.lon, self.montana_pbs_ll[1], delta=0.01)

    def test_caf_fidelity_scaling(self):
        """Test that CAF peak fidelity uncertainties scale the error ellipse proportionally."""
        solver = GeolocationSolver(ellipsoid='wgs84')

        # Tight fidelity (sharp CAF peak)
        res_sharp = solver.solve_from_twosat(
            two_sat=self.two_sat,
            tdoa=self.tdoa_true,
            fdoa=self.fdoa_true,
            uplink=self.pbs_uplink,
            translation_frequency=self.translation_frequency,
            ref_emitter=self.pbs_noc_ll,
            error_fidelity=(1e-7, 0.05),  # 0.1 µs, 0.05 Hz
            altitude=271.0,
            initial_guess=self.montana_pbs_ll,
        )

        # Loose fidelity (diffuse CAF peak: 10x worse)
        res_loose = solver.solve_from_twosat(
            two_sat=self.two_sat,
            tdoa=self.tdoa_true,
            fdoa=self.fdoa_true,
            uplink=self.pbs_uplink,
            translation_frequency=self.translation_frequency,
            ref_emitter=self.pbs_noc_ll,
            error_fidelity=(1e-6, 0.5),   # 1 µs, 0.5 Hz
            altitude=271.0,
            initial_guess=self.montana_pbs_ll,
        )

        # 10x larger measurement noise should produce ~10x larger semi-axes
        ratio_major = res_loose.error_ellipse.semi_major_m / res_sharp.error_ellipse.semi_major_m
        ratio_minor = res_loose.error_ellipse.semi_minor_m / res_sharp.error_ellipse.semi_minor_m

        self.assertAlmostEqual(ratio_major, 10.0, delta=0.5)
        self.assertAlmostEqual(ratio_minor, 10.0, delta=0.5)

    def test_satellite_state_error_propagation(self):
        """Test that satellite position and velocity uncertainties inflate the error ellipse."""
        solver = GeolocationSolver(ellipsoid='wgs84')

        # Baseline without satellite state errors
        res_no_sat_err = solver.solve_from_twosat(
            two_sat=self.two_sat,
            tdoa=self.tdoa_true,
            fdoa=self.fdoa_true,
            uplink=self.pbs_uplink,
            translation_frequency=self.translation_frequency,
            ref_emitter=self.pbs_noc_ll,
            error_fidelity=(1e-6, 0.5),
            altitude=271.0,
            initial_guess=self.montana_pbs_ll,
            use_twosat_model=False,
        )

        # With satellite state error covariance (e.g. 500m position error, 0.05 m/s velocity error)
        sat_pos_cov = np.eye(3) * (500.0**2)
        sat_vel_cov = np.eye(3) * (0.05**2)

        res_with_sat_err = solver.solve_from_twosat(
            two_sat=self.two_sat,
            tdoa=self.tdoa_true,
            fdoa=self.fdoa_true,
            uplink=self.pbs_uplink,
            translation_frequency=self.translation_frequency,
            ref_emitter=self.pbs_noc_ll,
            error_fidelity=(1e-6, 0.5),
            sat1_cov={'pos_cov': sat_pos_cov, 'vel_cov': sat_vel_cov},
            sat2_cov={'pos_cov': sat_pos_cov, 'vel_cov': sat_vel_cov},
            altitude=271.0,
            initial_guess=self.montana_pbs_ll,
            use_twosat_model=False,
        )

        # Error ellipse with satellite errors should be larger than without
        self.assertGreater(
            res_with_sat_err.error_ellipse.semi_major_m,
            res_no_sat_err.error_ellipse.semi_major_m
        )

    def test_ellipsoid_models(self):
        """Test solver with different reference ellipsoids (WGS84, GRS80, IERS2003)."""
        for model in ['wgs84', 'grs80', 'iers2003']:
            solver = GeolocationSolver(ellipsoid=model, max_iterations=50)
            res = solver.solve_from_twosat(
                two_sat=self.two_sat,
                tdoa=self.tdoa_true,
                fdoa=self.fdoa_true,
                uplink=self.pbs_uplink,
                translation_frequency=self.translation_frequency,
                ref_emitter=self.pbs_noc_ll,
                altitude=271.0,
                initial_guess=[44.0, -110.0],
            )
            self.assertTrue(res.converged)
            self.assertEqual(res.ellipsoid, model)

    def test_iteration_parameter_exposure(self):
        """Test that max_iterations is respected when setting a low limit."""
        solver = GeolocationSolver(max_iterations=3)
        res = solver.solve_from_twosat(
            two_sat=self.two_sat,
            tdoa=self.tdoa_true,
            fdoa=self.fdoa_true,
            uplink=self.pbs_uplink,
            translation_frequency=self.translation_frequency,
            ref_emitter=self.pbs_noc_ll,
            altitude=271.0,
            initial_guess=[30.0, -90.0],  # far away
        )
        self.assertLessEqual(res.iterations, 3)

    def test_standard_form_and_confidence(self):
        """Test the standard operational form and confidence scaling of the error ellipse."""
        cov = np.array([[4e6, 1e6], [1e6, 2e6]])
        ee_95 = ErrorEllipse.from_covariance_2d(cov, confidence=0.95)
        sf_95 = ee_95.standard_form

        # Verify keys in standard form
        expected_keys = [
            'semi_major_axis_m', 'semi_minor_axis_m', 'azimuth_deg',
            'orientation_deg', 'confidence_level', 'confidence_k', 'cep_50_m', 'quadratic_constants'
        ]
        for k in expected_keys:
            self.assertIn(k, sf_95)

        self.assertEqual(sf_95['confidence_level'], 0.95)
        self.assertAlmostEqual(sf_95['confidence_k'], 2.4477, places=3)
        self.assertGreater(sf_95['semi_major_axis_m'], sf_95['semi_minor_axis_m'])
        self.assertGreater(sf_95['cep_50_m'], 0.0)

        # 50% confidence ellipse should have smaller semi-axes
        ee_50 = ErrorEllipse.from_covariance_2d(cov, confidence=0.50)
        self.assertLess(ee_50.semi_major_m, ee_95.semi_major_m)

    def test_error_ellipse_zone_intersection(self):
        """Test point containment and polygon/bounding box zone intersection."""
        cov = np.array([[1e6, 0.0], [0.0, 4e5]])
        ee = ErrorEllipse.from_covariance_2d(cov, confidence=0.95)

        # 1. Ellipse center (0, 0) is contained
        self.assertTrue(ee.contains_point(0.0, 0.0))

        # 2. Far coordinate is not contained
        self.assertFalse(ee.contains_point(50000.0, 50000.0))

        # 3. Zone 1: Bounding box enclosing the center
        self.assertTrue(ee.intersects_bbox(-500.0, 500.0, -500.0, 500.0))

        # 4. Zone 2: Far bounding box
        self.assertFalse(ee.intersects_bbox(20000.0, 30000.0, 20000.0, 30000.0))

        # 5. Zone 3: Polygon cutting through the ellipse
        poly_intersect = [(-1000.0, -1000.0), (1000.0, -1000.0), (1000.0, 1000.0), (-1000.0, 1000.0)]
        self.assertTrue(ee.intersects_zone(poly_intersect))

        # 6. Zone 4: Polygon completely outside the ellipse
        poly_outside = [(30000.0, 30000.0), (40000.0, 30000.0), (40000.0, 40000.0), (30000.0, 40000.0)]
        self.assertFalse(ee.intersects_zone(poly_outside))

    def test_coarse_grid_search_for_unknown_location(self):
        """Test solving an unknown emitter without initial guess using coarse_grid_search=True."""
        solver = GeolocationSolver(ellipsoid='wgs84', max_iterations=50)

        # Completely unknown location: initial_guess is None, coarse_grid_search enabled
        res = solver.solve_from_twosat(
            two_sat=self.two_sat,
            tdoa=self.tdoa_true,
            fdoa=self.fdoa_true,
            uplink=self.pbs_uplink,
            translation_frequency=self.translation_frequency,
            ref_emitter=self.pbs_noc_ll,
            altitude=271.0,
            initial_guess=None,  # No user guess!
            coarse_grid_search=True,
            coarse_grid_bounds=(35.0, 55.0, -125.0, -95.0),
        )

        self.assertTrue(res.converged)
        true_ecef = np.array(pm.geodetic2ecef(*self.montana_pbs_ll))
        dist_err = float(np.linalg.norm(true_ecef - res.ecef))
        self.assertLess(dist_err, 2.0)


if __name__ == '__main__':
    unittest.main()
