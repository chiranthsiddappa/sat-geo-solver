import unittest
from datetime import timedelta

import numpy as np
from numpy.testing import assert_allclose
from skyfield.api import load as skyf_load
from skyfield.iokit import parse_tle_file

from sat_geo_solver.observe import (Observe, distance_to, light_seconds,
                                    range_and_rate)


class TestObserve(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load timescale
        cls.ts = skyf_load.timescale()

        # Load satellite from g19.tle
        with open('tests/g19.tle', 'rb') as f:
            cls.satellites = list(parse_tle_file(f, cls.ts))
        cls.sat = cls.satellites[0]

        # Observer location from notebook
        cls.cos_ll = [+38.4, -104.82]

        # Datetimes from notebook (±12h from epoch, 15m steps)
        epoch_dt = cls.sat.epoch.utc_datetime()
        start_dt = epoch_dt - timedelta(hours=12)
        end_dt = epoch_dt + timedelta(hours=12)

        cls.dt_observables = []
        curr_dt = start_dt
        while curr_dt <= end_dt:
            cls.dt_observables.append(curr_dt)
            curr_dt += timedelta(minutes=15)

        # Observe instance
        cls.observe = Observe(cls.sat, cls.dt_observables)
        cls.observe_epoch = Observe(cls.sat, cls.sat.epoch.utc_datetime())

    def test_distance_to_single_epoch(self):
        expected = distance_to(self.sat, self.cos_ll, self.sat.epoch.utc_datetime())
        actual = self.observe_epoch.distance_to(self.cos_ll)
        assert_allclose(actual, expected)

    def test_distance_to(self):
        # Compare Observe.distance_to with standalone distance_to
        expected = distance_to(self.sat, self.cos_ll, self.dt_observables)
        actual = self.observe.distance_to(self.cos_ll)

        assert_allclose(actual, expected)

    def test_range_and_rate_single_epoch(self):
        expected_range, expected_rate = range_and_rate(self.sat, self.cos_ll, self.sat.epoch.utc_datetime())
        actual_range, actual_rate = self.observe_epoch.range_and_rate(self.cos_ll)

        assert_allclose(actual_range, expected_range)
        assert_allclose(actual_rate, expected_rate)

    def test_range_and_rate(self):
        # Compare Observe.range_and_rate with standalone range_and_rate
        expected_range, expected_rate = range_and_rate(self.sat, self.cos_ll, self.dt_observables)
        actual_range, actual_rate = self.observe.range_and_rate(self.cos_ll)

        assert_allclose(actual_range, expected_range)
        assert_allclose(actual_rate, expected_rate)

    def test_light_seconds_single_epoch(self):
        expected = light_seconds(self.sat, self.cos_ll, self.sat.epoch.utc_datetime())
        actual = self.observe_epoch.light_seconds(self.cos_ll)
        assert_allclose(actual, expected)

    def test_light_seconds(self):
        # Compare Observe.light_seconds with standalone light_seconds
        expected = light_seconds(self.sat, self.cos_ll, self.dt_observables)
        actual = self.observe.light_seconds(self.cos_ll)

        assert_allclose(actual, expected)

    def test_pos_vel_xyz_output(self):
        # Verify that pos_xyz and vel_xyz return numpy arrays with xyz
        pos = self.observe.pos_xyz()
        vel = self.observe.vel_xyz()

        self.assertIsInstance(pos, np.ndarray)
        self.assertIsInstance(vel, np.ndarray)

        # Since it's a list of datetimes, shape should be (3, N) for skyfield position/velocity
        self.assertEqual(pos.shape, (3, len(self.dt_observables)))
        self.assertEqual(vel.shape, (3, len(self.dt_observables)))


if __name__ == '__main__':
    unittest.main()
