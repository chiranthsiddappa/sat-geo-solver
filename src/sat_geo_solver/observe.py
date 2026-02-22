from datetime import datetime
from typing import List, Tuple

import numpy as np
from skyfield.api import load as skyf_load
from skyfield.api import wgs84
from skyfield.constants import C as SPEED_OF_LIGHT
from skyfield.timelib import Time

__ts__ = skyf_load.timescale()


def doppler(frequency: float | np.ndarray, range_rate: float | np.ndarray) -> float | np.ndarray:
    """
    Classical (non-relativistic) Doppler: compute received frequency from a line-of-sight range rate.

    Sign convention (Skyfield):
      - range_rate > 0 means range increasing (receding) -> received frequency decreases

    :param frequency: Transmitted frequency in Hz
    :param range_rate: Line-of-sight range rate in m/s (positive = moving away)
    :return: Received frequency in Hz
    """
    return frequency * (1.0 - (range_rate / SPEED_OF_LIGHT))


def relativistic_doppler(frequency: float | np.ndarray, range_rate: float | np.ndarray) -> float | np.ndarray:
    """
    Relativistic Doppler: compute received frequency from a line-of-sight range rate.

    Sign convention (Skyfield):
      - range_rate > 0 means range increasing (receding)
      - Doppler beta is positive for approaching, so beta = -range_rate / c

    :param frequency: Transmitted frequency in Hz
    :param range_rate: Line-of-sight range rate in m/s (positive = moving away)
    :return: Received frequency in Hz
    """
    beta = -range_rate / SPEED_OF_LIGHT
    return frequency * np.sqrt((1.0 + beta) / (1.0 - beta))


def dt_to_ts(dt: datetime | List[datetime]) -> Time:
    """
    Convert datetime or list of datetimes to Skyfield Time.

    :param dt: Datetime or list of datetimes
    :return: Skyfield Time object
    """
    if isinstance(dt, datetime):
        return __ts__.from_datetime(dt)
    elif isinstance(dt, list):
        return __ts__.from_datetimes(dt)
    else:
        raise TypeError(f"Invalid type for 'dt': {type(dt)}")


def distance_to(sat, from_pos: list | np.ndarray, at: datetime | List[datetime]) -> float | List[float]:
    """
    Provide the distance to the satellite from a given position at a given time.

    :param sat: Satellite to observe
    :param from_pos: Position to observe from
    :param at: Time to observe at
    :return: Distance in meters
    """
    pos = wgs84.latlon(*from_pos)
    loc_diff = sat - pos
    observe_ts = dt_to_ts(at)
    return loc_diff.at(observe_ts).distance().km * 1000


def range_and_rate(sat, from_pos: list | np.ndarray, at: datetime | List[datetime]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the range and rate of a satellite from a given position at a given time.

    :param sat: Satellite to observe
    :param from_pos: Position to observe from
    :param at: Time to observe at
    :return: Tuple of range and rate in meters and meters per second
    """
    pos = wgs84.latlon(*from_pos)
    loc_diff = sat - pos
    observe_ts = dt_to_ts(at)
    _, _, range_distance, _, _, range_rate = loc_diff.at(observe_ts).frame_latlon_and_rates(pos)
    return range_distance.km * 1000, range_rate.km_per_s * 1000


def light_seconds(sat, from_pos: list | np.ndarray, at: datetime | List[datetime]) -> float | np.ndarray:
    """
    Calculate the light time between the observer and the satellite.

    :param sat: Satellite to observe
    :param from_pos: Position to calculate light time from
    :param at: Time to calculate light time at
    :return: Light time in seconds
    """
    pos = wgs84.latlon(*from_pos)
    loc_diff = sat - pos
    observe_ts = dt_to_ts(at)
    return loc_diff.at(observe_ts).distance().light_seconds()


class Observe:

    def __init__(self, sat, at: datetime | List[datetime]):
        self.sat = sat
        self.observe_ts = dt_to_ts(at)
        self.sat_at = sat.at(self.observe_ts)

    def distance_to(self, from_pos: list | np.ndarray) -> float | List[float]:
        """
        Provide the distance to the satellite from a given position.

        :param from_pos: Position to observe from
        :return: Distance to the satellite in meters
        """
        loc_diff = self.sat_at - wgs84.latlon(*from_pos).at(self.observe_ts)
        return loc_diff.distance().km * 1000

    def range_and_rate(self, from_pos: list | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the range and rate of the satellite from a given position.

        :param from_pos: Position to observe from
        :return: Range distance in meters, range rate in meters per second
        """
        pos = wgs84.latlon(*from_pos)
        loc_diff = self.sat_at - pos.at(self.observe_ts)
        _, _, range_distance, _, _, range_rate = loc_diff.frame_latlon_and_rates(pos)
        return range_distance.m, range_rate.m_per_s

    def pos_xyz(self) -> np.ndarray:
        """
        Produce the xyz position of the satellite in meters.

        :return: np.ndarray
        """
        return self.sat_at.position.km * 1000

    def vel_xyz(self) -> np.ndarray:
        """
        Produce the xyz velocity of the satellite in meters per second.

        :return: np.ndarray
        """
        return self.sat_at.velocity.km_per_s * 1000

    def light_seconds(self, from_pos: list | np.ndarray) -> float | np.ndarray:
        """
        Calculate the light time between the observer and the satellite.

        :param from_pos: Position to calculate light time from
        :return: Light time in seconds
        """
        pos = wgs84.latlon(*from_pos)
        loc_diff = self.sat_at - pos.at(self.observe_ts)
        return loc_diff.distance().light_seconds()

    def downlink_received_frequency(self,
                                    from_pos: list | np.ndarray,
                                    to_pos: list | np.ndarray,
                                    uplink: float,
                                    translation_frequency: float) -> float | np.ndarray:
        """
        Bent-pipe link budget in terms of frequency only.

        Steps:
          1) Uplink: ground transmitter -> satellite receiver (uplink Doppler)
          2) Translate at satellite: f_down_tx = f_up_rx - translation_frequency
          3) Downlink: satellite transmitter -> ground receiver (downlink Doppler)

        :param from_pos: Position from which the signal is emitted.
        :param to_pos: Position at which the signal is received.
        :param uplink: Uplink frequency in Hz.
        :param translation_frequency: Translation frequency in Hz.
        :return: Final received downlink frequency at the ground receiver in Hz
        """

        _, rr_uplink = self.range_and_rate(from_pos)
        f_up_rx_at_satellite = relativistic_doppler(uplink, rr_uplink)

        f_down_tx = f_up_rx_at_satellite - translation_frequency

        _, rr_downlink = self.range_and_rate(to_pos)
        return relativistic_doppler(f_down_tx, rr_downlink)

    def doppler_shift(self,
                      from_pos: list | np.ndarray,
                      to_pos: list | np.ndarray,
                      uplink: float,
                      translation_frequency: float) -> float | np.ndarray:
        """
        Final bent-pipe Doppler shift in Hz, relative to the nominal translated frequency:
            f_nominal = uplink - translation_frequency

        :param from_pos: Position from which the signal is emitted.
        :param to_pos: Position at which the signal is received.
        :param uplink: Uplink frequency in Hz.
        :param translation_frequency: Translation frequency in Hz.
        :return: (f_down_rx_at_ground - f_nominal) in Hz
        """
        f_down_rx_at_ground = self.downlink_received_frequency(
            from_pos=from_pos,
            to_pos=to_pos,
            uplink=uplink,
            translation_frequency=translation_frequency,
        )
        f_nominal = uplink - translation_frequency
        return f_down_rx_at_ground - f_nominal

    def fdoa(self,
             from_pos: list | np.ndarray,
             rx1_pos: list | np.ndarray,
             rx2_pos: list | np.ndarray,
             uplink: float,
             translation_frequency: float) -> float | np.ndarray:
        """
        Frequency Difference of Arrival (FDOA) for a bent-pipe satellite:
            FDOA = f_rx(rx1) - f_rx(rx2)

        :param from_pos: Position from which the signal is emitted.
        :param rx1_pos: Position of the first receiver.
        :param rx2_pos: Position of the second receiver.
        :param uplink: Uplink frequency in Hz.
        :param translation_frequency: Translation frequency in Hz.
        :return: Frequency difference in Hz (same shape as the underlying observation times)
        """
        f1 = self.downlink_received_frequency(
            from_pos=from_pos,
            to_pos=rx1_pos,
            uplink=uplink,
            translation_frequency=translation_frequency,
        )
        f2 = self.downlink_received_frequency(
            from_pos=from_pos,
            to_pos=rx2_pos,
            uplink=uplink,
            translation_frequency=translation_frequency,
        )
        return f1 - f2

    def __repr__(self):
        return f"Observe(sat={self.sat}, at={self.observe_ts})"
