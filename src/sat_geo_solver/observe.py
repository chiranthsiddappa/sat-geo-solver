from datetime import datetime
from typing import List, Tuple

import numpy as np
from skyfield.api import load as skyf_load
from skyfield.api import wgs84
from skyfield.timelib import Timescale

__ts__ = skyf_load.timescale()


def dt_to_ts(dt: datetime | List[datetime]) -> Timescale:
    """
    Convert datetime or list of datetimes to Skyfield Timescale.

    :param dt: Datetime or list of datetimes
    :return: Skyfield Timescale object
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