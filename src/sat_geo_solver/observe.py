from datetime import datetime
from typing import List

import numpy as np
from skyfield.api import load as skyf_load
from skyfield.api import wgs84

__ts__ = skyf_load.timescale()


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
    if isinstance(at, datetime):
        observe_ts = __ts__.from_datetime(at)
    elif isinstance(at, list):
        observe_ts = __ts__.from_datetimes(at)
    else:
        raise TypeError(f"Invalid type for 'at': {type(at)}")
    return loc_diff.at(observe_ts).distance().km * 1000