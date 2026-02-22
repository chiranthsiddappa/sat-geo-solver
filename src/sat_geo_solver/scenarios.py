from datetime import datetime
from typing import List

import numpy as np

from sat_geo_solver.observe import Observe


class TwoSat:

    def __init__(self, primary_satellite,
                 secondary_satellite,
                 at: datetime | List[datetime],
                 primary_receiver: list | np.ndarray,
                 secondary_receiver: list | np.ndarray,):
        self.primary = Observe(primary_satellite, at)
        self.secondary = Observe(secondary_satellite, at)
        self.at = at
        self.primary_receiver = primary_receiver
        self.secondary_receiver = secondary_receiver

    def dto(self, from_pos: list | np.ndarray) -> float | np.ndarray:
        """
        Calculate the differential time offset of a signal from a given position between the two satellites.
        Secondary - Primary

        :param from_pos: Position from which the signal is emitted. [lat_deg, lon_deg, elevation_m]
        :return: Time difference of arrival in light seconds
        """
        primary_signal_path = (
            self.primary.light_seconds(from_pos)
            + self.primary.light_seconds(self.primary_receiver)
        )
        secondary_signal_path = (
            self.secondary.light_seconds(from_pos)
            + self.secondary.light_seconds(self.secondary_receiver)
        )
        return secondary_signal_path - primary_signal_path

    def dfo(self,
            from_pos: list | np.ndarray,
            uplink: float,
            translation_frequency: float) -> float | np.ndarray:
        """
        Calculate the differential frequency offset between the two satellite paths for a bent-pipe link.
        Secondary - Primary

        This models the case where the same uplink is (primarily) received by the primary satellite,
        but some energy is also received by the secondary satellite (e.g., sidelobes), and each satellite
        downlinks to its own receiver.

        :param from_pos: Position from which the signal is emitted. [lat_deg, lon_deg, elevation_m]
        :param uplink: Uplink frequency in Hz.
        :param translation_frequency: Translation frequency in Hz.
        :return: Differential received frequency in Hz (secondary_rx - primary_rx)
        """
        f_primary = self.primary.downlink_received_frequency(
            from_pos=from_pos,
            to_pos=self.primary_receiver,
            uplink=uplink,
            translation_frequency=translation_frequency,
        )
        f_secondary = self.secondary.downlink_received_frequency(
            from_pos=from_pos,
            to_pos=self.secondary_receiver,
            uplink=uplink,
            translation_frequency=translation_frequency,
        )
        return f_secondary - f_primary