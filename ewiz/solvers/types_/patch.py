"""
Patch object implementation.

-------------------------------------------------------------------------------------
NOTE: In its current form, eWiz does not currently use the patch object. But it might
be useful in the future.
-------------------------------------------------------------------------------------

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/types/flow_patch.py
"""

import copy
import numpy as np

from dataclasses import dataclass

from typing import Any, Tuple


@dataclass
class Patch():
    """
    Patch object data class. Contains the patch coordinates (location in the image) and
    the pixel-based velocity (value of the flow at that location).

    Attributes
    ----------
    x : np.int16
        The x coordinate value of the patch's center (width dimension).
    y : np.int16
        The y coordinate value of the patch's center (height dimension).
    u : float
        The pixel displacement value of the patch along the x dimension.
    v : float
        The pixel displacement value of the patch along the y dimension.
    """

    # Center coordinates (x is width, y is height)
    x: np.int16
    y: np.int16

    shape: Tuple

    # Flow displacement
    u: float = 0.0
    v: float = 0.0

    @property
    def h(self) -> int:
        """
        Returns the patch height.
        """
        return self.shape[0]    

    @property
    def w(self) -> int:
        """
        Returns the patch width.
        """
        return self.shape[1]    

    @property
    def x_min(self) -> int:
        """
        Returns minimum x patch coordinate.
        """
        return int(self.x - np.ceil(self.w/2))

    @property
    def x_max(self) -> int:
        """
        Returns maximum x patch coordinate.
        """
        return int(self.x + np.floor(self.w/2))

    @property
    def y_min(self) -> int:
        """
        Returns minimum y patch coordinate.
        """
        return int(self.y - np.ceil(self.h/2))

    @property
    def y_max(self) -> int:
        """
        Returns maximum y patch coordinate.
        """
        return int(self.y + np.floor(self.h/2))

    @property
    def position(self) -> np.ndarray:
        """
        Returns patch coordinates.
        """
        return np.array([self.x, self.y])

    @property
    def flow(self) -> np.ndarray:
        """
        Returns patch flow.
        """
        return np.array([self.u, self.v])

    def update_flow(self, u: float, v: float) -> None:
        """
        Updates flow displacement values.
        """
        self.u = u
        self.v = v

    def new_ones(self) -> np.ndarray:
        """
        Creates numpy array of ones, with the same
        size as the shape of the patch.
        """
        return np.ones(self.shape)

    def copy(self) -> Any:
        """
        Creates a copy of the patch.
        """
        return copy.deepcopy(self)