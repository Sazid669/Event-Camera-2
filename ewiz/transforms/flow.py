"""
Optical flow transforms, compatible with the 'tonic' library.
"""

import numpy as np

from dataclasses import dataclass

from typing import Tuple


@dataclass(frozen=True)
class FlowCenterCrop():
    """
    Crops an optical flow array around its center.

    Attributes
    ----------
    out_size : Tuple[int, int]
        The output size of shape (H, W).
    """

    out_size: Tuple[int, int]

    def __call__(self, flow: np.ndarray) -> np.ndarray:
        """
        Function call.
        """
        # Get offsets
        off_x = int((flow.shape[2] - self.out_size[1])/2)
        off_y = int((flow.shape[1] - self.out_size[0])/2)

        # Crop flow
        flow = flow[..., off_y:-off_y, off_x:-off_x]

        return flow
