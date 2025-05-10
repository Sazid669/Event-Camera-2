"""
Transforms for event-based data, compatible with the 'tonic' library. These transforms
can be temporal or spatial, however, they can only be applied on structured events arrays
(read 'tonic' documentation and source code). Randomness is also compatible with 'tonic'
and 'eWiz' for reproducibility. Currently implemented transforms:
* Data Structure:
    - EventsToStructured
    - EventsToUnstructured
* Spatial Transforms:
    - EventsFlipXY
    - EventsRandomRotation
    - EventsCenterCrop

Usage Example
-------------
If you are using unstructured event-based data apply the transforms below:
    transforms = tonic.transforms.Compose([
        EventsToStructured(),
            ...
        EventsToUnstructured()
    ])
"""

import numpy as np
import numpy.lib.recfunctions as nlr

from dataclasses import dataclass

from typing import Tuple


@dataclass(frozen=True)
class EventsCenterCropUns():
    """
    Crop unstructured events around sensor's center.

    Attributes
    ----------
    sensor_size : Tuple[int, int]
        Size of the event-based sensor of shape (W, H).
    out_size : Tuple[int, int]
        Output size of the cropped image of shape (W, H).
    """

    sensor_size: Tuple[int, int]
    out_size: Tuple[int, int]

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """
        Function call.
        """
        # Get offsets
        off_x = int((self.sensor_size[0] - self.out_size[0])/2)
        off_y = int((self.sensor_size[1] - self.out_size[1])/2)

        # Crop events
        events = events.copy()
        events = events[np.logical_and(
            np.logical_and(events[:, 0] >= off_x, events[:, 0] < self.sensor_size[0] - off_x),
            np.logical_and(events[:, 1] >= off_y, events[:, 1] < self.sensor_size[1] - off_y)
        )]
        events[:, 0] = events[:, 0] - off_x
        events[:, 1] = events[:, 1] - off_y

        return events
