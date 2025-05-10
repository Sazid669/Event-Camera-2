import numpy as np

from typing import Tuple


class RendererBase():
    """
    Base rendering class.

    Attributes
    ----------
    image_size : Tuple[int, int]
        The sensor size with shape (H, W), where H is the height dimension, and W is the
        width dimension.
    events : Union[np.ndarray, h5py.Dataset]
        The events data which consists of one array of shape ('num_events', 4), where the
        second dimension corresponds to (x, y, t, p) respectively.
    grayscale_images : Union[np.ndarray, h5py.Dataset]
        The extracted grayscale images which consist of one array of shape ('num_frames', H, W).
    grayscale_timestamps : Union[np.ndarray, h5py.Dataset]
        The timestamp of each grayscale image in the form of a 1D array of shape ('num_frames').
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        events: np.ndarray,
        grayscale_images: np.ndarray = None,
        grayscale_timestamps: np.ndarray = None,
        *args,
        **kwargs
    ) -> None:
        # Initialize variables
        self._image_size = image_size
        self._events = events
        self._grayscale_images = grayscale_images
        self._grayscale_timestamps = grayscale_timestamps

    # TODO: Deprecated function
    def compute_mask(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ps: np.ndarray
    ) -> np.ndarray:
        """
        Computes mask.
        """
        reps = np.zeros(self._image_size, dtype=np.int32)
        
        # Compute representation
        coords = np.ravel_multi_index(
            (ys.astype(np.int32), xs.astype(np.int32)),
            self._image_size
        )

        reps = np.bincount(
            coords,
            weights=ps,
            minlength=self._image_size[0]*self._image_size[1]
        )
        reps = reps.reshape(self._image_size)

        return reps

    # TODO: This function is overridden
    def slide(self, *args, **kwargs) -> None:
        """
        Slides over a sequence of events.
        """
        raise NotImplementedError
    
    # TODO: This function is overridden
    def plot(self, *args, **kwargs) -> None:
        """
        Plots a sequence of events.
        """
        raise NotImplementedError
