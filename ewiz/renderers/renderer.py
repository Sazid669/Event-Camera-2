import numpy as np

from .formats import rendering_formats

from typing import List, Tuple


class Renderer():
    """
    Event-based data renderer. The renderer can either render event-based data in 3D (named
    'volume'), or as a 2D sequence, similar to the DAVIS renderer (named 'video').

    To slide over the whole sequence, choose the 'slide' method. To plot one specific sequence,
    use the 'plot' method:
    * For format 'volume': Both, 'slide' and 'plot' are available.
    * For format 'video': Only 'slide' is available.

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
    rendering_format : str
        The visualization format, you can either render event-based data in 3D by choosing
        'volume', or as a 2D sequence by choosing 'video'.
    rendering_method : str
        The rendering approach, you can either slide over the events sequence ('slide' option),
        or plot the whole sequence ('plot' option) at once.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        events: np.ndarray,
        grayscale_images: np.ndarray = None,
        grayscale_timestamps: np.ndarray = None,
        rendering_format: str = "video",
        rendering_method: str = "plot",
        *args,
        **kwargs
    ) -> None:
        # Initialize variables
        self._image_size = image_size
        self._events = events
        self._grayscale_images = grayscale_images
        self._grayscale_timestamps = grayscale_timestamps

        # Method variables
        self._rendering_format = rendering_format
        self._rendering_method = rendering_method

        # Initialize renderer
        self._renderer = rendering_formats[self._rendering_format](
            image_size=self._image_size,
            events=self._events,
            grayscale_images=self._grayscale_images,
            grayscale_timestamps=self._grayscale_timestamps,
            *args,
            **kwargs
        )

    def run(self) -> None:
        """
        Runs the renderer after initialization.
        """
        rendering_func = getattr(self._renderer, self._rendering_method)
        # Run renderer
        rendering_func(
            events=self._events,
            grayscale_images=self._grayscale_images,
            grayscale_timestamps=self._grayscale_timestamps
        )
