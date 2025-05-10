import cv2
import numpy as np

from .base import RendererBase

from typing import Tuple


class Renderer2D(RendererBase):
    """
    A 2D video renderer for event-based data, opens an OpenCV window to render the data
    stream.

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
    num_events : int
        The number of events to render in the same frame (defaults to 30000).
    refresh_rate : int
        The number of frames to render per second.
    """

    _name = "video"

    def __init__(
        self,
        image_size: Tuple[int, int],
        events: np.ndarray,
        grayscale_images: np.ndarray = None,
        grayscale_timestamps: np.ndarray = None,
        num_events: int = 30000,
        refresh_rate: int = 40,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            image_size=image_size,
            events=events,
            grayscale_images=grayscale_images,
            grayscale_timestamps=grayscale_timestamps,
            *args,
            **kwargs
        )
        # Initialize 2D renderer variables
        self._num_events = num_events
        self._refresh_rate = refresh_rate

    # ===== Common Functions ===== #
    def slide(self, *args, **kwargs) -> None:
        """
        Slides over a sequence of events.
        """
        xs = self._events[:, 0]
        ys = self._events[:, 1]
        ts = self._events[:, 2]
        ps = self._events[:, 3]

        # Get number of frames
        num_frames = xs.shape[0]//self._num_events
        # Initialize filters
        red_filter = np.zeros((*self._image_size, 3))
        red_filter[:, :, 2] = 255
        blue_filter = np.zeros((*self._image_size, 3))
        blue_filter[:, :, 0] = 255
        white_filter = np.zeros((*self._image_size, 3))
        white_filter[:, :, :] = 255

        # Iterate over frames
        for i in range(num_frames):
            xs_slice = xs[i*self._num_events:i*self._num_events + self._num_events]
            ys_slice = ys[i*self._num_events:i*self._num_events + self._num_events]
            ts_slice = ts[i*self._num_events:i*self._num_events + self._num_events]
            ps_slice = ps[i*self._num_events:i*self._num_events + self._num_events]

            # Mask of events
            events_mask = self.compute_mask(xs=xs_slice, ys=ys_slice, ps=ps_slice)

            # Mean timestamp
            t_mean = np.mean(ts_slice)
            
            # Render grayscale images
            if self._grayscale_images is not None and self._grayscale_timestamps is not None:
                gray_idx = np.searchsorted(self._grayscale_timestamps, t_mean)
                gray_image: np.ndarray = self._grayscale_images[gray_idx]

            # Convert to 'RGB' image with 3 channels
            events_mask = np.expand_dims(events_mask, axis=2)
            events_mask = np.repeat(events_mask, 3, axis=2)

            # Get pixel indices
            pos_ids = np.any(events_mask > 0, axis=2, keepdims=True)
            neg_ids = np.any(events_mask < 0, axis=2, keepdims=True)
            gray_ids = np.any(events_mask == 0, axis=2, keepdims=True)

            # Replace with event color
            events_mask = np.where(pos_ids, red_filter, events_mask)
            events_mask = np.where(neg_ids, blue_filter, events_mask)

            # Update image
            if self._grayscale_images is not None and self._grayscale_timestamps is not None:
                # Grayscale background
                gray_image = gray_image.astype(np.uint8)
                gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                events_mask = np.where(gray_ids, gray_image, events_mask)
            else:
                # White background
                events_mask = np.where(gray_ids, white_filter, events_mask)

            # Convert to image-compatible type
            events_mask = events_mask.astype(np.uint8)

            # Create OpenCV window
            cv2.namedWindow("eWiz: Video 2D Renderer", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("eWiz: Video 2D Renderer", 720, 720)

            # Insert timestamp text
            text = "t = " + "%.2f" % t_mean + " s"
            text_coords = (15, 30)
            events_mask = cv2.putText(
                events_mask, text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                2, cv2.LINE_AA
            )

            # Render frame
            cv2.imshow("eWiz: Video 2D Renderer", events_mask)
            cv2.waitKey(self._refresh_rate)

        # Destroy all windows
        cv2.destroyAllWindows()

    # TODO: Not available for 2D renderer
    def plot(self, *args, **kwargs) -> None:
        """
        Plots a sequence of events.
        """
        raise NotImplementedError
    # ============================ #


"""
## ===== Under Development ===== ##
violet_filter = np.zeros((*self._image_size, 3))
violet_filter[:, :, :] = np.array([127, 0, 255])
green_filter = np.zeros((*self._image_size, 3))
green_filter[:, :, 1] = 255
pos_ids = np.any(events_mask == 2, axis=2, keepdims=True)
neg_ids = np.any(events_mask == -2, axis=2, keepdims=True)
events_mask = np.where(pos_ids, violet_filter, events_mask)
events_mask = np.where(pos_ids, green_filter, events_mask)
## ============================= ##
"""
