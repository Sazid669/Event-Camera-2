"""
Visualization functions.

Adapted from 'Optical Flow Visualization' and 'Secrets of Event-based Optical Flow':
https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis/flow_vis.py
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/visualizer.py
"""

import os

import cv2
import numpy as np
import torch

from PIL import Image

from typing import Tuple, List


# ================================ #
# ----- OpenCV Visualization ----- #
# ================================ #
class VizWindowManager():
    """
    An OpenCV window manager for all visualization functions.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        grid_size: Tuple[int, int],
        window_names: List[str],
        refresh_rate: int = 2
    ) -> None:
        self._image_size = image_size
        self._grid_size = grid_size
        self._window_names = window_names
        self._num_windows = len(window_names)
        self._refresh_rate = refresh_rate

    def render_iter(self, *args, **kwargs) -> None:
        """
        Renders windows for one iteration.
        """
        # Initialize indices
        h = 0
        w = 0
        for i in range(self._num_windows):
            # Create window
            cv2.namedWindow(self._window_names[i], 0)
            cv2.moveWindow(
                self._window_names[i],
                int(w*self._image_size[1]*1.8 + 100),
                int(h*self._image_size[0]*1.8 + 100)
            )
            # Create and show image
            image = self._convert_image(image=args[i])
            cv2.imshow(self._window_names[i], image)
            # Update indices
            w += 1
            if w == self._grid_size[1]:
                w = 0
                h += 1
        # Refresh windows
        cv2.waitKey(self._refresh_rate)

    def render(self, *args, **kwargs) -> None:
        """
        Renders windows and keeps them on screen.
        """
        # Initialize indices
        h = 0
        w = 0
        for i in range(self._num_windows):
            # Create window
            cv2.namedWindow(self._window_names[i], 0)
            cv2.moveWindow(
                self._window_names[i],
                int(w*self._image_size[1]*1.2 + 100),
                int(h*self._image_size[0]*1.2 + 100)
            )
            # Create and show image
            image = self._convert_image(image=args[i])
            cv2.imshow(self._window_names[i], image)
            # Update indices
            w += 1
            if w == self._grid_size[1]:
                w = 0
                h += 1
        # Keep windows
        cv2.waitKey(0)

    def convert_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Converts Torch tensor to numpy array.
        """
        data = tensor.clone()
        data = data.detach().cpu().numpy()

        return data

    def _convert_image(self, image: Image.Image) -> np.ndarray:
        """
        Converts PIL image to a Numpy array.
        """
        image = image.convert("RGB")
        image: np.ndarray = np.array(image)
        image = image[:, :, ::-1].copy()

        return image


# ================================= #
# ----- Visualization Manager ----- #
# ================================= #
class Visualizer():
    """
    Applies visualizations for event-based data related to:
    * Events (2-dimensional, colorized polarity, etc.)
    * Grayscale images (intensity, or 'RGB', depending on the camera type)
    * Optical flow (uses a color wheel as reference)

    The visualizer serves multiple functions, such as:
    * Actively visualizing the optimization algorithm.
    * Saving images to a desired directory.
    * Provides an interface to study event-based data.

    Attributes
    ----------
    image_size : Tuple[int, int]
        The size of the output image (in pixels) with size ('height', 'width') (defaults
        to (256, 256)).
    out_dir : str
        The output directory to save the output images (defaults to '.', meaning that the
        images will be saved in the same directory).
    prefix : str
        The filename prefix for all saved images. An internal count is kept track of, to
        avoid overriding any other images (defaults to 'image').
    save_images : bool
        Flag to save the images (defaults to False).
    show_images : bool
        Flag to show the images (defaults to False).
    image_format : str
        The format of the saved images (defaults to 'png').
    transparency : float
        Transparency for overlay of optical flow on top of grayscale image (defaults to 0.25).
    override : bool
        Overrides the same image instead of creating a new file for each output (defaults
        to False).
    """
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        out_dir: str = ".",
        prefix: str = "image",
        save_images: bool = False,
        show_images: bool = False,
        image_format: str = "png",
        transparency: float = 0.25,
        override: bool = False
    ) -> None:
        self.modify_image_size(image_size=image_size)
        self.modify_out_dir(out_dir=out_dir)
        self._init_prefix(prefix=prefix)

        # Initialize variables
        self._save_images = save_images
        self._show_images = show_images
        self._image_format = image_format
        self._transparency = transparency
        self._override = override

    # ============================= #
    # ----- General Functions ----- #
    # ============================= #
    def modify_image_size(self, image_size: Tuple[int, int]) -> None:
        """
        Modifies the image size of the class instance.
        """
        self._image_size = image_size

    def modify_out_dir(self, out_dir: str) -> None:
        """
        Modifies the output directory.
        """
        self._out_dir = out_dir
        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir)

    def _init_prefix(self, prefix: str) -> None:
        """
        Initializes prefix and prefix count.
        """
        self._prefix = prefix
        self._prefix_count = 0

    def _get_save_dir(self) -> str:
        """
        Returns the desired save directory taking into account the prefix and its count.
        """
        if self._override:
            save_dir = os.path.join(
                self._out_dir,
                f"{self._prefix}.{self._image_format}"
            )
        else:
            save_dir = os.path.join(
                self._out_dir,
                f"{self._prefix}{self._prefix_count}.{self._image_format}"
            )
            self._prefix_count += 1

        return save_dir

    def _generate_image(self, image: Image.Image) -> None:
        """
        Shows or saves the resulting image depending on the chosen 'show_images' and
        'save_images' parameters.
        """
        if self._show_images:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.show()

        if self._save_images:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image_save_dir = self._get_save_dir()
            image.save(image_save_dir)

    # ================================ #
    # ----- High-level Functions ----- #
    # ================================ #
    # ----- Grayscale Images ----- #
    def visualize_grayscale_image(self, image: np.ndarray) -> None:
        """
        Visualizes a grayscale image.
        """
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        self._generate_image(image=image)

        return image

    # ----- Events Images ----- #
    def visualize_events_image(
        self,
        events: np.ndarray,
        background_color: int = 255
    ) -> None:
        """
        Visualizes an events image.
        """
        image = self._generate_events_image(
            events=events,
            background_color=background_color
        )
        self._generate_image(image=image)

        return image

    # ----- Optical Flow ----- #
    def visualize_optical_flow(
        self,
        flow: np.ndarray,
        events_mask: np.ndarray = None
    ) -> None:
        """
        Visualizes an optical flow.
        """
        image = self._generate_flow_image(
            flow=flow,
            events_mask=events_mask
        )
        self._generate_image(image=image)

        return image

    # =============================== #
    # ----- Low-level Functions ----- #
    # =============================== #
    # ----- Events ----- #
    def _generate_events_image(
        self,
        events: np.ndarray,
        background_color: int = 255
    ) -> Image.Image:
        """
        Generates an events image.
        """
        events = self._clip_events(events=events)

        # Create 2-channel events image
        events_image = self._convert_events_to_image(
            events=events,
            image_size=self._image_size
        )
        # Create 'RGB' events image
        events_image = self._generate_events_image_rgb(
            events_image=events_image,
            background_color=background_color
        )

        image = Image.fromarray(events_image)

        return image

    def _clip_events(self, events: np.ndarray) -> np.ndarray:
        """
        Clips events within bounds of the image size.
        """
        events[:, 0] = np.clip(events[:, 0], 0, self._image_size[1] - 1)
        events[:, 1] = np.clip(events[:, 1], 0, self._image_size[0] - 1)

        return events

    def _convert_events_to_image(
        self,
        events: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Converts a sequence of events to a 2-channel events image.

        Parameters
        ----------
        events : np.ndarray
            Events array, extracted from the '.hdf5' file.
        image_size : Tuple[int, int]
            The size of the output image, equal to the size of the sensor of format (H, W).

        Returns
        -------
        events_image : np.ndarray
            Output events image of shape (2, H, W).
        """
        pos_events = events[events[:, 3] > 0]
        neg_events = events[events[:, 3] < 0]

        pos_xs = pos_events[:, 0].astype(np.int32)
        pos_ys = pos_events[:, 1].astype(np.int32)
        neg_xs = neg_events[:, 0].astype(np.int32)
        neg_ys = neg_events[:, 1].astype(np.int32)

        events_image = np.zeros(((2,) + image_size))

        # Create events image
        np.add.at(events_image[0, ...], (pos_ys, pos_xs), 1)
        np.add.at(events_image[1, ...], (neg_ys, neg_xs), 1)

        return events_image.astype(np.uint8)

    def _generate_events_image_rgb(
        self,
        events_image: np.ndarray,
        background_color: int = 255
    ) -> np.ndarray:
        """
        Generates an 'RGB' image from an events image.
        """
        image_size = (events_image.shape[1], events_image.shape[2])
        image = np.zeros(((3,) + image_size), dtype=np.uint8) + background_color

        # Get polarity indices
        pos_ids = np.where(events_image[0, ...] > 0)
        neg_ids = np.where(events_image[1, ...] > 0)

        # Create 'RGB' image
        image[:, pos_ids[0], pos_ids[1]] = np.array([255, 0, 0])[:, None]
        image[:, neg_ids[0], neg_ids[1]] = np.array([0, 0, 255])[:, None]
        image = np.transpose(image, (1, 2, 0))

        return image

    # ----- Optical Flow ----- #
    def _generate_flow_image(
        self,
        flow: np.ndarray,
        events_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Generates an 'RGB' optical flow image.
        """
        magnitudes = np.linalg.norm(flow, axis=0)
        angles = np.arctan2(flow[1, :, :], flow[0, :, :])

        # Process angles
        angles += np.pi
        angles *= 180.0/np.pi/2.0
        angles = angles.astype(np.uint8)

        # Color map
        color_map = np.zeros([flow.shape[1], flow.shape[2], 3], dtype=np.uint8)
        color_map[:, :, 0] = np.mod(angles, 180)
        color_map[:, :, 1] = 255
        color_map[:, :, 2] = cv2.normalize(magnitudes, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to 'RGB' image
        flow_image = cv2.cvtColor(color_map, cv2.COLOR_HSV2BGR)

        if events_mask is not None:
            flow_image = flow_image*events_mask[0, 0, ...]

        # Convert to 'PIL' image
        flow_image = Image.fromarray(flow_image)

        return flow_image

    def visualize_image_of_warped_events(
        self,
        IWE: torch.Tensor
    ) -> None:
        image = np.zeros(self._image_size, dtype=np.uint8)
        image = (IWE/10)*255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        self._generate_image(image=image)

        return image