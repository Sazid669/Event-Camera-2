"""
Imager functions, responsible for the creation of various representations of event-based
data.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/event_image_converter.py
"""

import torch
import numpy as np

from torchvision.transforms.functional import gaussian_blur
from scipy.ndimage.filters import gaussian_filter

from typing import Union, Tuple


# ========================= #
# ----- Torch Version ----- #
# ========================= #
class ImagerTorch():
    """
    Imager functions are responsible for the creation of various image representations of
    event-based data.

    Currently supported features include:
    * Image of Warped Events (IWE): Create an IWE based on the number of events that took
    place at each pixel. In case the warped events do not have integer values for their pixel
    IDs, bilinear interpolation can be used to determine the value of the resulting pixel image.
    * Events Mask: A mask of locations where events took place. Such information is valuable
    for computing evaluation metrics.

    Attributes
    ----------
    image_size : Tuple[int, int]
        Image size of shape (H, W).
    image_padding : Tuple[int, int]
        Image padding of shape (H, W) (defaults to (0, 0)).
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        image_padding: Tuple[int, int] = (0, 0)
    ) -> None:
        self._update_image_properties(image_size=image_size, image_padding=image_padding)

    def generate_mask_of_events(self, events: torch.Tensor) -> torch.Tensor:
        """
        Generates mask of events.
        """
        mask = (
            self.generate_image_of_events(events=events, sigma_blur=0) != 0
        )[..., None, :, :]

        return mask

    def generate_image_of_events(
        self,
        events: torch.Tensor,
        method: str = "bilinear",
        sigma_blur: float = 1.0,
        weights: Union[float, torch.Tensor] = 1.0
    ) -> torch.Tensor:
        """
        Generates the Image of Warped Events (IWE).
        """
        if method == "count":
            image = self._count(events=events)
        elif method == "bilinear":
            image = self._bilinear_vote(events=events, weights=weights)
        else:
            error = f"The method '{method}' is not supported."
            raise NotImplementedError(error)

        if sigma_blur > 0.0:
            if len(image.shape) == 2:
                image = image[None, None, ...]
            elif len(image.shape) == 3:
                image = image[:, None, ...]
            image = gaussian_blur(image, kernel_size=3, sigma=sigma_blur)

        return torch.squeeze(image)

    def _update_image_properties(
        self,
        image_size: Tuple[int, int],
        image_padding: Tuple[int, int] = (0, 0)
    ) -> None:
        """
        Updates the image size and its padding simultaneously.
        """
        self._image_size = image_size
        self._padding = image_padding
        self._image_size = tuple(
            int(s + 2*p) for s, p in zip(self._image_size, self._padding)
        )

    def _count(self, events: torch.Tensor) -> torch.Tensor:
        """
        Applies a count operation for creating the IWE.
        """
        # Add batch dimension
        if len(events.shape) == 2:
            events = events[None, ...]

        batch_size = len(events)
        pad_h, pad_w = self._padding
        h, w = self._image_size

        # Create initial image
        image = events.new_zeros((batch_size, h*w))

        coords = torch.floor(events[..., 2] + 1e-6)

        xs = coords[..., 0] + pad_w
        ys = coords[..., 1] + pad_h

        pos_ids = torch.cat(
            [
                 xs      +  ys     *w,
                 xs      + (ys + 1)*w,
                (xs + 1) +  ys     *w,
                (xs + 1) + (ys + 1)*w
            ],
            dim=-1
        )
        mask_ids = torch.cat(
            [
                (0 <= xs)    *(xs < w)    *(0 <= ys)    *(ys < h),
                (0 <= xs)    *(xs < w)    *(0 <= ys + 1)*(ys + 1 < h),
                (0 <= xs + 1)*(xs + 1 < w)*(0 <= ys)    *(ys < h),
                (0 <= xs + 1)*(xs + 1 < w)*(0 <= ys + 1)*(ys + 1 < h)
            ],
            dim=-1
        )

        # Get the pixel values
        pixel_vals = torch.ones_like(pos_ids)
        pos_ids = (pos_ids*mask_ids).long()
        pixel_vals = pixel_vals*mask_ids

        # Construct the image
        image.scatter_add_(1, pos_ids, pixel_vals)

        # TODO: Check batch size
        return image.reshape((batch_size,), self._image_size).squeeze()

    def _bilinear_vote(
        self,
        events: torch.Tensor,
        weights: Union[float, torch.Tensor] = 1.0
    ) -> torch.Tensor:
        """
        Applies a bilinear vote for creating the IWE.
        """
        # Either float or torch.Tensor can be given as weights
        if type(weights) == torch.Tensor:
            assert weights.shape == events.shape[:-1], (
                "Incompatible shapes for 'weights' and 'events'."
            )

        # Add batch dimension
        if len(events.shape) == 2:
            events = events[None, ...]

        batch_size = len(events)
        pad_h, pad_w = self._padding
        h, w = self._image_size

        # Create initial image
        image = events.new_zeros((batch_size, h*w))

        coords = torch.floor(events[..., :2] + 1e-6)
        diff_coords = events[..., :2] - coords
        coords = coords.long()

        xs = coords[..., 0] + pad_w
        ys = coords[..., 1] + pad_h

        pos_ids = torch.cat(
            [
                 xs      +  ys     *w,
                 xs      + (ys + 1)*w,
                (xs + 1) +  ys     *w,
                (xs + 1) + (ys + 1)*w
            ],
            dim=-1
        )
        mask_ids = torch.cat(
            [
                (0 <= xs)    *(xs < w)    *(0 <= ys)    *(ys < h),
                (0 <= xs)    *(xs < w)    *(0 <= ys + 1)*(ys + 1 < h),
                (0 <= xs + 1)*(xs + 1 < w)*(0 <= ys)    *(ys < h),
                (0 <= xs + 1)*(xs + 1 < w)*(0 <= ys + 1)*(ys + 1 < h)
            ],
            dim=-1
        )

        pos0 = (1 - diff_coords[..., 1])*(1 - diff_coords[..., 0])*weights
        pos1 =  diff_coords[..., 1]     *(1 - diff_coords[..., 0])*weights
        pos2 = (1 - diff_coords[..., 1])* diff_coords[..., 0]     *weights
        pos3 =  diff_coords[..., 1]     * diff_coords[..., 0]     *weights

        # Get the pixel values
        pixel_vals = torch.cat([pos0, pos1, pos2, pos3], dim=-1)
        pos_ids = (pos_ids*mask_ids).long()
        pixel_vals = pixel_vals*mask_ids

        # Construct the image
        image.scatter_add_(1, pos_ids, pixel_vals)

        # TODO: Check batch size
        return image.reshape((batch_size,) + self._image_size).squeeze()


# ========================= #
# ----- Numpy Version ----- #
# ========================= #
class ImagerNumpy():
    """
    Imager functions are responsible for the creation of various image representations of
    event-based data.

    Currently supported features include:
    * Image of Warped Events (IWE): Create an IWE based on the number of events that took
    place at each pixel. In case the warped events do not have integer values for their pixel
    IDs, bilinear interpolation can be used to determine the value of the resulting pixel image.
    * Events Mask: A mask of locations where events took place. Such information is valuable
    for computing evaluation metrics.

    Attributes
    ----------
    image_size : Tuple[int, int]
        Image size of shape (H, W).
    image_padding : Tuple[int, int]
        Image padding of shape (H, W) (defaults to (0, 0)).
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        image_padding: Tuple[int, int] = (0, 0)
    ) -> None:
        self._update_image_properties(image_size=image_size, image_padding=image_padding)

    def generate_mask_of_events(self, events: np.ndarray) -> np.ndarray:
        """
        Generates mask of events.
        """
        mask = (
            self.generate_image_of_events(events=events, sigma_blur=0) != 0
        )[..., None, :, :]

        return mask

    def generate_image_of_events(
        self,
        events: np.ndarray,
        method: str = "bilinear",
        sigma_blur: float = 1.0,
        weights: Union[float, np.ndarray] = 1.0
    ) -> np.ndarray:
        """
        Generates the Image of Warped Events (IWE).
        """
        if method == "count":
            image = self._count(events=events)
        elif method == "bilinear":
            image = self._bilinear_vote(events=events, weights=weights)
        else:
            error = f"The method '{method}' is not supported."
            raise NotImplementedError(error)

        if sigma_blur > 0.0:
            image = gaussian_filter(image, sigma_blur)

        return image

    def _update_image_properties(
        self,
        image_size: Tuple[int, int],
        image_padding: Tuple[int, int] = (0, 0)
    ) -> None:
        """
        Updates the image size and its padding simultaneously.
        """
        self._image_size = image_size
        self._padding = image_padding
        self._image_size = tuple(
            int(s + 2*p) for s, p in zip(self._image_size, self._padding)
        )

    def _count(self, events: np.ndarray) -> np.ndarray:
        """
        Applies a count operation for creating the IWE.
        """
        # Add batch dimension
        if len(events.shape) == 2:
            events = events[None, ...]

        batch_size = len(events)
        pad_h, pad_w = self._padding
        h, w = self._image_size

        # Create initial image
        image = np.zeros((batch_size, h*w), dtype=np.float64)

        coords = np.floor(events[..., 2] + 1e-6)

        xs = coords[..., 0] + pad_w
        ys = coords[..., 1] + pad_h

        pos_ids = np.concatenate(
            [
                 xs      +  ys     *w,
                 xs      + (ys + 1)*w,
                (xs + 1) +  ys     *w,
                (xs + 1) + (ys + 1)*w
            ],
            axis=-1
        )
        mask_ids = np.concatenate(
            [
                (0 <= xs)    *(xs < w)    *(0 <= ys)    *(ys < h),
                (0 <= xs)    *(xs < w)    *(0 <= ys + 1)*(ys + 1 < h),
                (0 <= xs + 1)*(xs + 1 < w)*(0 <= ys)    *(ys < h),
                (0 <= xs + 1)*(xs + 1 < w)*(0 <= ys + 1)*(ys + 1 < h)
            ],
            axis=-1
        )

        # Get the pixel values
        pixel_vals = np.ones_like(pos_ids)
        pos_ids = (pos_ids*mask_ids).astype(np.int64)
        pixel_vals = pixel_vals*mask_ids

        # Construct the image
        for i in range(batch_size):
            np.add.at(image[i], pos_ids[i], pixel_vals[i])

        # TODO: Check batch size
        return image.reshape((batch_size,), self._image_size).squeeze()

    def _bilinear_vote(
        self,
        events: np.ndarray,
        weights: Union[float, np.ndarray] = 1.0
    ) -> np.ndarray:
        """
        Applies a bilinear vote for creating the IWE.
        """
        # Either float or np.ndarray can be given as weights
        if type(weights) == np.ndarray:
            assert weights.shape == events.shape[:-1], (
                "Incompatible shapes for 'weights' and 'events'."
            )

        # Add batch dimension
        if len(events.shape) == 2:
            events = events[None, ...]

        batch_size = len(events)
        pad_h, pad_w = self._padding
        h, w = self._image_size

        # Create initial image
        image = np.zeros((batch_size, h*w), dtype=np.float64)

        coords = np.floor(events[..., 2] + 1e-6)
        diff_coords = events[..., 2] - coords

        xs = coords[..., 0] + pad_w
        ys = coords[..., 1] + pad_h

        pos_ids = np.concatenate(
            [
                 xs      +  ys     *w,
                 xs      + (ys + 1)*w,
                (xs + 1) +  ys     *w,
                (xs + 1) + (ys + 1)*w
            ],
            axis=-1
        )
        mask_ids = np.concatenate(
            [
                (0 <= xs)    *(xs < w)    *(0 <= ys)    *(ys < h),
                (0 <= xs)    *(xs < w)    *(0 <= ys + 1)*(ys + 1 < h),
                (0 <= xs + 1)*(xs + 1 < w)*(0 <= ys)    *(ys < h),
                (0 <= xs + 1)*(xs + 1 < w)*(0 <= ys + 1)*(ys + 1 < h)
            ],
            axis=-1
        )

        pos0 = (1 - diff_coords[..., 1])*(1 - diff_coords[..., 0])*weights
        pos1 =  diff_coords[..., 1]     *(1 - diff_coords[..., 0])*weights
        pos2 = (1 - diff_coords[..., 1])* diff_coords[..., 0]     *weights
        pos3 =  diff_coords[..., 1]     * diff_coords[..., 0]     *weights

        # Get the pixel values
        pixel_vals = np.concatenate([pos0, pos1, pos2, pos3], axis=-1)
        pos_ids = (pos_ids*mask_ids).astype(np.int64)
        pixel_vals = pixel_vals*mask_ids

        # Construct the image
        for i in range(batch_size):
            np.add.at(image[i], pos_ids[i], pixel_vals[i])

        # TODO: Check batch size
        return image.reshape((batch_size,), self._image_size).squeeze()
