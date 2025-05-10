"""
Vanilla gradient magnitude.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/costs/gradient_magnitude.py
"""

import torch

from ..base import LossBase
from ..sobel import SobelTorch

from typing import Dict


class GradientMagnitude(LossBase):
    """
    Vanilla gradient magnitude loss function.

    Attributes
    ----------
    direction : str
        Desired direction of optimization, can choose between 'minimize', or 'maximize'
        (defaults to 'minimize').
    store_history : bool
        Stores the loss history internally (defaults to False).
    precision : str
        Floating point precision to use, can choose between '64' or '32' (defaults to '64').
    use_cuda : bool
        Use CUDA for computations (defaults to True).
    """

    name = "gradient_magnitude"
    required_keys = ["iwe", "omit_boundary"]

    def __init__(
        self,
        direction: str = "minimize",
        store_history: bool = False,
        precision: str = "64",
        use_cuda: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            direction=direction,
            store_history=store_history
        )
        self._precision = precision
        self._use_cuda = use_cuda

        # Initialize Sobel operator
        self._sobel_torch = SobelTorch(
            in_channels=1,
            kernel_size=3,
            precision=self._precision,
            use_cuda=self._use_cuda
        )

    @LossBase.save_history
    @LossBase.catch_key_error
    def calculate(self, args: Dict) -> torch.Tensor:
        iwe: torch.Tensor = args["iwe"]
        omit_boundary: bool = args["omit_boundary"]

        # Add batch and channel dimensions
        if len(iwe.shape) == 2:
            iwe = iwe[None, None, ...]
        elif len(iwe.shape) == 3:
            iwe = iwe[:, None, ...]

        # Apply precision conversion
        if self._precision == "64":
            iwe = iwe.double()

        # Calculate Sobel image
        sobel_image = self._sobel_torch.forward(iwe)/8.0
        sobel_x = sobel_image[:, 0]
        sobel_y = sobel_image[:, 1]

        if omit_boundary:
            sobel_x = sobel_x[..., 1:-1, 1:-1]
            sobel_y = sobel_y[..., 1:-1, 1:-1]

        # Calculate loss
        # TODO: For now this code only works for a batch size of 1
        loss = torch.mean(torch.square(sobel_x) + torch.square(sobel_y))

        if self._direction == "minimize":
            return -loss

        return loss
