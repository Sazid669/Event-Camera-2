"""
Flow regularizer.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/costs/total_variation.py
"""

import torch

from ..base import LossBase
from ..sobel import SobelTorch

from typing import Dict


class Regularizer(LossBase):
    """
    Flow regularizer, ensures the optical flow is smoothened out across the image.

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

    name = "regularizer"
    required_keys = ["flow", "omit_boundary"]

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
            in_channels=2,
            kernel_size=3,
            precision=self._precision,
            use_cuda=self._use_cuda
        )

    @LossBase.save_history
    @LossBase.catch_key_error
    def calculate(self, args: Dict) -> torch.Tensor:
        flow: torch.Tensor = args["flow"]
        omit_boundary: bool = args["omit_boundary"]

        # Add batch dimensions
        if len(flow.shape) == 3:
            flow = flow[None, ...]

        # Apply precision conversion
        if self._precision == "64":
            flow = flow.double()

        # Calculate Sobel image
        sobel_image = self._sobel_torch.forward(flow)/8.0

        # TODO: If very few patches are available, we should not omit boundary
        if omit_boundary:
            if sobel_image.shape[2] > 2 and sobel_image.shape[3] > 2:
                sobel_image = sobel_image[..., 1:-1, 1:-1]

        # Calculate loss
        # TODO: For now this code only works for a batch size of 1
        loss = torch.mean(torch.abs(sobel_image))

        if self._direction == "minimize":
            return loss

        return -loss
