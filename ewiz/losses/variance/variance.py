"""
Vanilla image variance.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/costs/image_variance.py
"""

import torch

from ..base import LossBase

from typing import Dict


class ImageVariance(LossBase):
    """
    Vanilla image variance loss function.

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

    name = "image_variance"
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

        if omit_boundary:
            iwe = iwe[..., 1:-1, 1:-1]

        # Calculate loss
        # TODO: For now this code only works for a batch size of 1
        loss = torch.var(iwe)

        if self._direction == "minimize":
            return -loss

        return loss
