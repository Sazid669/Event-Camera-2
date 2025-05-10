"""
Normalized image variance.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/costs/normalized_image_variance.py
"""

import torch

from ..base import LossBase

from typing import Dict


class NormalizedImageVariance(LossBase):
    """
    Normalized image variance loss function.

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

    name = "normalized_image_variance"
    required_keys = ["ie", "iwe", "omit_boundary"]

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
        ie: torch.Tensor = args["ie"]
        iwe: torch.Tensor = args["iwe"]
        omit_boundary: bool = args["omit_boundary"]

        # Add batch and channel dimensions
        if len(ie.shape) == 2 or len(iwe.shape) == 2:
            ie = ie[None, None, ...]
            iwe = iwe[None, None, ...]
        elif len(ie.shape) == 3 or len(iwe.shape) == 3:
            ie = ie[None, None, ...]
            iwe = iwe[:, None, ...]

        # Apply precision conversion
        if self._precision == "64":
            ie = ie.double()
            iwe = iwe.double()

        if omit_boundary:
            ie = ie[..., 1:-1, 1:-1]
            iwe = iwe[..., 1:-1, 1:-1]

        # Calculate loss
        # TODO: For now this code only works for a batch size of 1
        loss_ie = torch.var(ie)
        loss_iwe = torch.var(iwe)

        if self._direction == "minimize":
            return loss_ie/loss_iwe

        return loss_iwe/loss_ie
