"""
Normalized gradient magnitude.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/costs/normalized_gradient_magnitude.py
"""

import torch

from ..base import LossBase
from .gradient import GradientMagnitude

from typing import Dict


class NormalizedGradientMagnitude(LossBase):
    """
    Normalized gradient magnitude loss function.

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

    name = "normalized_gradient_magnitude"
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

        # Initialize gradient magnitude loss
        self._gradient_magnitude = GradientMagnitude(
            direction=direction,
            store_history=store_history,
            precision=precision,
            use_cuda=use_cuda
        )

    @LossBase.save_history
    @LossBase.catch_key_error
    def calculate(self, args: Dict) -> torch.Tensor:
        ie: torch.Tensor = args["ie"]
        iwe: torch.Tensor = args["iwe"]
        omit_boundary: bool = args["omit_boundary"]

        loss_iwe = self._gradient_magnitude.calculate(
            args={
                "iwe": iwe,
                "omit_boundary": omit_boundary
            }
        )
        loss_ie = self._gradient_magnitude.calculate(
            args={
                "iwe": ie,
                "omit_boundary": omit_boundary
            }
        )

        if self._direction == "minimize":
            return loss_ie/loss_iwe

        return loss_iwe/loss_ie
