"""
Multi-focal normalized image variance.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/costs/multi_focal_normalized_image_variance.py
"""

import torch

from ..base import LossBase
from .normalized import NormalizedImageVariance

from typing import Dict


class MultifocalNormalizedImageVariance(LossBase):
    """
    Multi-focal normalized image variance loss function.

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

    name = "multifocal_normalized_image_variance"
    required_keys = ["ie", "start_iwe", "middle_iwe", "end_iwe", "omit_boundary"]

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

        # Initialize normalized image variance
        self._normalized_image_variance = NormalizedImageVariance(
            direction=direction,
            store_history=store_history,
            precision=precision,
            use_cuda=use_cuda
        )

    @LossBase.save_history
    @LossBase.catch_key_error
    def calculate(self, args: Dict) -> torch.Tensor:
        ie: torch.Tensor = args["ie"]
        start_iwe: torch.Tensor = args["start_iwe"]
        middle_iwe: torch.Tensor = args["middle_iwe"]
        end_iwe: torch.Tensor = args["end_iwe"]
        omit_boundary: bool = args["omit_boundary"]

        loss_start = self._normalized_image_variance.calculate(
            args={
                "ie": ie,
                "iwe": start_iwe,
                "omit_boundary": omit_boundary
            }
        )
        loss_middle = self._normalized_image_variance.calculate(
            args={
                "ie": ie,
                "iwe": middle_iwe,
                "omit_boundary": omit_boundary
            }
        )
        loss_end = self._normalized_image_variance.calculate(
            args={
                "ie": ie,
                "iwe": end_iwe,
                "omit_boundary": omit_boundary
            }
        )

        loss = loss_start + loss_middle*2 + loss_end

        if self._direction == "minimize":
            return loss

        return -loss
