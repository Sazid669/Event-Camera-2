"""
Hybrid loss.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/costs/hybrid.py
"""

import torch

from . import loss_functions
from .base import LossBase

from typing import Union, Dict, List


class LossHybrid(LossBase):
    """
    Hybrid class for all loss functions. Allows for merging of multiple loss functions at
    the same time.

    Attributes
    ----------
    losses : List[str]
        List of all desired loss functions.
    weights : List[float]
        Weights corresponding to each desired loss function. Note that the weight of each
        loss is related to its respective function by its index in the 'loss' list.
    batch_size : int
        If it is desired to compute the loss in batches increase the batch size more than
        1 (defaults to 1).
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

    name = "hybrid"

    def __init__(
        self,
        losses: List[str],
        weights: List[float],
        batch_size: int = 1,
        direction: str = "minimize",
        store_history: bool = False,
        precision: str = "64",
        use_cuda: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._losses = losses
        self._weights = weights
        self._batch_size = batch_size
        self._loss_functions = {
            name: {
                "func": loss_functions[name](
                    direction=direction,
                    store_history=store_history,
                    precision=precision,
                    use_cuda=use_cuda,
                    *args,
                    **kwargs
                ),
                "weight": weights[i]
            }
            for i, name in enumerate(self._losses)
        }
        super().__init__(
            direction=direction,
            store_history=store_history
        )

        # List of required keys encapsulates the required keys of all loss functions
        self.required_keys = []
        for name in self._loss_functions.keys():
            self.required_keys.extend(
                self._loss_functions[name]["func"].required_keys
            )

    def update_weights(self, losses: List[str], weights: List[str]) -> None:
        """
        Updates weights of all loss functions.
        """
        for i, name in enumerate(losses):
            self._loss_functions[name]["weight"] = weights[i]

    # TODO: History code needs some modifications
    def clear_history(self) -> None:
        """
        Clears the loss history for all loss functions.
        """
        self._history: Dict[str, List] = {"loss": []}
        for name in self._loss_functions.keys():
            self._loss_functions[name]["func"].clear_history()

    # TODO: History code needs some modifications
    def get_history(self) -> Dict[str, List]:
        """
        Returns the loss history for all loss functions.
        """
        all_histories = self._history.copy()
        for name in self._loss_functions.keys():
            all_histories.update(
                {name: self._loss_functions[name]["func"].get_history()["loss"]}
            )

        return all_histories

    # TODO: History code needs some modifications
    def enable_history(self) -> None:
        """
        Enables history for all loss functions.
        """
        self._store_history = True
        for name in self._loss_functions.keys():
            self._loss_functions[name]["func"]._store_history = True

    # TODO: History code needs some modifications
    def disable_history(self) -> None:
        """
        Disables history for all loss functions.
        """
        self._store_history = False
        for name in self._loss_functions.keys():
            self._loss_functions[name]["func"]._store_history = False

    @LossBase.save_history
    @LossBase.catch_key_error
    def calculate(self, args: Dict) -> Union[float, torch.Tensor]:
        """
        Main hybrid loss function, adds all loss values together with respect to their
        weights.
        """
        # TODO: CUDA flag should not be hard-coded
        loss = torch.zeros(self._batch_size).cuda()

        for name in self._loss_functions.keys():
            _loss = (
                self._loss_functions[name]["weight"]
                *self._loss_functions[name]["func"].calculate(args=args)
            )
            loss += _loss

        # TODO: Modify this with algorithm, especially when using higher batch sizes
        return loss
