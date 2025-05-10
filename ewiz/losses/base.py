"""
Base class for all loss functions.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/costs/base.py
"""

import torch

from typing import Union, Dict, List, Any


class LossBase():
    """
    Base class for all loss functions.

    Attributes
    ----------
    direction : str
        Desired direction of optimization, can choose between 'minimize', or 'maximize'
        (defaults to 'minimize').
    store_history : bool
        Stores the loss history internally (defaults to False).
    """

    required_keys: List[str] = []

    def __init__(
        self,
        direction: str = "minimize",
        store_history: bool = False,
        *args,
        **kwargs
    ) -> None:
        self._direction = direction
        self._store_history = store_history
        self.clear_history()

        if self._direction not in ["minimize", "maximize"]:
            error = (
                f"Direction should either be 'minimize' or "
                "'maximize', got {self._direction} instead."
            )
            raise ValueError(error)

    def catch_key_error(func: Any) -> Any:
        """
        Checks for the requires keys of the loss function. Each loss functions has its own
        set of required keys.
        """
        def wrapper(self, args: Dict) -> Any:
            try:
                return func(self, args)
            except KeyError as error:
                print("The cost function requires the following keys:")
                print(self.required_keys)
                raise error

        return wrapper

    def save_history(func: Any) -> Any:
        """
        Saves the loss value in the history register.
        """
        def wrapper(self, args: Dict) -> Any:
            loss = func(self, args)
            if self._store_history:
                self._history["loss"].append(self.get_item(loss))
            return loss

        return wrapper

    def get_item(self, loss: Union[float, torch.Tensor]) -> float:
        """
        Get the value of the cost function.
        """
        if isinstance(loss, torch.Tensor):
            return loss.item()

        return loss

    def clear_history(self) -> None:
        """
        Clears the loss history.
        """
        self._history: Dict[str, List] = {"loss": []}

    def get_history(self) -> Dict[str, List]:
        """
        Returns the loss history.
        """
        return self._history.copy()

    def enable_history(self) -> None:
        """
        Enables history.
        """
        self._store_history = True

    def disable_history(self) -> None:
        """
        Disables history.
        """
        self._store_history = False

    @save_history
    @catch_key_error
    def calculate(self, args: Dict) -> Union[float, torch.Tensor]:
        """
        Main function to calculate the loss. This function should be overridden by any
        custom implementation.
        """
        raise NotImplementedError

    save_history = staticmethod(save_history)
    catch_key_error = staticmethod(catch_key_error)
