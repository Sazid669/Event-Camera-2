"""
Events warping functions, supports batched events.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/warp.py
"""

import torch
import numpy as np

from typing import Tuple, Union


class Warper():
    """
    Warper class warps events to create an Image of Warped Events (IWE).

    Attributes
    ----------
    image_size : Tuple[int, int]
        Image size of shape (H, W).
    """

    def __init__(self, image_size: Tuple[int, int]) -> None:
        self._image_size = image_size

    def _reference_timestamp(
        self,
        events: torch.Tensor,
        direction: Union[str, float] = "start"
    ) -> torch.Tensor:
        """
        Calculates the reference timestamp.
        """
        if type(direction) is float:
            max_time = torch.max(events[..., 2], dim=-1).values
            min_time = torch.min(events[..., 2], dim=-1).values
            delta_time = max_time - min_time

            return min_time + delta_time*direction

        # Convert string input to float
        elif direction == "start":
            return torch.min(events[..., 2], dim=-1).values
        elif direction == "middle":
            return self._reference_timestamp(events=events, direction=0.5)
        elif direction == "end":
            return torch.max(events[..., 2], dim=-1).values
        elif direction == "random":
            random_pos = np.random.uniform(low=0.0, high=1.0)
            return self._reference_timestamp(events=events, direction=random_pos)
        elif direction == "before":
            return self._reference_timestamp(events=events, direction=-1.0)
        elif direction == "after":
            return self._reference_timestamp(events=events, direction=2.0)

        error = f"Direction '{direction}' is not supported."
        raise ValueError(error)

    # TODO: Check batch support
    def _warp_events_from_dense_flow(
        self,
        events: torch.Tensor,
        flow: torch.Tensor,
        reference_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Warps events using a dense optical flow.
        """
        # Calculate delta time
        delta_time = events[..., 2] - reference_time

        # Add batch dimension
        if len(events.shape) == 2:
            events = events[None, ...]
            flow = flow[None, ...]
            reference_time = reference_time[None, ...]
            delta_time = delta_time[None, ...]

        # Check shape compatibility
        assert (len(delta_time.shape) + 1 == len(flow.shape) - 1 == 3), (
            "Incompatible shapes in 'Warper' object."
        )

        # Initialize warped events
        warped_events = events.clone()

        # TODO: Optimize algorithm
        reshaped_flow = flow.reshape((flow.shape[0], 2, -1))
        events_ids = events[..., 1].long()*self._image_size[0] + events[..., 0].long()

        # Override warped events
        warped_events[..., 0] = events[..., 0] - delta_time*torch.gather(
            reshaped_flow[:, 0], 1, events_ids
        )
        warped_events[..., 1] = events[..., 1] - delta_time*torch.gather(
            reshaped_flow[:, 1], 1, events_ids
        )
        warped_events[..., 2] = delta_time

        return warped_events.squeeze()

    def warp_events(
        self,
        events: torch.Tensor,
        flow: torch.Tensor,
        direction: Union[str, float],
        motion_type: str
    ) -> torch.Tensor:
        """
        Main function to warp events.
        """
        # Get reference timestamp
        reference_time = self._reference_timestamp(events=events, direction=direction)

        # TODO: Add more motion types
        if motion_type == "dense":
            return self._warp_events_from_dense_flow(
                events=events,
                flow=flow,
                reference_time=reference_time
            )
