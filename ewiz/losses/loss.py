import torch
import numpy as np

from . import loss_functions
from .hybrid import LossHybrid

from .warper import Warper
from .imager import ImagerTorch
from ..visualizers.visualizer import Visualizer, VizWindowManager

from typing import Optional, Tuple, List, Dict


class MotionCompensationLoss():
    """
    Motion compensation loss, used in conjunction with the included optimizer.

    Attributes
    ----------
    image_size : Tuple[int, int]
        Size of the image of shape (H, W).
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
    gt_flow : np.ndarray
        Ground truth flow of shape (2, H, W) (defaults to None). Only used for visualization.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        losses: List[str],
        weights: List[float],
        batch_size: int = 1,
        direction: str = "minimize",
        store_history: bool = False,
        precision: str = "64",
        use_cuda: bool = True,
        gt_flow: np.ndarray = None
    ) -> None:
        # TODO: Change visualization logic
        self.loss = 0.0
        if gt_flow is not None:
            self._gt_flow = gt_flow
        self._window_manager = VizWindowManager(
            image_size=image_size,
            grid_size=(2, 3),
            window_names=[
                "Events Image (Before Warp)",
                "Events Image (After Warp)",
                "IWE",
                "Compensation Flow",
                "Ground Truth Flow"
            ],
            refresh_rate=2
        )
        # Initialize hybrid loss function
        self._loss_function = LossHybrid(
            losses=losses,
            weights=weights,
            batch_size=batch_size,
            direction=direction,
            store_history=store_history,
            precision=precision,
            use_cuda=use_cuda
        )
        # Initialize other modules
        self._warper = Warper(image_size=image_size)
        self._imager = ImagerTorch(image_size=image_size, image_padding=(0, 0))
        self._visualizer = Visualizer(
            image_size=image_size,
            show_images=False,
            save_images=False
        )

    # TODO: Patch-based flow is optional
    def _parse_motion_compensation_args(
        self,
        events: torch.Tensor,
        flow: torch.Tensor,
        patch_flow: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Loss argument parser.
        """
        # Visualization images
        self.images = []

        loss_args = {"omit_boundary": True}

        if "ie" in self._loss_function.required_keys:
            ie = self._imager.generate_image_of_events(events=events)
            loss_args.update({"ie": ie})
            # Visualize 'Events Image (Before Warp)'
            _events = self._window_manager.convert_numpy(tensor=events)
            image = self._visualizer.visualize_events_image(events=_events)
            self.images.append(image)

        if "iwe" in self._loss_function.required_keys:
            warped_events = self._warper.warp_events(
                events=events,
                flow=flow,
                direction="start",
                motion_type="dense"
            )
            iwe = self._imager.generate_image_of_events(events=warped_events)
            loss_args.update({"iwe": iwe})

        if "start_iwe" in self._loss_function.required_keys:
            warped_events = self._warper.warp_events(
                events=events,
                flow=flow,
                direction="start",
                motion_type="dense"
            )
            start_iwe = self._imager.generate_image_of_events(events=warped_events)
            loss_args.update({"start_iwe": start_iwe})
            # Visualize 'Events Image (After Warp)'
            _events = self._window_manager.convert_numpy(tensor=warped_events)
            image = self._visualizer.visualize_events_image(events=_events)
            self.images.append(image)
            # Visualize 'IWE'
            iwee = self._window_manager.convert_numpy(tensor=start_iwe)
            image = self._visualizer.visualize_image_of_warped_events(IWE=iwee)
            self.images.append(image)

        if "middle_iwe" in self._loss_function.required_keys:
            warped_events = self._warper.warp_events(
                events=events,
                flow=flow,
                direction="middle",
                motion_type="dense"
            )
            middle_iwe = self._imager.generate_image_of_events(events=warped_events)
            loss_args.update({"middle_iwe": middle_iwe})

        if "end_iwe" in self._loss_function.required_keys:
            warped_events = self._warper.warp_events(
                events=events,
                flow=flow,
                direction="end",
                motion_type="dense"
            )
            end_iwe = self._imager.generate_image_of_events(events=warped_events)
            loss_args.update({"end_iwe": end_iwe})

        if "flow" in self._loss_function.required_keys:
            if patch_flow is None:
                loss_args.update({"flow": flow})
            else:
                loss_args.update({"flow": patch_flow})

        # Visualize 'Compensation Flow'
        _flow = self._window_manager.convert_numpy(tensor=flow)
        image = self._visualizer.visualize_optical_flow(flow=_flow)
        self.images.append(image)
        # Visualize 'Ground Truth Flow'
        image = self._visualizer.visualize_optical_flow(flow=self._gt_flow)
        self.images.append(image)

        # Render images
        self._window_manager.render_iter(*self.images)

        return loss_args

    # TODO: Merge with batch size
    def calculate(
        self,
        events: torch.Tensor,
        flow: torch.Tensor,
        patch_flow: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Loss calculator.
        """
        # Get arguments
        loss_args = self._parse_motion_compensation_args(
            events=events,
            flow=flow,
            patch_flow=patch_flow
        )
        # Calculate loss
        loss: torch.Tensor = self._loss_function.calculate(args=loss_args)

        # TODO: Adapt to notebook
        print("Loss =", loss.item(), end="\r")
        self.loss = loss.item()

        return loss
