"""
Pyramidal patch-based motion compensation.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/solver/patch_contrast_pyramid.py
"""

import numpy as np

import torch
import skimage.transform

from autograd_minimize import minimize

from .patch import PatchMotionCompensation
from .optimizers import SCIPY_OPTIMIZERS

from typing import Tuple, Dict, Any, List


class PyramidalPatchMotionCompensation(PatchMotionCompensation):
    """
    Pyramidal patch-based motion compensation.

    Attributes
    ----------
    image_size : Tuple[int, int]
        Image size of shape (H, W) (defaults to (256, 256)).
    optimizer : str
        Type of Scipy optimizer (defaults to 'BFGS'). Check Scipy documentation for a list
        of possible optimizers.
    init_method : str
        Method of initialization. You can choose between 'random', or 'zero' (defaults to
        'random').
    random_inits : Tuple[float, float]
        Random initialization boundaries (defaults to (-20.0, 20.0)).
    loss_function : Any
        Desired loss function (defaults to None).
    """
    SUPPORTED_OPTIMIZERS=[
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP',
    'trust-constr',
    'dogleg',
    'trust-ncg',
    'trust-krylov',
    'trust-exact'
]

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        optimizer: List[str] = [
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP',
    'trust-constr',
    'dogleg',
    'trust-ncg',
    'trust-krylov',
    'trust-exact'
]
,  # List of optimizers
        selected_optimizer: str = 'BFGS', #parent optimizer
        init_method: str = "random",
        random_inits: Tuple[float, float] = (-20.0, 20.0),
        loss_function: Any = None,
        scale_range: Tuple[int, int] = (1,5)
    ) -> None:
        super().__init__(
            image_size=image_size,
            optimizer=selected_optimizer,
            init_method=init_method,
            random_inits=random_inits,
            loss_function=loss_function
        )
        # Validate selected optimizer
        if selected_optimizer not in self.SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Selected optimizer '{selected_optimizer}' is not supported.")

        self.optimizers = optimizer
        self.selected_optimizer = selected_optimizer

        print("Initializing pyramidal patch-based motion compensation...")

        # Placeholder variables
        self.patch_size = (0, 0)
        self.patch_stride = (0, 0)
        self.patch_grid_size = (0, 0)
        self.patches = {}
        self.num_patches = 0

        # TODO: No need for this
        self.patch_shift = (
            (self.image_size[0] - self.image_size[0])//2,
            (self.image_size[0] - self.image_size[0])//2
        )
        self.cropped_image_height = 260
        self.cropped_image_width = 346
        self.cropped_image_size = (self.cropped_image_height, self.cropped_image_width)

        # Placeholder variables
        self.scaled_patch_size = {}
        self.scaled_patch_stride = {}
        self.scaled_patch_grid_size = {}
        self.scaled_patches = {}
        self.scaled_num_patches = {}
        self.total_num_patches = 0

        # TODO: Add as configs
        self.coarsest_scale = scale_range[0]
        self.finest_scale = scale_range[1]

        # TODO: Add support for time-based optimization
        self.optimized_patch_flows = None

        # Prepare patches
        self.prepare_pyramidal_patches(
            image_size=self.image_size,
            coarsest_scale=self.coarsest_scale,
            finest_scale=self.finest_scale
        )

    # =============================== #
    # -------- Patches Setup -------- #
    # =============================== #
    def prepare_pyramidal_patches(
        self,
        image_size: Tuple[int, int],
        coarsest_scale: int,
        finest_scale: int
    ) -> None:
        """
        Prepares pyramidal patches.
        """
        self.scaled_patch_size = {}
        self.scaled_patch_stride = {}
        self.scaled_patch_grid_size = {}
        self.scaled_patches = {}
        self.scaled_num_patches = {}
        self.total_num_patches = 0

        # Get patches for all scales
        for i in range(coarsest_scale, finest_scale):
            scaled_size = (image_size[0]//(2**i), image_size[1]//(2**i))
            self.scaled_patch_size[i] = scaled_size
            self.scaled_patch_stride[i] = scaled_size
            self.scaled_patches[i], self.scaled_patch_grid_size[i] = self.prepare_patches(
                image_size=image_size,
                patch_size=scaled_size,
                patch_stride=scaled_size
            )
            self.scaled_num_patches[i] = len(self.scaled_patches[i].keys())
            self.total_num_patches += self.scaled_num_patches[i]

    # ======================================== #
    # -------- Patches Configurations -------- #
    # ======================================== #
    def overload_patches_configs(self, scale: int) -> None:
        """
        Load configurations of desired scale.
        """
        self.current_scale = scale

        self.patch_size = self.scaled_patch_size[scale]
        self.patch_grid_size = self.scaled_patch_grid_size[scale]
        self.patch_stride = self.scaled_patch_stride[scale]
        self.patches = self.scaled_patches[scale]
        self.num_patches = self.scaled_num_patches[scale]

    # ====================================== #
    # -------- Patches Manipulation -------- #
    # ====================================== #
    def update_fine_to_coarse_flow(self, patch_flows: Dict) -> Dict:
        """
        Refines coarsest patch flows from finer flows.
        """
        coarsest_scale = min(patch_flows.keys())
        finest_scale = max(patch_flows.keys())

        refined_patch_flows = {finest_scale: patch_flows[finest_scale]}

        for i in range(finest_scale, coarsest_scale - 1, -1):
            refined_patch_flows[i - 1] = skimage.transform.pyramid_reduce(
                patch_flows[i], channel_axis=0
            )

        return refined_patch_flows

    def convert_patch_to_dense_flow(self, patch_flows: Dict) -> torch.Tensor:
        """
        Patch-based flow to dense flow converter, only for pyramidal flows.

        Returns
        -------
        dense_flow : torch.Tensor
            Dense flow at the current scale.
        """
        # Extract flow at current scale
        current_flow = patch_flows[self.current_scale]

        # TODO: Add Numpy implementation
        if isinstance(current_flow, torch.Tensor):
            dense_flow = self.patch_to_dense(patch_flow=current_flow)
        else:
            error = (
                f"Patch flow type '{type(current_flow)}' is not supported, "
                "it should be 'torch.Tensor' instead."
            )
            raise TypeError(error)

        return dense_flow

    # ===================================== #
    # -------- Objective Functions -------- #
    # ===================================== #
    def objective_function(
        self,
        patch_flow: np.ndarray,
        events: np.ndarray,
        patch_flows: Dict
    ) -> torch.Tensor:
        """
        Objective function for optimization.
        """
        assert self.current_scale not in patch_flows.keys(), (
            f"Flow already computed at '{self.current_scale}', check your code."
        )
        
        # Compute dense flow from patch-based prediction
        _patch_flows = patch_flows.copy()
        _patch_flows.update({self.current_scale: patch_flow})
        dense_flow = self.convert_patch_to_dense_flow(patch_flows=_patch_flows)

        # Patch flow regularization
        patch_flow = -_patch_flows[self.current_scale].reshape(
            (1, 2) + self.patch_grid_size
        )[0]

        # Calculate loss
        # TODO: Check time scale option
        loss = self.loss_function.calculate(
            flow=dense_flow,
            events=events,
            patch_flow=patch_flow
        )

        return loss

    # ======================================== #
    # -------- Optimization Functions -------- #
    # ======================================== #
    def optimize(self, events: np.ndarray) -> Tuple[Dict, Any]:
        """
        Main optimization function.
        """
        print("Starting optimizer...")
        print(
            f"Total degrees of freedom (DOFs): '{2*self.total_num_patches}' "
            "for all scales."
        )

        # Run algorithm
        patch_flows, optimizer_results = self.run_for_all_scales(events=events)
        print("Done with optimizer.")

        # Refine flow
        patch_flows = self.update_fine_to_coarse_flow(patch_flows=patch_flows)
        print("Flow refined.")

        return patch_flows, optimizer_results

    def run_for_all_scales(self, events: np.ndarray) -> Tuple[Dict, Any]:
        """
        Optimization for all scales.
        """
        patch_flows = {}

        # TODO: Add device option
        if self.optimizer in SCIPY_OPTIMIZERS:
            events = torch.from_numpy(events).double().requires_grad_().to("cuda")

        for scale in range(self.coarsest_scale, self.finest_scale):
            self.overload_patches_configs(scale=scale)
            print(f"The current scale is '{scale}'.")

            if self.optimizer in SCIPY_OPTIMIZERS:
                optimizer_results = self.run(
                    events=events,
                    patch_flows=patch_flows
                )
                # Update patch flows
                patch_flows[scale] = optimizer_results.x.reshape(
                    ((2,) + self.patch_grid_size)
                )
            else:
                error = (
                    f"Optimizer '{self.optimizer}' is not supported, "
                    "check Scipy documentation."
                )
                raise NotImplementedError(error)

        return patch_flows, optimizer_results

    def run(self, events: np.ndarray, patch_flows: np.ndarray) -> Any:
        """
        Optimization for one scale.
        """
        # TODO: Add support for time-based optimization
        if (self.optimized_patch_flows is not None 
            and self.current_scale == self.coarsest_scale):
            print("Use previously optimized patch flows.")

            # Initialized flow
            flow0 = np.copy(self.optimized_patch_flows[self.current_scale])

        # TODO: Check condition here
        elif self.current_scale > self.coarsest_scale:
            print("Use coarser patch flow.")

            # Initialized flow
            flow0 = skimage.transform.pyramid_expand(
                patch_flows[self.current_scale - 1], channel_axis=0
            ).reshape(-1)

            # TODO: Add support for time-based optimization
            if self.optimized_patch_flows is not None:
                flow0 = (
                    flow0 + self.optimized_patch_flows[self.current_scale].reshape(-1)
                )/2

            # TODO: Might yield better results without random initialization
            # if self.optimized_patch_flows is not None:
            #     pass
        else:
            if self.init_method == "random":
                flow0 = self._init_random()
            elif self.init_method == "zero":
                flow0 = self._init_zero()

        # Run optimization
        # TODO: Add more options as arguments
        opt_opts = {
            "gtol": 1e-5,
            "disp": True,
            "maxiter": 80,
            "eps": 1
        }

        optimizer_results = minimize(
            fun=lambda x: self.objective_function(x, events, patch_flows),
            x0=flow0,
            method=self.optimizer,
            options=opt_opts,
            backend="torch",
            precision="float64",
            torch_device="cuda"
        )

        return optimizer_results
