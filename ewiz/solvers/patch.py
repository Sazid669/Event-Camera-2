"""
Patch-based motion compensation.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/solver/patch_contrast_base.py
"""

import numpy as np

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from .types_.patch import Patch
from ..losses import LossHybrid

from typing import Any, Dict, Tuple,List


class PatchMotionCompensation():
    """
    Patch-based motion compensation.

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
        loss_function: Any = None
    ) -> None:
        
        # Initialize variables
        self.image_size = image_size
        self.optimizer = optimizer
        self.init_method = init_method
        self.random_inits = random_inits
        self.loss_function = loss_function

        # Placeholder variables
        self.patch_size = (0, 0)
        self.patch_stride = (0, 0)
        self.patch_grid_size = (0, 0)
        self.patches = {}
        self.num_patches = 0
        if selected_optimizer not in self.SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Selected optimizer '{selected_optimizer}' is not supported.")

        self.optimizers = optimizer
        self.selected_optimizer = selected_optimizer

        print("Initializing pyramidal patch-based motion compensation...")
    def _init_random(self) -> np.ndarray:
        """
        Random patch initialization.
        """
        print("Random patch initialization...")
        flow0 = np.random.rand(2, self.num_patches).astype(np.float64)
        flow0[0] = (
            flow0[0]*(self.random_inits[1] - self.random_inits[0]) + self.random_inits[0]
        )
        flow0[1] = (
            flow0[1]*(self.random_inits[1] - self.random_inits[0]) + self.random_inits[0]
        )

        return flow0

    def _init_zero(self) -> np.ndarray:
        """
        Zero patch initialization.
        """
        print("Zero patch initialization...")
        flow0 = np.random.rand(2, self.num_patches).astype(np.float64)

        return flow0

    def prepare_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        patch_stride: Tuple[int, int]
    ) -> Tuple[Dict[int, Patch], Tuple[int, int]]:
        """
        Prepares patches.

        Parameters
        ----------
        image_size : Tuple[int, int]
            Image size of shape (H, W).
        patch_size : Tuple[int, int]
            Patch size of shape (H, W).
        patch_stride : Tuple[int, int]
            Patch stride of shape (H, W).

        Returns
        -------
        patches : Dict[int, Patch]
            Dictionary containing the patch ID as key, and its corresponding patch object.
        patch_grid_size : Tuple[int, int]
            Patch grid size, can be explained as patch-based flow resolution.
        """
        h, w = image_size
        patch_h, patch_w = patch_size
        stride_h, stride_w = patch_stride

        centers_x = np.arange(0, w - patch_w + stride_w, stride_w) + patch_w//2
        centers_y = np.arange(0, h - patch_h + stride_h, stride_h) + patch_h//2

        grid_x, grid_y = np.meshgrid(centers_x, centers_y)

        patch_grid_size = grid_x.shape

        coords_x = grid_x.reshape(-1)
        coords_y = grid_y.reshape(-1)

        # Create patches
        patches = {
            i: Patch(
                x=coords_x[i], y=coords_y[i], shape=patch_size, u=0.0, v=0.0
            )
            for i in range(0, len(coords_x))
        }

        return patches, patch_grid_size

    def patch_to_dense(self, patch_flow: torch.Tensor) -> torch.Tensor:
        """
        Patch-based flow to dense flow converter.

        Parameters
        ----------
        patch_flow : torch.Tensor
            Patch-based flow of shape (2, 'num_patches').

        Returns
        -------
        dense_flow : torch.Tensor
            Dense flow with the same shape as the 'image_size'.
        """
        # TODO: Old padding method
        # patch_grid_flow = torch.nn.functional.pad(
        #     -patch_flow.reshape((1, 2) + self.patch_grid_size),
        #     (1, 1, 1, 1),
        #     mode="replicate"
        # )[0]

        patch_grid_flow = -patch_flow.reshape((1, 2) + self.patch_grid_size)[0]

        # TODO: Add more interpolation options
        interpol_mode = InterpolationMode.BILINEAR
        dense_flow = F.resize(
            patch_grid_flow,
            self.image_size,
            interpol_mode,
            antialias=True
        )
        self.dense_flow = dense_flow

        return dense_flow
