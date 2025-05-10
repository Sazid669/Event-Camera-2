"""
Sobel Torch operator.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/utils/stat_utils.py
"""

import torch

from torch import nn


class SobelTorch(nn.Module):
    """
    Sobel operator (compatible with PyTorch).

    Attributes
    ----------
    in_channels : int
        Input channels (defaults to 2).
    kernel_size : int
        Kernel size (defaults to 3).
    precision : str
        Floating point precision to use, can choose between '64' or '32' (defaults to '64').
    use_cuda : bool
        Use CUDA for computations (defaults to True).
    """
    def __init__(
        self,
        in_channels: int = 2,
        kernel_size: int = 3,
        precision: str = "64",
        use_cuda: bool = True
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._precision = precision
        self._use_cuda = use_cuda

        # Create two separate filters, for each dimension (x and y)
        self.filter_dx = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=1,
            kernel_size=self._kernel_size,
            stride=1,
            padding=1,
            bias=False
        )
        self.filter_dy = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=1,
            kernel_size=self._kernel_size,
            stride=1,
            padding=1,
            bias=False
        )

        # Create kernels, for each direction
        # ('64' precision corresponds to double)
        if self._precision == "64":
            kernel_dx = torch.tensor(
                [[-1.0, -2.0, -1.0],
                 [ 0.0,  0.0,  0.0],
                 [ 1.0,  2.0,  1.0]]
            ).double()
            kernel_dy = torch.tensor(
                [[-1.0, 0.0, 1.0],
                 [-2.0, 0.0, 2.0],
                 [-1.0, 0.0, 1.0]]
            ).double()
        else:
            kernel_dx = torch.tensor(
                [[-1.0, -2.0, -1.0],
                 [ 0.0,  0.0,  0.0],
                 [ 1.0,  2.0,  1.0]]
            )
            kernel_dy = torch.tensor(
                [[-1.0, 0.0, 1.0],
                 [-2.0, 0.0, 2.0],
                 [-1.0, 0.0, 1.0]]
            )

        if self._use_cuda:
            kernel_dx = kernel_dx.cuda()
            kernel_dy = kernel_dy.cuda()

        # Update the convolutional kernels
        self.filter_dx.weight = nn.Parameter(
            kernel_dx.unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )
        self.filter_dy.weight = nn.Parameter(
            kernel_dy.unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the Sobel operator.
        """
        if self._in_channels == 2:
            sobel_xx = self.filter_dx(image[..., [0], :, :])
            sobel_yy = self.filter_dy(image[..., [1], :, :])
            sobel_yx = self.filter_dx(image[..., [1], :, :])
            sobel_xy = self.filter_dy(image[..., [0], :, :])

            return torch.cat([sobel_xx, sobel_yy, sobel_yx, sobel_xy], dim=1)

        elif self._in_channels == 1:
            sobel_x = self.filter_dx(image[..., [0], :, :])
            sobel_y = self.filter_dy(image[..., [0], :, :])

            return torch.cat([sobel_x, sobel_y], dim=1)
