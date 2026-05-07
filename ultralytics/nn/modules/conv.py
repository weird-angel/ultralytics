# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "DWOConv1d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "OAConv",
    "RepConv",
    "SpatialAttention",
    "make_cyclic_angles",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class OAConv(nn.Module):
    """Oriented Axial Convolution module.

    Replaces a standard k×k 2D depthwise convolution with a pair of oriented 1D depthwise convolutions:
    one horizontal (1×k) and one vertical (k×1). The outputs are summed, followed by batch normalization
    and an activation function. This captures directional spatial context with fewer parameters than a
    full 2D depthwise kernel.

    Inspired by "Convolutional Networks with Oriented 1D Kernels" (ICCV 2023).
    https://arxiv.org/abs/2309.15812

    Attributes:
        conv_h (nn.Conv2d): Horizontal 1×k depthwise convolution.
        conv_v (nn.Conv2d): Vertical k×1 depthwise convolution.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation (SiLU).
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=7, act=True):
        """Initialize OAConv with horizontal and vertical 1D depthwise convolutions.

        Args:
            c1 (int): Number of input channels. Must equal c2 (depthwise operation).
            c2 (int): Number of output channels. Must equal c1.
            k (int): Length of the 1D kernel. Must be odd. Defaults to 7.
            act (bool | nn.Module): Activation function. True uses default SiLU.
        """
        super().__init__()
        assert c1 == c2, "OAConv requires equal input and output channels for depthwise operation"
        assert k % 2 == 1, f"OAConv kernel size k must be odd for symmetric same-size padding, got k={k}"
        self.conv_h = nn.Conv2d(c1, c1, (1, k), stride=1, padding=(0, k // 2), groups=c1, bias=False)
        self.conv_v = nn.Conv2d(c1, c1, (k, 1), stride=1, padding=(k // 2, 0), groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply oriented 1D depthwise convolutions, batch normalization and activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_h(x) + self.conv_v(x)))

    def forward_fuse(self, x):
        """Apply oriented 1D depthwise convolutions and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_h(x) + self.conv_v(x))


def make_cyclic_angles(channels: int, N: int = 8, layer_offset: float = 0.0) -> list:
    """Compute per-channel initial angles cycling evenly through N orientations.

    Implements the ``channel_cycle_offset`` + ``theta_offset`` convention from the Oriented1D paper
    (Kirchmeyer & Deng, ICCV 2023, https://arxiv.org/abs/2309.15812).  Channels are divided into N
    equal groups; all channels in group ``n`` share angle ``n/N * π`` radians.  An optional
    ``layer_offset`` of ``0.5`` adds a 90° shift for odd-depth layers, matching
    ``layer_wise_rotation_offset(enable_layer_cycle=True)`` from the reference code.

    Args:
        channels (int): Number of channels.
        N (int): Number of equally-spaced direction groups in ``[0, π)``. Default: 8.
        layer_offset (float): Fractional orientation offset added before angle conversion.
            ``0.0`` = no shift, ``0.5`` = +90° shift for alternating layers. Default: 0.0.

    Returns:
        (list[float]): Angle in radians for each channel, length ``channels``.

    Examples:
        >>> angles = make_cyclic_angles(64, N=8)                    # 8 groups of 8 channels
        >>> angles = make_cyclic_angles(64, N=8, layer_offset=0.5)  # same, shifted 90°
    """
    return [
        math.pi
        * (
            # Wrap to [0, 1) BEFORE multiplying by π, then convert to radians.
            # Group index n: channels split into N groups; n = floor((N*(c+1)-1) / channels)
            (((N * (c + 1) - 1) // channels) / N + layer_offset)
            % 1
        )
        for c in range(channels)
    ]


class DWOConv1d(nn.Module):
    """Depthwise Oriented 1D Convolution — pure-PyTorch implementation.

    Applies a 1D depthwise convolution kernel at a **per-channel learnable angle** theta. Each channel
    has its own angle parameter that is trained end-to-end via backpropagation through ``torch.sin`` /
    ``torch.cos`` and ``F.grid_sample``.  This is a differentiable, CUDA-free equivalent of the three
    ``dwoconv1d`` / ``dwoconv1d_reference`` / ``dwoconv1d_specialized`` CUDA modules from:

        "Convolutional Networks with Oriented 1D Kernels", Kirchmeyer & Deng, ICCV 2023.
        https://arxiv.org/abs/2309.15812

    **How it works**: for each channel ``c`` with learnable angle ``theta_c``, the module samples the
    input along the direction ``(cos(theta_c), sin(theta_c))`` at ``k`` evenly-spaced positions centred
    on each output pixel, then multiplies by the 1×k learned weight and sums — exactly like a 1-D
    depthwise convolution but applied along a rotated axis.  Sampling uses bilinear interpolation
    (``F.grid_sample``), making the angle gradient available automatically.

    The three CUDA variants in the reference repository correspond to:
        * ``dwoconv1d``            – optimised custom CUDA kernel (fastest, GPU-only).
        * ``dwoconv1d_reference``  – simpler reference CUDA kernel (slower, used for correctness tests).
        * ``dwoconv1d_specialized``– pre-compiled CUDA for fixed spatial dimensions (fastest at known size).
    This class provides an equivalent that runs on any device (CPU or CUDA) without custom extensions.

    Attributes:
        weight (nn.Parameter): 1D depthwise kernel of shape ``(channels, 1, 1, k)``.
        theta (nn.Parameter): Per-channel rotation angle in radians, shape ``(channels,)``.
        bn (nn.BatchNorm2d): Batch normalisation applied to the output.
        act (nn.Module): Activation function (default SiLU).
        channels (int): Number of input/output channels.
        k (int): Kernel length (must be odd).
    """

    default_act = nn.SiLU()

    def __init__(self, channels, k=7, angle=0.0, act=True):
        """Initialise DWOConv1d.

        Args:
            channels (int): Number of input and output channels (depthwise — in == out).
            k (int): 1D kernel length.  Must be odd.  Defaults to 7.
            angle (float | list | torch.Tensor): Initial rotation angle(s) in radians.  A scalar
                initialises all channels to the same angle; a sequence of length ``channels`` sets
                per-channel angles.  Use :func:`make_cyclic_angles` to initialise channels across
                N evenly-spaced directions following the Oriented1D paper convention.
                Defaults to 0.0 (all channels start horizontal).
            act (bool | nn.Module): Activation function.  ``True`` uses default SiLU.
        """
        super().__init__()
        assert k % 2 == 1, f"DWOConv1d kernel size k must be odd, got k={k}"
        self.channels = channels
        self.k = k

        # 1-D depthwise weight: horizontal kernel shape (C, 1, 1, k)
        self.weight = nn.Parameter(torch.empty(channels, 1, 1, k))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Learnable per-channel rotation angle (radians)
        if isinstance(angle, (float, int)):
            angle_t = torch.full((channels,), float(angle))
        else:
            angle_t = torch.as_tensor(list(angle), dtype=torch.float32).reshape(channels)
        self.theta = nn.Parameter(angle_t)

        self.bn = nn.BatchNorm2d(channels)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def _oriented_sample(self, x):
        """Vectorized oriented sampling: single F.grid_sample call across all k positions.

        Builds a combined sampling grid of shape ``(N*C*k, H, W, 2)`` and calls ``F.grid_sample``
        once, then multiplies by the 1-D weights and sums across the kernel dimension.

        Args:
            x (torch.Tensor): Input of shape ``(N, C, H, W)``.

        Returns:
            (torch.Tensor): Accumulated output of shape ``(N, C, H, W)``.
        """
        N, C, H, W = x.shape
        k = self.k
        half = k // 2
        device = x.device

        norm_w = W / 2.0  # normalisation factor for x-axis grid coords
        norm_h = H / 2.0  # normalisation factor for y-axis grid coords

        offsets = torch.arange(-half, half + 1, dtype=x.dtype, device=device)  # (k,)
        cos_t = torch.cos(self.theta).to(x.dtype)  # (C,)
        sin_t = torch.sin(self.theta).to(x.dtype)  # (C,)

        # Per-channel, per-kernel-position offsets in normalised [-1, 1] grid coords
        dx_ck = (cos_t.unsqueeze(1) * offsets.unsqueeze(0)) / norm_w  # (C, k)
        dy_ck = (sin_t.unsqueeze(1) * offsets.unsqueeze(0)) / norm_h  # (C, k)

        # Base normalised pixel-centre grid: (H, W)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, dtype=x.dtype, device=device),
            torch.linspace(-1.0, 1.0, W, dtype=x.dtype, device=device),
            indexing="ij",
        )

        # Build all (C×k) sampling grids simultaneously: (C, k, H, W, 2)
        sx = grid_x[None, None] + dx_ck[:, :, None, None]  # (C, k, H, W)
        sy = grid_y[None, None] + dy_ck[:, :, None, None]  # (C, k, H, W)
        grid_all = torch.stack([sx, sy], dim=-1).reshape(C * k, H, W, 2)  # (C*k, H, W, 2)

        # Expand grid for batch dimension: (N*C*k, H, W, 2)
        grid_all = grid_all.unsqueeze(0).expand(N, -1, -1, -1, -1).reshape(N * C * k, H, W, 2)

        # Expand input: each channel is repeated k times → (N*C*k, 1, H, W)
        x_exp = x.unsqueeze(2).expand(-1, -1, k, -1, -1).reshape(N * C * k, 1, H, W)

        # Single vectorized grid_sample: (N*C*k, 1, H, W) → (N, C, k, H, W)
        sampled = torch.nn.functional.grid_sample(
            x_exp, grid_all, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(N, C, k, H, W)

        # Weighted sum over kernel dimension: weights (1, C, k, 1, 1)
        w = self.weight[:, 0, 0, :].reshape(1, C, k, 1, 1)
        return (sampled * w).sum(dim=2)  # (N, C, H, W)

    def forward(self, x):
        """Apply per-channel learnable-angle 1D depthwise convolution with BN and activation.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, channels, H, W)``.

        Returns:
            (torch.Tensor): Output tensor of shape ``(N, channels, H, W)``.
        """
        return self.act(self.bn(self._oriented_sample(x)))

    def forward_fuse(self, x):
        """Apply per-channel learnable-angle 1D depthwise convolution and activation (no BN).

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, channels, H, W)``.

        Returns:
            (torch.Tensor): Output tensor of shape ``(N, channels, H, W)``.
        """
        return self.act(self._oriented_sample(x))


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
