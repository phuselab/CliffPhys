#================================================================================================================================
#==            'CliffPhys: Camera-based Respiratory Measurement using Clifford Neural Networks' (Paper ID #11393)              ==
#================================================================================================================================

"""
Code containing functions and classes for the definition of the CliffPhys family of prediction models.

CLIFFORD:
    This script contains utility functions and classes for the definition of the CLI. 
    It includes implementations of the Clifford product for different algebra signatures (Clifford kernels), 
    the implementation of the Linear and 2D, 3D convolutional Clifford layers.
    It also provides the Negative Pearson loss implementation and the Processor() class to use any CliffPhys model in prediction mode.
    It provides the implementation of each model in the CliffPhys family, 4 models working using depth information, 4 depth-lacking models.

        MODELS:                                                     training version choices: 
                'CliffPhys02_d'         'CliffPhys02'                                 'PT-scamps_XYZ_FT-cohface_XYZ'                   
                'CliffPhys03_d'         'CliffPhys03'
                'CliffPhys30_d'         'CliffPhys30'
                'CliffPhys20_d'         'CliffPhys20'
"""


import os
import numpy as np
import torch
import torch.nn as nn
import os
import numpy as np
import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple
from typing import Callable, Optional, Tuple, Union

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Neg_Pearson(nn.Module):
    """
    Custom PyTorch module to compute the negative Pearson correlation coefficient loss.

    Attributes:
        None
    """
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        pearson = cos(preds - preds.mean(dim=0, keepdim=True), labels - labels.mean(dim=0, keepdim=True))
        return torch.mean(1 - pearson)

    
def _w_assert(w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]) -> torch.Tensor:
    """Convert Clifford weights to tensor .
    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Clifford weights.
    Raises:
        ValueError: Unknown weight type.
    Returns:
        torch.Tensor: Clifford weights as torch.Tensor.
    """
    if type(w) in (tuple, list):
        w = torch.stack(w)
        return w
    elif isinstance(w, torch.Tensor):
        return w
    elif isinstance(w, nn.Parameter):
        return w
    elif isinstance(w, nn.ParameterList):
        return w
    else:
        raise ValueError("Unknown weight type.")

class CliffordSignature:
    def __init__(self, g: Union[tuple, list, torch.Tensor]):
        super().__init__()
        self.g = self._g_tensor(g)
        self.dim = self.g.numel()
        if self.dim == 1:
            self.n_blades = 2
        elif self.dim == 2:
            self.n_blades = 4
        elif self.dim == 3:
            self.n_blades = 8
        else:
            raise NotImplementedError("Wrong Clifford signature.")

    def _g_tensor(self, g: Union[tuple, list, torch.Tensor]) -> torch.Tensor:
        """Convert Clifford signature to tensor.
        Args:
            g (Union[tuple, list, torch.Tensor]): Clifford signature.
        Raises:
            ValueError: Unknown metric.
        Returns:
            torch.Tensor: Clifford signature as torch.Tensor.
        """
        if type(g) in (tuple, list):
            g = torch.as_tensor(g, dtype=torch.float32)
        elif isinstance(g, torch.Tensor):
            pass
        else:
            raise ValueError("Unknown signature.")
        if not torch.any(abs(g) == 1.0):
            raise ValueError("Clifford signature should have at least one element as 1.")
        return g

def get_2d_clifford_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 2d Clifford algebras, g = [-1, -1] corresponds to a quaternion kernel.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(4, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of shape `(d~output~ * 4, d~input~ * 4, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 2
    w = _w_assert(w)
    assert len(w) == 4

    k0 = torch.cat([w[0], g[0] * w[1], g[1] * w[2], -g[0] * g[1] * w[3]], dim=1)
    k1 = torch.cat([w[1], w[0], -g[1] * w[3], g[1] * w[2]], dim=1)
    k2 = torch.cat([w[2], g[0] * w[3], w[0], -g[0] * w[1]], dim=1)
    k3 = torch.cat([w[3], w[2], -w[1], w[0]], dim=1)
    k = torch.cat([k0, k1, k2, k3], dim=0)
    return 4, k

def get_3d_clifford_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 3d Clifford algebras, g = [-1, -1, -1] corresponds to an octonion kernel.
    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(8, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of dimension `(d~output~ * 8, d~input~ * 8, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 3
    w = _w_assert(w)
    assert len(w) == 8

    k0 = torch.cat([w[0], w[1] * g[0], w[2] * g[1], w[3] * g[2], -w[4] * g[0] * g[1], -w[5] * g[0] * g[2], -w[6] * g[1] * g[2], -w[7] * g[0] * g[1] * g[2],], dim=1,)
    k1 = torch.cat([w[1], w[0], -w[4] * g[1], -w[5] * g[2], w[2] * g[1], w[3] * g[2], -w[7] * g[1] * g[2], -w[6] * g[2] * g[1]], dim=1,)
    k2 = torch.cat([w[2], w[4] * g[0], w[0], -w[6] * g[2], -w[1] * g[0], w[7] * g[0] * g[2], w[3] * g[2], w[5] * g[2] * g[0]], dim=1,)
    k3 = torch.cat([w[3], w[5] * g[0], w[6] * g[1], w[0], -w[7] * g[0] * g[1], -w[1] * g[0], -w[2] * g[1], -w[4] * g[0] * g[1]], dim=1,)
    k4 = torch.cat([w[4], w[2], -w[1], g[2] * w[7], w[0], -w[6] * g[2], w[5] * g[2], w[3] * g[2]], dim=1)
    k5 = torch.cat([w[5], w[3], -w[7] * g[1], -w[1], w[6] * g[1], w[0], -w[4] * g[1], -w[2] * g[1]], dim=1)
    k6 = torch.cat([w[6], w[7] * g[0], w[3], -w[2], -w[5] * g[0], w[4] * g[0], w[0], w[1] * g[0]], dim=1)
    k7 = torch.cat([w[7], w[6], -w[5], w[4], w[3], -w[2], w[1], w[0]], dim=1)
    k = torch.cat([k0, k1, k2, k3, k4, k5, k6, k7], dim=0)
    return 8, k

def clifford_convnd(conv_fn: Callable, x: torch.Tensor, output_blades: int, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs,
                    ) -> torch.Tensor:
    """Apply a Clifford convolution to a tensor.
    Args:
        conv_fn (Callable): The convolution function to use.
        x (torch.Tensor): Input tensor.
        output_blades (int): The output blades of the Clifford algebra.
        Different from the default n_blades when using encoding and decoding layers.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor, optional): Bias tensor. Defaults to None.
    Returns:
        torch.Tensor: Convolved output tensor.
    """
    # Reshape x such that the convolution function can be applied.
    #print('x shape: '+str(x.shape))
    B, *_ = x.shape
    B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
    x = x.permute(B_dim, -1, C_dim, *D_dims)
    x = x.reshape(B, -1, *x.shape[3:])
    # Apply convolution function
    output = conv_fn(x, weight, bias=bias, **kwargs)
    #print('\n weights shape: '+str(weight.shape))
    # Reshape back.
    output = output.view(B, output_blades, -1, *output.shape[2:])
    B_dim, I_dim, C_dim, *D_dims = range(len(output.shape))
    output = output.permute(B_dim, C_dim, *D_dims, I_dim)
    return output


class _CliffordConvNd(nn.Module):
    """Base class for all Clifford convolution modules."""

    def __init__(self, g: Union[tuple, list, torch.Tensor], in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 padding: int, dilation: int, groups: int, bias: bool, padding_mode: str, rotation: bool = False, ) -> None:
        super().__init__()
        sig = CliffordSignature(g)
        # register as buffer as we want the tensor to be moved to the same device as the module
        self.register_buffer("g", sig.g)
        self.dim = sig.dim
        self.n_blades = sig.n_blades
        if rotation:
            assert (
                self.dim == 2
            ), "2d rotational Clifford layers are only available for g = [-1, -1]. Make sure you have the right signature."

        if self.dim == 2:
            self._get_kernel = get_2d_clifford_kernel
        elif self.dim == 3:
            self._get_kernel = get_3d_clifford_kernel
        else:
            raise NotImplementedError(
                f"Clifford convolution not implemented for {self.dim} dimensions. Wrong Clifford signature."
            )

        if padding_mode != "zeros":
            raise NotImplementedError(f"Padding mode {padding_mode} not implemented.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.rotation = rotation

        self.weight = nn.ParameterList(
            [nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size)) for _ in range(self.n_blades)]
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)

        if rotation:
            self.scale_param = nn.Parameter(torch.Tensor(self.weight[0].shape))
            self.zero_kernel = nn.Parameter(torch.zeros(self.weight[0].shape), requires_grad=False)
            self.weight.append(self.scale_param)
            self.weight.append(self.zero_kernel)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialization of the Clifford convolution weight and bias tensors.
        The number of blades is taken into account when calculated the bounds of Kaiming uniform.
        """
        for blade, w in enumerate(self.weight):
            # Weight initialization for Clifford weights.
            if blade < self.n_blades:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    torch.Tensor(
                        self.out_channels, int(self.in_channels * self.n_blades / self.groups), *self.kernel_size
                    )
                )
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(w, -bound, bound)
            # Extra weights for 2d Clifford rotation layer.
            elif blade == self.n_blades:
                assert self.rotation is True
                # Default channel_in / channel_out initialization for scaling params.
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            elif blade == self.n_blades + 1:
                # Nothing to be done for zero kernel.
                pass
            else:
                raise ValueError(
                    f"Wrong number of Clifford weights. Expected {self.n_blades} weight tensors, and 2 extra tensors for rotational kernels."
                )

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                torch.Tensor(self.out_channels, int(self.in_channels * self.n_blades / self.groups), *self.kernel_size)
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, conv_fn: callable) -> torch.Tensor:
        if self.bias is not None:
            b = self.bias.view(-1)
        else:
            b = None
        output_blades, w = self._get_kernel(self.weight, self.g)
        return clifford_convnd(conv_fn, x, output_blades, w, b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"

        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"

        return s.format(**self.__dict__)

class CliffordConv2d(_CliffordConvNd):
    """2d Clifford convolution (dim(g)=2).
    Args:
        g (Union[tuple, list, torch.Tensor]): Clifford signature.
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int]]): padding added to both sides of the input.
        dilation (Union[int, Tuple[int, int]]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        padding_mode (str): Padding to use.
        rotation (bool): If True, enables the rotation kernel for Clifford convolution.
    """

    def __init__(self, g: Union[tuple, list, torch.Tensor], in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", rotation: bool = False,):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)

        super().__init__(g, in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, groups, bias, padding_mode, rotation,)
        if not self.dim == 2 and not self.dim == 3:
            raise NotImplementedError("Wrong Clifford signature for CliffordConv2d.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")
        return super().forward(x, F.conv2d)

class CliffordConv3d(_CliffordConvNd):
    """3d Clifford convolution.
    Args:
        g (Union[tuple, list, torch.Tensor]): Clifford signature.
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int, int]]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int, int]]): padding added to all sides of the input.
        dilation (Union[int, Tuple[int, int, int]]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        padding_mode (str): Padding to use.
    """

    def __init__(self, g: Union[tuple, list, torch.Tensor], in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros",):
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)

        super().__init__(g, in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, groups, bias, padding_mode,)
        if not self.dim == 2 and not self.dim == 3:
            raise NotImplementedError("Wrong Clifford signature for CliffordConv3d.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")
        return super().forward(x, F.conv3d)


class CliffordLinear(nn.Module):
    """Clifford linear layer.
    Args:
        g (Union[List, Tuple]): Clifford signature tensor.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
    """

    def __init__(self, g, in_channels: int, out_channels: int, bias: bool = True,) -> None:
        super().__init__()
        sig = CliffordSignature(g)

        self.register_buffer("g", sig.g)
        self.dim = sig.dim
        self.n_blades = sig.n_blades

        if self.dim == 2:
            self._get_kernel = get_2d_clifford_kernel
        elif self.dim == 3:
            self._get_kernel = get_3d_clifford_kernel
        else:
            raise NotImplementedError(
                f"Clifford linear layers are not implemented for {self.dim} dimensions. Wrong Clifford signature."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(self.n_blades, out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization of the Clifford linear weight and bias tensors.
        # The number of blades is taken into account when calculated the bounds of Kaiming uniform.
        nn.init.kaiming_uniform_(
            self.weight.view(self.out_channels, self.in_channels * self.n_blades),
            a=math.sqrt(5),
        )
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weight.view(self.out_channels, self.in_channels * self.n_blades)
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape x such that the Clifford kernel can be applied.
        B, _, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")
        B_dim, C_dim, I_dim = range(len(x.shape))
        x = x.permute(B_dim, -1, C_dim)
        x = x.reshape(B, -1)
        # Get Clifford kernel, apply it.
        _, weight = self._get_kernel(self.weight, self.g)
        output = F.linear(x, weight, self.bias.view(-1))
        # Reshape back.
        output = output.view(B, I, -1)
        B_dim, I_dim, C_dim = range(len(output.shape))
        output = output.permute(B_dim, C_dim, I_dim)
        return output


class CliffPhys20(nn.Module): 

    """CliffordPhys20 model.

    Args:
        model_params (dict): Dictionary containing model parameters.
        dropout_rate1 (float, optional): Dropout rate for the first dropout layer. Defaults to 0.25.
        dropout_rate2 (float, optional): Dropout rate for the second dropout layer. Defaults to 0.5.
        pool_size (Tuple, optional): Size of the pooling window. Defaults to (2, 2).
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """

    def __init__(self, model_params, dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), device='cpu'):

        super(CliffPhys20, self).__init__()
        self.img_size = model_params['img_size']
        self.num_frames = model_params['num_frames']

        self.in_channels = 1
        self.hidden_channels = 1
        self.out_channels = self.img_size * self.img_size
        self.kernel_size_1 = (15, 3, 3)    # (frames, height, width) A:(15, 3, 3) B:(101, 3, 3)
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size

        self.g = [1, 1]
        self.stride = (1, 1, 1)             # (frames, height, width)
        self.padding_1 = (7, 1, 1)          # (frames, height, width) A:(7, 1, 1) B:(50, 1, 1)
        self.num_groups = 1

        self.device = device
                
        # Motion branch
        self.motion_conv1 = CliffordConv3d(self.g, self.in_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
        self.motion_conv2 = CliffordConv3d(self.g, self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
                
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.final_dense_1 = CliffordLinear(self.g, self.out_channels, 128, bias=True)
        self.dropout_2 = nn.Dropout(self.dropout_rate2)
        self.final_dense_2 = CliffordLinear(self.g, 128, 1, bias=True)
        self.dropout_3 = nn.Dropout(self.dropout_rate2)
        self.final_dense_3 = nn.Linear(4, 1, bias=True)
    
    def forward(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        S, F, C, H, W = inputs.size()

        m = inputs.reshape(S, F, H, W, C)
        scalar_pure_coeff = torch.zeros(S, F, H, W, 1)
        e_coeff = torch.zeros(S, F, H, W, 1)
        m = torch.cat((scalar_pure_coeff.to(self.device), m, e_coeff), dim=-1)

        Q = m.size(-1)
        m = m.reshape(S, 1, F, H, W, Q)

        d1 = self.motion_conv1(m)
        d2 = self.motion_conv2(d1)

        d2 = d2.reshape(S*F, Q, H, W)
        d2 = self.dropout_1(d2)

        d2 = d2.view(S*F, H*W, Q)
        d3 = self.final_dense_1(d2)
        d3 = self.dropout_2(d3)
        d4 = self.final_dense_2(d3)
        d4 = self.dropout_3(d4)
        out = self.final_dense_3(d4)
        out = out.view(S,F)

        return out

class CliffPhys30(nn.Module): 

    """CliffordPhys30 model.

    Args:
        model_params (dict): Dictionary containing model parameters.
        dropout_rate1 (float, optional): Dropout rate for the first dropout layer. Defaults to 0.25.
        dropout_rate2 (float, optional): Dropout rate for the second dropout layer. Defaults to 0.5.
        pool_size (Tuple, optional): Size of the pooling window. Defaults to (2, 2).
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """

    def __init__(self, model_params, dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), device='cpu'):

        super(CliffPhys30, self).__init__()
        self.img_size = model_params['img_size']
        self.num_frames = model_params['num_frames']

        self.in_channels = 1
        self.hidden_channels = 1
        self.out_channels = self.img_size * self.img_size
        self.kernel_size_1 = (15, 3, 3)    # (frames, height, width) A:(15, 3, 3) B:(101, 3, 3)
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size

        self.g = [1, 1, 1]
        self.stride = (1, 1, 1)             # (frames, height, width)
        self.padding_1 = (7, 1, 1)          # (frames, height, width) A:(7, 1, 1) B:(50, 1, 1)
        self.num_groups = 1

        self.device = device
                
        # Motion branch
        self.motion_conv1 = CliffordConv3d(self.g, self.in_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
        self.motion_conv2 = CliffordConv3d(self.g, self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
                
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.final_dense_1 = CliffordLinear(self.g, self.out_channels, 128, bias=True)
        self.dropout_2 = nn.Dropout(self.dropout_rate2)
        self.final_dense_2 = CliffordLinear(self.g, 128, 1, bias=True)
        self.dropout_3 = nn.Dropout(self.dropout_rate2)
        self.final_dense_3 = nn.Linear(8, 1, bias=True)
    
    def forward(self, inputs, params=None):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        S, F, C, H, W = inputs.size()

        m = inputs.reshape(S, F, H, W, C)
        scalar_pure_coeff = torch.zeros(S, F, H, W, 1)
        e_coeff = torch.zeros(S, F, H, W, 5)
        m = torch.cat((scalar_pure_coeff.to(self.device), m, e_coeff), dim=-1)

        Q = m.size(-1)
        m = m.reshape(S, 1, F, H, W, Q)

        d1 = self.motion_conv1(m)
        d2 = self.motion_conv2(d1)

        d2 = d2.reshape(S*F, Q, H, W)
        d2 = self.dropout_1(d2)

        d2 = d2.view(S*F, H*W, Q)
        d3 = self.final_dense_1(d2)
        d3 = self.dropout_2(d3)
        d4 = self.final_dense_2(d3)
        d4 = self.dropout_3(d4)
        out = self.final_dense_3(d4)
        out = out.view(S,F)

        return out

class CliffPhys02(nn.Module): 

    """CliffordPhys02 model.

    Args:
        model_params (dict): Dictionary containing model parameters.
        dropout_rate1 (float, optional): Dropout rate for the first dropout layer. Defaults to 0.25.
        dropout_rate2 (float, optional): Dropout rate for the second dropout layer. Defaults to 0.5.
        pool_size (Tuple, optional): Size of the pooling window. Defaults to (2, 2).
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """

    def __init__(self, model_params, dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), device='cpu'):

        super(CliffPhys02, self).__init__()
        self.img_size = model_params['img_size']
        self.num_frames = model_params['num_frames']

        self.in_channels = 1
        self.hidden_channels = 1
        self.out_channels = self.img_size * self.img_size
        self.kernel_size_1 = (15, 3, 3)    # (frames, height, width) A:(15, 3, 3) B:(101, 3, 3)
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size

        self.g = [-1, -1]
        self.stride = (1, 1, 1)             # (frames, height, width)
        self.padding_1 = (7, 1, 1)          # (frames, height, width) A:(7, 1, 1) B:(50, 1, 1)
        self.num_groups = 1

        self.device = device
                
        # Motion branch
        self.motion_conv1 = CliffordConv3d(self.g, self.in_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
        self.motion_conv2 = CliffordConv3d(self.g, self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
                
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.final_dense_1 = CliffordLinear(self.g, self.out_channels, 128, bias=True)
        self.dropout_2 = nn.Dropout(self.dropout_rate2)
        self.final_dense_2 = CliffordLinear(self.g, 128, 1, bias=True)
        self.dropout_3 = nn.Dropout(self.dropout_rate2)
        self.final_dense_3 = nn.Linear(4, 1, bias=True)
    
    def forward(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        S, F, C, H, W = inputs.size()

        m = inputs.reshape(S, F, H, W, C)
        scalar_pure_coeff = torch.zeros(S, F, H, W, 1)
        e_coeff = torch.zeros(S, F, H, W, 1)
        m = torch.cat((scalar_pure_coeff.to(self.device), m, e_coeff), dim=-1)

        Q = m.size(-1)
        m = m.reshape(S, 1, F, H, W, Q)

        d1 = self.motion_conv1(m)
        d2 = self.motion_conv2(d1)

        d2 = d2.reshape(S*F, Q, H, W)
        d2 = self.dropout_1(d2)

        d2 = d2.view(S*F, H*W, Q)
        d3 = self.final_dense_1(d2)
        d3 = self.dropout_2(d3)
        d4 = self.final_dense_2(d3)
        d4 = self.dropout_3(d4)
        out = self.final_dense_3(d4)
        out = out.view(S,F)

        return out

class CliffPhys03(nn.Module): 

    """CliffordPhys03 model.

    Args:
        model_params (dict): Dictionary containing model parameters.
        dropout_rate1 (float, optional): Dropout rate for the first dropout layer. Defaults to 0.25.
        dropout_rate2 (float, optional): Dropout rate for the second dropout layer. Defaults to 0.5.
        pool_size (Tuple, optional): Size of the pooling window. Defaults to (2, 2).
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """

    def __init__(self, model_params, dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), device='cpu'):

        super(CliffPhys03, self).__init__()
        self.img_size = model_params['img_size']
        self.num_frames = model_params['num_frames']

        self.in_channels = 1
        self.hidden_channels = 1
        self.out_channels = self.img_size * self.img_size
        self.kernel_size_1 = (15, 3, 3)    # (frames, height, width) A:(15, 3, 3) B:(101, 3, 3)
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size

        self.g = [-1, -1, -1]
        self.stride = (1, 1, 1)             # (frames, height, width)
        self.padding_1 = (7, 1, 1)          # (frames, height, width) A:(7, 1, 1) B:(50, 1, 1)
        self.num_groups = 1

        self.device = device
                
        # Motion branch
        self.motion_conv1 = CliffordConv3d(self.g, self.in_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
        self.motion_conv2 = CliffordConv3d(self.g, self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
                
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.final_dense_1 = CliffordLinear(self.g, self.out_channels, 128, bias=True)
        self.dropout_2 = nn.Dropout(self.dropout_rate2)
        self.final_dense_2 = CliffordLinear(self.g, 128, 1, bias=True)
        self.dropout_3 = nn.Dropout(self.dropout_rate2)
        self.final_dense_3 = nn.Linear(8, 1, bias=True)
    
    def forward(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        S, F, C, H, W = inputs.size()

        m = inputs.reshape(S, F, H, W, C)
        scalar_pure_coeff = torch.zeros(S, F, H, W, 1)
        e_coeff = torch.zeros(S, F, H, W, 5)
        m = torch.cat((scalar_pure_coeff.to(self.device), m, e_coeff), dim=-1)

        Q = m.size(-1)
        m = m.reshape(S, 1, F, H, W, Q)

        d1 = self.motion_conv1(m)
        d2 = self.motion_conv2(d1)

        d2 = d2.reshape(S*F, Q, H, W)
        d2 = self.dropout_1(d2)

        d2 = d2.view(S*F, H*W, Q)
        d3 = self.final_dense_1(d2)
        d3 = self.dropout_2(d3)
        d4 = self.final_dense_2(d3)
        d4 = self.dropout_3(d4)
        out = self.final_dense_3(d4)
        out = out.view(S,F)

        return out

class CliffPhys30_d(nn.Module):

    """CliffordPhys30_d model, processing also depth information.

    Args:
        model_params (dict): Dictionary containing model parameters.
        dropout_rate1 (float, optional): Dropout rate for the first dropout layer. Defaults to 0.25.
        dropout_rate2 (float, optional): Dropout rate for the second dropout layer. Defaults to 0.5.
        pool_size (Tuple, optional): Size of the pooling window. Defaults to (2, 2).
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """ 

    def __init__(self, model_params, dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), device='cpu'):

        super(CliffPhys30_d, self).__init__()
        self.img_size = model_params['img_size']
        self.num_frames = model_params['num_frames']

        self.in_channels = 1
        self.hidden_channels = 1
        self.out_channels = self.img_size * self.img_size
        self.kernel_size_1 = (15, 3, 3)    # (frames, height, width) A:(15, 3, 3) B:(101, 3, 3)
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size

        self.g = [1, 1, 1]
        self.stride = (1, 1, 1)             # (frames, height, width)
        self.padding_1 = (7, 1, 1)          # (frames, height, width) A:(7, 1, 1) B:(50, 1, 1)
        self.num_groups = 1

        self.device = device
                
        # Motion branch
        self.motion_conv1 = CliffordConv3d(self.g, self.in_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
        self.motion_conv2 = CliffordConv3d(self.g, self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
                
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.final_dense_1 = CliffordLinear(self.g, self.out_channels, 128, bias=True)
        self.dropout_2 = nn.Dropout(self.dropout_rate2)
        self.final_dense_2 = CliffordLinear(self.g, 128, 1, bias=True)
        self.dropout_3 = nn.Dropout(self.dropout_rate2)
        self.final_dense_3 = nn.Linear(8, 1, bias=True)
    
    def forward(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        S, F, C, H, W = inputs.size()

        m = inputs.reshape(S, F, H, W, C)
        e_coeff = torch.zeros(S, F, H, W, 5)
        m = torch.cat((m, e_coeff), dim=-1)

        Q = m.size(-1)
        m = m.reshape(S, 1, F, H, W, Q)

        d1 = self.motion_conv1(m)
        d2 = self.motion_conv2(d1)

        d2 = d2.reshape(S*F, Q, H, W)
        d2 = self.dropout_1(d2)

        d2 = d2.view(S*F, H*W, Q)
        d3 = self.final_dense_1(d2)
        d3 = self.dropout_2(d3)
        d4 = self.final_dense_2(d3)
        d4 = self.dropout_3(d4)
        out = self.final_dense_3(d4)
        out = out.view(S,F)

        return out

class CliffPhys20_d(nn.Module): 
    """CliffordPhys20_d model, processing also depth information.

    Args:
        model_params (dict): Dictionary containing model parameters.
        dropout_rate1 (float, optional): Dropout rate for the first dropout layer. Defaults to 0.25.
        dropout_rate2 (float, optional): Dropout rate for the second dropout layer. Defaults to 0.5.
        pool_size (Tuple, optional): Size of the pooling window. Defaults to (2, 2).
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """ 

    def __init__(self, model_params, dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), device='cpu'):

        super(CliffPhys20_d, self).__init__()
        self.img_size = model_params['img_size']
        self.num_frames = model_params['num_frames']

        self.in_channels = 1
        self.hidden_channels = 1
        self.out_channels = self.img_size * self.img_size
        self.kernel_size_1 = (15, 3, 3)    # (frames, height, width) A:(15, 3, 3) B:(101, 3, 3)
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size

        self.g = [1, 1]
        self.stride = (1, 1, 1)             # (frames, height, width)
        self.padding_1 = (7, 1, 1)          # (frames, height, width) A:(7, 1, 1) B:(50, 1, 1)
        self.num_groups = 1

        self.device = device
                
        # Motion branch
        self.motion_conv1 = CliffordConv3d(self.g, self.in_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
        self.motion_conv2 = CliffordConv3d(self.g, self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
                
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.final_dense_1 = CliffordLinear(self.g, self.out_channels, 128, bias=True)
        self.dropout_2 = nn.Dropout(self.dropout_rate2)
        self.final_dense_2 = CliffordLinear(self.g, 128, 1, bias=True)
        self.dropout_3 = nn.Dropout(self.dropout_rate2)
        self.final_dense_3 = nn.Linear(4, 1, bias=True)
    
    def forward(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        S, F, C, H, W = inputs.size()

        m = inputs.reshape(S, F, H, W, C)
        e_coeff = torch.zeros(S, F, H, W, 1)
        m = torch.cat((m, e_coeff), dim=-1)

        Q = m.size(-1)
        m = m.reshape(S, 1, F, H, W, Q)

        d1 = self.motion_conv1(m)
        d2 = self.motion_conv2(d1)

        d2 = d2.reshape(S*F, Q, H, W)
        d2 = self.dropout_1(d2)

        d2 = d2.view(S*F, H*W, Q)
        d3 = self.final_dense_1(d2)
        d3 = self.dropout_2(d3)
        d4 = self.final_dense_2(d3)
        d4 = self.dropout_3(d4)
        out = self.final_dense_3(d4)
        out = out.view(S,F)

        return out

class CliffPhys02_d(nn.Module): 
    """CliffordPhys02_d model, processing also depth information.

    Args:
        model_params (dict): Dictionary containing model parameters.
        dropout_rate1 (float, optional): Dropout rate for the first dropout layer. Defaults to 0.25.
        dropout_rate2 (float, optional): Dropout rate for the second dropout layer. Defaults to 0.5.
        pool_size (Tuple, optional): Size of the pooling window. Defaults to (2, 2).
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """ 

    def __init__(self, model_params, dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), device='cpu'):

        super(CliffPhys02_d, self).__init__()
        self.img_size = model_params['img_size']
        self.num_frames = model_params['num_frames']

        self.in_channels = 1
        self.hidden_channels = 1
        self.out_channels = self.img_size * self.img_size
        self.kernel_size_1 = (15, 3, 3)    # (frames, height, width) A:(15, 3, 3) B:(101, 3, 3)
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size

        self.g = [-1, -1]
        self.stride = (1, 1, 1)             # (frames, height, width)
        self.padding_1 = (7, 1, 1)          # (frames, height, width) A:(7, 1, 1) B:(50, 1, 1)
        self.num_groups = 1

        self.device = device
                
        # Motion branch
        self.motion_conv1 = CliffordConv3d(self.g, self.in_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
        self.motion_conv2 = CliffordConv3d(self.g, self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
                
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.final_dense_1 = CliffordLinear(self.g, self.out_channels, 128, bias=True)
        self.dropout_2 = nn.Dropout(self.dropout_rate2)
        self.final_dense_2 = CliffordLinear(self.g, 128, 1, bias=True)
        self.dropout_3 = nn.Dropout(self.dropout_rate2)
        self.final_dense_3 = nn.Linear(4, 1, bias=True)
    
    def forward(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        S, F, C, H, W = inputs.size()

        m = inputs.reshape(S, F, H, W, C)
        k_pure_coeff = torch.zeros(S, F, H, W, 1)
        m = torch.cat((m, k_pure_coeff.to(self.device)), dim=-1)

        Q = m.size(-1)
        m = m.reshape(S, 1, F, H, W, Q)

        d1 = self.motion_conv1(m)
        d2 = self.motion_conv2(d1)

        d2 = d2.reshape(S*F, Q, H, W)
        d2 = self.dropout_1(d2)

        d2 = d2.view(S*F, H*W, Q)
        d3 = self.final_dense_1(d2)
        d3 = self.dropout_2(d3)
        d4 = self.final_dense_2(d3)
        d4 = self.dropout_3(d4)
        out = self.final_dense_3(d4)
        out = out.view(S,F)

        return out

class CliffPhys03_d(nn.Module):
    """CliffordPhys03_d model, processing also depth information.

    Args:
        model_params (dict): Dictionary containing model parameters.
        dropout_rate1 (float, optional): Dropout rate for the first dropout layer. Defaults to 0.25.
        dropout_rate2 (float, optional): Dropout rate for the second dropout layer. Defaults to 0.5.
        pool_size (Tuple, optional): Size of the pooling window. Defaults to (2, 2).
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """ 

    def __init__(self, model_params, dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), device='cpu'):

        super(CliffPhys03_d, self).__init__()
        self.img_size = model_params['img_size']
        self.num_frames = model_params['num_frames']

        self.in_channels = 1
        self.hidden_channels = 1
        self.out_channels = self.img_size * self.img_size
        self.kernel_size_1 = (15, 3, 3)     # (15, 3, 3)
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size

        self.g = [-1, -1, -1]
        self.stride = (1, 1, 1)             # (frames, height, width)
        self.padding_1 = (7, 1, 1)            # (7, 3, 3)
        self.norm = True
        self.num_groups = 1

        self.device = device
                
        # Motion branch
        self.motion_conv1 = CliffordConv3d(self.g, self.in_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
        self.motion_conv2 = CliffordConv3d(self.g, self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_size_1, stride=self.stride, 
                                    padding=self.padding_1, bias=True)
        # self.motion_avg_pooling = nn.AvgPool2d(self.pool_size)
                
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.final_dense_1 = CliffordLinear(self.g, self.out_channels, 128, bias=True)
        self.dropout_2 = nn.Dropout(self.dropout_rate2)
        self.final_dense_2 = CliffordLinear(self.g, 128, 1, bias=True)
        self.dropout_3 = nn.Dropout(self.dropout_rate2)
        self.final_dense_3 = nn.Linear(8, 1, bias=True)
    
    def forward(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        S, F, C, H, W = inputs.size()

        m = inputs.reshape(S, F, H, W, C)
        e_coeff = torch.zeros(S, F, H, W, 5)
        m = torch.cat((m, e_coeff.to(self.device)), dim=-1)

        Q = m.size(-1)
        m = m.reshape(S, 1, F, H, W, Q)

        d1 = self.motion_conv1(m)
        d2 = self.motion_conv2(d1)

        d2 = d2.reshape(S*F, Q, H, W)
        d2 = self.dropout_1(d2)

        d2 = d2.view(S*F, H*W, Q)
        d3 = self.final_dense_1(d2)
        d3 = self.dropout_2(d3)
        d4 = self.final_dense_2(d3)
        d4 = self.dropout_3(d4)
        out = self.final_dense_3(d4)
        out = out.view(S,F)

        return out


class Processor():
    """
    Generic processor class for PyTorch implemented CliffPhys models.

    Initializing Parameters:
        model_class (class): Class of the model to be used.
        model_params (dict): Parameters for initializing the model.
        load_path (str): Path to the directory containing the saved model files.
        use_last_epoch (bool, optional): Flag indicating whether to use the last epoch's model or the best model. Defaults to False.
    """
    
    def __init__(self, model_class, model_params, load_path, use_last_epoch=False):
        self.load_path = load_path                        
        self.use_last_epoch=use_last_epoch
        self.device = torch.device("cpu")
        self.model = model_class(model_params, device=self.device).to(self.device)
        self.criterion = Neg_Pearson()
        if self.use_last_epoch:
            last_epoch_model_path = os.path.join(self.load_path, 'Epoch_' + str(self.get_last_epoch()) + '.pth')
            self.model.load_state_dict(torch.load(last_epoch_model_path))
        else:
            best_model_path = os.path.join(self.load_path, 'Best_epoch.pth')
            self.model.load_state_dict(torch.load(best_model_path))

    def predict(self, data_loader):    
        """
        Computes the prediction given the input batch of XYZ videos (data_loader).

        Parameters:
            data_loader (DataLoader): PyTorch DataLoader object containing a batch of XYZ input videos.

        Returns:
            numpy.ndarray: Batch predictions.
        """   
        self.model = self.model.to(self.device)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                data = batch[0].to(
                    self.device)
                preds = self.model(data)

                predictions.append(preds.cpu())

        return np.concatenate(predictions)
    
    def score(self, pred, label):
        """
        Computes the loss between predictions and labels using the Negative Pearson loss.

        Parameters:
            pred (numpy.ndarray): Predictions array.
            label (numpy.ndarray): Labels array.

        Returns:
            float: Loss value.
        """
        loss = self.criterion(torch.FloatTensor(pred), torch.FloatTensor(label))
        return loss
    
    def get_last_epoch(self):
        """
        Retrieves the index of the last epoch from the saved model files.

        Returns:
            int: Index of the last epoch.
        """
        pth_files = [f for f in os.listdir(self.load_path) if f.endswith('.pth') and 'Epoch_' in f]
        return max([int(f.split('Epoch_')[1].split('.pth')[0]) for f in pth_files])
