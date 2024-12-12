import torch
import math
import numpy as np
from typing import List, Tuple, Union
from KANLinear import KANLinear

#1d
def add_padding_1d(array: np.ndarray, padding: int) -> np.ndarray:
    """Adds padding to a 1D array."""
    n = array.shape[0]
    padded_array = np.zeros(n + 2 * padding)
    padded_array[padding: n + padding] = array
    return padded_array


def calc_out_dims_1d(array, kernel_size, stride, dilation, padding):
    """Calculate output dimensions for 1D convolution."""
    batch_size, n_channels, n = matrix.shape
    out_size = np.floor((n + 2 * padding - kernel_size - (kernel_size - 1) * (dilation - 1)) / stride).astype(int) + 1
    return out_size, batch_size, n_channels


def multiple_convs_kan_conv1d(array,
                               kernels,
                               kernel_size,
                               out_channels,
                               stride=1,
                               dilation=1,
                               padding=0,
                               device="cuda") -> torch.Tensor:
    """Performs a 1D convolution with multiple kernels on the input array using specified stride, dilation, and padding.

    Args:
        array (torch.Tensor): 1D tensor of shape (batch_size, channels, length).
        kernels (list): List of kernel functions to be applied.
        kernel_size (int): Size of the 1D kernel.
        out_channels (int): Number of output channels.
        stride (int): Stride along the length of the array. Default is 1.
        dilation (int): Dilation rate along the length of the array. Default is 1.
        padding (int): Number of elements to pad on each side. Default is 0.
        device (str): Device to perform calculations on. Default is "cuda".

    Returns:
        torch.Tensor: Feature map after convolution with shape (batch_size, out_channels, length_out).
    """
    length_out, batch_size = calc_out_dims_1d(array, kernel_size, stride, dilation, padding)
    n_convs = len(kernels)

    array_out = torch.zeros((batch_size, out_channels, length_out)).to(device)

    array = F.pad(array, (padding, padding), mode='constant', value=0)
    conv_groups = array.unfold(2, kernel_size, stride)
    conv_groups = conv_groups.contiguous()

    kern_per_out = len(kernels) // out_channels

    for c_out in range(out_channels):
        out_channel_accum = torch.zeros((batch_size, length_out), device=device)

        for k_idx in range(kern_per_out):
            kernel = kernels[c_out * kern_per_out + k_idx]
            conv_result = kernel(conv_groups.view(-1, 1, kernel_size))
            out_channel_accum += conv_result.view(batch_size, length_out)

        array_out[:, c_out, :] = out_channel_accum

    return array_out

def kan_conv1d(matrix: torch.Tensor,
               kernel,
               kernel_size: int,
               stride: int = 1,
               dilation: int = 1,
               padding: int = 0,
               device: str = "cpu") -> torch.Tensor:
    """
    Performs a 1D convolution with the given kernel over a 1D matrix using the defined stride, dilation, and padding.

    Args:
        matrix (torch.Tensor): 3D tensor (batch_size, channels, width) to be convolved.
        kernel (function): Kernel function to apply on the 1D patches of the matrix.
        kernel_size (int): Size of the kernel (assumed to be square).
        stride (int, optional): Stride along the width axis. Defaults to 1.
        dilation (int, optional): Dilation along the width axis. Defaults to 1.
        padding (int, optional): Padding along the width axis. Defaults to 0.
        device (str): "cuda" or "cpu".

    Returns:
        torch.Tensor: 1D Feature map after convolution.
    """

    batch_size, n_channels, width_in = matrix.shape
    width_out = ((width_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    matrix_out = torch.zeros((batch_size, n_channels, width_out), device=device)

    matrix_padded = torch.nn.functional.pad(matrix, (padding, padding))

    for i in range(width_out):

        start = i * stride
        end = start + kernel_size * dilation
        patch = matrix_padded[:, :, start:end:dilation]

        matrix_out[:, :, i] = kernel.forward(patch).squeeze(-1)

    return matrix_out


class KAN_Convolutional_Layer_1D(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, device="cuda"):
        super(KAN_Convolutional_Layer_1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device
        self.convs = torch.nn.ModuleList([KAN_Convolution_1D(kernel_size, stride, padding, dilation, device) for _ in range(in_channels * out_channels)])

    def forward(self, x: torch.Tensor):
        return torch.cat([conv(x[:, i, :].unsqueeze(1)) for i, conv in enumerate(self.convs)], dim=1)


class KAN_Convolutional_Layer_1D(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 2,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device: str = "cpu"
        ):
        super(KAN_Convolutional_Layer_1D, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride


        self.convs = torch.nn.ModuleList()
        for _ in range(in_channels * out_channels):
            self.convs.append(
                KAN_Convolution_1D(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor):
        batch_size, in_channels, length = x.shape
        output_length = (length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        output = torch.zeros((batch_size, self.out_channels, output_length), device=x.device)


        for i in range(self.out_channels):
            output_accum = torch.zeros((batch_size, output_length), device=x.device)
            for j in range(self.in_channels):
                kernel_idx = i * self.in_channels + j
                conv_result = self.convs[kernel_idx].forward(x[:, j, :].unsqueeze(1))
                output_accum += conv_result.squeeze(1)  # Squeeze
            output[:, i, :] = output_accum  # A to output channel

        return output

class KAN_Convolution_1D(torch.nn.Module):
    def __init__(
            self,
            kernel_size: int = 2,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            grid_size: int = 50,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1]
        ):
        super(KAN_Convolution_1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv = KANLinear(
            in_features = kernel_size,
            out_features = 1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def forward(self, x: torch.Tensor):
        self.device = x.device
        return kan_conv1d(x, self.conv, self.kernel_size,self.stride, self.dilation, self.padding, self.device)

#2d

def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size,n_channels,n, m = matrix.shape

    h_out =  np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    b = [kernel_side // 2, kernel_side// 2]
    return h_out,w_out,batch_size,n_channels

def multiple_convs_kan_conv2d(matrix, #but as torch tensors. Kernel side asume q el kernel es cuadrado
             kernels, 
             kernel_side,
             out_channels,
             stride= (1, 1), 
             dilation= (1, 1), 
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out,batch_size,n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size,out_channels,h_out,w_out)).to(device)#estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)
    conv_groups = unfold(matrix[:,:,:,:]).view(batch_size, n_channels,  kernel_side*kernel_side, h_out*w_out).transpose(2, 3)#reshape((batch_size,n_channels,h_out,w_out))
    #for channel in range(n_channels):
    kern_per_out = len(kernels)//out_channels
    #print(len(kernels),out_channels)
    for c_out in range(out_channels):
        out_channel_accum = torch.zeros((batch_size, h_out, w_out), device=device)

        # Aggregate outputs from each kernel assigned to this output channel
        for k_idx in range(kern_per_out):
            kernel = kernels[c_out * kern_per_out + k_idx]
            conv_result = kernel.conv.forward(conv_groups[:, k_idx, :, :].flatten(0, 1))  # Apply kernel with non-linear function
            out_channel_accum += conv_result.view(batch_size, h_out, w_out)

        matrix_out[:, c_out, :, :] = out_channel_accum  # Store results in output tensor
    
    return matrix_out
def add_padding(matrix: np.ndarray, 
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix. 

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding
    
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix
    
    return padded_matrix


class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order:int = 3,
            scale_noise:float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device: str = "cpu"
        ):
        """
        Kan Convolutional Layer with multiple convolutions
        
        Args:
            n_convs (int): Number of convolutions to apply
            kernel_size (tuple): Size of the kernel
            stride (tuple): Stride of the convolution
            padding (tuple): Padding of the convolution
            dilation (tuple): Dilation of the convolution
            grid_size (int): Size of the grid
            spline_order (int): Order of the spline
            scale_noise (float): Scale of the noise
            scale_base (float): Scale of the base
            scale_spline (float): Scale of the spline
            base_activation (torch.nn.Module): Activation function
            grid_eps (float): Epsilon of the grid
            grid_range (tuple): Range of the grid
            device (str): Device to use
        """


        super(KAN_Convolutional_Layer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        # self.device = device
        self.dilation = dilation
        self.padding = padding
        self.convs = torch.nn.ModuleList()
        self.stride = stride

        
        # Create n_convs KAN_Convolution objects
        for _ in range(in_channels*out_channels):
            self.convs.append(
                KAN_Convolution(
                    kernel_size= kernel_size,
                    stride = stride,
                    padding=padding,
                    dilation = dilation,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    # device = device ## changed device to be allocated as per the input device for pytorch DDP
                )
            )

    def forward(self, x: torch.Tensor):
        # If there are multiple convolutions, apply them all
        self.device = x.device
        #if self.n_convs>1:
        return multiple_convs_kan_conv2d(x, self.convs,self.kernel_size[0],self.out_channels,self.stride,self.dilation,self.padding,self.device)
        
        # If there is only one convolution, apply it
        #return self.convs[0].forward(x)
        

class KAN_Convolution(torch.nn.Module):
    def __init__(
            self,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device = "cpu"
        ):
        """
        Args
        """
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.device = device
        self.conv = KANLinear(
            in_features = math.prod(kernel_size),
            out_features = 1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def forward(self, x: torch.Tensor):
        self.device = x.device
        return kan_conv2d(x, self.conv,self.kernel_size[0],self.stride,self.dilation,self.padding,self.device)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum( layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)



