import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch
import numpy as np
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm3d(num_features=out_channels))
    result.add_module('relu', nn.ReLU())
    return result

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm3d(num_features=out_channels))
    return result

class SELayer3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer3D, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)  # Squeeze
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)  # Excitation
        return x * y.expand_as(x)  # Scale input feature maps


class RepVGGplusBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros',
                 deploy=False, use_post_se=False):
        super(RepVGGplusBlock3D, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels


        assert padding == 1



        self.nonlinearity = nn.ReLU()

        if use_post_se:
            self.post_se = SEBlock3D(out_channels, internal_neurons=out_channels // 4)
        else:
            self.post_se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, 
                                         bias=True, padding_mode=padding_mode)
        else:
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm3d(num_features=out_channels)
            else:
                self.rbr_identity = None

            self.rbr_dense = conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups)
            if isinstance(kernel_size, list):
                padding_11 = [padding - k // 2 for k in kernel_size]  # Create a list of paddings for each kernel size
            else:
                padding_11 = padding - kernel_size // 2
            self.rbr_1x1 = conv_bn(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, x):
        if self.deploy:
            return self.post_se(self.nonlinearity(self.rbr_reparam(x)))

        id_out = 0 if self.rbr_identity is None else self.rbr_identity(x)
        out = self.rbr_dense(x) + self.rbr_1x1(x) + id_out
        out = self.post_se(self.nonlinearity(out))
        return out

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel, running_mean, running_var, gamma, beta, eps = (
                branch.conv.weight, branch.bn.running_mean, branch.bn.running_var, 
                branch.bn.weight, branch.bn.bias, branch.bn.eps
            )
        else:
            assert isinstance(branch, nn.BatchNorm3d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel, running_mean, running_var, gamma, beta, eps = (
                self.id_tensor, branch.running_mean, branch.running_var, 
                branch.weight, branch.bias, branch.eps
            )
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv3d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class RepVGGplusStage(nn.Module):

    def __init__(self, in_planes, planes, num_blocks, stride, use_checkpoint, use_post_se=False, deploy=False):
        super().__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        self.in_planes = in_planes
        for stride in strides:
            cur_groups = 1
            blocks.append(RepVGGplusBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=deploy, use_post_se=use_post_se))
            self.in_planes = planes
        self.blocks = nn.ModuleList(blocks)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x

def maybe_convert_scalar_to_list(conv_op, scalar):
    """
    useful for converting, for example, kernel_size=3 to [3, 3, 3] in case of nn.Conv3d
    :param conv_op:
    :param scalar:
    :return:
    """
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        if conv_op == nn.Conv2d:
            return [scalar] * 2
        elif conv_op == nn.Conv3d:
            return [scalar] * 3
        elif conv_op == nn.Conv1d:
            return [scalar] * 1
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))
    else:
        return scalar

class StackedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 use_checkpoint,
                 deploy: bool = False
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param deploy:
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs
        blocks = [RepVGGplusBlock3D(
                input_channels, output_channels[0], kernel_size, initial_stride, deploy = deploy
            )]
        for i in range(1, num_convs):
            blocks.append(RepVGGplusBlock3D(
                    output_channels[i - 1], output_channels[i], kernel_size, 1, deploy = deploy
                ))
        self.blocks = nn.ModuleList(blocks)
        # self.convs = nn.Sequential(
        #     RepVGGplusBlock3D(
        #         input_channels, output_channels[0], kernel_size, initial_stride, deploy = deploy
        #     ),
        #     *[
        #         RepVGGplusBlock3D(
        #             output_channels[i - 1], output_channels[i], kernel_size, 1, deploy = deploy
        #         )
        #         for i in range(1, num_convs)
        #     ]
        # )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        #return self.convs(x)
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output
