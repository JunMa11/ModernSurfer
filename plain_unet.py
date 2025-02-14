from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

setting = {'input_channels': 1, 'segmentation_heads': 9, 'arch_init_args': {'n_stages': 6, 'features_per_stage': [32, 64, 128, 256, 320, 320], 'conv_op': torch.nn.modules.conv.Conv3d, 'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], 'n_conv_per_stage': [2, 2, 2, 2, 2, 2], 'n_conv_per_stage_decoder': [2, 2, 2, 2, 2], 'conv_bias': True, 'norm_op': nn.InstanceNorm3d, 'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': torch.nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True}}, 'arch_init_args_req_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']}
setting['arch_init_args']['norm_op'] = nn.BatchNorm3d
# setting['arch_init_args_req_import'] = ['conv_op', 'dropout_op', 'nonlin']

class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.input_channels = input_channels
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


def build_fix_plain_unet():
    input_channels = setting['input_channels']
    arch_init_args = setting['arch_init_args']
    n_stages = arch_init_args['n_stages']
    features_per_stage = arch_init_args['features_per_stage']
    conv_op = arch_init_args['conv_op']
    kernel_sizes = arch_init_args['kernel_sizes']
    strides = arch_init_args['strides']
    n_conv_per_stage = arch_init_args['n_conv_per_stage']
    num_classes = setting['segmentation_heads']
    n_conv_per_stage_decoder = arch_init_args['n_conv_per_stage_decoder']
    conv_bias = arch_init_args['conv_bias']
    norm_op = arch_init_args['norm_op']
    norm_op_kwargs = arch_init_args['norm_op_kwargs']
    dropout_op = arch_init_args['dropout_op']
    dropout_op_kwargs = arch_init_args['dropout_op_kwargs']
    nonlin = arch_init_args['nonlin']
    nonlin_kwargs = arch_init_args['nonlin_kwargs']
    

    network = PlainConvUNet(
                 input_channels= input_channels,
                 n_stages=n_stages,
                 features_per_stage=features_per_stage,
                 conv_op=conv_op,
                 kernel_sizes=kernel_sizes,
                 strides=strides,
                 n_conv_per_stage=n_conv_per_stage,
                 num_classes=num_classes,
                 n_conv_per_stage_decoder=n_conv_per_stage_decoder,
                 conv_bias=conv_bias,
                 norm_op=norm_op,
                 norm_op_kwargs=norm_op_kwargs,
                 dropout_op=dropout_op,
                 dropout_op_kwargs=dropout_op_kwargs,
                 nonlin=nonlin,
                 nonlin_kwargs=nonlin_kwargs,
                 deep_supervision = False,
                 nonlin_first = False
                 )
    if hasattr(network, 'initialize'):
        network.apply(network.initialize)
    return network