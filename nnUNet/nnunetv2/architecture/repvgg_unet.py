import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
import torch.nn.functional as F

class SEBlock(nn.Module):

	def __init__(self, input_channels, internal_neurons):
		super(SEBlock, self).__init__()
		self.down = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
		self.up = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
		self.input_channels = input_channels

	def forward(self, inputs):
		x = F.avg_pool3d(inputs, kernel_size=inputs.size(-1))
		x = self.down(x)
		x = F.relu(x)
		x = self.up(x)
		x = torch.sigmoid(x)
		x = x.view(-1, self.input_channels, 1, 1, 1)
		return inputs * x

class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

class StackedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1):
	result = nn.Sequential()
	result.add_module('conv', nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
	                                    kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
	result.add_module('bn', nn.BatchNorm3d(num_features=out_channels))
	return result

class RepVGGBlock(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size,
	             stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
		super(RepVGGBlock, self).__init__()
		self.deploy = deploy
		self.groups = groups
		self.in_channels = in_channels

		assert kernel_size == 3

		#   Considering dilation, the actuall size of rbr_dense is  kernel_size + 2*(dilation - 1)
		#   For the same output size:     (padding - padding_11) ==  (kernel_size + 2*(dilation - 1) - 1) // 2
		padding_11 = padding - (kernel_size + 2*(dilation - 1) - 1) // 2
		assert padding_11 >= 0, 'It seems that your configuration of kernelsize (k), padding (p) and dilation (d) will ' \
		                        'reduce the output size. In this case, you should crop the input of conv1x1. ' \
		                        'Since this is not a common case, we do not consider it. But it is easy to implement (e.g., self.rbr_1x1(inputs[:,:,1:-1,1:-1])). ' \
		                        'The common combinations are (k=3,p=1,d=1) (no dilation), (k=3,p=2,d=2) and (k=3,p=4,d=4) (PSPNet).'

		self.nonlinearity = nn.ReLU()

		if use_se:
			self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
		else:
			self.se = nn.Identity()

		if deploy:
			self.rbr_reparam = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
			                             padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

		else:
			self.rbr_identity = nn.BatchNorm3d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
			self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
			self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
		# print('RepVGG Block, identity = ', self.rbr_identity)


	def forward(self, inputs):
		if hasattr(self, 'rbr_reparam'):
			return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

		if self.rbr_identity is None:
			id_out = 0
		else:
			id_out = self.rbr_identity(inputs)

		return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))



	#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
	#   You can get the equivalent kernel and bias at any time and do whatever you want,
	#   for example, apply some penalties or constraints during training, just like you do to the other models.
	#   May be useful for quantization or pruning.
	def get_equivalent_kernel_bias(self):
		kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
		kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
		kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
		return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

	def _pad_1x1_to_3x3_tensor(self, kernel1x1):
		if kernel1x1 is None:
			return 0
		else:
			return torch.nn.functional.pad(kernel1x1, [1,1,1,1,1,1]) #for 3DCNN, original: torch.nn.functional.pad(kernel1x1, [1,1,1,1])

	def _fuse_bn_tensor(self, branch):
		if branch is None:
			return 0, 0
		if isinstance(branch, nn.Sequential):
			kernel = branch.conv.weight
			running_mean = branch.bn.running_mean
			running_var = branch.bn.running_var
			gamma = branch.bn.weight
			beta = branch.bn.bias
			eps = branch.bn.eps
		else:
			assert isinstance(branch, nn.BatchNorm3d)
			if not hasattr(self, 'id_tensor'):
				input_dim = self.in_channels // self.groups
				# for 3DCNN, original: kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
				kernel_value = np.zeros((self.in_channels, input_dim, 3, 3, 3), dtype=np.float32)
				for i in range(self.in_channels):
					kernel_value[i, i % input_dim, 1, 1, 1] = 1 #for 3DCNN, original: kernel_value[i, i % input_dim, 1, 1] = 1
				self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
			kernel = self.id_tensor
			running_mean = branch.running_mean
			running_var = branch.running_var
			gamma = branch.weight
			beta = branch.bias
			eps = branch.eps
		std = (running_var + eps).sqrt()
		t = (gamma / std).reshape(-1, 1, 1, 1, 1) # for 3DCNN, original: t = (gamma / std).reshape(-1, 1, 1, 1)
		return kernel * t, beta - running_mean * gamma / std

	def switch_to_deploy(self):
		if hasattr(self, 'rbr_reparam'):
			return
		kernel, bias = self.get_equivalent_kernel_bias()
		self.rbr_reparam = nn.Conv3d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
		                             kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
		                             padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
		self.rbr_reparam.weight.data = kernel
		self.rbr_reparam.bias.data = bias
		for para in self.parameters():
			para.detach_()
		self.__delattr__('rbr_dense')
		self.__delattr__('rbr_1x1')
		if hasattr(self, 'rbr_identity'):
			self.__delattr__('rbr_identity')


class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 convs_per_stage: List[str] = None,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 use_se: bool = False,
                 deploy: bool = False,
                 override_groups_map = None,
                 pool: str = 'conv'
                 ):

        super(PlainConvEncoder, self).__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()

            if convs_per_stage:
                if convs_per_stage[s] == 'repvgg':
                    stage_modules.append(self._make_stage(
                        n_conv_per_stage[s], input_channels, features_per_stage[s], conv_stride, deploy, use_se
                    ))
                else:
                    stage_modules.append(StackedConvBlocks(
                        n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                        conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, False
                    ))
            else:
                stage_modules.append(self._make_stage(
                    n_conv_per_stage[s], input_channels, features_per_stage[s], conv_stride, deploy, use_se
                ))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.kernel_sizes = kernel_sizes


    def _make_stage(self, num_blocks, in_planes, planes, stride, deploy, use_se):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock(in_channels=in_planes, out_channels=planes, kernel_size=3,
                                        stride=stride, padding=1, groups=1, deploy=deploy, use_se=use_se))
            in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 use_se: bool = False,
                 deploy: bool = False,
                 override_groups_map = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.use_se = use_se
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=True
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(self._make_stage(
                n_conv_per_stage[s-1], 2 * input_features_skip, input_features_skip, 1, deploy, use_se
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
    
    def _make_stage(self, num_blocks, in_planes, planes, stride, deploy, use_se):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock(in_channels=in_planes, out_channels=planes, kernel_size=3,
                                        stride=stride, padding=1, groups=1, deploy=deploy, use_se=use_se))
            in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


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
                 convs_per_stage: List[str] = None,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 use_se: bool = False,
                 deploy: bool = False
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
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, convs_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True, use_se=use_se,
                                        deploy=deploy)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, use_se=use_se,
                                   deploy=deploy)

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


def plain_unet(output_channels, convs_per_stage, deep_supervision=True, deploy=False):
    input_channels = 1
    n_stages = 7
    features_per_stage = [
                    32,
                    64,
                    128,
                    256,
                    320,
                    320,
                    320
                ]
    kernel_sizes = [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
    n_conv_per_stage = [
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2
                ]
    n_conv_per_stage_decoder = [
                    2,
                    2,
                    2,
                    2,
                    2,
                    2
                ]

    conv_bias = True
    norm_op = torch.nn.modules.instancenorm.InstanceNorm3d
    norm_op_kwargs = {
                        "eps": 1e-05,
                        "affine": True
                    }
    dropout_op = None
    dropout_op_kwargs = None
    nonlin = torch.nn.LeakyReLU
    nonlin_kwargs = {
                        "inplace": True
                    }

    network = PlainConvUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=output_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        convs_per_stage = convs_per_stage,
        conv_bias = conv_bias,
        norm_op = norm_op,
        norm_op_kwargs = norm_op_kwargs,
        dropout_op = dropout_op,
        dropout_op_kwargs = dropout_op_kwargs,
        nonlin = nonlin,
        nonlin_kwargs = nonlin_kwargs,
        deep_supervision=deep_supervision,
        deploy=deploy
        )
    return network

def plain_unet_702(output_channels, convs_per_stage, deep_supervision=True, deploy=False):
    input_channels = 1
    n_stages = 6
    features_per_stage = [
                        32,
                        64,
                        128,
                        256,
                        320,
                        320
                    ]
    kernel_sizes = [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    n_conv_per_stage = [
                    2,
                    2,
                    2,
                    2,
                    2,
                    2
                ]
    n_conv_per_stage_decoder = [
                    2,
                    2,
                    2,
                    2,
                    2
                ]

    conv_bias = True
    norm_op = torch.nn.modules.instancenorm.InstanceNorm3d
    norm_op_kwargs = {
                        "eps": 1e-05,
                        "affine": True
                    }
    dropout_op = None
    dropout_op_kwargs = None
    nonlin = torch.nn.LeakyReLU
    nonlin_kwargs = {
                        "inplace": True
                    }

    network = PlainConvUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=output_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        convs_per_stage = convs_per_stage,
        conv_bias = conv_bias,
        norm_op = norm_op,
        norm_op_kwargs = norm_op_kwargs,
        dropout_op = dropout_op,
        dropout_op_kwargs = dropout_op_kwargs,
        nonlin = nonlin,
        nonlin_kwargs = nonlin_kwargs,
        deep_supervision=deep_supervision,
        deploy=deploy
        )
    return network
        
def plain_unet_S4(output_channels, deep_supervision=True, deploy=False):
    input_channels = 1
    n_stages = 4
    features_per_stage = [
                32,
                64,
                128,
                256
            ]
    kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    n_conv_per_stage = [
                    2,
                    2,
                    2,
                    2
                ]
    n_conv_per_stage_decoder = [
                    2,
                    2,
                    2
                ]

    network = PlainConvUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=output_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        deep_supervision=deep_supervision,
        deploy=deploy
        )
    return network

def plain_unet_S5(output_channels, deep_supervision=True, deploy=False):
    input_channels = 1
    n_stages = 5
    features_per_stage = [
                32,
                64,
                128,
                256,
                320
            ]
    kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    n_conv_per_stage = [
                    2,
                    2,
                    2,
                    2,
                    2
                ]
    n_conv_per_stage_decoder = [
                    2,
                    2,
                    2,
                    2
                ]
    
    network = PlainConvUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=output_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        deep_supervision=deep_supervision,
        use_se = False,
        deploy=deploy
        )
    return network





if __name__ == "__main__":
    import time
    # Example usage of the PlainConvUNet
    # ckpt_path = '/home/jma/Documents/Ching-Yuan/miccai/nnUNet_data/nnUNet_results/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg_S4__nnUNetPlans__3d_fullres_S4/fold_all/checkpoint_final.pth'
    # ckpt = torch.load(ckpt_path, weights_only=False)
    # ckpt = ckpt['network_weights']
    # ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    #model = create_RepVGG(14, 1, False)
    convs_per_stage = ['stackedconv', 'stackedconv', 'stackedconv', 'repvgg', 'repvgg', 'repvgg', 'repvgg']
    #convs_per_stage = ['stackedconv', 'stackedconv', 'stackedconv', 'stackedconv', 'stackedconv', 'stackedconv', 'stackedconv']
    model = plain_unet(14,convs_per_stage, False, False)
    #model.load_state_dict(ckpt)
    model.eval()
    # for module in model.modules():
    #     if isinstance(module, torch.nn.BatchNorm3d):
    #         nn.init.uniform_(module.running_mean, 0, 0.1)
    #         nn.init.uniform_(module.running_var, 0, 0.1)
    #         nn.init.uniform_(module.weight, 0, 0.1)
    #         nn.init.uniform_(module.bias, 0, 0.1)
    data = torch.rand((1, 1, 128, 512, 512))  # Example input
    output = model(data)


    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    t0 = time.time()
    deploy_y = model(data)
    print(time.time() - t0)
    print('========================== The diff is')
    print(((output - deploy_y) ** 2).sum())
    # print('========================== The diff is')
    # for i in range(len(skips)):
    #     print(((skips[i] - deploy_skips[i]) ** 2).sum())
    # import json

    # with open('arch_dict.json', 'w') as json_file:
    #     json.dump(arch, json_file)
