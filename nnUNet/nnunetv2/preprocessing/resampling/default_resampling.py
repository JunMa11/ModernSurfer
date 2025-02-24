from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import sklearn
import torch
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
from skimage.transform import resize
from nnunetv2.configuration import ANISO_THRESHOLD


def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], anisotropy_threshold=ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape



def determine_do_sep_z_and_axis(
        force_separate_z: bool,
        current_spacing,
        new_spacing,
        separate_z_anisotropy_threshold: float = ANISO_THRESHOLD) -> Tuple[bool, Union[int, None]]:
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
            axis = None
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
            axis = None
        else:
            axis = axis[0]
    return do_separate_z, axis


def resample_data_or_seg_to_spacing(data: np.ndarray,
                                    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    is_seg: bool = False,
                                    order: int = 3, order_z: int = 0,
                                    force_separate_z: Union[bool, None] = False,
                                    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    shape = np.array(data.shape)
    new_shape = compute_new_shape(shape[1:], current_spacing, new_spacing)

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


def resample_data_or_seg_to_shape(data: Union[torch.Tensor, np.ndarray],
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    """
    needed for segmentation export. Stupid, I know
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


def resample_data_or_seg(data: np.ndarray, new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
                         is_seg: bool = False, axis: Union[None, int] = None, order: int = 3,
                         do_separate_z: bool = False, order_z: int = 0, dtype_out = None):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if dtype_out is None:
        dtype_out = data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    if np.any(shape != new_shape):
        data = data.astype(float, copy=False)
        if do_separate_z:
            # print("separate z, order in z is", order_z, "order inplane is", order)
            assert axis is not None, 'If do_separate_z, we need to know what axis is anisotropic'
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp)
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                    else:
                        reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape

                    # align_corners=False
                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')[None]
                    else:
                        unique_labels = np.sort(pd.unique(reshaped_here.ravel()))  # np.unique(reshaped_data)
                        for i, cl in enumerate(unique_labels):
                            reshaped_final[c][np.round(
                                map_coordinates((reshaped_here == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest')) > 0.5] = cl
                else:
                    reshaped_final[c] = reshaped_here
        else:
            # print("no separate z, order", order)
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        return reshaped_final
    else:
        # print("no resampling necessary")
        return data

def fast_resize_segmentation(segmentation, new_shape, mode="nearest"):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype

    if isinstance(segmentation, torch.Tensor):
        assert len(segmentation.shape[2:]) == len(new_shape), f"segmentation.shape = {segmentation.shape}, new_shape = {new_shape}"
    else:
        assert len(segmentation.shape[1:]) == len(new_shape), f"segmentation.shape = {segmentation.shape}, new_shape = {new_shape}"
        segmentation = torch.from_numpy(segmentation).unsqueeze(0).float()
    #if order == 0:
        #return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    if mode == "nearest":
        seg_torch = torch.nn.functional.interpolate(segmentation, new_shape, mode=mode)
        reshaped = seg_torch
    else:
        #reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
        unique_labels = torch.unique(segmentation)
        seg_torch = segmentation
        reshaped = torch.zeros([*seg_torch.shape[:2], *new_shape], dtype=seg_torch.dtype, device=seg_torch.device)
        for i, c in enumerate(unique_labels):
            #mask = segmentation == c
            #reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            mask = seg_torch == c
            reshaped_multihot = torch.nn.functional.interpolate(mask.float(), new_shape, mode=mode, align_corners=False)
            reshaped[reshaped_multihot >= 0.5] = c

    return reshaped


def fast_resample_data_or_seg_to_shape(data: Union[torch.Tensor, np.ndarray],
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")
    order_to_mode_map = {
        0: "nearest",
        1: "trilinear" if new_shape[0] > 1 else "bilinear",
        2: "trilinear" if new_shape[0] > 1 else "bilinear",
        3: "trilinear" if new_shape[0] > 1 else "bicubic",
        4: "trilinear" if new_shape[0] > 1 else "bicubic",
        5: "trilinear" if new_shape[0] > 1 else "bicubic",
    }
    
    if is_seg:
        print(f"seg.shape: {data.shape}")
        resize_fn = fast_resize_segmentation
        kwargs = {
            "mode": order_to_mode_map[order]
        }
    else:
        print(f"data.shape: {data.shape}")
        resize_fn = torch.nn.functional.interpolate
        kwargs = {
            'mode': order_to_mode_map[order],
            'align_corners': False
        }
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        if not isinstance(data, torch.Tensor):
            #torch_data = torch.from_numpy(data).float()
            torch_data = torch.as_tensor(data.get())
        else:
            torch_data = data.float()
        if new_shape[0] == 1:
            torch_data = torch_data.transpose(1, 0)
            new_shape = new_shape[1:]
        else:
            torch_data = torch_data.unsqueeze(0)
        
        torch_data = resize_fn(torch_data.to(device), tuple(new_shape), **kwargs)

        if new_shape[0] == 1:
            torch_data = torch_data.transpose(1, 0)
        else:
            torch_data = torch_data.squeeze(0)

        # if use_gpu:
        #     torch_data = torch_data.cpu()
        reshaped_final_data = torch_data
        # if isinstance(data, np.ndarray):
        #     reshaped_final_data = torch_data.numpy().astype(dtype_data)
        # else:
        #     reshaped_final_data = torch_data.to(dtype_data)
        
        print(f"Reshaped data from {shape} to {new_shape}")
        print(f"reshaped_final_data shape: {reshaped_final_data.shape}")
        assert reshaped_final_data.ndim == 4, f"reshaped_final_data.shape = {reshaped_final_data.shape}"
        return reshaped_final_data
    else:
        print("no resampling necessary")
        return data


if __name__ == '__main__':
    input_array = np.random.random((1, 42, 231, 142))
    output_shape = (52, 256, 256)
    out = resample_data_or_seg(input_array, output_shape, is_seg=False, axis=3, order=1, order_z=0, do_separate_z=True)
    print(out.shape, input_array.shape)
