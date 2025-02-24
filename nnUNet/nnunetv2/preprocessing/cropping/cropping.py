import numpy as np
from typing import List
# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
import cupy as cp
from cupyx.scipy import ndimage

# Assume you have a binary array 'mask' on the GPU
def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    filled_mask = ndimage.binary_fill_holes(nonzero_mask)
    return filled_mask

def get_bbox_from_mask(mask: cp.ndarray) -> List[List[int]]:
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision 
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    
    This implementation uses CuPy for GPU acceleration. The mask should be a CuPy array.
    
    Args:
        mask (cp.ndarray): 3D mask array on GPU
        
    Returns:
        List[List[int]]: Bounding box coordinates as [[minz, maxz], [minx, maxx], [miny, maxy]]
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    
    # Create range arrays on GPU
    zidx = cp.arange(Z)
    xidx = cp.arange(X)
    yidx = cp.arange(Y)
    
    # Z dimension
    for z in zidx.get():  # .get() to iterate over CPU array
        if cp.any(mask[z]).get():  # .get() to get boolean result to CPU
            minzidx = z
            break
    for z in zidx[::-1].get():
        if cp.any(mask[z]).get():
            maxzidx = z + 1
            break
            
    # X dimension
    for x in xidx.get():
        if cp.any(mask[:, x]).get():
            minxidx = x
            break
    for x in xidx[::-1].get():
        if cp.any(mask[:, x]).get():
            maxxidx = x + 1
            break
            
    # Y dimension
    for y in yidx.get():
        if cp.any(mask[:, :, y]).get():
            minyidx = y
            break
    for y in yidx[::-1].get():
        if cp.any(mask[:, :, y]).get():
            maxyidx = y + 1
            break
            
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def bounding_box_to_slice(bounding_box: List[List[int]]):
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73
    """
    return tuple([slice(*i) for i in bounding_box])

def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox


