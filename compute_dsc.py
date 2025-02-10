import os 
join = os.path.join
import numpy as np
import SimpleITK as sitk
import pandas as pd
from collections import OrderedDict

def dice_multi_class(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape
    labels = np.unique(targets)[1:]
    dices = []
    for label in labels:
        pred = preds == label
        target = targets == label
        intersection = (pred * target).sum()
        dices.append((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
    return np.mean(dices)

seg_metric = OrderedDict()
seg_metric['name'] = []
seg_metric['dice'] = []

seg_path = ''
gt_path = ''
names = os.listdir(seg_path)
names = [name for name in names if os.path.isfile(join(seg_path, name))]    
for name in names[:3]:
    seg_sitk = sitk.ReadImage(join(seg_path, name))
    gt_sitk = sitk.ReadImage(join(gt_path, name))
    seg = sitk.GetArrayFromImage(seg_sitk)
    gt = sitk.GetArrayFromImage(gt_sitk)
    seg_metric['name'].append(name)
    seg_metric['dice'].append(dice_multi_class(seg, gt))

df = pd.DataFrame(seg_metric)
df.to_csv('seg_metric.csv', index=False)


