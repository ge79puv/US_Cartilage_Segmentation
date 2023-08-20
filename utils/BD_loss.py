import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from typing import Iterable, List, Set, Tuple





# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape                # Tuple[int, int, int, int]
    assert simplex(probs)
 
    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)
 
    return res
 
 
def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)
 
    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)
 
    return res
 
 
def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)
 
 
def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])
 
    # Assert utils
 
 
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())
 
 
def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)



def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:                         # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    # assert sset(seg, list(range(C)))
 
    b, w, h = seg.shape                             # Tuple[int, int, int]
 
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)
 
    return res
 

# The idea is to leave blank the negative classes
# since this is one-hot encoded, another class will supervise that pixel

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=1)       # , axis=0
    # C: int = len(seg)       # C = 2, but doesn't mean batch, C = classes
    batch_size = seg.shape[0]
    C = seg.shape[1]

    res = np.zeros_like(seg)
    # res = res.astype(np.float64)
    for i in range(batch_size):
        for c in range(C):
            posmask = seg[i][c].astype(np.bool)
            # batch 1: 0 (240, 320) 71797 1 (240, 320) 5003  71797+5003 = 76800 = 240*320
            # batch 2: 0 (240, 320) 71778 1 (240, 320) 5022   
            # print(i, c, distance(posmask), distance(posmask).sum())
            if posmask.any():
                negmask = ~posmask
                res[i][c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask     # distance()输出为正值
                # 求BD Loss只需要看第二个channel，忽略 0 channel，上一行实现了：ground boundary以外的distance是正数，以内的是负数
                # 正值的时候对应于distance(negmask) * negmask，负值的时候对应于 - (distance(posmask) - 1) * posmask
    return res

 
class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]  # 这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)
 
        pc = probs[:, self.idc, ...].type(torch.float32)      # self.idc = [1] to keep the shape, torch.Size([2, 1, 240, 320])
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multiplied = einsum("bcwh,bcwh->bcwh", pc, dc)
 
        loss = multiplied.mean()
 
        return loss


def one_hot2hd_dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                    dtype=None) -> np.ndarray:
    """
    Used for https://arxiv.org/pdf/1904.10030.pdf,
    implementation from https://github.com/JunMa11/SegWithDistMap
    """
    # Relasx the assertion to allow computation live on only a
    # subset of the classes
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            res[k] = distance(posmask, sampling=resolution)

    return res




def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res




