import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import measure
from typing import Sequence, Union
from torch import Tensor, einsum
from torchmetrics.utilities.distributed import reduce
# from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
import cv2
from typing import List, cast
from utils.BD_loss import one_hot2hd_dist, simplex, probs2one_hot
from torch.autograd import Variable






def dice_loss(predict, target):			

    assert predict.size() == target.size(), "the size of predict and target must be equal."

    smooth = 1e-5           

    y_true_f = target.contiguous().view(target.shape[0], -1)        
    y_pred_f = predict.contiguous().view(predict.shape[0], -1)      

    intersection = torch.sum(torch.mul(y_pred_f, y_true_f), dim=1) + smooth     
    union = torch.sum(y_pred_f, dim=1) + torch.sum(y_true_f, dim=1) + smooth

    dice_score = (2.0 * intersection) / union
    dice_loss = 1 - dice_score

    return dice_loss.mean()


class Dice_Loss(nn.Module):
    def __init__(self):
        super(Dice_Loss, self).__init__()

    def forward(self, predicts, target):

        preds = torch.softmax(predicts, dim=1)
        
        dice_loss0 = dice_loss(preds[:, 0, :, :], 1 - target[:, 0, :, :])		
        dice_loss1 = dice_loss(preds[:, 1, :, :], target[:, 1, :, :])
        loss_D = (dice_loss0.mean() + dice_loss1.mean())/2.0

        return loss_D	



def kl_div_zmuv(mu: Tensor, logvar: Tensor) -> Tensor:
    """Computes the KL divergence between a specified distribution and a N(0,1) Gaussian distribution.

    It is the standard loss to use for the reparametrization trick when training a variational autoencoder.

    Notes:
        - 'zmuv' stands for Zero Mean, Unit Variance.

    Args:
        mu: (N, Z), Mean of the distribution to compare to N(0,1).
        logvar: (N, Z) Log variance of the distribution to compare to N(0,1).

    Returns:
        (1,), KL divergence term of the VAE's loss.
    """
    kl_div_by_samples = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  
	
    return kl_div_by_samples     





class HausdorffLoss(nn.Module):
    def __init__(self, p=2):
        super(HausdorffLoss, self).__init__()
        self.p = p

    def torch2D_Hausdorff_distance(self, x, y, p=2):  				
        x = x.float()
        y = y.float()
        distance_matrix = torch.cdist(x, y, p=p)  					# p=2 means Euclidean Distance

        # print(distance_matrix.shape)    # torch.Size([2, 2, 240, 240])

        value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
        value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

        value = torch.cat((value1, value2), dim=1)

        return value.max(1)[0].mean()

    def forward(self, x, y):  										# Input be like (Batch,height,width)
        loss = self.torch2D_Hausdorff_distance(x, y, self.p)
        return loss




class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)        # np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)      # 必须使用numpy进行操作，但是会导致field.requires_grad = False
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu())).float()       # .cpu().numpy()  未参与计算
        target_dt = torch.from_numpy(self.distance_field(target.cpu())).float()   # .cpu().numpy()
        
        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance.cuda()
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:

            return loss



class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().numpy(), target.cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss



class HausdorffLossNew():
    """
    Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]    # kwargs["idc"]
        # print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs)
        assert simplex(target)
        assert probs.shape == target.shape

        B, K, *xy = probs.shape  # type: ignore

        pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
        tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
        assert pc.shape == tc.shape == (B, len(self.idc), *xy)

        target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
                                              for b in range(B)], axis=0)
        assert target_dm_npy.shape == tc.shape == pc.shape
        tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

        pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
        pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
                                            for b in range(B)], axis=0)
        assert pred_dm_npy.shape == tc.shape == pc.shape
        pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

        delta = (pc - tc)**2
        dtm = tdm**2 + pdm**2

        multipled = einsum("bkwh,bkwh->bkwh", delta, dtm)

        loss = multipled.mean()

        return loss



class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask
        
        input = flatten(input)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)  # 2
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))   # (1, 0, 2, 3)
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()    # torch.Size([2, 16, 240, 320])
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)



class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]
        # print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        # assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss    


def dice_loss_vnet(prediction, target, epsilon=1e-6):
    """
    prediction is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the prediction
    """
    assert prediction.size() == target.size(), "prediction sizes must be equal."
    assert prediction.dim() == 4, "prediction must be a 4D Tensor."
    uniques = np.unique(target.cpu().data.numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(prediction, dim=1) # channel/classwise
   
    num = probs * target # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)
    # num = torch.sum(num, dim=1) # b,c
    
    den1 = probs * probs # --p^2
    den1 = torch.sum(den1, dim=3) #b,c,h
    den1 = torch.sum(den1, dim=2) #b,c,h
    # den1 = torch.sum(den1, dim=1) 
  
    
    den2 = target * target # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)
    # den2 = torch.sum(den2, dim=1) 
    den = (den1+den2+epsilon)#.clamp(min=epsilon)
    dice=2*(num/den)
    dice_eso=dice #[:,1:]#we ignore background dice val, and take the foreground
    dice_total=torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return 1 - dice_total


class Dice_Loss_Vnet(nn.Module):
    def __init__(self):
        super(Dice_Loss_Vnet, self).__init__()

    def forward(self, predicts, target):

        preds = torch.softmax(predicts, dim=1)
        # preds = predicts
        dice_loss0 = dice_loss_vnet(preds[:, [0], :, :], 1 - target[:, [0], :, :])		# here is 1 - target[:, 0, :, :], opposite number
        dice_loss1 = dice_loss_vnet(preds[:, [1], :, :], target[:, [1], :, :])
        loss_D = (dice_loss0.mean() + dice_loss1.mean())/2.0

		# loss_R = rank_loss(preds[:, 1, :, :], target)

        return loss_D	#, loss_R



