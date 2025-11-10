import torch
import torch.nn.functional as F

# ---------------------------
# soft-clDice implementation (2D)
# ---------------------------
def soft_erode2d(img):
    # img: (B, C, H, W)  float in [0,1]
    p1 = -F.max_pool2d(-img, kernel_size=(3,1), stride=(1,1), padding=(1,0))
    p2 = -F.max_pool2d(-img, kernel_size=(1,3), stride=(1,1), padding=(0,1))
    return torch.min(p1, p2)

def soft_dilate2d(img):
    return F.max_pool2d(img, kernel_size=(3,3), stride=(1,1), padding=(1,1))

def soft_open2d(img):
    return soft_dilate2d(soft_erode2d(img))

def soft_skel2d(img, iterations=10):
    """
    Differentiable soft skeletonization (2D) from clDice supplemental.
    img: (B, C, H, W)  values in [0,1]
    iterations: number of erosions
    returns: soft skeleton same shape as img
    """
    img1 = soft_open2d(img)
    skel = F.relu(img - img1)
    for _ in range(iterations):
        img = soft_erode2d(img)
        img1 = soft_open2d(img)
        delta = F.relu(img - img1)
        # accumulate new skeleton parts, avoid double counting using formula from supp.
        skel = skel + F.relu(delta - skel * delta)
    return skel

def cldice_from_probabilities(pred_prob, gt_prob, skel_iter=10, eps=1e-6):
    """
    pred_prob, gt_prob: tensors (B, 1, H, W), values in [0,1], float
    returns: clDice score per-batch (mean of batch) in [0,1]
    """
    # compute soft skeletons
    S_pred = soft_skel2d(pred_prob, iterations=skel_iter)  # skeleton of prediction
    S_gt   = soft_skel2d(gt_prob, iterations=skel_iter)    # skeleton of ground-truth

    # topological precision: fraction of skeleton_pred that lies inside gt
    # tprec = sum( S_pred * gt ) / ( sum(S_pred) )
    tprec_num = (S_pred * gt_prob).sum(dim=(1,2,3))
    tprec_den = S_pred.sum(dim=(1,2,3)).clamp_min(eps)
    tprec = tprec_num / tprec_den

    # topological sensitivity: fraction of skeleton_gt that lies inside pred
    tsens_num = (S_gt * pred_prob).sum(dim=(1,2,3))
    tsens_den = S_gt.sum(dim=(1,2,3)).clamp_min(eps)
    tsens = tsens_num / tsens_den

    # clDice per-sample (harmonic mean)
    cldice = (2 * tprec * tsens) / (tprec + tsens + eps)
    # return mean over batch
    return cldice.mean()

# region dice loss helper (uses probabilities)
def dice_loss_from_probs(pred_prob, gt_prob, eps=1e-6):
    # pred_prob, gt_prob shape (B,1,H,W)
    inter = (pred_prob * gt_prob).sum(dim=(1,2,3))
    denom = pred_prob.sum(dim=(1,2,3)) + gt_prob.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (denom + eps)
    return (1 - dice).mean()

class SoftCLDiceLoss(torch.nn.Module):
    """
    Lc(α) = α * region_loss + (1 - α) * (1 - clDice)
    - region: 'dice' or 'bce'
    - skel_iter: iterations for soft skeleton (try 3~10; higher=better skeleton but slower)
    """
    def __init__(self, alpha=0.5, region='dice', skel_iter=10):
        super().__init__()
        assert region in ('dice', 'bce')
        self.alpha = alpha
        self.region = region
        self.skel_iter = skel_iter
        if region == 'bce':
            # use BCEWithLogits if passing logits. But here we will pass probabilities to clDice,
            # so compute BCE on logits separately or use probabilities (stable to use BCEWithLogits on logits).
            # We'll compute BCE on logits in forward if logits provided.
            self.bce = torch.nn.BCEWithLogitsLoss()  # expects logits
        # if region == 'dice', we'll use dice_loss_from_probs

    def forward(self, logits, gt):
        """
        logits: model output (B,1,H,W) (raw logits)
        gt: ground-truth mask (B,1,H,W)  values 0/1 (float or long)
        """
        # ensure gt is float
        gt = gt.float()
        # probabilities for skeleton & dice computations
        probs = torch.sigmoid(logits)

        # region loss
        if self.region == 'dice':
            region_l = dice_loss_from_probs(probs, gt)
        else:
            # BCE computed on logits for numerical stability
            region_l = self.bce(logits, gt)

        # clDice part
        cldice_score = cldice_from_probabilities(probs, gt, skel_iter=self.skel_iter)
        cldice_loss = 1.0 - cldice_score

        loss = self.alpha * region_l + (1.0 - self.alpha) * cldice_loss
        return loss
