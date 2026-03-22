import torch
import torch.nn as nn
import torch.nn.functional as F


class PointPillarsLoss(nn.Module):
    """
    Kombinovaná loss:
      cls_loss — Focal loss pre klasifikáciu
      reg_loss — SmoothL1 pre bounding box regrésiu
      dir_loss — CrossEntropy pre smer objektu
    """

    def __init__(self, alpha=0.25, gamma=2.0,
                 w_cls=1.0, w_reg=2.0, w_dir=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.w_cls = w_cls
        self.w_reg = w_reg
        self.w_dir = w_dir

    def focal_loss(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        bce      = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        p_t      = pred_sig * target + (1 - pred_sig) * (1 - target)
        alpha_t  = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss     = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()

    def forward(self, preds, gt_boxes, batch_size):
        cls_pred = preds['cls']   # (B, num_anchors*num_classes, H, W)
        reg_pred = preds['reg']   # (B, num_anchors*7, H, W)
        dir_pred = preds['dir']   # (B, num_anchors*2, H, W)

        B, _, H, W = cls_pred.shape

        # Zjednodušené targets — náhodné pre overenie tréningovej slučky
        # V produkcii tu bude anchor matching s gt_boxes
        cls_target = torch.zeros_like(cls_pred)
        reg_target = torch.zeros_like(reg_pred)
        dir_target = torch.zeros(B, reg_pred.shape[1] // 7, H, W,
                                 dtype=torch.long, device=cls_pred.device)

        cls_loss = self.focal_loss(cls_pred, cls_target)
        reg_loss = F.smooth_l1_loss(reg_pred, reg_target)
        dir_loss = F.cross_entropy(dir_pred, dir_target.squeeze(1))

        total = (self.w_cls * cls_loss +
                 self.w_reg * reg_loss +
                 self.w_dir * dir_loss)

        return {
            'total':   total,
            'cls':     cls_loss,
            'reg':     reg_loss,
            'dir':     dir_loss,
        }
