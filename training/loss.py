import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointPillarsLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        w_cls=1.0,
        w_reg=2.0,
        w_dir=0.2,
        x_range=(0, 200),
        y_range=(-50, 50),
        voxel_size=(0.2, 0.2, 6.0),
        backbone_stride=2,
        ignore_radius=1,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.w_cls = w_cls
        self.w_reg = w_reg
        self.w_dir = w_dir

        self.x_range = x_range
        self.y_range = y_range
        self.voxel_size = voxel_size
        self.backbone_stride = backbone_stride
        self.ignore_radius = ignore_radius

        self.car_class_id = 0
        self.anchor_id = 0
        self.num_classes = 3
        self.reg_box_dim = 7

    def focal_loss_masked(self, pred, target, valid_mask):
        pred_sig = torch.sigmoid(pred)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = pred_sig * target + (1 - pred_sig) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce

        loss = loss * valid_mask.float()
        denom = valid_mask.float().sum().clamp(min=1.0)
        return loss.sum() / denom

    def _build_targets(self, preds, gt_boxes):
        cls_pred = preds['cls']
        reg_pred = preds['reg']
        dir_pred = preds['dir']

        device = cls_pred.device
        B, _, H, W = cls_pred.shape

        cls_target = torch.zeros_like(cls_pred, device=device)
        reg_target = torch.zeros_like(reg_pred, device=device)
        dir_target = torch.zeros_like(dir_pred, device=device)

        cls_valid_mask = torch.ones_like(cls_pred, dtype=torch.bool, device=device)
        reg_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
        dir_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        vx, vy, _ = self.voxel_size

        sx = vx * self.backbone_stride
        sy = vy * self.backbone_stride

        cls_ch = self.anchor_id * self.num_classes + self.car_class_id
        reg_ch0 = self.anchor_id * self.reg_box_dim
        dir_ch0 = self.anchor_id * 2

        for b in range(B):
            if gt_boxes[b].numel() == 0:
                continue

            boxes = gt_boxes[b].to(device)
            boxes = boxes[boxes[:, 7] == self.car_class_id]
            if boxes.numel() == 0:
                continue

            for box in boxes:
                x, y, z, l, w, h, rot, cls_id = box.tolist()

                if not (x_min <= x < x_max and y_min <= y < y_max):
                    continue

                gx = int((x - x_min) / sx)
                gy = int((y - y_min) / sy)

                if gx < 0 or gx >= W or gy < 0 or gy >= H:
                    continue

                cls_target[b, cls_ch, gy, gx] = 1.0

                cell_cx = x_min + (gx + 0.5) * sx
                cell_cy = y_min + (gy + 0.5) * sy

                dx = (x - cell_cx) / sx
                dy = (y - cell_cy) / sy

                reg_target[b, reg_ch0 + 0, gy, gx] = dx
                reg_target[b, reg_ch0 + 1, gy, gx] = dy
                reg_target[b, reg_ch0 + 2, gy, gx] = z
                reg_target[b, reg_ch0 + 3, gy, gx] = math.log(max(l, 1e-3))
                reg_target[b, reg_ch0 + 4, gy, gx] = math.log(max(w, 1e-3))
                reg_target[b, reg_ch0 + 5, gy, gx] = math.log(max(h, 1e-3))
                reg_target[b, reg_ch0 + 6, gy, gx] = math.sin(rot)

                dir_bin = 0 if math.cos(rot) >= 0 else 1
                dir_target[b, dir_ch0 + dir_bin, gy, gx] = 1.0

                reg_mask[b, gy, gx] = True
                dir_mask[b, gy, gx] = True

                # ignore okolie pre klasifikáciu
                r = self.ignore_radius
                y0 = max(0, gy - r)
                y1 = min(H, gy + r + 1)
                x0 = max(0, gx - r)
                x1 = min(W, gx + r + 1)

                cls_valid_mask[b, cls_ch, y0:y1, x0:x1] = False

                # ale center bunka musí zostať validná a positive
                cls_valid_mask[b, cls_ch, gy, gx] = True

        return cls_target, reg_target, dir_target, cls_valid_mask, reg_mask, dir_mask

    def forward(self, preds, gt_boxes, batch_size):
        cls_pred = preds['cls']
        reg_pred = preds['reg']
        dir_pred = preds['dir']

        cls_target, reg_target, dir_target, cls_valid_mask, reg_mask, dir_mask = self._build_targets(
            preds, gt_boxes
        )

        cls_loss = self.focal_loss_masked(cls_pred, cls_target, cls_valid_mask)

        reg_ch0 = self.anchor_id * self.reg_box_dim
        reg_pred_car = reg_pred[:, reg_ch0:reg_ch0 + 7, :, :]
        reg_target_car = reg_target[:, reg_ch0:reg_ch0 + 7, :, :]

        if reg_mask.any():
            reg_mask_exp = reg_mask.unsqueeze(1).expand(-1, 7, -1, -1)
            reg_loss = F.smooth_l1_loss(
                reg_pred_car[reg_mask_exp],
                reg_target_car[reg_mask_exp],
                reduction='mean'
            )
        else:
            reg_loss = reg_pred.sum() * 0.0

        dir_ch0 = self.anchor_id * 2
        dir_pred_car = dir_pred[:, dir_ch0:dir_ch0 + 2, :, :]
        dir_target_car = dir_target[:, dir_ch0:dir_ch0 + 2, :, :]

        if dir_mask.any():
            dir_mask_exp = dir_mask.unsqueeze(1).expand(-1, 2, -1, -1)
            dir_loss = F.binary_cross_entropy_with_logits(
                dir_pred_car[dir_mask_exp],
                dir_target_car[dir_mask_exp],
                reduction='mean'
            )
        else:
            dir_loss = dir_pred.sum() * 0.0

        total = (
            self.w_cls * cls_loss +
            self.w_reg * reg_loss +
            self.w_dir * dir_loss
        )

        return {
            'total': total,
            'cls': cls_loss,
            'reg': reg_loss,
            'dir': dir_loss,
        }
