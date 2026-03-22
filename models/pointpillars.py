import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Konfigurácia ──────────────────────────────────────────────────
class PointPillarsConfig:
    x_range = (0, 200)
    y_range = (-50, 50)
    z_range = (-3, 3)

    voxel_size             = (0.2, 0.2, 6.0)
    max_points_per_pillar  = 32
    max_pillars            = 12000

    num_classes  = 3
    class_names  = ['Car', 'Pedestrian', 'Cyclist']

    # Anchor rozmery [l, w, h, uhol] pre každú triedu
    anchors = {
        'Car':         [4.3,  1.95, 1.6,  0],
        'Pedestrian':  [0.55, 0.55, 1.65, 0],
        'Cyclist':     [1.76, 0.6,  1.65, 0],
    }

    # Tréning
    learning_rate = 1e-3
    batch_size    = 4
    num_epochs    = 50


# ── PillarFeatureNet ──────────────────────────────────────────────
class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn     = nn.BatchNorm1d(out_channels)

    def forward(self, pillars, num_points):
        # pillars: (B, P, N, 9)
        B, P, N, _ = pillars.shape

        x = self.linear(pillars)             # (B, P, N, 64)

        # BatchNorm1d očakáva (*, 64) → reshapuj na (B*P*N, 64)
        x = x.reshape(B * P * N, -1)        # (B*P*N, 64)
        x = self.bn(x)
        x = F.relu(x)

        # Späť na (B, P, N, 64) → max pooling cez body v pilieri
        x = x.reshape(B, P, N, -1)          # (B, P, N, 64)
        x = x.max(dim=2).values             # (B, P, 64)
        x = x.permute(0, 2, 1)             # (B, 64, P)
        return x


# ── PointPillarScatter ────────────────────────────────────────────
class PointPillarScatter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nx = int((cfg.x_range[1] - cfg.x_range[0]) / cfg.voxel_size[0])
        self.ny = int((cfg.y_range[1] - cfg.y_range[0]) / cfg.voxel_size[1])

    def forward(self, pillar_features, coords, batch_size):
        # pillar_features: (B, 64, P)
        # coords:          (B, P, 2)
        B, C, P = pillar_features.shape
        canvas  = torch.zeros(
            B, C, self.ny, self.nx,
            dtype  = pillar_features.dtype,
            device = pillar_features.device
        )

        # Vektorizované scatter — žiadny Python loop
        ix   = coords[:, :, 0].long().clamp(0, self.nx - 1)  # (B, P)
        iy   = coords[:, :, 1].long().clamp(0, self.ny - 1)  # (B, P)
        flat = iy * self.nx + ix                               # (B, P)

        # Rozšír pre všetky kanály
        flat_exp = flat.unsqueeze(1).expand(B, C, P)          # (B, C, P)

        canvas_flat = canvas.view(B, C, -1)                    # (B, C, H*W)
        canvas_flat.scatter_(2, flat_exp, pillar_features)

        return canvas_flat.view(B, C, self.ny, self.nx)


# ── Backbone ──────────────────────────────────────────────────────
class Backbone2D(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.block1 = self._make_block(in_channels, 64,  4, stride=2)
        self.block2 = self._make_block(64,          128, 6, stride=2)
        self.block3 = self._make_block(128,         256, 6, stride=2)

        # 1x1 konvolúcie na zjednotenie počtu kanálov
        self.up1 = nn.Conv2d(64,  128, kernel_size=1)
        self.up2 = nn.Conv2d(128, 128, kernel_size=1)
        self.up3 = nn.Conv2d(256, 128, kernel_size=1)

    def _make_block(self, in_ch, out_ch, num_layers, stride):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.block1(x)    # /2
        x2 = self.block2(x1)   # /4
        x3 = self.block3(x2)   # /8

        # Upsample x2 a x3 na veľkosť x1 pomocou interpolate
        target_h, target_w = x1.shape[2], x1.shape[3]

        u1 = self.up1(x1)
        u2 = F.interpolate(self.up2(x2), size=(target_h, target_w),
                           mode='bilinear', align_corners=False)
        u3 = F.interpolate(self.up3(x3), size=(target_h, target_w),
                           mode='bilinear', align_corners=False)

        return torch.cat([u1, u2, u3], dim=1)   # (B, 384, H, W)


# ── Detection Head ────────────────────────────────────────────────
class DetectionHead(nn.Module):
    def __init__(self, in_channels=384, num_anchors=6, num_classes=3):
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 7, 1)
        self.dir_head = nn.Conv2d(in_channels, num_anchors * 2, 1)

    def forward(self, x):
        return self.cls_head(x), self.reg_head(x), self.dir_head(x)


# ── Hlavný model ──────────────────────────────────────────────────
class PointPillars(nn.Module):
    def __init__(self, cfg: PointPillarsConfig = None):
        super().__init__()
        self.cfg      = cfg or PointPillarsConfig()
        self.pfn      = PillarFeatureNet(in_channels=9, out_channels=64)
        self.scatter  = PointPillarScatter(self.cfg)
        self.backbone = Backbone2D(in_channels=64)
        self.head     = DetectionHead(
            in_channels=384,
            num_anchors=self.cfg.num_classes * 2,
            num_classes=self.cfg.num_classes,
        )

    def forward(self, pillars, coords, num_points, batch_size):
        features     = self.pfn(pillars, num_points)
        bev_map      = self.scatter(features, coords, batch_size)
        backbone_out = self.backbone(bev_map)
        cls, reg, dir = self.head(backbone_out)
        return {'cls': cls, 'reg': reg, 'dir': dir}
