import torch
import torch.nn as nn
import torch.nn.functional as F

class PointPillarsConfig:
    x_range = (-50, 50)
    y_range = (-50, 50)
    z_range = (-3, 3)

    voxel_size = (0.2, 0.2, 6.0)   
    max_points_per_pillar = 32    
    max_pillars = 12000             

    num_classes = 3                 
    class_names = ['car', 'pedestrian', 'cyclist']

    # Anchor boxy pre každú triedu [dĺžka, šírka, výška, uhol]
    anchors = {
        'car':         [3.9, 1.6, 1.56, 0],
        'pedestrian':  [0.8, 0.6, 1.73, 0],
        'cyclist':     [1.76, 0.6, 1.73, 0],
    }


class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, pillars, num_points_per_pillar):
        """
        pillars: (B, P, N, C)  - batch, piliere, body, features
        vráti:   (B, C_out, P) - features pre každý pilier
        """
        x = self.linear(pillars)                          # (B, P, N, 64)
        x = x.permute(0, 3, 1, 2)                        # (B, 64, P, N)
        B, C, P, N = x.shape
        x = self.bn(x.reshape(B * C, P * N)).reshape(B, C, P, N)
        x = F.relu(x)

        x = x.max(dim=3).values                          # (B, 64, P)
        return x

class PointPillarScatter(nn.Module):
    def __init__(self, cfg: PointPillarsConfig):
        super().__init__()
        self.cfg = cfg
        self.nx = int((cfg.x_range[1] - cfg.x_range[0]) / cfg.voxel_size[0])
        self.ny = int((cfg.y_range[1] - cfg.y_range[0]) / cfg.voxel_size[1])

    def forward(self, pillar_features, coords, batch_size):
        """
        pillar_features: (B, C, P)
        coords:          (B, P, 2)  - [ix, iy] index piliera
        """
        C = pillar_features.shape[1]
        canvas = torch.zeros(
            batch_size, C, self.ny, self.nx,
            device=pillar_features.device
        )
        for b in range(batch_size):
            ix = coords[b, :, 0].long()
            iy = coords[b, :, 1].long()
            mask = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            canvas[b, :, iy[mask], ix[mask]] = pillar_features[b, :, mask]
        return canvas                                     # (B, C, H, W)

class Backbone2D(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.block1 = self._make_block(in_channels, 64,  num_layers=4, stride=2)
        self.block2 = self._make_block(64,          128, num_layers=6, stride=2)
        self.block3 = self._make_block(128,         256, num_layers=6, stride=2)

        self.up1 = nn.ConvTranspose2d(64,  128, kernel_size=1, stride=1)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4)

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
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        
        out = torch.cat([self.up1(x1), self.up2(x2), self.up3(x3)], dim=1)
        return out                                        # (B, 384, H, W)

class DetectionHead(nn.Module):
    def __init__(self, in_channels=384, num_anchors=6, num_classes=3):
        super().__init__()
        
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 7, 1)
        # 7 = [dx, dy, dz, dw, dl, dh, dθ]
        self.dir_head = nn.Conv2d(in_channels, num_anchors * 2, 1)

    def forward(self, x):
        cls = self.cls_head(x)   
        reg = self.reg_head(x)  
        dir = self.dir_head(x)   
        return cls, reg, dir

class PointPillars(nn.Module):
    def __init__(self, cfg: PointPillarsConfig = None):
        super().__init__()
        self.cfg = cfg or PointPillarsConfig()

        self.pfn    = PillarFeatureNet(in_channels=9, out_channels=64)
        self.scatter = PointPillarScatter(self.cfg)
        self.backbone = Backbone2D(in_channels=64)
        self.head   = DetectionHead(
            in_channels=384,
            num_anchors=self.cfg.num_classes * 2,
            num_classes=self.cfg.num_classes
        )

    def forward(self, pillars, coords, num_points, batch_size):
       
        features = self.pfn(pillars, num_points)          # (B, 64, P)
        
        bev_map  = self.scatter(features, coords, batch_size)  # (B, 64, H, W)
      
        backbone_out = self.backbone(bev_map)             # (B, 384, H, W)
    
        cls, reg, dir = self.head(backbone_out)
        return {'cls': cls, 'reg': reg, 'dir': dir}


if __name__ == '__main__':
    cfg = PointPillarsConfig()
    model = PointPillars(cfg)
    print(f"Parametrov: {sum(p.numel() for p in model.parameters()):,}")

    B, P, N, C = 1, 8000, 32, 9
    pillars    = torch.randn(B, P, N, C)
    coords     = torch.randint(0, 500, (B, P, 2))
    num_points = torch.randint(1, N, (B, P))

    out = model(pillars, coords, num_points, batch_size=B)
    print("cls:", out['cls'].shape)
    print("reg:", out['reg'].shape)
