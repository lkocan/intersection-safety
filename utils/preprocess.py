import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset


# ── Konfigurácia ciest ────────────────────────────────────────────
DAIR_ROOT  = '/Users/lkocan/Documents/MATLAB/data/single-infrastructure-side'
PCD_DIR    = f'{DAIR_ROOT}/single-infrastructure-side-velodyne'
LABEL_DIR  = f'{DAIR_ROOT}/label/virtuallidar'
SPLIT_FILE = f'{DAIR_ROOT}/split_data.json'


# ── Dátová štruktúra ──────────────────────────────────────────────
@dataclass
class Box3D:
    obj_type:  str
    x:         float
    y:         float
    z:         float
    length:    float
    width:     float
    height:    float
    rotation:  float

    CLASS_MAP = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    @property
    def class_id(self) -> int:
        return self.CLASS_MAP.get(self.obj_type, -1)


# ── Načítanie PCD ─────────────────────────────────────────────────
def load_pcd(pcd_path: str) -> np.ndarray:
    """Vráti (N, 4): x, y, z, intensity"""
    pcd    = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float32)

    if pcd.has_colors():
        intensity = np.asarray(pcd.colors)[:, 0:1].astype(np.float32)
    else:
        intensity = np.zeros((len(points), 1), dtype=np.float32)

    return np.concatenate([points, intensity], axis=1)


# ── Načítanie labelov ─────────────────────────────────────────────
def load_labels(label_path: str) -> List[Box3D]:
    """
    Číta label/virtuallidar/*.json z DAIR-V2X-I.
    Všetky hodnoty sú stringy → konvertujeme na float.
    """
    with open(label_path, encoding='utf-8') as f:
        data = json.load(f)

    boxes = []
    for obj in data:
        obj_type = obj.get('type', '')
        if obj_type not in Box3D.CLASS_MAP:
            continue                        # preskočí 'Unknown', 'TrafficCone' atď.

        loc  = obj['3d_location']
        dims = obj['3d_dimensions']

        boxes.append(Box3D(
            obj_type = obj_type,
            x        = float(loc['x']),
            y        = float(loc['y']),
            z        = float(loc['z']),
            length   = float(dims['l']),
            width    = float(dims['w']),
            height   = float(dims['h']),
            rotation = float(obj.get('rotation', 0.0)),
        ))
    return boxes


# ── Filter ROI ────────────────────────────────────────────────────
def filter_pointcloud(
    points:  np.ndarray,
    x_range: Tuple = (-70, 70),   # DAIR má objekty až do ~55m
    y_range: Tuple = (-40, 40),
    z_range: Tuple = (-3,  3),
) -> np.ndarray:
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]


# ── Tvorba pilárov ────────────────────────────────────────────────
def create_pillars(
    points:                 np.ndarray,
    voxel_size:             Tuple = (0.2, 0.2),
    x_range:                Tuple = (-70, 70),
    y_range:                Tuple = (-40, 40),
    max_points_per_pillar:  int   = 32,
    max_pillars:            int   = 12000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vráti:
        pillars    (P, N, 9)  — features bodov
        coords     (P, 2)     — [ix, iy] index v mriežke
        num_points (P,)       — počet platných bodov v pilieri
    """
    dx, dy   = voxel_size
    x_min, _ = x_range
    y_min, _ = y_range

    ix = ((points[:, 0] - x_min) / dx).astype(np.int32)
    iy = ((points[:, 1] - y_min) / dy).astype(np.int32)

    nx = int((x_range[1] - x_range[0]) / dx)
    ny = int((y_range[1] - y_range[0]) / dy)

    valid   = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    points  = points[valid]
    ix, iy  = ix[valid], iy[valid]

    pillar_key           = iy * nx + ix
    unique_keys, inverse = np.unique(pillar_key, return_inverse=True)

    if len(unique_keys) > max_pillars:
        unique_keys = unique_keys[:max_pillars]
        mask        = inverse < max_pillars
        points      = points[mask]
        inverse     = inverse[mask]

    P              = len(unique_keys)
    pillar_out     = np.zeros((P, max_points_per_pillar, 9), dtype=np.float32)
    num_points_out = np.zeros(P, dtype=np.int32)

    for pid, key in enumerate(unique_keys):
        pt_mask = inverse == pid
        pts     = points[pt_mask]
        n       = min(len(pts), max_points_per_pillar)
        num_points_out[pid] = n

        cx = x_min + (key % nx + 0.5) * dx
        cy = y_min + (key // nx + 0.5) * dy
        cz = pts[:n, 2].mean()
        p  = pts[:n]

        pillar_out[pid, :n, 0] = p[:, 0]                        # x
        pillar_out[pid, :n, 1] = p[:, 1]                        # y
        pillar_out[pid, :n, 2] = p[:, 2]                        # z
        pillar_out[pid, :n, 3] = p[:, 3]                        # intensity
        pillar_out[pid, :n, 4] = p[:, 0] - cx                   # Δx od stredu piliera
        pillar_out[pid, :n, 5] = p[:, 1] - cy                   # Δy od stredu piliera
        pillar_out[pid, :n, 6] = p[:, 2] - cz                   # Δz od stredu piliera
        pillar_out[pid, :n, 7] = p[:, 0] - points[:, 0].mean()  # globálna pozícia x
        pillar_out[pid, :n, 8] = p[:, 1] - points[:, 1].mean()  # globálna pozícia y

    coords = np.stack([unique_keys % nx, unique_keys // nx], axis=1).astype(np.int32)
    return pillar_out, coords, num_points_out


# ── Dataset ───────────────────────────────────────────────────────
class DAIRDataset(Dataset):
    """
    Použitie:
        train_ds = DAIRDataset(split='train')
        val_ds   = DAIRDataset(split='val')
    """

    def __init__(self, split: str = 'train'):
        assert split in ('train', 'val', 'test')

        with open(SPLIT_FILE) as f:
            self.ids = json.load(f)[split]          # ['000000', '000001', ...]

        # Filtruj — nechaj len IDs kde existuje PCD aj label
        self.ids = [
            fid for fid in self.ids
            if os.path.exists(f'{PCD_DIR}/{fid}.pcd')
            and os.path.exists(f'{LABEL_DIR}/{fid}.json')
        ]
        print(f'[DAIRDataset] {split}: {len(self.ids)} platných vzoriek')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]

        # Point cloud
        points = load_pcd(f'{PCD_DIR}/{fid}.pcd')
        points = filter_pointcloud(points)

        # Labely
        boxes = load_labels(f'{LABEL_DIR}/{fid}.json')

        # Piliere
        pillars, coords, num_pts = create_pillars(points)

        # GT boxy → (M, 8): x,y,z,l,w,h,rot,class_id
        if boxes:
            gt_boxes = np.array([
                [b.x, b.y, b.z, b.length, b.width, b.height, b.rotation, b.class_id]
                for b in boxes if b.class_id >= 0
            ], dtype=np.float32)
        else:
            gt_boxes = np.zeros((0, 8), dtype=np.float32)

        return {
            'pillars':    torch.from_numpy(pillars),    # (P, 32, 9)
            'coords':     torch.from_numpy(coords),     # (P, 2)
            'num_points': torch.from_numpy(num_pts),    # (P,)
            'gt_boxes':   torch.from_numpy(gt_boxes),   # (M, 8)
            'frame_id':   fid,
        }


# ── Rýchly test lokálne (bez GPU) ────────────────────────────────
if __name__ == '__main__':
    ds     = DAIRDataset(split='train')
    sample = ds[0]

    print(f"\nFrame ID:    {sample['frame_id']}")
    print(f"Pillars:     {sample['pillars'].shape}")     # (P, 32, 9)
    print(f"Coords:      {sample['coords'].shape}")      # (P, 2)
    print(f"GT boxes:    {sample['gt_boxes'].shape}")    # (M, 8)

    # Ukáž triedy v sample
    if len(sample['gt_boxes']) > 0:
        classes = sample['gt_boxes'][:, 7].int().tolist()
        names   = ['Car', 'Pedestrian', 'Cyclist']
        print(f"Objekty:     {[names[c] for c in classes]}")
