import os
import json
import struct
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

# ── Konfigurácia ciest ────────────────────────────────────────────
DAIR_ROOT  = '/Users/lkocan/Documents/intersection-safety-1/data'
PCD_DIR    = f'{DAIR_ROOT}/pcd'
LABEL_DIR  = f'{DAIR_ROOT}/label/virtuallidar'
SPLIT_FILE = f'{DAIR_ROOT}/split_data.json'

# ── Dátová štruktúra ──────────────────────────────────────────────
@dataclass
class Box3D:
    obj_type: str
    x:        float
    y:        float
    z:        float
    length:   float
    width:    float
    height:   float
    rotation: float

    CLASS_MAP = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    @property
    def class_id(self) -> int:
        return self.CLASS_MAP.get(self.obj_type, -1)

# ── LZF dekompresor ───────────────────────────────────────────────
def lzf_decompress(data: bytes, output_size: int) -> bytes:
    output  = bytearray(output_size)
    in_pos  = 0
    out_pos = 0
    while in_pos < len(data):
        ctrl = data[in_pos]
        in_pos += 1
        if ctrl < 32:
            length = ctrl + 1
            output[out_pos: out_pos + length] = data[in_pos: in_pos + length]
            in_pos  += length
            out_pos += length
        else:
            length = (ctrl >> 5)
            if length == 7:
                length += data[in_pos]
                in_pos += 1
            length += 2
            ref_offset = ((ctrl & 0x1F) << 8) + data[in_pos] + 1
            in_pos += 1
            ref_pos = out_pos - ref_offset
            for i in range(length):
                output[out_pos] = output[ref_pos + i]
                out_pos += 1
    return bytes(output)

# ── Načítanie PCD ─────────────────────────────────────────────────
def load_pcd(pcd_path: str) -> np.ndarray:
    with open(pcd_path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if not line: continue
            if line.startswith('DATA'):
                data_type = line.split()[1].strip()
                break
            parts = line.split()
            if len(parts) >= 2:
                header[parts[0]] = parts[1:]

        num_points = int(header.get('POINTS', [0])[0])
        fields     = header.get('FIELDS', ['x', 'y', 'z'])
        sizes      = [int(s) for s in header.get('SIZE',  ['4'] * len(fields))]
        types      = header.get('TYPE',  ['F'] * len(fields))
        counts     = [int(c) for c in header.get('COUNT', ['1'] * len(fields))]

        col_offset = {}
        offset = 0
        for name, count in zip(fields, counts):
            col_offset[name] = offset
            offset += count
        total_cols = offset

        fmt_char = {
            ('F', 4): 'f', ('F', 8): 'd',
            ('I', 4): 'i', ('I', 2): 'h', ('I', 1): 'b',
            ('U', 4): 'I', ('U', 2): 'H', ('U', 1): 'B',
        }

        if data_type == 'ascii':
            rows = []
            for _ in range(num_points):
                row = f.readline().decode('utf-8', errors='ignore').strip().split()
                if row: rows.append([float(v) for v in row])
            data = np.array(rows, dtype=np.float32) if rows else np.zeros((0, total_cols), dtype=np.float32)

        elif data_type == 'binary_compressed':
            compressed_size   = struct.unpack('<I', f.read(4))[0]
            uncompressed_size = struct.unpack('<I', f.read(4))[0]
            compressed_data   = f.read(compressed_size)
            raw = lzf_decompress(compressed_data, uncompressed_size)
            columns  = []
            byte_pos = 0
            for t, s, c, name in zip(types, sizes, counts, fields):
                ch        = fmt_char.get((t, s), 'f')
                col_bytes = s * c * num_points
                col_raw   = raw[byte_pos: byte_pos + col_bytes]
                byte_pos += col_bytes
                col_data = np.frombuffer(col_raw, dtype=np.dtype('<' + ch))
                col_data = col_data.reshape(num_points, c).astype(np.float32)
                columns.append(col_data)
            data = np.concatenate(columns, axis=1)
        else:
            row_fmt  = '<' + ''.join(fmt_char.get((t, s), 'f') * c for t, s, c in zip(types, sizes, counts))
            row_size = struct.calcsize(row_fmt)
            raw      = f.read(row_size * num_points)
            usable   = (len(raw) // row_size) * row_size
            if usable == 0: return np.zeros((0, 4), dtype=np.float32)
            data = np.array(list(struct.iter_unpack(row_fmt, raw[:usable])), dtype=np.float32)

    if data.size == 0: return np.zeros((0, 4), dtype=np.float32)

    x = data[:, col_offset.get('x', 0)]
    y = data[:, col_offset.get('y', 1)]
    z = data[:, col_offset.get('z', 2)]
    if 'intensity' in col_offset:
        intensity = data[:, col_offset['intensity']]
        if intensity.max() > 1.0: intensity = intensity / 255.0
    else:
        intensity = np.zeros(len(x), dtype=np.float32)
    return np.stack([x, y, z, intensity], axis=1)

# ── Načítanie labelov ─────────────────────────────────────────────
def load_labels(label_path: str) -> List[Box3D]:
    with open(label_path, encoding='utf-8') as f:
        data = json.load(f)
    boxes = []
    for obj in data:
        obj_type = obj.get('type', '')
        if obj_type not in Box3D.CLASS_MAP: continue
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

# ── Filtrovanie ROI ───────────────────────────────────────────────
def filter_pointcloud(
    points:  np.ndarray,
    x_range: Tuple = (0, 200),
    y_range: Tuple = (-50, 50),
    z_range: Tuple = (-3,  3),
) -> np.ndarray:
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]

# ── GT SAMPLER (AUGMENTÁCIA) ──────────────────────────────────────
class GTSampler:
    def __init__(self, db_path='gt_database'):
        self.db_path = db_path
        info_path = os.path.join(db_path, 'gt_database_info.pkl')
        if not os.path.exists(info_path):
            print(f"⚠️ GTSampler: {info_path} nenájdený!")
            self.db_info = {}
        else:
            with open(info_path, 'rb') as f:
                self.db_info = pickle.load(f)
        self.class_map = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    def sample(self, existing_boxes, max_ped=8, max_cyc=8):
        sampled_pts = []
        sampled_boxes = []
        all_boxes = existing_boxes.copy() if len(existing_boxes) > 0 else np.empty((0, 8))
        
        for cls_name, max_count in [('Pedestrian', max_ped), ('Cyclist', max_cyc)]:
            available = self.db_info.get(cls_name, [])
            if not available: continue
            
            num_to_sample = min(len(available), max_count)
            choices = np.random.choice(len(available), num_to_sample, replace=False)
            
            for idx in choices:
                info = available[idx]
                pts = np.load(os.path.join(self.db_path, info['filepath']))
                l, w, h = info['box_dims']
                
                x = np.random.uniform(10.0, 70.0) # ROI pre vkladanie
                y = np.random.uniform(-35.0, 35.0)
                z = -1.5 
                yaw = np.random.uniform(0, 2 * np.pi)
                
                if len(all_boxes) > 0:
                    centers = all_boxes[:, :2]
                    dists = np.sqrt(np.sum((centers - [x, y])**2, axis=1))
                    if np.any(dists < 3.5): continue
                
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                R = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
                
                new_pts = pts.copy()
                new_pts[:, :3] = np.dot(pts[:, :3], R.T)
                new_pts[:, 0] += x
                new_pts[:, 1] += y
                new_pts[:, 2] += z
                
                new_box = np.array([x, y, z, l, w, h, yaw, self.class_map[cls_name]])
                sampled_pts.append(new_pts)
                sampled_boxes.append(new_box)
                all_boxes = np.vstack([all_boxes, new_box])
        return sampled_pts, sampled_boxes

# ── Tvorba pilárov ────────────────────────────────────────────────
def create_pillars(
    points:                np.ndarray,
    voxel_size:            Tuple = (0.2, 0.2),
    x_range:               Tuple = (0, 200),
    y_range:               Tuple = (-50, 50),
    max_points_per_pillar: int   = 32,
    max_pillars:           int   = 12000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx, dy = voxel_size
    x_min, x_max = x_range
    y_min, y_max = y_range
    nx = int((x_max - x_min) / dx)
    ny = int((y_max - y_min) / dy)

    pillar_out     = np.zeros((max_pillars, max_points_per_pillar, 9), dtype=np.float32)
    coords_out     = np.zeros((max_pillars, 2), dtype=np.int32)
    num_points_out = np.zeros((max_pillars,), dtype=np.int32)

    if points.shape[0] == 0: return pillar_out, coords_out, num_points_out

    ix = ((points[:, 0] - x_min) / dx).astype(np.int32)
    iy = ((points[:, 1] - y_min) / dy).astype(np.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    points = points[valid]
    ix, iy = ix[valid], iy[valid]

    if points.shape[0] == 0: return pillar_out, coords_out, num_points_out

    pillar_key = iy * nx + ix
    unique_keys, inverse = np.unique(pillar_key, return_inverse=True)

    if len(unique_keys) > max_pillars:
        counts = np.bincount(inverse, minlength=len(unique_keys))
        top_idx = np.argsort(-counts)[:max_pillars]
        keep_mask_keys = np.zeros(len(unique_keys), dtype=bool)
        keep_mask_keys[top_idx] = True
        keep_mask_points = keep_mask_keys[inverse]
        points = points[keep_mask_points]
        inverse = inverse[keep_mask_points]
        old_to_new = -np.ones(len(unique_keys), dtype=np.int32)
        old_to_new[top_idx] = np.arange(len(top_idx), dtype=np.int32)
        inverse = old_to_new[inverse]
        unique_keys = unique_keys[top_idx]

    P = len(unique_keys)
    px, py = (unique_keys % nx).astype(np.int32), (unique_keys // nx).astype(np.int32)
    coords_out[:P, 0], coords_out[:P, 1] = px, py
    cx, cy = x_min + (px + 0.5) * dx, y_min + (py + 0.5) * dy
    global_mean_x, global_mean_y = points[:, 0].mean(), points[:, 1].mean()

    sort_idx = np.argsort(inverse, kind='stable')
    points, inverse = points[sort_idx], inverse[sort_idx]
    counts = np.bincount(inverse, minlength=P)
    offsets = np.concatenate([[0], np.cumsum(counts)])

    for pid in range(P):
        start, end = offsets[pid], offsets[pid+1]
        pts = points[start:end][:max_points_per_pillar]
        n = len(pts)
        num_points_out[pid] = n
        cz = pts[:, 2].mean()
        pillar_out[pid, :n, 0:4] = pts[:, 0:4]
        pillar_out[pid, :n, 4] = pts[:, 0] - cx[pid]
        pillar_out[pid, :n, 5] = pts[:, 1] - cy[pid]
        pillar_out[pid, :n, 6] = pts[:, 2] - cz
        pillar_out[pid, :n, 7] = pts[:, 0] - global_mean_x
        pillar_out[pid, :n, 8] = pts[:, 1] - global_mean_y

    return pillar_out, coords_out, num_points_out

# ── Dataset ───────────────────────────────────────────────────────
class DAIRDataset(Dataset):
    def __init__(self, split: str = 'train'):
        assert split in ('train', 'val', 'test')
        self.split = split
        with open(SPLIT_FILE) as f:
            all_ids = json.load(f)[split]
        self.ids = [fid for fid in all_ids if os.path.exists(f'{PCD_DIR}/{fid}.pcd') and os.path.exists(f'{LABEL_DIR}/{fid}.json')]
        print(f'[DAIRDataset] {split}: {len(self.ids)} vzoriek')
        
        if self.split == 'train':
            self.gt_sampler = GTSampler(db_path='gt_database')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        points = load_pcd(f'{PCD_DIR}/{fid}.pcd')
        points = filter_pointcloud(points)
        boxes = load_labels(f'{LABEL_DIR}/{fid}.json')

        valid_boxes_list = [b for b in boxes if b.class_id >= 0]
        if valid_boxes_list:
            gt_boxes = np.array([[b.x, b.y, b.z, b.length, b.width, b.height, b.rotation, b.class_id] for b in valid_boxes_list], dtype=np.float32)
        else:
            gt_boxes = np.zeros((0, 8), dtype=np.float32)

        # 👇 AUGMENTÁCIA (iba pre train)
        if self.split == 'train' and hasattr(self, 'gt_sampler'):
            s_pts, s_boxes = self.gt_sampler.sample(gt_boxes)
            if len(s_pts) > 0:
                points = np.vstack([points] + s_pts)
                gt_boxes = np.vstack([gt_boxes, s_boxes])

        pillars, coords, num_pts = create_pillars(points)

        return {
            'pillars':    torch.from_numpy(pillars),
            'coords':     torch.from_numpy(coords),
            'num_points': torch.from_numpy(num_pts),
            'gt_boxes':   torch.from_numpy(gt_boxes),
            'frame_id':   fid,
            'raw_points': points
        }

class DAIRDatasetCached(Dataset):
    def __init__(self, split: str = 'train', cache_dir: str = '/content/dair/cache'):
        with open(SPLIT_FILE) as f:
            all_ids = json.load(f)[split]
        self.cache_dir = cache_dir
        self.ids = [fid for fid in all_ids if os.path.exists(f'{cache_dir}/{fid}.npz')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid  = self.ids[idx]
        data = np.load(f'{self.cache_dir}/{fid}.npz')
        return {
            'pillars':    torch.from_numpy(data['pillars']),
            'coords':     torch.from_numpy(data['coords']),
            'num_points': torch.from_numpy(data['num_points']),
            'gt_boxes':   torch.from_numpy(data['gt_boxes']),
            'frame_id':   fid,
        }

if __name__ == '__main__':
    ds = DAIRDataset(split='train')
    sample = ds[0]
    print(f"Frame ID: {sample['frame_id']} | Pillars: {sample['pillars'].shape} | GT: {sample['gt_boxes'].shape}")
