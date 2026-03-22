import os
import json
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
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
    """Pure Python LZF dekompresor — nevyžaduje žiadne knižnice."""
    output  = bytearray(output_size)
    in_pos  = 0
    out_pos = 0

    while in_pos < len(data):
        ctrl = data[in_pos]
        in_pos += 1

        if ctrl < 32:
            # Literál — skopíruj ctrl+1 bytov priamo
            length = ctrl + 1
            output[out_pos: out_pos + length] = data[in_pos: in_pos + length]
            in_pos  += length
            out_pos += length

        else:
            # Spätná referencia
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
    """
    Číta .pcd súbory bez open3d.
    Podporuje ASCII, binary aj binary_compressed (LZF).
    Vráti (N, 4): x, y, z, intensity
    """
    with open(pcd_path, 'rb') as f:

        # ── Parsuj header ─────────────────────────────────────────
        header = {}
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
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

        # Mapovanie názvov stĺpcov na offsety (s ohľadom na COUNT)
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

        # ── ASCII ─────────────────────────────────────────────────
        if data_type == 'ascii':
            rows = []
            for _ in range(num_points):
                row = f.readline().decode('utf-8', errors='ignore').strip().split()
                if row:
                    rows.append([float(v) for v in row])
            data = np.array(rows, dtype=np.float32) if rows \
                   else np.zeros((0, total_cols), dtype=np.float32)

        # ── Binary compressed (LZF) ───────────────────────────────
        elif data_type == 'binary_compressed':
            compressed_size   = struct.unpack('<I', f.read(4))[0]
            uncompressed_size = struct.unpack('<I', f.read(4))[0]
            compressed_data   = f.read(compressed_size)

            raw = lzf_decompress(compressed_data, uncompressed_size)

            # binary_compressed ukladá dáta PO STĹPCOCH nie po riadkoch
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

        # ── Binary (nekomprimovaný) ───────────────────────────────
        else:
            row_fmt  = '<' + ''.join(
                fmt_char.get((t, s), 'f') * c
                for t, s, c in zip(types, sizes, counts)
            )
            row_size = struct.calcsize(row_fmt)
            raw      = f.read(row_size * num_points)
            usable   = (len(raw) // row_size) * row_size
            if usable == 0:
                return np.zeros((0, 4), dtype=np.float32)
            data = np.array(
                list(struct.iter_unpack(row_fmt, raw[:usable])),
                dtype=np.float32
            )

    if data.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # ── Vytiahni x, y, z, intensity ──────────────────────────────
    x = data[:, col_offset.get('x', 0)]
    y = data[:, col_offset.get('y', 1)]
    z = data[:, col_offset.get('z', 2)]

    if 'intensity' in col_offset:
        intensity = data[:, col_offset['intensity']]
        if intensity.max() > 1.0:
            intensity = intensity / 255.0
    else:
        intensity = np.zeros(len(x), dtype=np.float32)

    return np.stack([x, y, z, intensity], axis=1)   # (N, 4)


# ── Načítanie labelov ─────────────────────────────────────────────
def load_labels(label_path: str) -> List[Box3D]:
    """
    Číta label/virtuallidar/*.json z DAIR-V2X-I.
    Hodnoty sú stringy — konvertujeme na float.
    """
    with open(label_path, encoding='utf-8') as f:
        data = json.load(f)

    boxes = []
    for obj in data:
        obj_type = obj.get('type', '')
        if obj_type not in Box3D.CLASS_MAP:
            continue

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
    """Orezá point cloud na oblasť záujmu."""
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]


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

    ix = ((points[:, 0] - x_min) / dx).astype(np.int32)
    iy = ((points[:, 1] - y_min) / dy).astype(np.int32)

    valid  = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    points = points[valid]
    ix     = ix[valid]
    iy     = iy[valid]

    if len(points) == 0:
        empty = np.zeros((1, max_points_per_pillar, 9), dtype=np.float32)
        return empty, np.zeros((1, 2), dtype=np.int32), np.zeros(1, dtype=np.int32)

    pillar_key           = iy * nx + ix
    unique_keys, inverse = np.unique(pillar_key, return_inverse=True)

    if len(unique_keys) > max_pillars:
        # Nechaj piliere s najviac bodmi
        counts     = np.bincount(inverse, minlength=len(unique_keys))
        top_idx    = np.argsort(-counts)[:max_pillars]
        key_map    = np.full(len(unique_keys), -1, dtype=np.int32)
        key_map[top_idx] = np.arange(max_pillars, dtype=np.int32)
        new_inverse = key_map[inverse]
        mask        = new_inverse >= 0
        points      = points[mask]
        inverse     = new_inverse[mask]
        unique_keys = unique_keys[top_idx]

    P = len(unique_keys)

    # ── Vektorizovaná verzia bez Python for loop ──────────────────
    # Zoraď body podľa piliera
    sort_idx = np.argsort(inverse, kind='stable')
    points   = points[sort_idx]
    inverse  = inverse[sort_idx]

    # Počet bodov v každom pilieri
    num_points_out = np.bincount(inverse, minlength=P).astype(np.int32)
    num_points_out = np.minimum(num_points_out, max_points_per_pillar)

    # Centroidy pilárov (vektorizovane)
    cx = x_min + (unique_keys % nx + 0.5) * dx
    cy = y_min + (unique_keys // nx + 0.5) * dy

    # Globálne priemery
    global_mean_x = points[:, 0].mean()
    global_mean_y = points[:, 1].mean()

    # Naplň pilier_out vektorizovane
    pillar_out = np.zeros((P, max_points_per_pillar, 9), dtype=np.float32)

    # Kumulatívne offsety pre každý pilier
    offsets = np.concatenate([[0], np.cumsum(np.bincount(inverse, minlength=P))])

    for pid in range(P):
        start = offsets[pid]
        n     = num_points_out[pid]
        p     = points[start: start + n]

        cz = p[:, 2].mean()

        pillar_out[pid, :n, 0] = p[:, 0]
        pillar_out[pid, :n, 1] = p[:, 1]
        pillar_out[pid, :n, 2] = p[:, 2]
        pillar_out[pid, :n, 3] = p[:, 3]
        pillar_out[pid, :n, 4] = p[:, 0] - cx[pid]
        pillar_out[pid, :n, 5] = p[:, 1] - cy[pid]
        pillar_out[pid, :n, 6] = p[:, 2] - cz
        pillar_out[pid, :n, 7] = p[:, 0] - global_mean_x
        pillar_out[pid, :n, 8] = p[:, 1] - global_mean_y

    coords = np.stack(
        [unique_keys % nx, unique_keys // nx], axis=1
    ).astype(np.int32)

    return pillar_out, coords, num_points_out


# ── Dataset ───────────────────────────────────────────────────────
class DAIRDataset(Dataset):
    """
    PyTorch Dataset pre DAIR-V2X-I.

    Použitie:
        train_ds = DAIRDataset(split='train')
        val_ds   = DAIRDataset(split='val')
    """

    def __init__(self, split: str = 'train'):
        assert split in ('train', 'val', 'test'), \
            f"split musí byť 'train', 'val' alebo 'test', nie '{split}'"

        with open(SPLIT_FILE) as f:
            all_ids = json.load(f)[split]

        self.ids = [
            fid for fid in all_ids
            if os.path.exists(f'{PCD_DIR}/{fid}.pcd')
            and os.path.exists(f'{LABEL_DIR}/{fid}.json')
        ]
        print(f'[DAIRDataset] {split}: {len(self.ids)} platných vzoriek '
              f'(z {len(all_ids)} celkom)')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]

        points = load_pcd(f'{PCD_DIR}/{fid}.pcd')
        points = filter_pointcloud(points)

        boxes = load_labels(f'{LABEL_DIR}/{fid}.json')

        pillars, coords, num_pts = create_pillars(points)

        valid_boxes = [b for b in boxes if b.class_id >= 0]
        if valid_boxes:
            gt_boxes = np.array([
                [b.x, b.y, b.z, b.length, b.width,
                 b.height, b.rotation, b.class_id]
                for b in valid_boxes
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


# ── Rýchly test ───────────────────────────────────────────────────
if __name__ == '__main__':
    ds     = DAIRDataset(split='train')
    sample = ds[0]
    print(f"Frame ID:  {sample['frame_id']}")
    print(f"Pillars:   {sample['pillars'].shape}")
    print(f"GT boxes:  {sample['gt_boxes'].shape}")
