import sys
sys.path.append('.')

import numpy as np
from utils.preprocess import (
    load_pcd, load_labels, filter_pointcloud,
    create_pillars, DAIRDataset,
    PCD_DIR, LABEL_DIR,
)

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']

# ── Test 1: PCD reader ────────────────────────────────────────────
print("=" * 50)
print("TEST 1: Načítanie PCD súboru")
print("=" * 50)

test_pcd = f'{PCD_DIR}/008065.pcd'
points   = load_pcd(test_pcd)
print(f"Shape:       {points.shape}")
print(f"Prvé 3 body:\n{points[:3]}")
print(f"X rozsah:    {points[:,0].min():.1f} až {points[:,0].max():.1f} m")
print(f"Y rozsah:    {points[:,1].min():.1f} až {points[:,1].max():.1f} m")
print(f"Z rozsah:    {points[:,2].min():.1f} až {points[:,2].max():.1f} m")
print(f"Intensity:   {points[:,3].min():.3f} až {points[:,3].max():.3f}")

# ── Test 2: Filtrovanie ───────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 2: Filtrovanie ROI")
print("=" * 50)

filtered = filter_pointcloud(points)
print(f"Pred filtrom: {len(points):,} bodov")
print(f"Po filtri:    {len(filtered):,} bodov")

# ── Test 3: Labely ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 3: Načítanie labelov")
print("=" * 50)

label_path = f'{LABEL_DIR}/008065.json'
boxes      = load_labels(label_path)
print(f"Počet objektov: {len(boxes)}")
for b in boxes:
    print(f"  {b.obj_type:12s}  "
          f"x={b.x:6.1f}  y={b.y:6.1f}  z={b.z:5.1f}  "
          f"l={b.length:.2f}  w={b.width:.2f}  h={b.height:.2f}  "
          f"class_id={b.class_id}")

# ── Test 4: Piliere ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 4: Tvorba pilárov")
print("=" * 50)

pillars, coords, num_pts = create_pillars(filtered)
print(f"Pillars shape:    {pillars.shape}")
print(f"Coords shape:     {coords.shape}")
print(f"Num points shape: {num_pts.shape}")
print(f"Priemerný počet bodov v pilieri: {num_pts.mean():.1f}")

# ── Test 5: Dataset ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 5: DAIRDataset")
print("=" * 50)

ds     = DAIRDataset(split='train')
sample = ds[0]
print(f"\nFrame ID:   {sample['frame_id']}")
print(f"Pillars:    {sample['pillars'].shape}")
print(f"Coords:     {sample['coords'].shape}")
print(f"GT boxes:   {sample['gt_boxes'].shape}")

if len(sample['gt_boxes']) > 0:
    classes = sample['gt_boxes'][:, 7].int().tolist()
    print(f"Objekty:    {[CLASS_NAMES[c] for c in classes]}")
else:
    print("Objekty:    žiadne anotované objekty v tomto frame")

print("\n✓ Všetky testy prebehli úspešne!")
