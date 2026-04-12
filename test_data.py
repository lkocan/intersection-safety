import os
import sys
import torch
import numpy as np
import open3d as o3d
import json

# 1. NASTAVENIE PROSTREDIA
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from utils.preprocess import DAIRDataset
from tracking.tracker import Tracker3D
from utils.roi_utils import calculate_risk_score

def load_roi(path):
    """Načíta ROI súradnice a vytvorí 3D LineSet pre vizualizáciu."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        roi_data = json.load(f).get('roi', [])
    
    if not roi_data:
        return None

    # Vytvorenie bodov pre LineSet (Z = -1.5m pre úroveň cesty)
    points = [[p[0], p[1], -1.5] for p in roi_data]
    points.append(points[0]) # Uzavretie polygónu
    
    lines = [[i, i+1] for i in range(len(points)-1)]
    
    roi_lineset = o3d.geometry.LineSet()
    roi_lineset.points = o3d.utility.Vector3dVector(points)
    roi_lineset.lines = o3d.utility.Vector2iVector(lines)
    roi_lineset.paint_uniform_color([0.2, 0.2, 1.0]) # Modrá farba pre ROI
    
    return roi_lineset

def get_box_lineset(t, color=(0, 1, 0)):
    """Vytvorí 3D box pre detekciu."""
    pos = t.kf.x[:3]
    dim = [t.l, t.w, t.h]
    yaw = t.yaw
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    box = o3d.geometry.OrientedBoundingBox(pos, R, dim)
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    ls.paint_uniform_color(color)
    return ls

def main():
    dataset = DAIRDataset(split='val')
    
    # Načítanie ROI konfigurácie
    roi_path = os.path.join(PROJECT_ROOT, 'roi_config.json')
    roi_visual = load_roi(roi_path)
    
    tracker = Tracker3D(roi_coords=None) # ROI riešime vizuálne

    # 2. OPEN3D SETUP
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="M4 3D Safety Monitor - ROI Mode", width=1280, height=720)
    
    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)
    
    # Pridanie statickej ROI do scény
    if roi_visual:
        vis.add_geometry(roi_visual)
    
    current_boxes = []

    for i in range(len(dataset)):
        data = dataset[i]
        
        # 3. POINT CLOUD
        points = data['points'][:, :3].numpy()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        
        # Dynamické farbenie podľa intenzity alebo výšky
        z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-6)
        colors = np.zeros_like(points)
        colors[:, 1] = z_norm  # Zelený kanál podľa výšky
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
        
        vis.update_geometry(pcd_o3d)

        # 4. TRACKING
        gt = data['gt_boxes'].numpy()
        detections = np.hstack([gt, np.ones((gt.shape[0], 1))]) if gt.shape[0] > 0 else gt
        confirmed = tracker.update(detections)

        # 5. BOXES
        for b in current_boxes: vis.remove_geometry(b, reset_bounding_box=False)
        current_boxes.clear()

        for t in confirmed:
            risk = calculate_risk_score(t, confirmed)
            t.update_risk(risk)
            color = (t.smoothed_risk, 1 - t.smoothed_risk, 0)
            
            box_ls = get_box_lineset(t, color=color)
            vis.add_geometry(box_ls, reset_bounding_box=False)
            current_boxes.append(box_ls)

        if not vis.poll_events(): break
        vis.update_renderer()

    vis.destroy_window()

if __name__ == "__main__":
    main()
