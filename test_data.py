import os
import sys
import torch
import numpy as np
import open3d as o3d
import json

# 1. NASTAVENIE PROSTREDIA
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from utils.preprocess import DAIRDataset, DAIR_ROOT
from models.pointpillars import PointPillars, PointPillarsConfig
from tracking.tracker import Tracker3D
from utils.roi_utils import calculate_risk_score

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def get_box_lineset(t, color=(0, 1, 0)):
    """Vytvorí 3D drôtený model (LineSet) pre bounding box z tracku."""
    # Parametre z trackera
    pos = t.kf.x[:3]  # x, y, z
    dim = [t.l, t.w, t.h] # dĺžka, šírka, výška
    yaw = t.yaw

    # Vytvorenie orientovaného boxu v Open3D
    center = np.array([pos[0], pos[1], pos[2]])
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    box = o3d.geometry.OrientedBoundingBox(center, R, dim)
    
    # Prevod na LineSet (čiary), aby sme mohli meniť farbu a hrúbku
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    lineset.paint_uniform_color(color)
    return lineset

def main():
    device = get_device()
    cfg = PointPillarsConfig()
    dataset = DAIRDataset(split='val')
    tracker = Tracker3D()

    # 2. INICIALIZÁCIA OPEN3D VIZUALIZÉRA
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="M4 3D Safety Monitor", width=1280, height=720)
    
    # Príprava mračna bodov
    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)
    
    # Kontajner pre boxy (aby sme ich vedeli mazať každý snímok)
    current_boxes = []

    print("--- Spúšťam interaktívne 3D (Ovládanie: Myš + Shift) ---")

    for i in range(len(dataset)):
        data = dataset[i]
        
        # 3. AKTUALIZÁCIA BODov (Point Cloud)
        # Predpokladáme, že dataset vracia body v 'points' (N, 3 alebo N, 4)
        points = data['points'][:, :3].numpy()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        
        # Farbenie bodov podľa výšky (ako v Matlabe)
        z_values = points[:, 2]
        colors = np.zeros((len(z_values), 3))
        colors[:, 2] = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-6)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
        
        vis.update_geometry(pcd_o3d)

        # 4. TRACKING
        gt = data['gt_boxes'].numpy()
        # Padding na 9 stĺpcov pre tracker
        detections = np.hstack([gt, np.ones((gt.shape[0], 1))]) if gt.shape[0] > 0 else gt
        confirmed = tracker.update(detections)

        # 5. AKTUALIZÁCIA 3D BOXOV
        for b in current_boxes:
            vis.remove_geometry(b, reset_bounding_box=False)
        current_boxes.clear()

        for t in confirmed:
            risk = calculate_risk_score(t, confirmed)
            t.update_risk(risk)
            
            # Farba od zelenej (0,1,0) po červenú (1,0,0)
            color = (t.smoothed_risk, 1 - t.smoothed_risk, 0)
            
            box_lineset = get_box_lineset(t, color=color)
            vis.add_geometry(box_lineset, reset_bounding_box=False)
            current_boxes.append(box_lineset)

        # 6. RENDER
        if not vis.poll_events(): break
        vis.update_renderer()

    vis.destroy_window()

if __name__ == "__main__":
    main()
