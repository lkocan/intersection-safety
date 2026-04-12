import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils.preprocess import DAIRDataset
# Ak používaš tracker, odkomentuj riadok nižšie:
# from tracking.tracker import Tracker3D

def main():
    # 1. Inicializácia datasetu (uisti sa, že cesty v configu sú správne)
    dataset = DAIRDataset(split='val')
    print(f"[DAIRDataset] Načítaných {len(dataset)} vzoriek")

    # 2. Nastavenie Open3D vizualizácie
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="M4 PointPillars Monitor", width=1280, height=720)
    
    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)
    
    # Pridanie súradnicového kríža pre lepšiu orientáciu (X-červená, Y-zelená, Z-modrá)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(axes)

    # Nastavenie pohľadu (voliteľné)
    view_ctl = vis.get_view_control()
    view_ctl.set_front([ -0.9, 0.1, 0.4 ])
    view_ctl.set_lookat([ 0, 0, 0 ])
    view_ctl.set_up([ 0, 0, 1 ])
    view_ctl.set_zoom(0.3)

    print("Spúšťam vizualizáciu... (stlač 'Q' v okne pre ukončenie)")

    for i in range(len(dataset)):
        data = dataset[i]
        
        # --- REKONŠTRUKCIA BODOV Z PILLAROV ---
        # Pillars tvar: [12000, 32, 9] -> berieme prvé 3 kanály (X, Y, Z)
        pillars = data['pillars']
        num_points = data['num_points']
        
        # Sploštenie na [N, 3]
        all_xyz = pillars[:, :, :3].reshape(-1, 3)
        
        # Vytvorenie masky pre reálne body (num_points hovorí, koľko bodov v pillari nie je nula)
        # Vytvoríme 2D masku [12000, 32] a sploštíme ju
        p_mask = torch.zeros(pillars.shape[0], pillars.shape[1], dtype=torch.bool)
        for p_idx in range(pillars.shape[0]):
            p_mask[p_idx, :num_points[p_idx]] = True
        p_mask = p_mask.reshape(-1)

        # Finálne body pre Open3D
        points_np = all_xyz[p_mask].cpu().numpy()

        if len(points_np) > 0:
            # --- FARBENIE PODĽA PILLAROV ---
            # Každý pillar dostane farbu z mapy 'tab20' (20 kontrastných farieb)
            p_indices = torch.arange(pillars.shape[0]).repeat_interleave(pillars.shape[1])
            active_p_indices = p_indices[p_mask].cpu().numpy()
            
            cmap = plt.get_cmap('tab20')
            normalized_indices = (active_p_indices % 20) / 20.0
            colors_np = cmap(normalized_indices)[:, :3]

            # Aktualizácia dát v Open3D
            pcd_o3d.points = o3d.utility.Vector3dVector(points_np)
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors_np)
            
            # Informovanie vizualizátora o zmene geometrie
            vis.update_geometry(pcd_o3d)
        
        # Vykreslenie snímky
        vis.poll_events()
        vis.update_renderer()
        
        # Malá pauza, aby si stíhal vnímať pohyb (cca 20 FPS)
        import time
        time.sleep(0.05)

    vis.destroy_window()

if __name__ == "__main__":
    main()
