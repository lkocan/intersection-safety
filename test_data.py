import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils.preprocess import DAIRDataset

def main():
    dataset = DAIRDataset(split='val')
    print(f"[DAIRDataset] Načítaných {len(dataset)} vzoriek")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="M4 PointPillars Monitor", width=1280, height=720)
    # Zväčšenie bodov (na Macu odporúčam 2.0 až 5.0)
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    render_option.background_color = np.array([0.05, 0.05, 0.05]) # Tmavošedé pozadie
    
    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)
    
    # Menší súradnicový kríž, aby nezavadzal
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(axes)

    print("Spúšťam vizualizáciu... (v okne stlač 'Q' pre koniec, 'R' pre reset pohľadu)")

    for i in range(len(dataset)):
        data = dataset[i]
        
        # --- ROBUSTNÁ EXTRAKCIA BODOV ---
        # Pillars tvar: [12000, 32, 9]
        pillars = data['pillars']
        
        # Sploštíme všetkých 12000 * 32 bodov do jedného zoznamu [N, 3]
        all_xyz = pillars[:, :, :3].reshape(-1, 3)
        
        # Namiesto 'num_points' použijeme masku: bod je platný, ak má aspoň jednu súradnicu nenulovú
        # (Väčšina padding bodov sú čisté nuly [0, 0, 0])
        mask = torch.any(all_xyz != 0, dim=1)
        points_np = all_xyz[mask].cpu().numpy()

        if len(points_np) > 0:
            # Dynamické farbenie (podľa vzdialenosti od stredu pre lepší efekt)
            dist = np.linalg.norm(points_np, axis=1)
            norm_dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
            
            # Použijeme farby z 'plasma' mapy
            cmap = plt.get_cmap('plasma')
            colors_np = cmap(norm_dist)[:, :3]

            pcd_o3d.points = o3d.utility.Vector3dVector(points_np)
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors_np)
            
            vis.update_geometry(pcd_o3d)
            
            # Pri prvom nájdenom snímku nastavíme kameru na body
            if i == 0 or 'camera_set' not in locals():
                vis.poll_events()
                vis.update_renderer()
                vis.get_view_control().set_zoom(0.5)
                camera_set = True
                
            print(f"Snímka {i}: Vykresľujem {len(points_np)} bodov.")
        else:
            print(f"Snímka {i}: PRÁZDNA (0 bodov po filtrácii)")

        vis.poll_events()
        vis.update_renderer()
        
        import time
        time.sleep(0.02) # Rýchlejšie prehrávanie (50 FPS)

    vis.destroy_window()

if __name__ == "__main__":
    main()
