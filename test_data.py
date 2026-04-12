import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils.preprocess import DAIRDataset

def main():
    dataset = DAIRDataset(split='val')
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="M4 LiDAR Monitor", width=1280, height=720)
    
    # --- NASTAVENIE PRE TOUCHPAD A RETINA DISPLEJ ---
    render_opt = vis.get_render_option()
    render_opt.point_size = 3.5  # Zväčšenie bodov
    render_opt.background_color = np.array([0.1, 0.1, 0.1]) # Tmavošedá
    
    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)
    
    # Malý kríž súradníc v strede
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(axes)

    print("\n💡 OVLÁDANIE PRE TOUCHPAD:")
    print("  [R] - Reset pohľadu (namieri kameru na body)")
    print("  [+/-] - Zväčšiť/Zmenšiť body")
    print("  [Q] - Ukončiť\n")

    for i in range(len(dataset)):
        data = dataset[i]
        
        # Extrakcia nenulových bodov
        all_xyz = data['pillars'][:, :, :3].reshape(-1, 3)
        mask = torch.any(all_xyz != 0, dim=1)
        points_np = all_xyz[mask].cpu().numpy()

        if len(points_np) > 0:
            # Farbenie podľa vzdialenosti
            dist = np.linalg.norm(points_np, axis=1)
            norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
            colors = plt.get_cmap('viridis')(norm)[:, :3]

            pcd_o3d.points = o3d.utility.Vector3dVector(points_np)
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
            
            vis.update_geometry(pcd_o3d)
            
            # Pri prvej snímke automaticky vycentruj kameru
            if i == 0:
                vis.reset_view_point(True)
                
            print(f"Snímka {i}: Vykresľujem {len(points_np)} bodov.")

        # Spracovanie okna
        keep_running = vis.poll_events()
        vis.update_renderer()
        
        if not keep_running:
            break
            
        import time
        time.sleep(0.05)

    vis.destroy_window()

if __name__ == "__main__":
    main()
