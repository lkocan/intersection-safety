import os
import sys
import torch
import numpy as np
import cv2
import json

# 1. DEFINÍCIA CIEST (Uprav PROJECT_ROOT, ak máš projekt inde)
# Tento skript automaticky hľadá dáta v priečinku 'data' vedľa seba
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# TU SA DEFINUJE CESTA K DÁTAM
# Ak máš dáta inde (napr. na externom disku), zmeň tento riadok:
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')

from models.pointpillars import PointPillars, PointPillarsConfig
from tracking.tracker import Tracker3D
from utils.roi_utils import calculate_risk_score
from utils.preprocess import DAIRDataset

def get_device():
    """Detekcia hardvéru (Mac M4 používa mps)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(cfg, checkpoint_path, device):
    """Načítanie modelu s podporou prenosu váh."""
    model = PointPillars(cfg).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Model načítaný na: {device}")
    else:
        print(f"Pozor: Checkpoint nenájdený v {checkpoint_path}. Bežím s náhodnými váhami.")
    model.eval()
    return model

def main():
    device = get_device()
    cfg = PointPillarsConfig()
    
    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'last.pth')
    model = load_model(cfg, ckpt_path, device)

    # Inicializácia Trackera a ROI
    roi_path = os.path.join(PROJECT_ROOT, 'roi_config.json')
    roi_coords = None
    if os.path.exists(roi_path):
        with open(roi_path, 'r') as f:
            roi_coords = json.load(f).get('roi')
    
    tracker = Tracker3D(roi_coords=roi_coords)
    
    # Inicializácia Loggera
    from utils.incident_logger import IncidentLogger
    logger = IncidentLogger(log_dir=os.path.join(PROJECT_ROOT, 'logs/incidents'))

    # 4. NAČÍTANIE SUROVÉHO DATASETU
    try:
    # Ak tvoj __init__ v preprocess.py vyzerá takto: def __init__(self, split):
        dataset = DAIRDataset(split='val')
        print(f"Dataset úspešne inicializovaný.")
        print(f"Cesty v preprocess.py: {DAIR_ROOT}")
    except Exception as e:
        print(f" Chyba pri načítaní datasetu: {e}")
        print("Skontroluj v preprocess.py, či __init__ vôbec prijíma nejaké argumenty.")
    return

    # 5. TESTOVACIA SLUČKA
    for i in range(len(dataset)):
        data = dataset[i]
        
        # Príprava tensorov pre Mac (mps)
        pillars = data['pillars'].unsqueeze(0).to(device)
        coords = data['coords'].unsqueeze(0).to(device)
        n_pts = data['num_points'].unsqueeze(0).to(device)

        with torch.no_grad():
            # Inferencia modelu
            # Tu by mal byť tvoj model(pillars, ...), ale pre test trackera 
            # odporúčam najprv použiť Ground Truth boxy:
            detections = data['gt_boxes'].numpy() 

        # Aktualizácia trackera a rizika
        confirmed = tracker.update(detections)
        for t in confirmed:
            risk = calculate_risk_score(t, confirmed)
            t.update_risk(risk)
            if t.smoothed_risk > 0.8:
                logger.save_incident(t.id, t.smoothed_risk)

        # 6. VIZUALIZÁCIA
        bev = np.zeros((600, 600, 3), dtype=np.uint8)
        for t in confirmed:
            # Prepočet na pixely (1m = 5px)
            px = int(300 - t.kf.x[1] * 5)
            py = int(300 - t.kf.x[0] * 5)
            
            risk = t.smoothed_risk
            color = (0, int(255 * (1 - risk)), int(255 * risk)) # BGR
            cv2.circle(bev, (px, py), 6, color, -1)
            cv2.putText(bev, f"ID:{t.id}", (px+10, py), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("M4 Intersection Safety Monitor", bev)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
