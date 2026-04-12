import os
import sys
import torch
import numpy as np
import cv2
import json

# 1. NASTAVENIE PROSTREDIA
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Importy z tvojich modulov
from utils.preprocess import DAIRDataset, DAIR_ROOT
from models.pointpillars import PointPillars, PointPillarsConfig
from tracking.tracker import Tracker3D
from utils.roi_utils import calculate_risk_score
from utils.incident_logger import IncidentLogger

def get_device():
    """Detekcia hardvéru (Priorita: MPS pre Mac M4)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_device()
    cfg = PointPillarsConfig()
    
    # 2. NAČÍTANIE MODELU
    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'last.pth')
    model = PointPillars(cfg).to(device)
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Model načítaný na: {device}")
    else:
        print(f"Pozor: Checkpoint nenájdený. Bežím s náhodnými váhami.")

    model.eval()

    # 3. INICIALIZÁCIA KOMPONENTOV
    # Načítanie ROI ak existuje
    roi_path = os.path.join(PROJECT_ROOT, 'roi_config.json')
    roi_coords = None
    if os.path.exists(roi_path):
        with open(roi_path, 'r') as f:
            roi_coords = json.load(f).get('roi')
    
    tracker = Tracker3D(roi_coords=roi_coords)
    logger = IncidentLogger(log_dir=os.path.join(PROJECT_ROOT, 'logs/incidents'))

    # 4. NAČÍTANIE DATASETU
    try:
        # Používame základný DAIRDataset (číta PCD priamo)
        dataset = DAIRDataset(split='val')
        print(f"Dataset pripravený ({len(dataset)} vzoriek).")
    except Exception as e:
        print(f"Chyba pri načítaní datasetu: {e}")
        return

    # 5. TESTOVACIA SLUČKA
    print("--- Spúšťam vizualizáciu (Stlač 'q' pre ukončenie) ---")
    
    for i in range(len(dataset)):
        data = dataset[i]
        
        # Príprava dát pre MPS
        pillars = data['pillars'].unsqueeze(0).to(device)
        coords = data['coords'].unsqueeze(0).to(device)
        n_pts = data['num_points'].unsqueeze(0).to(device)

        with torch.no_grad():
            # Získanie Ground Truth boxov (N, 8)
            gt = data['gt_boxes'].numpy() 
            
            # OPRAVA INDEXU: Tracker očakáva 9 hodnôt [x,y,z,w,l,h,yaw,class,score]
            # DAIR GT má 8 hodnôt [x,y,z,w,l,h,yaw,class]
            if gt.shape[0] > 0 and gt.shape[1] == 8:
                # Pridáme stĺpec s istotou 1.0
                scores = np.ones((gt.shape[0], 1), dtype=np.float32)
                detections = np.hstack([gt, scores])
            else:
                detections = gt

        # 6. TRACKING & RISK ANALYSIS
        confirmed = tracker.update(detections)
        logger.add_frame(confirmed)

        for t in confirmed:
            risk = calculate_risk_score(t, confirmed)
            t.update_risk(risk)
            # Automatické logovanie pri vysokom riziku
            if t.smoothed_risk > 0.8:
                logger.save_incident(t.id, t.smoothed_risk)

        # 7. VIZUALIZÁCIA (Bird's Eye View)
        bev = np.zeros((600, 600, 3), dtype=np.uint8)
        
        # Kreslenie pomocnej mriežky
        cv2.line(bev, (300, 0), (300, 600), (40, 40, 40), 1)
        cv2.line(bev, (0, 300), (600, 300), (40, 40, 40), 1)

        for t in confirmed:
            # Prepočet metrov na pixely (stred 300,300 | 1m = 5px)
            px = int(300 - t.kf.x[1] * 5)
            py = int(300 - t.kf.x[0] * 5)
            
            # Farba podľa rizika (Zelená -> Červená)
            risk_val = getattr(t, 'smoothed_risk', 0)
            color = (0, int(255 * (1 - risk_val)), int(255 * risk_val))
            
            cv2.circle(bev, (px, py), 8, color, -1)
            cv2.putText(bev, f"ID:{t.id} R:{risk_val:.2f}", (px + 10, py), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("M4 Safety Monitor", bev)
        
        # Ukončenie klávesou 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Test ukončený.")

if __name__ == "__main__":
    main()
