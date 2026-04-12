import os
import sys
import torch
import numpy as np
import cv2
import json

# 1. NASTAVENIE PROSTREDIA
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Importujeme dataset a tvoje preddefinované cesty priamo z preprocess
from utils.preprocess import DAIRDataset, DAIR_ROOT, PCD_DIR, LABEL_DIR
from models.pointpillars import PointPillars, PointPillarsConfig
from tracking.tracker import Tracker3D
from utils.roi_utils import calculate_risk_score
from utils.incident_logger import IncidentLogger

def get_device():
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
        print(f"Pozor: Checkpoint nenájdený v {ckpt_path}.")

    model.eval()

    # 3. INICIALIZÁCIA KOMPONENTOV
    roi_path = os.path.join(PROJECT_ROOT, 'roi_config.json')
    roi_coords = None
    if os.path.exists(roi_path):
        with open(roi_path, 'r') as f:
            roi_coords = json.load(f).get('roi')
    
    tracker = Tracker3D(roi_coords=roi_coords)
    logger = IncidentLogger(log_dir=os.path.join(PROJECT_ROOT, 'logs/incidents'))

    # 4. NAČÍTANIE DATASETU
    try:
        # Používame tvoj funkčný inicializátor
        dataset = DAIRDataset(split='val')
        print(f"Dataset pripravený. Používam root: {DAIR_ROOT}")
    except Exception as e:
        print(f"Chyba pri načítaní datasetu: {e}")
        return

    # 5. TESTOVACIA SLUČKA
    print("--- Spúšťam vizualizáciu (Stlač 'q' pre ukončenie) ---")
    for i in range(len(dataset)):
        data = dataset[i]
        
        # Presun dát na M4 GPU (mps)
        pillars = data['pillars'].unsqueeze(0).to(device)
        coords = data['coords'].unsqueeze(0).to(device)
        n_pts = data['num_points'].unsqueeze(0).to(device)

        with torch.no_grad():
            # PRE TEST: Používame Ground Truth (pravdivé dáta), aby sme videli tracker v akcii
            detections = data['gt_boxes'].numpy() 
            
            # Ak chceš skúsiť reálny model, odkomentuj toto:
            # preds = model(pillars, coords, n_pts)
            # detections = post_process(preds) 

        # Aktualizácia trackera
        confirmed = tracker.update(detections)
        logger.add_frame(confirmed)

        # Analýza rizika
        for t in confirmed:
            risk = calculate_risk_score(t, confirmed)
            t.update_risk(risk)
            if t.smoothed_risk > 0.8:
                logger.save_incident(t.id, t.smoothed_risk)

        # 6. VIZUALIZÁCIA (BEV)
        bev = np.zeros((600, 600, 3), dtype=np.uint8)
        # Nakreslíme kríž v strede (0,0)
        cv2.line(bev, (300, 0), (300, 600), (50, 50, 50), 1)
        cv2.line(bev, (0, 300), (600, 300), (50, 50, 50), 1)

        for t in confirmed:
            # Mierka: 1m = 5 pixelov
            px = int(300 - t.kf.x[1] * 5)
            py = int(300 - t.kf.x[0] * 5)
            
            color = (0, int(255 * (1 - t.smoothed_risk)), int(255 * t.smoothed_risk))
            cv2.circle(bev, (px, py), 7, color, -1)
            cv2.putText(bev, f"ID:{t.id} R:{t.smoothed_risk:.1f}", (px+10, py), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("M4 Safety Monitor", bev)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
