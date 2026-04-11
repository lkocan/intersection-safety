import os
import sys
import torch
import numpy as np
import cv2
import json

# Pridanie ciest k modulom (podľa toho, kde máš projekt na disku)
# Ak spúšťaš priamo z priečinka projektu, stačí '.'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from models.pointpillars import PointPillars, PointPillarsConfig
from tracking.tracker import Tracker3D
from utils.roi_utils import calculate_risk_score

def get_device():
    """Inteligentná detekcia hardvéru."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Špecifické pre tvoj Mac M4
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(cfg, checkpoint_path, device):
    """Bezpečné načítanie modelu naprieč platformami."""
    model = PointPillars(cfg).to(device)
    
    if os.path.exists(checkpoint_path):
        # Kľúčové: map_location zabezpečí, že váhy z CUDA (Colab) 
        # sa správne namapujú na MPS (Mac) alebo CPU
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Model úspešne načítaný na zariadenie: {device}")
    else:
        print(f"Checkpoint '{checkpoint_path}' nenájdený. Model beží s náhodnými váhami.")
    
    model.eval()
    return model

def main():
    device = get_device()
    cfg = PointPillarsConfig()
    
    # Nastav cestu k tvojmu checkpointu (relatívne k projektu)
    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'last.pth')
    
    model = load_model(cfg, ckpt_path, device)
    tracker = Tracker3D()

    print("--- Systém pripravený na testovanie ---")
    
    # Simulácia vizualizácie
    # V Colabe by sme použili cv2_imshow, na Macu klasické cv2.imshow
    is_colab = 'google.colab' in sys.modules

    # Tu by nasledovala tvoja slučka spracovania dát...
    # frame_data = dataset[0]
    # detections = model_inference(model, frame_data, device)
    # confirmed = tracker.update(detections)

if __name__ == "__main__":
    main()
