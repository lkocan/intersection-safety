import sys
import os
import time
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# --- UNIVERZÁLNE NASTAVENIE CIEST ---
# Zistí root projektu bez ohľadu na to, kde skript beží
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.pointpillars import PointPillars, PointPillarsConfig
from training.loss import PointPillarsLoss
from utils.preprocess import DAIRDataset

def collate_fn(batch):
    """Zlučuje vzorky do batchu, filtruje raw_points pre úsporu RAM."""
    return {
        'pillars': torch.stack([b['pillars'] for b in batch], dim=0).float(),
        'coords': torch.stack([b['coords'] for b in batch], dim=0).int(),
        'num_points': torch.stack([b['num_points'] for b in batch], dim=0).int(),
        'gt_boxes': [b['gt_boxes'].float() for b in batch],
        'frame_id': [b['frame_id'] for b in batch],
    }

def train():
    cfg = PointPillarsConfig()
    
    # --- AUTOMATICKÁ DETEKCIA HARDVÉRU ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True  # Nvidia GPU - zapíname zmiešanú presnosť
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False # Apple Silicon - stabilnejšie v FP32
    else:
        device = torch.device("cpu")
        use_amp = False

    print(f"\n" + "="*40)
    print(f"ŠTART TRÉNINGU")
    print(f"Zariadenie: {device} | AMP: {use_amp}")
    print(f"Batch Size: {cfg.batch_size} | Epochy: {cfg.num_epochs}")
    print("="*40 + "\n")

    # --- DATA LOADERY ---
    # num_workers: 8 pre M4 (10 jadier), v Colabe (T4) dajte radšej 4
    n_workers = 8 if device.type != 'cpu' else 0
    if os.environ.get('COLAB_GPU'): n_workers = 4 # Optimalizácia pre Colab

    train_ds = DAIRDataset(split='train')
    val_ds   = DAIRDataset(split='val')

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=n_workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(n_workers > 0)
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=n_workers, collate_fn=collate_fn
    )

    # --- MODEL & STRATA ---
    model = PointPillars(cfg).to(device)
    criterion = PointPillarsLoss().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.learning_rate, 
        steps_per_epoch=len(train_loader), epochs=cfg.num_epochs
    )

    # Používame modernejší API pre AMP
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Ukladanie do rootu projektu
    save_dir = BASE_DIR / 'checkpoints_universal'
    save_dir.mkdir(exist_ok=True)

    best_val_loss = float('inf')

    # --- TRÉNINGOVÝ CYKLUS ---
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_losses = []
        start_t = time.time()

        for i, batch in enumerate(train_loader):
            pillars = batch['pillars'].to(device, non_blocking=True)
            coords = batch['coords'].to(device, non_blocking=True)
            num_points = batch['num_points'].to(device, non_blocking=True)
            gt_boxes = [b.to(device) for b in batch['gt_boxes']]

            optimizer.zero_grad()
            
            # Autocast pre Nvidiu, na Macu/CPU neaktívny
            with torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=use_amp):
                preds = model(pillars, coords, num_points, batch_size=pillars.shape[0])
                losses = criterion(preds, gt_boxes, batch_size=pillars.shape[0])

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_losses.append(losses['total'].item())

            if i % 20 == 0:
                print(f"E{epoch+1} [{i}/{len(train_loader)}] Loss: {losses['total'].item():.4f} | "
                      f"Cls: {losses['cls'].item():.4f} | Reg: {losses['reg'].item():.4f}")

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        
        # --- VALIDÁCIA ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pillars, coords, n_pts = batch['pillars'].to(device), batch['coords'].to(device), batch['num_points'].to(device)
                gt_boxes = [b.to(device) for b in batch['gt_boxes']]
                preds = model(pillars, coords, n_pts, batch_size=pillars.shape[0])
                v_losses = criterion(preds, gt_boxes, batch_size=pillars.shape[0])
                val_losses.append(v_losses['total'].item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"\n🏁 Epoch {epoch+1} | Time: {time.time()-start_t:.1f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- UKLADANIE ---
        checkpoint = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'val_loss': avg_val_loss}
        torch.save(checkpoint, save_dir / 'last.pth')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, save_dir / 'best.pth')
            print(f"⭐ NOVÝ NAJLEPŠÍ MODEL (Loss: {avg_val_loss:.4f})")

if __name__ == '__main__':
    train()
