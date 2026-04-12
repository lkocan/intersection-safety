import sys
import os

# Pridanie koreňového adresára do cesty pre správne importy
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader
# Pozor: Na Macu (MPS) sa GradScaler v niektorých verziách torch správa inak, 
# ale ponechávame ho pre kompatibilitu.
from torch.cuda.amp import GradScaler, autocast 

from models.pointpillars import PointPillars, PointPillarsConfig
from training.loss import PointPillarsLoss
# Importujeme klasický DAIRDataset (nie Cached), aby fungoval GTSampler
from utils.preprocess import DAIRDataset

def collate_fn(batch):
    """
    Zlučuje vzorky do batchu. 
    Vynechávame 'raw_points', pretože majú rôzne veľkosti a nepotrebujeme ich v GPU.
    """
    return {
        'pillars': torch.stack([b['pillars'] for b in batch], dim=0).float(),
        'coords': torch.stack([b['coords'] for b in batch], dim=0).int(),
        'num_points': torch.stack([b['num_points'] for b in batch], dim=0).int(),
        'gt_boxes': [b['gt_boxes'].float() for b in batch],
        'frame_id': [b['frame_id'] for b in batch],
    }

def train(cfg, device, save_dir='./checkpoints_v2', start_epoch=0):
    os.makedirs(save_dir, exist_ok=True)

    # Inicializujeme dataset (GTSampler sa zapne automaticky vnútri pre split='train')
    train_ds = DAIRDataset(split='train')
    val_ds   = DAIRDataset(split='val')

    train_loader = DataLoader(
        train_ds,
        batch_size       = cfg.batch_size,
        shuffle          = True,
        num_workers      = 8,                 # Prispôsob podľa počtu jadier na Macu
        collate_fn       = collate_fn,
        pin_memory       = (device.type == 'cuda'), # Iba pre NVIDIA GPU
        persistent_workers = True,
        prefetch_factor  = 2,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size       = cfg.batch_size,
        shuffle          = False,
        num_workers      = 8,
        collate_fn       = collate_fn,
    )

    model     = PointPillars(cfg).to(device)
    # Loss funkcia už obsahuje tvoje nové váhy pre chodcov/cyklistov
    criterion = PointPillarsLoss().to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = 0.01,
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = cfg.learning_rate,
        steps_per_epoch = len(train_loader),
        epochs          = cfg.num_epochs,
    )
    
    # scaler na Macu (MPS) niekedy nie je podporovaný, ošetríme to nižšie
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)

    print("-" * 30)
    print(f"🚀 ŠTART TRÉNINGU")
    print(f"Zariadenie: {device}")
    print(f"Počet epoch: {cfg.num_epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Model parametrov: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 30)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        train_losses = []

        for i, batch in enumerate(train_loader):
            pillars    = batch['pillars'].to(device,    non_blocking=True)
            coords     = batch['coords'].to(device,     non_blocking=True)
            num_points = batch['num_points'].to(device, non_blocking=True)
            gt_boxes   = [boxes.to(device) for boxes in batch['gt_boxes']]
            B          = pillars.shape[0]

            optimizer.zero_grad()

            # Na Macu (MPS) autocast zatiaľ nie je plne stabilný, používame len ak máme CUDA
            if use_amp:
                with autocast():
                    preds  = model(pillars, coords, num_points, batch_size=B)
                    losses = criterion(preds, gt_boxes, batch_size=B)
                scaler.scale(losses['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Štandardný pass pre Mac (MPS) / CPU
                preds  = model(pillars, coords, num_points, batch_size=B)
                losses = criterion(preds, gt_boxes, batch_size=B)
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

            scheduler.step()
            train_losses.append(losses['total'].item())

            if i % 20 == 0:
                print(f"E{epoch+1} [{i}/{len(train_loader)}] "
                      f"Loss: {losses['total'].item():.4f} | "
                      f"Cls: {losses['cls'].item():.4f} | "
                      f"Reg: {losses['reg'].item():.4f}")

        avg_train = sum(train_losses) / len(train_losses)

        # ── Validácia ─────────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                pillars    = batch['pillars'].to(device)
                coords     = batch['coords'].to(device)
                num_points = batch['num_points'].to(device)
                gt_boxes   = [boxes.to(device) for boxes in batch['gt_boxes']]
                B          = pillars.shape[0]

                preds  = model(pillars, coords, num_points, batch_size=B)
                losses = criterion(preds, gt_boxes, batch_size=B)
                val_losses.append(losses['total'].item())

        avg_val = sum(val_losses) / len(val_losses)
        print(f"\n✅ Epoch {epoch+1} FINISHED | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Ukladanie
        checkpoint = {
            'epoch':      epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'val_loss':   avg_val,
        }
        torch.save(checkpoint, f'{save_dir}/last.pth')

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(checkpoint, f'{save_dir}/best.pth')
            print(f"⭐ NOVÝ NAJLEPŠÍ MODEL ULOŽENÝ (Val Loss: {avg_val:.4f})\n")

if __name__ == '__main__':
    cfg    = PointPillarsConfig()
    # Detekcia Apple Silicon (MPS) alebo CUDA
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    train(cfg, device)
