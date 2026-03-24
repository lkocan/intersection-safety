import sys
sys.path.append('.')

import os
import torch
from torch.utils.data    import DataLoader
from torch.cuda.amp      import GradScaler, autocast   # ← FP16

from models.pointpillars import PointPillars, PointPillarsConfig
from training.loss       import PointPillarsLoss
from utils.preprocess import DAIRDatasetCached as DAIRDataset


def collate_fn(batch):
    return {
        'pillars':    torch.stack([b['pillars']    for b in batch]),
        'coords':     torch.stack([b['coords']     for b in batch]),
        'num_points': torch.stack([b['num_points'] for b in batch]),
        'gt_boxes':   [b['gt_boxes'] for b in batch],
        'frame_id':   [b['frame_id'] for b in batch],
    }


def train(cfg, device, save_dir='/content/checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    train_ds = DAIRDataset(split='train')
    val_ds   = DAIRDataset(split='val')

    train_loader = DataLoader(
        train_ds,
        batch_size       = cfg.batch_size,
        shuffle          = True,
        num_workers      = 4,              # ← zvýšené
        collate_fn       = collate_fn,
        pin_memory       = True,
        persistent_workers = True,         # ← workers ostanú nažive
        prefetch_factor  = 2,              # ← načítaj dopredu
    )
    val_loader = DataLoader(
        val_ds,
        batch_size       = cfg.batch_size,
        shuffle          = False,
        num_workers      = 4,
        collate_fn       = collate_fn,
        pin_memory       = True,
        persistent_workers = True,
    )

    model     = PointPillars(cfg).to(device)
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
    scaler = GradScaler()                  # ← FP16 scaler

    print(f"Model parametrov: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Tréning: {len(train_ds)} | Val: {len(val_ds)} | Device: {device}\n")

    best_val_loss = float('inf')

    for epoch in range(cfg.num_epochs):
        # ── Tréning ───────────────────────────────────────────────
        model.train()
        train_losses = []

        for i, batch in enumerate(train_loader):
            pillars    = batch['pillars'].to(device,    non_blocking=True)
            coords     = batch['coords'].to(device,     non_blocking=True)
            num_points = batch['num_points'].to(device, non_blocking=True)
            gt_boxes   = batch['gt_boxes']
            B          = pillars.shape[0]

            optimizer.zero_grad()

            with autocast():               # ← FP16 forward pass
                preds  = model(pillars, coords, num_points, batch_size=B)
                losses = criterion(preds, gt_boxes, batch_size=B)

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(losses['total'].item())

            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{cfg.num_epochs} "
                      f"[{i}/{len(train_loader)}]  "
                      f"loss={losses['total'].item():.4f}  "
                      f"cls={losses['cls'].item():.4f}  "
                      f"reg={losses['reg'].item():.4f}")

        avg_train = sum(train_losses) / len(train_losses)

        # ── Validácia ─────────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                pillars    = batch['pillars'].to(device,    non_blocking=True)
                coords     = batch['coords'].to(device,     non_blocking=True)
                num_points = batch['num_points'].to(device, non_blocking=True)
                B          = pillars.shape[0]

                with autocast():
                    preds  = model(pillars, coords, num_points, batch_size=B)
                    losses = criterion(preds, batch['gt_boxes'], batch_size=B)

                val_losses.append(losses['total'].item())

        avg_val = sum(val_losses) / len(val_losses)
        print(f"\n→ Epoch {epoch+1} | train={avg_train:.4f} | val={avg_val:.4f}\n")

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
            print(f"✓ Najlepší model uložený (val_loss={avg_val:.4f})")


if __name__ == '__main__':
    cfg    = PointPillarsConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(cfg, device)
