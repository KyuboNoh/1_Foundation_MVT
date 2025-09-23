# src/gfm4mpm/training/train_ssl.py
from typing import Iterable
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        return self.mse(pred, target)

def train_ssl(model, dataloader: DataLoader, epochs=30, lr=2.5e-4, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95))
    criterion = MAELoss()
    model.train()
    for ep in range(1, epochs+1):
        running = 0.0
        for batch in tqdm(dataloader, desc=f"SSL epoch {ep}"):
            x = batch.to(device)
            pred, _ = model(x)
            loss = criterion(pred, x)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * x.size(0)
        print(f"[SSL] epoch {ep} loss: {running/len(dataloader.dataset):.4f}")
    return model
