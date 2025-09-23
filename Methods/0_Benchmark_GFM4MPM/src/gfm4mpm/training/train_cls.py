# src/gfm4mpm/training/train_cls.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, matthews_corrcoef
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

def train_classifier(encoder, mlp, train_loader, val_loader, epochs=50, lr=1e-3, device='cuda'):
    encoder.eval().to(device)
    mlp.to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr)
    bce = torch.nn.BCELoss()
    best = {"f1": -1, "state_dict": None}
    for ep in range(1, epochs+1):
        mlp.train()
        for x, y in tqdm(train_loader, desc=f"CLS epoch {ep}"):
            x, y = x.to(device), y.float().to(device)
            with torch.no_grad():
                z = encoder.encode(x)
            p = mlp(z)
            loss = bce(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
        # validate
        f1, mcc, auprc, auroc = eval_classifier(encoder, mlp, val_loader, device)
        if f1 > best["f1"]:
            best = {"f1": f1, "state_dict": mlp.state_dict()}
        print(f"[VAL] f1={f1:.3f} mcc={mcc:.3f} auprc={auprc:.3f} auroc={auroc:.3f}")
    mlp.load_state_dict(best["state_dict"])
    return mlp

def eval_classifier(encoder, mlp, loader, device='cuda'):
    encoder.eval().to(device)
    mlp.eval().to(device)
    ys, ps = [], []
    auroc = BinaryAUROC()
    auprc = BinaryAveragePrecision()
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            z = encoder.encode(x)
            p = mlp(z).cpu()
        ys.append(y)
        ps.append(p)
        auroc.update(p, y)
        auprc.update(p, y)
    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(ps).numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    from sklearn.metrics import f1_score, matthews_corrcoef
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return f1, mcc, float(auprc.compute()), float(auroc.compute())
