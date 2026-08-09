# train_fast.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import TransformerTCN
from tqdm import tqdm
import multiprocessing

def train(num_workers=0, pin_memory=False):
    EMB_DIR = 'embeddings_cache'
    X_PATH = os.path.join(EMB_DIR, 'X.npy')
    Y_PATH = os.path.join(EMB_DIR, 'y.npy')

    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError("Run precompute_embeddings.py first to generate X.npy and y.npy")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print('Loaded shapes:', X.shape, y.shape)

    # convert
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, dim)
    y_t = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)

    # if num_workers is None -> try to pick reasonable value, else use provided
    loader_workers = num_workers
    if loader_workers is None:
        loader_workers = 2

    try:
        loader = DataLoader(dataset, batch_size=64, shuffle=True,
                            num_workers=loader_workers, pin_memory=pin_memory)
        # quick smoke test to catch worker errors early
        it = iter(loader)
        _ = next(it)
    except Exception as e:
        print("⚠️ DataLoader with workers failed:", e)
        print("Falling back to num_workers=0 (Windows safe).")
        loader = DataLoader(dataset, batch_size=64, shuffle=True,
                            num_workers=0, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(np.unique(y))
    model = TransformerTCN(input_dim=X.shape[1], hidden_dim=256, num_classes=num_classes).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    epochs = 6
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=(loader.num_workers>0 and pin_memory))
            yb = yb.to(device, non_blocking=(loader.num_workers>0 and pin_memory))
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=total_loss/len(dataset))
        print(f'Epoch {epoch+1} avg loss: {total_loss/len(dataset):.4f}')

    os.makedirs('models', exist_ok=True)
    out_path = 'models/model_fast_distil.pth'
    torch.save(model.state_dict(), out_path)
    print('Saved model to', out_path)

if __name__ == "__main__":
    # Windows multiprocessing safety
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass

    # Detect if GPU exists; adjust pin_memory accordingly (pin_memory only helps with GPU)
    pin_memory = torch.cuda.is_available()
    # Start with a small number of workers for speed; if that fails we'll fallback to 0
    preferred_workers = 2  # set to 0 if you want to avoid workers entirely
    print(f"Starting training (preferred_workers={preferred_workers}, pin_memory={pin_memory})")
    train(num_workers=preferred_workers, pin_memory=pin_memory)
