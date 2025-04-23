import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. 👇 Define your model here
class YourModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # TODO: build your layers

    def forward(self, x):
        # TODO: implement forward
        return x

def main():
    # ⚙️ 2. Setup device(s)
    ngpu = torch.cuda.device_count()
    print(f"Available GPUs: {ngpu}")
    device = torch.device('cuda' if ngpu > 0 else 'cpu')

    # ⚙️ 3. Instantiate model & wrap for multi-GPU
    model = YourModel(...)
    if ngpu > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # ⚙️ 4. Prepare your dataset & DataLoader
    # from your_dataset import YourDataset
    # train_ds = YourDataset(...)
    # train_loader = DataLoader(train_ds, batch_size=..., shuffle=True, num_workers=4)

    # ⚙️ 5. Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ⚙️ 6. Training loop
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg = running_loss / len(train_loader)
        print(f'Epoch {epoch}/{NUM_EPOCHS} — Loss: {avg:.4f}')

if __name__ == '__main__':
    main()
