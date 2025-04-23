import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import argparse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=3)
    return p.parse_args()

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(9216, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def main():
    args = get_args()

    # 1️⃣ Detect and report GPUs
    ngpu = torch.cuda.device_count()
    print(f"Found {ngpu} GPU(s).")

    device = torch.device('cuda' if ngpu > 0 else 'cpu')

    # 2️⃣ Build model and wrap in DataParallel
    model = SimpleCNN()
    if ngpu > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # 3️⃣ Data loaders (MNIST example)
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 4️⃣ Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{args.epochs} — Loss: {total_loss/len(train_loader):.4f}")

if __name__ == '__main__':
    main()
