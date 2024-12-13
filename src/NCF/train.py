import torch
from torch.utils.data import DataLoader
from .model import NCF

def train_model(model, train_loader, epochs, lr, device):
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (user, item, label) in enumerate(train_loader):
            user, item, label = user.to(device), item.to(device), label.to(device).float()

            optimizer.zero_grad()
            preds = model(user, item)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(train_loader):.4f}", flush=True)