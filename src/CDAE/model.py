import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class CDAEModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, dropout_rate=0.1):
        super(CDAEModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.hidden_layer1 = nn.Linear(embedding_dim, 1024)
        self.hidden_layer2 = nn.Linear(1024, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, num_items)
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input).squeeze(1)
        item_embedded = self.item_embedding(item_input).squeeze(1)
        hidden = self.leaky_relu(self.hidden_layer1(item_embedded))
        hidden = self.leaky_relu(self.hidden_layer2(hidden))
        combined = user_embedded + hidden
        output = torch.sigmoid(self.output_layer(combined))
        return output

class InteractionDataset(Dataset):
    def __init__(self, data, num_items):
        self.data = data
        self.num_items = num_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_id = row['user_id']
        item_id = int(row['item_id'])
        label = torch.zeros(self.num_items)
        label[item_id] = row['watched']
        return user_id, item_id, label

def save_checkpoint(model, optimizer, epoch, filepath="model_checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, filepath)

def train_model(model, train_data, num_items, num_epochs=10, batch_size=128, learning_rate=0.001, device='cuda'):
    dataset = InteractionDataset(train_data, num_items)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for user_input, item_input, labels in dataloader:
            user_input = user_input.to(device).long()
            item_input = item_input.to(device).long()
            labels = labels.to(device)

            outputs = model(user_input, item_input)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader):.5f}")

    save_checkpoint(model, optimizer, epoch + 1)