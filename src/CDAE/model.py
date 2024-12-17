import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class CDAEModel(nn.Module):
    """
    CDAE(Convolutional Denoising Autoencoder) 모델 클래스.

    Attributes:
        user_embedding: 사용자 임베딩 레이어.
        item_embedding: 아이템 임베딩 레이어.
        hidden_layer1: 첫 번째 은닉층.
        hidden_layer2: 두 번째 은닉층.
        output_layer: 출력층.
        dropout: 드롭아웃 레이어.
        leaky_relu: LeakyReLU 활성화 함수.
    """
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
        """
        모델의 순전파를 정의합니다.

        Args:
            user_input: 사용자 ID 텐서.
            item_input: 아이템 ID 텐서.

        Returns:
            Tensor: 예측된 아이템의 확률 분포.
        """
        user_embedded = self.user_embedding(user_input).squeeze(1)
        item_embedded = self.item_embedding(item_input).squeeze(1)
        hidden = self.leaky_relu(self.hidden_layer1(item_embedded))
        hidden = self.leaky_relu(self.hidden_layer2(hidden))
        combined = user_embedded + hidden
        output = torch.sigmoid(self.output_layer(combined))
        return output

class InteractionDataset(Dataset):
    """
    사용자-아이템 상호작용 데이터셋 클래스.

    Attributes:
        data: 상호작용 데이터프레임.
        num_items: 총 아이템 수.
    """
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
    """
    모델 체크포인트를 저장합니다.

    Args:
        model: 모델 객체.
        optimizer: 옵티마이저 객체.
        epoch: 현재 에포크.
        filepath: 체크포인트 파일 경로.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, filepath)

def train_model(model, train_data, num_items, num_epochs=10, batch_size=128, learning_rate=0.001, device='cuda'):
    """
    모델을 학습합니다.

    Args:
        model: 모델 객체.
        train_data: 학습 데이터프레임.
        num_items: 총 아이템 수.
        num_epochs: 학습 에포크 수.
        batch_size: 배치 크기.
        learning_rate: 학습률.
        device: 모델을 실행할 장치 ('cuda' 또는 'cpu').

    Returns:
        tuple: 학습된 모델, 옵티마이저, 마지막 에포크.
    """
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