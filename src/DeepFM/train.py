import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset

class PairwiseRecommendationDataset(Dataset):
    def __init__(self, user_col, item_col, user_item_dict, all_items, item_probs, user_item_embeddings, num_negatives=1):
        self.user_col = user_col.cpu()
        self.item_col = item_col.cpu()
        self.user_item_dict = user_item_dict
        self.all_items = np.array(all_items)
        self.item_probs = item_probs
        self.num_negatives = num_negatives
        self.user_item_embeddings = user_item_embeddings


class LambdaRankLoss(nn.Module):
    def __init__(self):
        super(LambdaRankLoss, self).__init__()

    def forward(self, pos_preds, neg_preds):
        """
        pos_preds: Positive 샘플의 예측값 (batch_size,)
        neg_preds: Negative 샘플의 예측값 (batch_size, num_negatives)
        """
        # Difference between positive and negative predictions
        diff = pos_preds.unsqueeze(1) - neg_preds  # (batch_size, num_negatives)
        # LambdaRank Loss 계산
        loss = -torch.log(torch.sigmoid(diff) + 1e-8).sum(dim=1).mean()
        return loss
    
def train_pairwise(model, train_loader, optimizer, device, genre_col, all_years, epochs=10):
    model.train()
    criterion = LambdaRankLoss()

    for epoch in range(epochs):
        total_loss = 0
        for user, pos_item, neg_items in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            user = user.to(device, dtype=torch.long)
            pos_item = pos_item.to(device, dtype=torch.long)
            neg_items = neg_items.to(device, dtype=torch.long)

            optimizer.zero_grad()

            # Positive 샘플
            pos_genre = genre_col[pos_item]
            pos_year = torch.tensor(all_years[pos_item.cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(1)
            pos_continuous = torch.cat([pos_genre, pos_year], dim=1)
            x_categorical_pos = torch.stack([user, pos_item], dim=1)
            pos_preds = model(x_categorical_pos, pos_continuous)

            # Negative 샘플
            neg_items_flat = neg_items.view(-1)
            user_neg = user.unsqueeze(1).expand(-1, neg_items.shape[1]).reshape(-1)
            neg_genre = genre_col[neg_items_flat]
            neg_year = torch.tensor(all_years[neg_items_flat.cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(1)
            neg_continuous = torch.cat([neg_genre, neg_year], dim=1)
            x_categorical_neg = torch.stack([user_neg, neg_items_flat], dim=1)
            neg_preds = model(x_categorical_neg, neg_continuous).view(neg_items.shape[0], neg_items.shape[1])

            # LambdaRank Loss 계산
            loss = criterion(pos_preds, neg_preds)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
