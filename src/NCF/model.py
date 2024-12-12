import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=16, hidden_dim=64):
        super(NCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        self.gmf_layer = nn.Sequential()
        
        self.mlp_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embed_dim)  # GMF와 차원 일치
        )
        
        # GMF와 MLP 결합
        self.final_layer = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        gmf_output = user_embed * item_embed
        
        # MLP: 임베딩 결합 후 Fully Connected Network 처리
        mlp_input = torch.cat([user_embed, item_embed], dim=-1)
        mlp_output = self.mlp_layer(mlp_input)
        
        final_input = gmf_output + mlp_output  # 또는 torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.final_layer(final_input)
        return self.sigmoid(output).squeeze()