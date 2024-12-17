import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    NCF (Neural Collaborative Filtering) 모델 클래스.
    이 모델은 GMF(Generalized Matrix Factorization)와 MLP(Multi-Layer Perceptron)를 결합하여
    추천 시스템을 구현합니다. GMF와 MLP는 사용자-아이템 임베딩을 결합하여 추천 예측을 합니다.
    
    Args:
        num_users (int): 총 사용자 수.
        num_items (int): 총 아이템 수.
        embed_dim (int, optional): 사용자 및 아이템 임베딩의 차원 (기본값은 16).
        hidden_dim (int, optional): MLP의 히든 레이어 차원 (기본값은 64).
    """
    
    def __init__(self, num_users, num_items, embed_dim=16, hidden_dim=64):
        super(NCF, self).__init__()

        # 사용자 및 아이템 임베딩 레이어 정의
        self.user_embedding = nn.Embedding(num_users, embed_dim)  # 사용자 임베딩
        self.item_embedding = nn.Embedding(num_items, embed_dim)  # 아이템 임베딩
        
        # GMF (Generalized Matrix Factorization) 레이어: 임베딩 차원에 맞춰 element-wise 곱셈 수행
        self.gmf_layer = nn.Sequential()
        
        # MLP (Multi-Layer Perceptron) 레이어
        self.mlp_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # 사용자 및 아이템 임베딩을 결합하여 입력
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embed_dim)  # GMF 출력과 동일한 차원으로 맞추기
        )
        
        # GMF와 MLP 출력을 결합하여 최종 예측값을 계산하는 레이어
        self.final_layer = nn.Linear(embed_dim, 1)
        
        # 시그모이드 활성화 함수 (출력을 확률 값으로 변환)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        """
        모델의 순전파(Forward pass) 함수.
        
        Args:
            user_indices (Tensor): 배치의 사용자 인덱스 (Tensor).
            item_indices (Tensor): 배치의 아이템 인덱스 (Tensor).
        
        Returns:
            Tensor: 사용자-아이템 간 상호작용에 대한 예측 확률 값 (0과 1 사이의 값).
        """
        # 사용자 및 아이템 임베딩 조회
        user_embed = self.user_embedding(user_indices)  # 사용자 임베딩
        item_embed = self.item_embedding(item_indices)  # 아이템 임베딩
        
        # GMF 경로: 사용자 임베딩과 아이템 임베딩의 element-wise 곱
        gmf_output = user_embed * item_embed
        
        # MLP 경로: 사용자 및 아이템 임베딩을 결합 후 MLP 처리
        mlp_input = torch.cat([user_embed, item_embed], dim=-1)  # 사용자 임베딩과 아이템 임베딩 결합
        mlp_output = self.mlp_layer(mlp_input)
        
        # GMF와 MLP 출력 결합 (합산 또는 결합 후 처리 가능)
        final_input = gmf_output + mlp_output  # GMF와 MLP를 더하는 방식
        
        # 최종 예측값 계산
        output = self.final_layer(final_input)
        
        # 시그모이드 활성화 함수 적용 (예측 확률 값으로 변환)
        return self.sigmoid(output).squeeze()  # 예측 결과는 0과 1 사이의 값으로 출력
