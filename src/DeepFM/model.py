import torch
import torch.nn as nn
import numpy as np

class DeepFM(nn.Module):
    """
    DeepFM 모델 클래스.

    Attributes:
        input_dims: 입력 차원 리스트.
        embedding_dim: 임베딩 차원.
        mlp_dims: MLP 레이어 차원 리스트.
        drop_rate: 드롭아웃 비율.
    """
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        
        self.total_input_dim = sum(input_dims)
        self.embedding_dim = embedding_dim
        # 범주형 변수의 개수
        self.num_categorical_features = len(input_dims)
        continuous_columns = ['genre_embedding', 'year']
        # 연속형 변수의 총 차원 수
        self.total_continuous_feature_dim = sum(1 if col in ['year'] else all_data[col].iloc[0].shape[0] for col in continuous_columns if col in all_data.columns)

        print(f"Input Dims: {input_dims}")
        print(f"num_categorical_features: {self.num_categorical_features}")
        print(f"total_continuous_feature_dim: {self.total_continuous_feature_dim}")
        print(f"embedding_dim: {self.embedding_dim}")

        # Embedding layer for categorical variables
        self.embedding = nn.Embedding(self.total_input_dim, embedding_dim)

        # FM components
        self.fc = nn.Embedding(self.total_input_dim, 1)
        self.bias = nn.Parameter(torch.zeros((1,)))

        # Continuous features' linear transformation
        self.continuous_linear = nn.Linear(self.total_continuous_feature_dim, self.total_continuous_feature_dim * embedding_dim)
        self.embedding_dim_total = self.num_categorical_features * embedding_dim + self.total_continuous_feature_dim * embedding_dim

        # MLP components
        self.mlp_input_dim = (self.num_categorical_features * embedding_dim) + (self.total_continuous_feature_dim * embedding_dim)
        print(f"Calculated MLP Input Dimension: {self.mlp_input_dim}")
        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i == 0:
                mlp_layers.append(nn.Linear(self.mlp_input_dim, dim))
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x_categorical, x_continuous):
        """
        FM 컴포넌트를 계산합니다.

        Args:
            x_categorical: 범주형 입력 텐서.
            x_continuous: 연속형 입력 텐서.

        Returns:
            Tensor: FM 출력.
        """
        # Embedding lookup for categorical variables
        embed_x = self.embedding(x_categorical)  # (batch_size, num_categorical_features, embedding_dim) = (2048, 2, 64)

        # Transform continuous features
        x_continuous_transformed = self.continuous_linear(x_continuous.float())  # (batch_size, total_continuous_feature_dim * embedding_dim) = (2048, 20*64=1280)
        x_continuous_transformed = x_continuous_transformed.view(
            -1, self.total_continuous_feature_dim, self.embedding_dim
        )  # (batch_size, total_continuous_feature_dim, embedding_dim) = (2048, 20, 64)

        # Concatenate embeddings and continuous features
        fm_input = torch.cat([embed_x, x_continuous_transformed], dim=1)  # (batch_size, total_features, embedding_dim) = (2048, 22, 64)

        # Linear term
        linear_part = torch.sum(self.fc(x_categorical), dim=1) + self.bias  # (batch_size, 1)
        linear_part += torch.sum(x_continuous, dim=1, keepdim=True)

        # Pairwise interaction term
        square_of_sum = torch.sum(fm_input, dim=1) ** 2  # (batch_size, embedding_dim) = (2048, 64)
        sum_of_square = torch.sum(fm_input ** 2, dim=1)  # (batch_size, embedding_dim) = (2048, 64)
        interaction_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)  # (batch_size, 1)

        fm_y = linear_part + interaction_part
        return fm_y

    def mlp(self, x_categorical, x_continuous):
        """
        MLP 컴포넌트를 계산합니다.

        Args:
            x_categorical: 범주형 입력 텐서.
            x_continuous: 연속형 입력 텐서.

        Returns:
            Tensor: MLP 출력.
        """
        # Embedding lookup for categorical variables
        embed_x = self.embedding(x_categorical)  # (batch_size, num_categorical_features, embedding_dim)
        # print(f"MLP - Embed_x Shape: {embed_x.shape}")  # 확인: (1024, num_categorical_features, embedding_dim)
        embed_x = embed_x.view(embed_x.size(0), -1)  # Flatten embeddings
        # print(f"MLP - Flattened Embed_x Shap!!!e: {embed_x.shape}")  # 확인: (1024, num_categorical_features * embedding_dim)

        # Transform continuous features
        x_continuous_transformed = self.continuous_linear(x_continuous)  # (batch_size, total_continuous_feature_dim * embedding_dim)
        # print(f"MLP - x_continuous_transformed Shape: {x_continuous_transformed.shape}")  # 확인: (1024, total_continuous_feature_dim * embedding_dim)

        # Concatenate embeddings and continuous features
        combined_features = torch.cat([embed_x, x_continuous_transformed], dim=1)  # (batch_size, mlp_input_dim) = (1024, (4+68)*64=4608)
        # print(f"MLP - Combined Features Shape: {combined_features.shape}")  # 확인: (1024, mlp_input_dim)
        
        # MLP forward pass
        mlp_y = self.mlp_layers(combined_features)  # (batch_size, 1)
        # print(f"MLP - MLP Output Shape: {mlp_y.shape}")  # 확인: (1024, 1)
        return mlp_y

    def forward(self, x_categorical, x_continuous):
        """
        모델의 순전파를 정의합니다.

        Args:
            x_categorical: 범주형 입력 텐서.
            x_continuous: 연속형 입력 텐서.

        Returns:
            Tensor: 모델 출력.
        """ 
        # x_categorical: (batch_size, num_categorical_features)
        # x_continuous: (batch_size, total_continuous_feature_dim)

        # FM component
        fm_y = self.fm(x_categorical, x_continuous)  # (batch_size, 1)
        
        # MLP component
        mlp_y = self.mlp(x_categorical, x_continuous)  # (batch_size, 1)
        
        # Combine FM and MLP components
        y = fm_y + mlp_y
        
        return y.squeeze(1)  # (batch_size,) 

