import torch
import torch.nn as nn

class FeaturesLinear(nn.Module):
    def __init__(self, input_dim, bias=True):
        """
        선형적 중요도를 학습하는 레이어
        input_dim: 입력 벡터의 차원 (임베딩된 벡터의 총 길이)
        bias: Bias 사용 여부
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, 1, bias=bias)  # 모든 벡터 요소에 대해 선형 가중치 학습
        self._initialize_weights()

    def _initialize_weights(self):
        """
        가중치 초기화 함수
        """
        if hasattr(self.fc, 'weight'):
            # Linear 레이어의 가중치를 Xavier 초기화
            nn.init.xavier_uniform_(self.fc.weight)

        if hasattr(self.fc, 'bias') and self.fc.bias is not None:
            # Bias를 0으로 초기화
            nn.init.constant_(self.fc.bias, 0)


    def forward(self, x: torch.Tensor):
        """
        입력 텐서의 각 요소에 대해 선형 가중치를 적용
        x: (batch_size, input_dim)
        """
        return self.fc(x).squeeze(1)  # (batch_size, 1) → (batch_size)


class FMLayer_Dense(nn.Module):
    def __init__(self):
        """
        2차 상호작용을 계산하는 FM 레이어
        """
        super().__init__()

    def square(self, x: torch.Tensor):
        return torch.pow(x, 2)

    def forward(self, x: torch.Tensor):
        """
        2차 상호작용을 계산
        x: (batch_size, num_fields, embed_dim)
        """
        # 각 벡터의 합을 제곱
        square_of_sum = self.square(torch.sum(x, dim=1))  # (batch_size, embed_dim)
        # 각 벡터의 제곱을 합산
        sum_of_square = torch.sum(self.square(x), dim=1)  # (batch_size, embed_dim)
        # 상호작용 계산
        interaction = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)  # (batch_size)
        return interaction


class FactorizationMachine(nn.Module):
    def __init__(self, input_dim):
        """
        input_dim: 전체 입력 벡터의 차원.
        """
        super().__init__()
        self.linear = FeaturesLinear(input_dim)  # 선형 항 계산
        self.fm = FMLayer_Dense()  # 2차 상호작용 계산

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, input_dim)
        """
        # 선형 항 계산
        linear_part = self.linear(x)
        # 2차 상호작용 계산
        fm_part = self.fm(x.unsqueeze(1))  # (batch_size, num_fields, embed_dim)
        return linear_part + fm_part 