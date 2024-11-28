import torch
import torch.nn as nn
from numpy import cumsum

# FM
class FeaturesLinear(nn.Module):
    def __init__(self, field_dims:list, output_dim:int=1, bias:bool=True):
        super().__init__()
        self.feature_dims = sum(field_dims)
        self.output_dim = output_dim
        self.offsets = [0, *cumsum(field_dims)[:-1]]

        self.fc = nn.Embedding(self.feature_dims, self.output_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty((self.output_dim,)), requires_grad=True)
    
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start
            if isinstance(m, nn.Parameter):
                nn.init.constant_(m, 0)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(self.fc(x), dim=1) + self.bias if hasattr(self, 'bias') \
                else torch.sum(self.fc(x), dim=1)
    

class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims:list, embed_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = [0, *cumsum(field_dims)[:-1]]
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return self.embedding(x)  # (batch_size, num_fields, embed_dim)
    

class FMLayer_Dense(nn.Module):
    def __init__(self):
        super().__init__()

    def square(self, x:torch.Tensor):
        return torch.pow(x,2)

    def forward(self, x):
        # square_of_sum =   # FILL HERE : Use `torch.sum()` and `self.square()` #
        # sum_of_square =   # FILL HERE : Use `torch.sum()` and `self.square()` #
        square_of_sum = self.square(torch.sum(x, dim=1))
        sum_of_square = torch.sum(self.square(x), dim=1)
        
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)


class FMLayer_Sparse(nn.Module):
    def __init__(self, field_dims:list, factor_dim:int):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, factor_dim)
        self.fm = FMLayer_Dense()


    def square(self, x):
        return torch.pow(x,2)
    

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.fm(x)
        
        return x
    

class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.embedding = nn.Embedding(input_dim+1, embed_dim)
        self.fm = FMLayer_Dense()
    def forward(self, x):
        x = torch.clamp(x, min=0, max=self.embedding.num_embeddings - 1)
        linear_part = self.linear(x)  # 선형 항 계산
        embedded_x = self.embedding(x.long())  # 임베딩 계산
        fm_part = self.fm(embedded_x)  # FM 상호작용 항
        return linear_part.squeeze(1) + fm_part
    