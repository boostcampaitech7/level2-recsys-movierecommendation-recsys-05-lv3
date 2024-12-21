import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention 메커니즘을 구현한 클래스입니다

    Args:
        hidden_units (int): 입력 벡터의 차원 크기
        dropout_rate (float): 드롭아웃(dropout) 비율

    Returns:
        tuple(torch.Tensor, torch.Tensor):
            - 첫 번째 반환값: 어텐션을 적용한 후의 출력 텐서 (shape: [batch_size, num_heads, seq_len, hidden_units])
            - 두 번째 반환값: 어텐션 가중치(attention distribution) (shape: [batch_size, num_heads, seq_len, seq_len])
    """
    def __init__(self, hidden_units, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)  # dropout rate

    def forward(self, Q, K, V, mask):
        """
        Args:
            Q (torch.Tensor): 쿼리 텐서 (shape: [batch_size, num_heads, seq_len, hidden_units])
            K (torch.Tensor): 키 텐서 (shape: [batch_size, num_heads, seq_len, hidden_units])
            V (torch.Tensor): 밸류 텐서 (shape: [batch_size, num_heads, seq_len, hidden_units])
            mask (torch.Tensor): 마스크 텐서 (shape: [batch_size, 1, 1, seq_len]), 어텐션을 계산하지 않을 위치를 0으로 표시

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - output: 어텐션 적용 후 출력 (shape: [batch_size, num_heads, seq_len, hidden_units])
                - attn_dist: 어텐션 가중치 (shape: [batch_size, num_heads, seq_len, seq_len])
        """
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.hidden_units)
        attn_score = attn_score.masked_fill(
            mask == 0, -1e9
        )  # 유사도가 0인 지점은 -infinity로 보내 softmax 결과가 0이 되도록 함
        attn_dist = self.dropout(
            F.softmax(attn_score, dim=-1)
        )  # attention distribution
        output = torch.matmul(
            attn_dist, V
        )  # dim of output : batchSize x num_head x seqLen x hidden_units
        return output, attn_dist


class MultiHeadAttention(nn.Module):
    """
    멀티-헤드 어텐션(Multi-Head Attention) 모듈 구현 클래스입니다.

    Args:
        num_heads (int): 어텐션 헤드(head)의 수
        hidden_units (int): 입력 벡터의 차원 크기
        dropout_rate (float): 드롭아웃(dropout) 비율

    Returns:
        (MultiHeadAttention): nn.Module을 상속한 멀티-헤드 어텐션 모듈 객체로, forward 호출 시 (output, attn_dist)를 반환합니다

        - output (torch.Tensor): 멀티-헤드 어텐션을 통과한 후의 출력 텐서 (shape: [batch_size, seq_len, hidden_units])
        - attn_dist (torch.Tensor): 어텐션 가중치 텐서 (shape: [batch_size, num_heads, seq_len, seq_len])
    """
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # head의 수
        self.hidden_units = hidden_units

        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_O = nn.Linear(hidden_units, hidden_units, bias=False)

        self.attention = ScaledDotProductAttention(
            hidden_units, dropout_rate
        )  # scaled dot product attention module을 사용하여 attention 계산
        self.dropout = nn.Dropout(dropout_rate)  # dropout rate
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6)  # layer normalization

    def forward(self, enc, mask):
        """
        Args:
            enc (torch.Tensor): 입력 시퀀스 텐서 (shape: [batch_size, seq_len, hidden_units])
            mask (torch.Tensor): 마스크 텐서 (shape: [batch_size, 1, 1, seq_len]), 어텐션을 계산하지 않을 위치를 0으로 표시

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - output: 멀티-헤드 어텐션 적용 후의 결과 텐서 (shape: [batch_size, seq_len, hidden_units])
                - attn_dist: 어텐션 가중치 텐서 (shape: [batch_size, num_heads, seq_len, seq_len])
        """
        residual = enc  # residual connection을 위해 residual 부분을 저장
        batch_size, seqlen = enc.size(0), enc.size(1)

        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)

        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seqlen, -1)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist


class PositionwiseFeedForward(nn.Module):
    """
    위치별 피드포워드(Position-wise Feed Forward) 레이어 클래스입니다.
    입력 시퀀스의 각 위치에 대해 독립적으로(feed-forward) 비선형 변환을 수행합니다.

    Args
        hidden_units (int): 입력 벡터의 차원 크기
        dropout_rate (float): 드롭아웃(dropout) 비율

    Returns:
        (PositionwiseFeedForward): nn.Module을 상속한 Position-wise FFN 모듈 객체로, forward 호출 시 출력 텐서를 반환합니다
        
        - output (torch.Tensor): Position-wise FFN을 통과한 후의 출력 텐서 (shape: [batch_size, seq_len, hidden_units])
    """
    def __init__(self, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()

        # SASRec과의 dimension 차이가 있습니다.
        self.W_1 = nn.Linear(hidden_units, 4 * hidden_units)
        self.W_2 = nn.Linear(4 * hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6)  # layer normalization

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 텐서 (shape: [batch_size, seq_len, hidden_units])

        Returns:
            torch.Tensor: Position-wise FFN 적용 후의 결과 텐서 (shape: [batch_size, seq_len, hidden_units])
        """
        residual = x
        output = self.W_2(F.gelu(self.dropout(self.W_1(x))))  # activation: relu -> gelu
        output = self.layerNorm(self.dropout(output) + residual)
        return output


class BERT4RecBlock(nn.Module):
    """
    BERT4Rec 모델의 한 블록을 구성하는 클래스입니다
    Multi-Head Attention과 Position-wise Feed Forward 레이어를 포함하고 있습니다.

    Args:
        num_heads (int): 어텐션 헤드 수
        hidden_units (int): 임베딩 차원 크기
        dropout_rate (float): 드롭아웃 비율

    Returns:
        (BERT4RecBlock): nn.Module을 상속한 BERT4Rec 블록 객체로,forward 호출 시 (output_enc, attn_dist)를 반환합니다

        - output_enc (torch.Tensor): 블록 통과 후의 시퀀스 출력 텐서 (shape: [batch_size, seq_len, hidden_units])
        - attn_dist (torch.Tensor): 어텐션 가중치 텐서 (shape: [batch_size, num_heads, seq_len, seq_len])
    """
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(BERT4RecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, input_enc, mask):
        """
        Args:
            input_enc (torch.Tensor): 블록 입력 텐서 (shape: [batch_size, seq_len, hidden_units])
            mask (torch.Tensor): 마스크 텐서 (shape: [batch_size, 1, seq_len, seq_len]), 어텐션 계산 시 무시할 위치를 0으로 표시

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - output_enc: 블록 처리 후의 출력 텐서 (shape: [batch_size, seq_len, hidden_units])
                - attn_dist: 어텐션 가중치 텐서 (shape: [batch_size, num_heads, seq_len, seq_len])
        """
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist


class BERT4Rec(nn.Module):
    """
    BERT4Rec 모델을 구현한 클래스입니다
    아이템 시퀀스를 입력으로 받아 마스크된 아이템을 예측하는 모델로,
    BERT4RecBlock을 여러 층으로 쌓아 구현합니다

    Args:
        config (dict): 모델 및 학습에 필요한 설정을 담은 딕셔너리
        num_user (int): 전체 사용자 수
        num_item (int): 전체 아이템 수

    Returns:
        (BERT4Rec): nn.Module을 상속한 BERT4Rec 모델 객체로, forward 호출 시 out 텐서를 반환합니다

        - out (torch.Tensor): 모델의 최종 출력 텐서로, 마스크된 아이템을 예측한 로짓(logit) 값 (shape: [batch_size, seq_len, num_item+1])
    """
    def __init__(self, config, num_user, num_item):
        super(BERT4Rec, self).__init__()
        parameters = config["parameters"]
        self.hidden_units = parameters["hidden_units"]
        self.num_heads = parameters["num_heads"]
        self.num_layers = parameters["num_layers"]
        self.dropout_rate = parameters["dropout_rate"]
        self.max_len = parameters["max_len"]
        self.device = config["device"]
        self.num_user = num_user
        self.num_item = num_item

        self.item_emb = nn.Embedding(
            num_item + 2, self.hidden_units, padding_idx=0
        )  
        self.pos_emb = nn.Embedding(
            self.max_len, self.hidden_units
        )  # learnable positional encoding
        self.dropout = nn.Dropout(self.dropout_rate)
        self.emb_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-6)

        self.blocks = nn.ModuleList(
            [
                BERT4RecBlock(self.num_heads, self.hidden_units, self.dropout_rate)
                for _ in range(self.num_layers)
            ]
        )
        self.out = nn.Linear(
            self.hidden_units, num_item + 1
        )  

    def forward(self, log_seqs):
        """
        Args:
            log_seqs (numpy.ndarray or torch.Tensor): 아이템 시퀀스 입력 (shape: [batch_size, seq_len])

        Returns:
            torch.Tensor: BERT4Rec 모델의 출력 텐서로, 각 위치별 아이템에 대한 로짓(logit) 값을 담고 있습니다 (shape: [batch_size, seq_len, num_item+1])
        """
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_layernorm(self.dropout(seqs))

        mask = (
            torch.BoolTensor(log_seqs > 0)
            .unsqueeze(1)
            .repeat(1, log_seqs.shape[1], 1)
            .unsqueeze(1)
            .to(self.device)
        )  # mask for zero pad
        for block in self.blocks:
            seqs, attn_dist = block(seqs, mask)
        out = self.out(seqs)
        return out
