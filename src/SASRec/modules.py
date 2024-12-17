import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    """
    레이어 정규화(Layer Normalization)를 수행하는 모듈입니다.
    TensorFlow 스타일로, 분모 안에 epsilon 값을 넣어서 계산합니다.
    
    Args:
        hidden_size (int): 입력 텐서의 차원 크기.
        eps (float, optional): 분모에 더할 작은 값(기본값 1e-12).
    """
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        """
        입력 텐서에 레이어 정규화를 수행합니다.
        
        Args:
            x (Tensor): 입력 텐서, 차원은 [batch_size, hidden_size].
        
        Returns:
            Tensor: 정규화된 텐서.
        """
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """
    아이템 임베딩과 위치 임베딩을 생성하는 모듈입니다.
    
    Args:
        args (Namespace): 하이퍼파라미터를 포함한 네임스페이스 객체.
    """
    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        """
        아이템 시퀀스에 대해 임베딩을 생성하고 위치 임베딩을 추가합니다.
        
        Args:
            input_ids (Tensor): 아이템 ID로 이루어진 입력 시퀀스, 차원은 [batch_size, seq_length].
        
        Returns:
            Tensor: 임베딩이 적용된 시퀀스, 차원은 [batch_size, seq_length, hidden_size].
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    """
    셀프 어텐션(Self Attention) 메커니즘을 구현하는 모듈입니다.
    
    Args:
        args (Namespace): 하이퍼파라미터를 포함한 네임스페이스 객체.
    """
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        """
        어텐션 계산을 위한 텐서의 형태를 변환합니다.
        
        Args:
            x (Tensor): 텐서, 차원은 [batch_size, seq_length, hidden_size].
        
        Returns:
            Tensor: 변환된 텐서, 차원은 [batch_size, num_heads, seq_length, attention_head_size].
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        """
        셀프 어텐션을 수행합니다.
        
        Args:
            input_tensor (Tensor): 입력 텐서, 차원은 [batch_size, seq_length, hidden_size].
            attention_mask (Tensor): 어텐션 마스크, 차원은 [batch_size, 1, 1, seq_length].
        
        Returns:
            Tensor: 어텐션을 적용한 출력, 차원은 [batch_size, seq_length, hidden_size].
        """
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    """
    중간 피드포워드 네트워크를 구현하는 모듈입니다.
    
    Args:
        args (Namespace): 하이퍼파라미터를 포함한 네임스페이스 객체.
    """
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        """
        피드포워드 네트워크를 통해 입력 텐서를 처리합니다.
        
        Args:
            input_tensor (Tensor): 입력 텐서, 차원은 [batch_size, seq_length, hidden_size].
        
        Returns:
            Tensor: 처리된 출력 텐서, 차원은 [batch_size, seq_length, hidden_size].
        """
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    """
    셀프 어텐션과 피드포워드 네트워크를 포함한 하나의 레이어입니다.
    
    Args:
        args (Namespace): 하이퍼파라미터를 포함한 네임스페이스 객체.
    """
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        """
        레이어를 통과시켜 입력 텐서를 처리합니다.
        
        Args:
            hidden_states (Tensor): 입력 텐서, 차원은 [batch_size, seq_length, hidden_size].
            attention_mask (Tensor): 어텐션 마스크, 차원은 [batch_size, seq_length].
        
        Returns:
            Tensor: 처리된 출력 텐서, 차원은 [batch_size, seq_length, hidden_size].
        """
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    """
    여러 개의 레이어로 구성된 인코더입니다.
    
    Args:
        args (Namespace): 하이퍼파라미터를 포함한 네임스페이스 객체.
    """
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(args.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        입력 텐서를 여러 레이어를 통해 처리합니다.
        
        Args:
            hidden_states (Tensor): 입력 텐서, 차원은 [batch_size, seq_length, hidden_size].
            attention_mask (Tensor): 어텐션 마스크, 차원은 [batch_size, seq_length].
            output_all_encoded_layers (bool): 모든 레이어 출력을 반환할지 여부 (기본값 True).
        
        Returns:
            list: 처리된 모든 레이어의 출력 텐서들.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
