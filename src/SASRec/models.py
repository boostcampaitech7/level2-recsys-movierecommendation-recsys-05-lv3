import torch
import torch.nn as nn

from .modules import Encoder, LayerNorm


class S3RecModel(nn.Module):
    """
    연속 추천 시스템을 위한 심층 학습 모델로, 여러 손실 함수를 사용하여 다양한 작업을 수행합니다. 
    이 모델은 AAP(연관된 속성 예측), MIP(마스킹된 아이템 예측), MAP(마스킹된 속성 예측), SP(세그먼트 예측) 등의 작업을 다룹니다.

    Args:
        args: 하이퍼파라미터를 포함하는 네임스페이스 또는 객체. 예: item_size, attribute_size, hidden_size, max_seq_length, hidden_dropout_prob, mask_id, initializer_range.

    Attributes:
        item_embeddings: 아이템을 표현하는 임베딩 레이어.
        attribute_embeddings: 속성을 표현하는 임베딩 레이어.
        position_embeddings: 위치 인코딩을 위한 임베딩 레이어.
        item_encoder: 아이템 시퀀스를 처리하는 인코더 모듈.
        LayerNorm: 시퀀스 임베딩에 대한 레이어 정규화.
        dropout: 과적합을 방지하기 위한 드롭아웃 레이어.
        aap_norm: 연관된 속성 예측(AAP) 결과를 정규화하는 선형 레이어.
        mip_norm: 마스킹된 아이템 예측(MIP) 결과를 정규화하는 선형 레이어.
        map_norm: 마스킹된 속성 예측(MAP) 결과를 정규화하는 선형 레이어.
        sp_norm: 세그먼트 예측(SP) 결과를 정규화하는 선형 레이어.
        criterion: 훈련에 사용되는 손실 함수, BCELoss.
        args: 입력된 하이퍼파라미터를 담고 있는 객체.
    """
    def __init__(self, args):
        """
        S3RecModel 초기화.
        
        Args:
            args: 하이퍼파라미터를 포함하는 네임스페이스 객체.
        """
        super(S3RecModel, self).__init__()
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.attribute_embeddings = nn.Embedding(
            args.attribute_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.map_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # AAP
    def associated_attribute_prediction(self, sequence_output, attribute_embedding):
        """
        연관된 속성 예측(AAP)을 수행합니다.

        Args:
            sequence_output: 아이템 시퀀스 인코더의 출력, 형태는 [B, L, H].
            attribute_embedding: 속성 임베딩, 형태는 [attribute_num, H].

        Returns:
            score: 연관된 속성에 대한 예측 점수, 형태는 [B*L, attribute_num].
        """
        sequence_output = self.aap_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view(
            [-1, self.args.hidden_size, 1]
        )  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item):
        """
        마스킹된 아이템 예측(MIP)을 수행합니다.

        Args:
            sequence_output: 아이템 시퀀스 인코더의 출력, 형태는 [B, L, H].
            target_item: 예측하려는 대상 아이템, 형태는 [B, L, H].

        Returns:
            score: 대상 아이템에 대한 예측 점수, 형태는 [B*L].
        """
        sequence_output = self.mip_norm(
            sequence_output.view([-1, self.args.hidden_size])
        )  # [B*L H]
        target_item = target_item.view([-1, self.args.hidden_size])  # [B*L H]
        score = torch.mul(sequence_output, target_item)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    # MAP
    def masked_attribute_prediction(self, sequence_output, attribute_embedding):
        """
        마스킹된 속성 예측(MAP)을 수행합니다.

        Args:
            sequence_output: 아이템 시퀀스 인코더의 출력, 형태는 [B, L, H].
            attribute_embedding: 속성 임베딩, 형태는 [attribute_num, H].

        Returns:
            score: 마스킹된 속성에 대한 예측 점수, 형태는 [B*L, attribute_num].
        """
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view(
            [-1, self.args.hidden_size, 1]
        )  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # SP sample neg segment
    def segment_prediction(self, context, segment):
        """
        세그먼트 예측(SP)을 수행합니다.

        Args:
            context: 현재 세그먼트의 컨텍스트, 형태는 [B, H].
            segment: 예측하려는 세그먼트, 형태는 [B, H].

        Returns:
            score: 대상 세그먼트에 대한 예측 점수, 형태는 [B].
        """
        context = self.sp_norm(context)
        score = torch.mul(context, segment)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]

    #
    def add_position_embedding(self, sequence):
        """
        아이템 시퀀스에 위치 임베딩을 추가합니다.

        Args:
            sequence: 아이템 ID로 이루어진 입력 시퀀스, 형태는 [B, L].

        Returns:
            sequence_emb: 위치 인코딩이 추가된 아이템 임베딩, 형태는 [B, L, H].
        """
        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def pretrain(
        self,
        attributes,
        masked_item_sequence,
        pos_items,
        neg_items,
        masked_segment_sequence,
        pos_segment,
        neg_segment,
    ):
        """
        주어진 입력에 대해 모델을 사전 훈련합니다.

        Args:
            attributes: 속성 레이블을 나타내는 텐서, 형태는 [B, attribute_size].
            masked_item_sequence: 마스킹된 아이템 시퀀스를 나타내는 텐서, 형태는 [B, L].
            pos_items: 긍정적인 아이템을 나타내는 텐서, 형태는 [B, L].
            neg_items: 부정적인 아이템을 나타내는 텐서, 형태는 [B, L].
            masked_segment_sequence: 마스킹된 세그먼트 시퀀스를 나타내는 텐서, 형태는 [B, L].
            pos_segment: 긍정적인 세그먼트를 나타내는 텐서, 형태는 [B, L].
            neg_segment: 부정적인 세그먼트를 나타내는 텐서, 형태는 [B, L].

        Returns:
            aap_loss: 연관된 속성 예측(AAP)에 대한 손실 값.
            mip_loss: 마스킹된 아이템 예측(MIP)에 대한 손실 값.
            map_loss: 마스킹된 속성 예측(MAP)에 대한 손실 값.
            sp_loss: 세그먼트 예측(SP)에 대한 손실 값.
        """

        sequence_emb = self.add_position_embedding(masked_item_sequence)
        sequence_mask = (masked_item_sequence == 0).float() * -1e8
        sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1)

        encoded_layers = self.item_encoder(
            sequence_emb, sequence_mask, output_all_encoded_layers=True
        )
        # [B L H]
        sequence_output = encoded_layers[-1]

        attribute_embeddings = self.attribute_embeddings.weight
        # AAP
        aap_score = self.associated_attribute_prediction(
            sequence_output, attribute_embeddings
        )
        aap_loss = self.criterion(
            aap_score, attributes.view(-1, self.args.attribute_size).float()
        )
        # only compute loss at non-masked position
        aap_mask = (masked_item_sequence != self.args.mask_id).float() * (
            masked_item_sequence != 0
        ).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(
            mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)
        )
        mip_mask = (masked_item_sequence == self.args.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self.masked_attribute_prediction(
            sequence_output, attribute_embeddings
        )
        map_loss = self.criterion(
            map_score, attributes.view(-1, self.args.attribute_size).float()
        )
        map_mask = (masked_item_sequence == self.args.mask_id).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        segment_context = self.add_position_embedding(masked_segment_sequence)
        segment_mask = (masked_segment_sequence == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.item_encoder(
            segment_context, segment_mask, output_all_encoded_layers=True
        )

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]  # [B H]
        # pos_segment
        pos_segment_emb = self.add_position_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.item_encoder(
            pos_segment_emb, pos_segment_mask, output_all_encoded_layers=True
        )
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.add_position_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.item_encoder(
            neg_segment_emb, neg_segment_mask, output_all_encoded_layers=True
        )
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :]  # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        sp_loss = torch.sum(
            self.criterion(
                sp_distance, torch.ones_like(sp_distance, dtype=torch.float32)
            )
        )

        return aap_loss, mip_loss, map_loss, sp_loss

    # Fine tune
    # same as SASRec
    def finetune(self, input_ids):
        """
        주어진 입력 시퀀스에 대해 모델을 미세 조정합니다.

        Args:
            input_ids: 아이템 ID로 이루어진 입력 시퀀스, 형태는 [B, L].

        Returns:
            sequence_output: 인코딩된 최종 시퀀스 출력, 형태는 [B, L, H].
        """
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """
        모델 파라미터의 가중치를 초기화합니다.

        Args:
            module: 가중치를 초기화할 PyTorch 모듈 (예: nn.Linear, nn.Embedding 등).
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
