import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam

from .utils import ndcg_k, recall_at_k


class Trainer:
    """
    모델의 훈련, 검증, 테스트, 제출 과정을 처리하는 Trainer 클래스입니다.
    이 클래스는 모델의 저장 및 로딩, 평가, recall과 NDCG와 같은 메트릭 계산을 쉽게 할 수 있게 도와줍니다.

    Attributes:
        model (nn.Module): 훈련 및 평가할 모델.
        train_dataloader (DataLoader): 훈련 데이터를 위한 DataLoader.
        eval_dataloader (DataLoader): 검증 데이터를 위한 DataLoader.
        test_dataloader (DataLoader): 테스트 데이터를 위한 DataLoader.
        submission_dataloader (DataLoader): 제출 데이터를 위한 DataLoader.
        args (Namespace): 하이퍼파라미터와 설정을 포함한 명령행 인자.
        cuda_condition (bool): CUDA가 사용 가능한지 여부를 나타내는 플래그.
        device (torch.device): 모델이 배치될 장치 (CPU 또는 CUDA).
        optim (torch.optim.Adam): 모델 훈련을 위한 Adam 옵티마이저.
        criterion (nn.BCELoss): 이진 크로스 엔트로피 손실 함수.
    
    Methods:
        train(epoch):
            주어진 epoch에 대해 훈련을 한 번 수행합니다.
        
        valid(epoch):
            주어진 epoch에 대해 검증을 한 번 수행합니다.

        test(epoch):
            주어진 epoch에 대해 테스트를 한 번 수행합니다.

        submission(epoch):
            주어진 epoch에 대해 제출을 한 번 수행합니다.

        iteration(epoch, dataloader, mode="train"):
            반복 로직을 정의하는 추상 메서드. 서브클래스에서 구현해야 합니다.

        get_full_sort_score(epoch, answers, pred_list):
            주어진 answers와 pred_list에 대해 recall과 NDCG@k (k=5, 10)을 계산하고 출력합니다.

        save(file_name):
            모델의 현재 상태를 파일에 저장합니다.

        load(file_name):
            파일에서 모델의 가중치를 로드합니다.

        cross_entropy(seq_out, pos_ids, neg_ids):
            시퀀스 출력에 대한 긍정적 및 부정적 아이템 임베딩 간의 크로스 엔트로피 손실을 계산합니다.

        predict_full(seq_out):
            시퀀스 출력을 사용하여 모든 아이템에 대한 예측 평점을 계산합니다.

    """
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        """
        Trainer를 초기화하는 생성자입니다. 모델, 데이터로더, 하이퍼파라미터 등을 설정합니다.

        Args:
            model (nn.Module): 훈련 및 평가할 모델.
            train_dataloader (DataLoader): 훈련 데이터를 위한 DataLoader.
            eval_dataloader (DataLoader): 검증 데이터를 위한 DataLoader.
            test_dataloader (DataLoader): 테스트 데이터를 위한 DataLoader.
            submission_dataloader (DataLoader): 제출 데이터를 위한 DataLoader.
            args (Namespace): 하이퍼파라미터와 설정을 포함한 명령행 인자.
        """

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        """
        주어진 epoch에 대해 훈련을 한 번 수행합니다.

        Args:
            epoch (int): 현재 epoch 번호.
        """
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        """
        주어진 epoch에 대해 검증을 한 번 수행합니다.

        Args:
            epoch (int): 현재 epoch 번호.

        Returns:
            output: 검증 반복의 결과.
        """
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        """
        주어진 epoch에 대해 테스트를 한 번 수행합니다.

        Args:
            epoch (int): 현재 epoch 번호.

        Returns:
            output: 테스트 반복의 결과.
        """
        return self.iteration(epoch, self.test_dataloader, mode="test")

    def submission(self, epoch):
        """
        주어진 epoch에 대해 제출을 한 번 수행합니다.

        Args:
            epoch (int): 현재 epoch 번호.

        Returns:
            output: 제출 반복의 결과.
        """
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        """
        반복 로직을 정의하는 추상 메서드입니다. 서브클래스에서 구현해야 합니다.

        Args:
            epoch (int): 현재 epoch 번호.
            dataloader (DataLoader): 반복에 사용할 데이터로더.
            mode (str): 작업 모드 ("train", "valid", "test", 또는 "submission").

        Raises:
            NotImplementedError: 서브클래스에서 구현되지 않은 경우.
        """
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        """
        주어진 answers와 pred_list에 대해 recall과 NDCG@k (k=5, 10)을 계산하고 출력합니다.

        Args:
            epoch (int): 현재 epoch 번호.
            answers (Tensor): 평가를 위한 실제 값.
            pred_list (Tensor): 평가를 위한 예측 값.

        Returns:
            tuple: recall과 NDCG 값 리스트와 해당 값들의 문자열 표현.
        """
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch + 1,
            "RECALL@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "RECALL@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
        }
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def save(self, file_name):
        """
        모델의 현재 상태를 파일에 저장합니다.

        Args:
            file_name (str): 모델을 저장할 파일 경로.
        """
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        """
        파일에서 모델의 가중치를 로드합니다.

        Args:
            file_name (str): 모델을 로드할 파일 경로.
        """
        self.model.load_state_dict(torch.load(file_name, weights_only=True))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        """
        시퀀스 출력에 대한 긍정적 및 부정적 아이템 임베딩 간의 크로스 엔트로피 손실을 계산합니다.

        Args:
            seq_out (Tensor): 모델의 출력 시퀀스.
            pos_ids (Tensor): 긍정적 아이템의 인덱스.
            neg_ids (Tensor): 부정적 아이템의 인덱스.

        Returns:
            Tensor: 계산된 손실 값.
        """
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        )  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_full(self, seq_out):
        """
        시퀀스 출력을 사용하여 모든 아이템에 대한 예측 평점을 계산합니다.

        Args:
            seq_out (Tensor): 모델의 출력 시퀀스.

        Returns:
            Tensor: 모든 아이템에 대한 예측된 평점.
        """
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class PretrainTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def pretrain(self, epoch, pretrain_dataloader):

        desc = (
            f"AAP-{self.args.aap_weight}-"
            f"MIP-{self.args.mip_weight}-"
            f"MAP-{self.args.map_weight}-"
            f"SP-{self.args.sp_weight}"
        )

        pretrain_data_iter = tqdm.tqdm(
            enumerate(pretrain_dataloader),
            desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch + 1}",
            total=len(pretrain_dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        self.model.train()
        aap_loss_avg = 0.0
        mip_loss_avg = 0.0
        map_loss_avg = 0.0
        sp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            (
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            ) = batch

            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            )

            joint_loss = (
                self.args.aap_weight * aap_loss
                + self.args.mip_weight * mip_loss
                + self.args.map_weight * map_loss
                + self.args.sp_weight * sp_loss
            )

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            aap_loss_avg += aap_loss.item()
            mip_loss_avg += mip_loss.item()
            map_loss_avg += map_loss.item()
            sp_loss_avg += sp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        losses = {
            "epoch": epoch + 1,
            "aap_loss_avg": aap_loss_avg / num,
            "mip_loss_avg": mip_loss_avg / num,
            "map_loss_avg": map_loss_avg / num,
            "sp_loss_avg": sp_loss_avg / num,
        }
        print(desc)
        print(str(losses))
        return losses


class FinetuneTrainer(Trainer):
    """
    모델을 미세 조정(finetuning)하는 훈련을 담당하는 클래스입니다.
    `Trainer` 클래스를 상속받아, 모델 훈련, 검증, 테스트 및 제출 과정에서 필요한 세부 로직을 구현합니다.
    주로 추천 시스템에서 시퀀스 데이터를 기반으로 훈련 및 평가를 수행합니다.

    Attributes:
        model (nn.Module): 훈련 및 평가할 모델.
        train_dataloader (DataLoader): 훈련 데이터를 위한 DataLoader.
        eval_dataloader (DataLoader): 검증 데이터를 위한 DataLoader.
        test_dataloader (DataLoader): 테스트 데이터를 위한 DataLoader.
        submission_dataloader (DataLoader): 제출 데이터를 위한 DataLoader.
        args (Namespace): 하이퍼파라미터와 설정을 포함한 명령행 인자.
    
    Methods:
        iteration(epoch, dataloader, mode="train"):
            주어진 epoch에 대해 훈련, 검증, 또는 제출을 위한 데이터를 반복하고 처리하는 함수.

    """
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        """
        FinetuneTrainer를 초기화하는 생성자입니다. `Trainer` 클래스를 상속받아 모델과 데이터로더를 설정합니다.

        Args:
            model (nn.Module): 훈련 및 평가할 모델.
            train_dataloader (DataLoader): 훈련 데이터를 위한 DataLoader.
            eval_dataloader (DataLoader): 검증 데이터를 위한 DataLoader.
            test_dataloader (DataLoader): 테스트 데이터를 위한 DataLoader.
            submission_dataloader (DataLoader): 제출 데이터를 위한 DataLoader.
            args (Namespace): 하이퍼파라미터와 설정을 포함한 명령행 인자.
        """
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):
        """
        주어진 epoch에 대해 훈련, 검증, 또는 제출을 위한 데이터를 반복하고 처리하는 함수입니다.
        훈련 모드에서는 손실을 계산하고 역전파하여 가중치를 업데이트합니다. 검증 및 제출 모드에서는 
        모델 예측 결과를 계산하고 평가 메트릭을 제공합니다.

        Args:
            epoch (int): 현재 epoch 번호.
            dataloader (DataLoader): 데이터를 반복할 DataLoader.
            mode (str): 작업 모드 ("train", "valid", "test", 또는 "submission").

        Returns:
            If mode == "submission": 
                - 예측된 추천 리스트를 반환합니다.
            Otherwise:
                - recall과 NDCG 메트릭을 계산한 후 반환합니다.
        """
        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch + 1),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )
        if mode == "train":
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output = self.model.finetune(input_ids)
                loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch + 1,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": "{:.4f}".format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()

            pred_list = None
            answer_list = None
            for i, batch in rec_data_iter:

                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, _, target_neg, answers = batch
                recommend_output = self.model.finetune(input_ids)

                recommend_output = recommend_output[:, -1, :]

                rating_pred = self.predict_full(recommend_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                ind = np.argpartition(rating_pred, -10)[:, -10:]

                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

                batch_pred_list = ind[
                    np.arange(len(rating_pred))[:, None], arr_ind_argsort
                ]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(
                        answer_list, answers.cpu().data.numpy(), axis=0
                    )

            if mode == "submission":
                return pred_list
            else:
                return self.get_full_sort_score(epoch, answer_list, pred_list)
