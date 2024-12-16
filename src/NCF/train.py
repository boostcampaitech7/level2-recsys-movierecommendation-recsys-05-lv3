import torch
from torch.utils.data import DataLoader
from .model import NCF

def train_model(model, train_loader, epochs, lr, device):
    """
    주어진 데이터셋을 사용하여 모델을 학습하는 함수입니다.

    Args:
        model (torch.nn.Module): 학습할 모델 (NCF 모델).
        train_loader (torch.utils.data.DataLoader): 훈련 데이터를 배치 단위로 제공하는 데이터로더.
        epochs (int): 학습할 에포크 수.
        lr (float): 학습률 (Learning Rate).
        device (torch.device): 모델과 데이터를 배치할 디바이스 (예: 'cuda' 또는 'cpu').

    Returns:
        None: 학습 후 손실값을 출력합니다.
    """
    model.to(device)  # 모델을 지정된 디바이스로 이동
    criterion = torch.nn.BCELoss()  # 이진 교차 엔트로피 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam 최적화 알고리즘

    model.train()  # 모델을 학습 모드로 설정
    for epoch in range(epochs):
        total_loss = 0  # 에포크당 손실값 초기화
        for batch_idx, (user, item, label) in enumerate(train_loader):
            # 배치 단위로 데이터를 가져오고, 디바이스로 이동
            user, item, label = user.to(device), item.to(device), label.to(device).float()

            optimizer.zero_grad()  # 이전 기울기 초기화
            preds = model(user, item)  # 모델 예측값
            loss = criterion(preds, label)  # 손실 계산
            loss.backward()  # 기울기 계산
            optimizer.step()  # 모델 파라미터 업데이트

            total_loss += loss.item()  # 총 손실값에 더하기

        # 에포크마다 평균 손실값 출력
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(train_loader):.4f}", flush=True)
