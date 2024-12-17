from tqdm import tqdm

# Model Train
def train(model, epochs, dataloader, criterion, optimizer, device, valid_ratio):
    """
    모델을 학습하는 함수. 주어진 에포크 수만큼 훈련 데이터를 이용해 모델을 학습하고, 
    검증 데이터가 있을 경우 검증을 수행합니다.

    Args:
        model (torch.nn.Module): 학습할 PyTorch 모델.
        epochs (int): 학습할 에포크 수.
        dataloader (dict): 훈련 및 검증 데이터가 포함된 DataLoader 객체를 담고 있는 딕셔너리.
        criterion (torch.nn.Module): 손실 함수 (예: MSELoss, CrossEntropyLoss 등).
        optimizer (torch.optim.Optimizer): 모델 파라미터를 업데이트할 최적화 알고리즘 (예: Adam, SGD 등).
        device (torch.device): 학습에 사용할 장치 ('cpu' 또는 'cuda').
        valid_ratio (float): 검증 데이터 비율. 검증 데이터를 사용할지 여부를 결정합니다.

    Returns:
        model (torch.nn.Module): 학습이 완료된 모델.
    """
    for epoch in range(epochs):
        model.train()  # 모델 학습 모드로 변경
        total_loss, train_len = 0, len(dataloader['train_dataloader'])

        # 훈련 데이터로 모델 학습
        for data in tqdm(dataloader['train_dataloader'], desc=f'[Epoch {epoch+1:02d}/{epochs:02d}]'):
            x, y = data["features"].to(device).float(), data["rating"].to(device).float()
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 훈련 손실 출력
        msg = ''
        train_loss = total_loss / train_len
        msg += f'\tTrain Loss : {train_loss:.3f}'

        # 검증 데이터가 있을 경우 검증 수행
        if valid_ratio != 0:  
            valid_loss = valid(model, dataloader['valid_dataloader'], criterion, device)
            msg += f'\n\tValid Loss : {valid_loss:.3f}'
            print(msg)
        else:  # 검증 데이터가 없을 경우
            print(msg)
        
    return model


def valid(model, dataloader, criterion, device):
    """
    모델을 검증하는 함수. 검증 데이터셋에 대해 예측을 수행하고, 
    손실을 계산하여 반환합니다.

    Args:
        model (torch.nn.Module): 검증할 PyTorch 모델.
        dataloader (DataLoader): 검증 데이터에 대한 DataLoader 객체.
        criterion (torch.nn.Module): 손실 함수 (예: MSELoss, CrossEntropyLoss 등).
        device (torch.device): 검증에 사용할 장치 ('cpu' 또는 'cuda').

    Returns:
        float: 검증 데이터셋에 대한 평균 손실 값.
    """
    model.eval()  # 모델 평가 모드로 변경
    total_loss = 0

    # 검증 데이터로 손실 계산
    for data in dataloader:
        x, y = data["features"].to(device).float(), data["rating"].to(device).float()
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()

    # 검증 손실 반환
    return total_loss / len(dataloader)
