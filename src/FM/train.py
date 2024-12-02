from tqdm import tqdm

# Model Train
def train(model, epochs, dataloader, criterion, optimizer, device, valid_ratio):

    for epoch in range(epochs):
        model.train() # 모델 학습 모드로 변경
        total_loss, train_len = 0, len(dataloader['train_dataloader'])

        for data in tqdm(dataloader['train_dataloader'], desc=f'[Epoch {epoch+1:02d}/{epochs:02d}]'):
            x, y = data["features"].to(device).float(), data["rating"].to(device).float()
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        msg = ''
        train_loss = total_loss / train_len
        msg += f'\tTrain Loss : {train_loss:.3f}'

        if valid_ratio != 0:  # valid 데이터가 존재할 경우
            valid_loss = valid(model, dataloader['valid_dataloader'], criterion, device)
            msg += f'\n\tValid Loss : {valid_loss:.3f}'
            print(msg)
        else:  # valid 데이터가 없을 경우
            print(msg)
        
    return model


def valid(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    for data in dataloader:
        x, y = data["features"].to(device).float(), data["rating"].to(device).float()
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()

    return total_loss / len(dataloader)
