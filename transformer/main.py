import math
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import csv




# 1. 데이터 다운로드
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# 애플(AAPL) 데이터 다운로드
ticker = "NVDA"
start_date = "2022-01-02"
end_date = "2025-01-02"
raw_data = fetch_data(ticker, start_date, end_date)

# 2. 필요한 열 선택
processed_data = raw_data[['Close', 'Volume']].copy()

# 특성 생성
processed_data.loc[:, 'SMA_20'] = processed_data['Close'].rolling(window=20).mean()  # 20일 단기 이동평균
processed_data.loc[:, 'SMA_50'] = processed_data['Close'].rolling(window=50).mean()  # 50일 이동평균
processed_data.loc[:, 'EMA_20'] = processed_data['Close'].ewm(span=20, adjust=False).mean()  # 20일 지수이동평균
processed_data.loc[:, 'RSI'] = 100 - (100 / (1 + ((processed_data['Close'].diff(1).gt(0).rolling(window=14).sum()) /
                                 (processed_data['Close'].diff(1).lt(0).rolling(window=14).sum()))))  # RSI 계산
processed_data.loc[:, 'MACD'] = processed_data['Close'].ewm(span=12, adjust=False).mean() - processed_data['Close'].ewm(span=26, adjust=False).mean()  # MACD
processed_data.loc[:, 'MACD_signal'] = processed_data['MACD'].ewm(span=9, adjust=False).mean()  # MACD Signal Line


# 볼린저 밴드
rolling_std = processed_data['Close'].rolling(window=20).std()
rolling_std = rolling_std.squeeze()

processed_data['BB_upper'] = processed_data['SMA_20'] + (rolling_std * 2)
processed_data['BB_lower'] = processed_data['SMA_20'] - (rolling_std * 2)
processed_data['volume'] = processed_data['Volume']

# 3. 결측치 처리
processed_data.dropna(inplace=True)

# 4. 특성 데이터 준비 (X)
features = ['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Volume']
X = processed_data[features].values

# 정규화 (MinMaxScaler)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# 5. Dataset 클래스 정의
class StockDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Dataset 생성
seq_len = 60  # 과거 60일치 데이터를 입력으로
pred_len = 5  # 앞으로 5일치 데이터를 예측
dataset = StockDataset(X_scaled, seq_len, pred_len)

# DataLoader 생성
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 1. 데이터를 훈련 데이터와 테스트 데이터로 나누기
train_size = 0.8  # 80%를 훈련 데이터로, 20%를 테스트 데이터로
train_data, test_data = train_test_split(X_scaled, train_size=train_size, shuffle=False)

# 2. 훈련용 및 테스트용 데이터셋 생성
train_dataset = StockDataset(train_data, seq_len, pred_len)
test_dataset = StockDataset(test_data, seq_len, pred_len)

# 3. DataLoader 생성
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # 데이터 확인
# for x, y in dataloader:
#     print(f"X shape: {x.shape}, Y shape: {y.shape}")
#     break


# 6. 모델 정의
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Decoder = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


# 7. Configs 클래스 설정
class Configs:
    seq_len = seq_len  # 입력 길이
    pred_len = pred_len  # 예측 길이
    individual = False  # False: 채널 공통 모델, True: 채널별 모델
    enc_in = len(features)  # 동적으로 설정된 입력 채널 수

# 모델 초기화
configs = Configs()
model = Model(configs)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# 손실 함수 및 옵티마이저
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 8. 학습 루프
num_epochs = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델을 훈련 모드로 설정
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for x, y in train_dataloader:  # train_dataloader로 훈련 데이터 사용
        x, y = x.to(device), y.to(device)

        # Forward pass
        output = model(x)

        # 손실 계산
        loss = criterion(output, y)
        epoch_loss += loss.item()

        # Backward pass 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_dataloader):.4f}")

# 9. 평가
# 모델을 평가 모드로 설정
model.eval()
test_loss = 0.0
with torch.no_grad():
    for x, y in test_dataloader:  # test_dataloader로 테스트 데이터 사용
        x, y = x.to(device), y.to(device)

        # Forward pass
        predictions = model(x)

        # 손실 계산
        loss = criterion(predictions, y)
        test_loss += loss.item()

        # 출력 (예: 예측 값과 실제 값 출력)
        print(f"Predictions: {predictions}")
        print(f"Ground Truth: {y}")


        # 3D → 2D 변환: (batch_size, pred_len, feature_dim) → (batch_size * pred_len, feature_dim)
        predictions_2d = predictions.cpu().numpy().reshape(-1, len(features))
        ground_truth_2d = y.cpu().numpy().reshape(-1, len(features))

        # MinMaxScaler 역변환 (정규화 해제)
        predicted_prices = scaler.inverse_transform(predictions_2d)
        ground_truth_prices = scaler.inverse_transform(ground_truth_2d)

        # 변환된 데이터를 다시 원래 3D로 재구성
        predicted_prices = predicted_prices.reshape(-1, pred_len, len(features))
        ground_truth_prices = ground_truth_prices.reshape(-1, pred_len, len(features))

        print(f"Predicted Prices: {predicted_prices}")
        print(f"Actual Prices: {ground_truth_prices}")


# 테스트 손실 출력
print(f"Test Loss: {test_loss / len(test_dataloader):.4f}")



# CSV 파일 저장 경로
csv_filename = "predictions_vs_ground_truth2.csv"

# CSV 파일 헤더 생성
headers = ["Day", "Type"] + features  # 컬럼명: 날짜 + (예측값 / 실제값) + 피처명

# CSV 파일 생성
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # 헤더 작성

    # 모델 평가 (test_dataloader 사용)
    model.eval()
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            predictions = model(x)

            # 3D → 2D 변환
            predictions_2d = predictions.cpu().numpy().reshape(-1, len(features))
            ground_truth_2d = y.cpu().numpy().reshape(-1, len(features))

            # MinMaxScaler 역변환 (정규화 해제)
            predicted_prices = scaler.inverse_transform(predictions_2d)
            ground_truth_prices = scaler.inverse_transform(ground_truth_2d)

            # 저장할 데이터 리스트
            data_to_write = []

            for day in range(pred_len):  # 예측 기간(5일)
                pred_row = [day + 1, "Predicted"] + list(predicted_prices[day])
                gt_row = [day + 1, "Actual"] + list(ground_truth_prices[day])
                data_to_write.append(pred_row)
                data_to_write.append(gt_row)

            # CSV 파일에 저장
            writer.writerows(data_to_write)

print(f"CSV 파일 저장 완료: {csv_filename}")
