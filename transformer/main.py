import math
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime, timedelta
import json
import copy


###############################
# 유틸리티 및 공통 함수들
###############################

# JSON 직렬화를 위한 함수
def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


# yfinance를 통해 데이터를 다운로드
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


# Sliding Window 구간 생성 함수
def generate_sliding_windows(start_date, final_date, train_window, val_window, step):
    """
    start_date: 전체 데이터의 시작 날짜 (datetime)
    final_date: 전체 데이터의 종료 날짜 (datetime)
    train_window: 학습 기간 (예: timedelta(days=365*2) -> 2년)
    val_window: 검증 기간 (예: timedelta(days=180) -> 6개월)
    step: 윈도우 이동 간격 (예: timedelta(days=180) -> 6개월)
    """
    windows = []
    current_train_start = start_date
    while True:
        current_train_end = current_train_start + train_window - timedelta(days=1)
        validation_start = current_train_end + timedelta(days=1)
        validation_end = validation_start + val_window - timedelta(days=1)
        if validation_end > final_date:
            break
        windows.append({
            "train_start": current_train_start,
            "train_end": current_train_end,
            "val_start": validation_start,
            "val_end": validation_end
        })
        current_train_start += step
    return windows


###############################
# Dataset 및 모델 정의
###############################

class StockDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        dataset_length = len(self.data) - self.seq_len - self.pred_len + 1
        return max(0, dataset_length)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


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
        kernel_size = configs.kernel_size  # kernel_size를 configs에서 전달받음
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in  # 일반적으로 1
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                lin_s = nn.Linear(self.seq_len, self.pred_len)
                lin_s.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Seasonal.append(lin_s)
                lin_t = nn.Linear(self.seq_len, self.pred_len)
                lin_t.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend.append(lin_t)
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
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
        return x.permute(0, 2, 1)  # [Batch, pred_len, Channel]


class Configs:
    def __init__(self, seq_len, pred_len, individual, enc_in, kernel_size):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in
        self.kernel_size = kernel_size


###############################
# 평가 및 재학습 함수
###############################
def evaluate_and_retrain(model, dataloader, criterion, optimizer, train_dataloader, num_epochs, device, ticker,
                         retrain_threshold=0.05):
    if len(dataloader) == 0:
        print("테스트 데이터셋이 충분하지 않습니다. 평가를 건너뜁니다.")
        return np.array([]), np.array([])
    model.eval()
    total_loss = 0.0
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            predictions.append(output.cpu().numpy())
            ground_truth.append(y.cpu().numpy())
    print(f"Test Loss: {total_loss / len(dataloader):.4f}")
    predictions = np.concatenate(predictions, axis=0)  # (num_samples, pred_len, 1)
    ground_truth = np.concatenate(ground_truth, axis=0)
    prediction_diff = np.abs(predictions - ground_truth)
    max_diff = np.max(prediction_diff)
    if max_diff > retrain_threshold:
        print(f"Prediction error is too large ({max_diff:.4f}), retraining model...")
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Retrain Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")
        torch.save(model.state_dict(), f"{ticker}_model_retrained.pth")
        print(f"✅ 모델 재학습 완료! {ticker}_model_retrained.pth 저장됨.")
    return predictions, ground_truth


###############################
# JSON 저장 함수들
###############################

# Output File 1: 모든 히스토리 데이터 (차트용: open, high, low, close, volume)
def save_json_file1(filename, raw_df):
    json_data = {"ticker": None, "data": {}}
    for date in raw_df.index.strftime("%Y-%m-%d"):
        row = raw_df.loc[pd.to_datetime(date)]
        hist = {
            "open": round(float(row["Open"].iloc[0]) if hasattr(row["Open"], "iloc") else float(row["Open"]), 2),
            "high": round(float(row["High"].iloc[0]) if hasattr(row["High"], "iloc") else float(row["High"]), 2),
            "low": round(float(row["Low"].iloc[0]) if hasattr(row["Low"], "iloc") else float(row["Low"]), 2),
            "close": round(float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"]), 2),
            "volume": int(row["Volume"].iloc[0]) if hasattr(row["Volume"], "iloc") else int(row["Volume"])
        }
        json_data["data"][date] = hist
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, default=convert_to_serializable)
    print(f"✅ JSON 파일 저장 완료: {filename}")


# Output File 2: 예측 데이터
def save_json_file2(filename, dates, predictions, raw_df, shares_outstanding):
    json_data = {"ticker": None, "data": {}}
    for i, date in enumerate(dates):
        dt = pd.to_datetime(date)
        if dt in raw_df.index:
            row = raw_df.loc[dt]
            close_val = round(float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"]), 2)
            market_cap = int(round(close_val * shares_outstanding)) if shares_outstanding else None
            prev_data = raw_df.loc[raw_df.index < dt]
            if not prev_data.empty:
                prev_close = float(prev_data.iloc[-1]["Close"].iloc[0]) if hasattr(prev_data.iloc[-1]["Close"],
                                                                                   "iloc") else float(
                    prev_data.iloc[-1]["Close"])
                actual_percentage = round(((close_val - prev_close) / prev_close) * 100, 2)
                prediction_val = round(float(predictions[i]), 2)
                predict_percentage = round(((prediction_val - prev_close) / prev_close) * 100, 2)
            else:
                actual_percentage = None
                predict_percentage = None
            json_data["data"][date] = {
                "prediction": round(float(predictions[i]), 2),
                "actual_percentage": actual_percentage,
                "predict_percentage": predict_percentage,
                "market_capitalization": market_cap,
                "close": close_val
            }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, default=convert_to_serializable)
    print(f"✅ JSON 파일 저장 완료: {filename}")


# Output File 3: stocks.json에 티커 리스트 저장
def save_stocks_file(filename, tickers):
    json_data = {"stocks": tickers}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)
    print(f"✅ Stocks JSON 파일 저장 완료: {filename}")


###############################
# 전체 파이프라인 (티커별 처리)
###############################
def process_ticker(ticker, start_date, end_date):
    print(f"\n========== Processing {ticker} ==========")
    # 1. 데이터 다운로드
    raw_data = fetch_data(ticker, start_date, end_date)
    if raw_data.empty:
        print(f"{ticker} 데이터가 없습니다.")
        return
    # 2. 발행 주식수 및 전처리
    stock_obj = yf.Ticker(ticker)
    shares_outstanding = stock_obj.info.get("sharesOutstanding", 0)
    processed_data = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    processed_data.dropna(inplace=True)
    features_for_model = ['Close']
    X = processed_data[features_for_model].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # 3. Sliding Window 설정 및 모델 학습/검증
    seq_len = 5
    pred_len = 1
    batch_size = 32
    num_epochs = 50  # sliding window 단계별 학습 에폭 수
    kernel_size = 7
    a = 0.5  # 이전 데이터 반영 정도 (0 < a <= 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 전체 데이터의 기간 (start_date부터 processed_data의 마지막 날짜까지)
    overall_start = pd.to_datetime(start_date)
    overall_end = processed_data.index[-1]
    train_window = timedelta(days=60)  # 2년 학습
    val_window = timedelta(days=15)  # 6개월 검증
    step = timedelta(days=60)  # 6개월 단위 이동

    windows = generate_sliding_windows(overall_start, overall_end, train_window, val_window, step)
    print(f"총 {len(windows)}개의 sliding window가 생성되었습니다.")

    mse_list = []
    # 모델 초기화 (슬라이딩 윈도우 단계간 연속 학습)
    configs = Configs(seq_len, pred_len, individual=False, enc_in=len(features_for_model), kernel_size=kernel_size)
    model = Model(configs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    prev_state = copy.deepcopy(model.state_dict())

    for i, window in enumerate(windows):
        print(f"\n--- Sliding Window Step {i + 1} ---")
        print(f"Train: {window['train_start'].date()} ~ {window['train_end'].date()}")
        print(f"Validation: {window['val_start'].date()} ~ {window['val_end'].date()}")

        # 날짜 필터링을 통해 해당 윈도우의 데이터 슬라이싱
        train_mask = (processed_data.index >= window["train_start"]) & (processed_data.index <= window["train_end"])
        train_data = X_scaled[train_mask]
        val_mask = (processed_data.index >= window["val_start"]) & (processed_data.index <= window["val_end"])
        val_data = X_scaled[val_mask]

        if len(train_data) < seq_len + pred_len or len(val_data) < seq_len + pred_len:
            print("데이터 부족으로 이 윈도우는 건너뜁니다.")
            continue

        train_dataset = StockDataset(train_data, seq_len, pred_len)
        val_dataset = StockDataset(val_data, seq_len, pred_len)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 해당 윈도우에서 모델 학습
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Step {i + 1} Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")

        # 검증 데이터셋에 대한 평가 (MSE 계산)
        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                count += 1
        val_mse = val_loss / count if count > 0 else None
        mse_list.append(val_mse)
        print(f"Step {i + 1} - Validation MSE: {val_mse:.4f}")

        # 이전 모델 가중치와 현재 가중치를 a 비율로 혼합하여 업데이트
        current_state = model.state_dict()
        for key in current_state:
            current_state[key] = a * current_state[key] + (1 - a) * prev_state[key]
        model.load_state_dict(current_state)
        prev_state = copy.deepcopy(model.state_dict())

    print("\nSliding Window Validation 완료. 각 스텝의 MSE:", mse_list)

    # 4. 최종 예측: 2025-01-01부터 예측 (원래 코드와 유사)
    target_date = pd.to_datetime("2025-01-01")
    if target_date in processed_data.index:
        target_loc = processed_data.index.get_loc(target_date)
    else:
        target_loc = processed_data.index.get_indexer([target_date], method='nearest')[0]

    # 예측을 위해 target_date - seq_len부터 사용
    test_data = X_scaled[target_loc - seq_len:]
    test_index = processed_data.index[target_loc - seq_len:]

    test_dataset = StockDataset(test_data, seq_len, pred_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            output = model(x)
            predictions.append(output.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0).squeeze()  # (num_samples, pred_len)
    test_predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()

    # 예측 날짜 설정 (test_index[seq_len:]부터 2025-01-01에 해당하도록)
    pred_dates = test_index[seq_len: seq_len + len(test_dataset)]
    pred_dates = pred_dates.strftime("%Y-%m-%d").tolist()
    today = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))
    yesterday = today - pd.Timedelta(days=1)
    filtered_dates = []
    filtered_predictions = []
    for date, pred in zip(pred_dates, test_predictions_original):
        dt = pd.to_datetime(date)
        if dt <= yesterday:
            filtered_dates.append(date)
            filtered_predictions.append(pred)
    pred_dates = filtered_dates
    final_predictions = np.array(filtered_predictions)

    # 5. JSON 파일 저장
    output_file1 = f"{ticker}_data.json"
    output_file2 = f"{ticker}_prediction_29.json"
    save_json_file1(output_file1, processed_data)
    save_json_file2(output_file2, pred_dates, final_predictions, processed_data, shares_outstanding)


###############################
# 전체 처리: 여러 티커에 대해 실행
###############################
tickers = ["AAPL", "NVDA", "TSLA"]
for t in tickers:
    process_ticker(t, "2020-01-01", (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"))

save_stocks_file("stocks.json", tickers)
