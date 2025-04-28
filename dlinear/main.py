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
import os

os.makedirs('results', exist_ok=True)

# 재현성을 위한 시드 설정
torch.manual_seed(42)
np.random.seed(42)

# 필요 시, multi mode에서는 기술적 지표 계산을 위해 ta 라이브러리를 사용합니다.
try:
    import ta
except ImportError:
    print("Multi mode는 'ta' 라이브러리가 필요합니다. pip install ta 로 설치하세요.")

###############################
# 유틸리티 및 공통 함수들
###############################

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def generate_sliding_windows(start_date, final_date, train_window, val_window, step):
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
    def __init__(self, data, seq_len, pred_len, target_idx=0):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_idx = target_idx

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)


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
        return x.permute(0, 2, 1)


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
        kernel_size = configs.kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
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
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], device=seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], device=trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
            x_out = seasonal_output + trend_output
        else:
            x_out = self.Linear_Seasonal(seasonal_init) + self.Linear_Trend(trend_init)

        # 종가(첫 피처)만 선택
        return x_out[:, 0, :].unsqueeze(-1)


class Configs:
    def __init__(self, seq_len, pred_len, individual, enc_in, kernel_size):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in
        self.kernel_size = kernel_size

###############################
# JSON 저장 함수들
###############################

def save_json_file1(filename, raw_df, ticker):
    json_data = {"ticker": ticker, "data": {}}
    for date in raw_df.index.strftime("%Y-%m-%d"):
        row = raw_df.loc[pd.to_datetime(date)]
        hist = {
            "open": round(float(row["Open"].iloc[0]), 2),
            "high": round(float(row["High"].iloc[0]), 2),
            "low": round(float(row["Low"].iloc[0]), 2),
            "close": round(float(row["Close"].iloc[0]), 2),
            "volume": int(row["Volume"].iloc[0])
        }
        json_data["data"][date] = hist
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, default=convert_to_serializable)
    print(f"JSON 파일 저장 완료: {filename}")


def save_json_file2(filename, dates, predictions, raw_df, shares_outstanding, ticker):
    json_data = {"ticker": ticker, "data": {}}
    for i, date in enumerate(dates):
        dt = pd.to_datetime(date)
        row = raw_df.loc[dt]
        close_val = round(float(row["Close"].iloc[0]), 2)
        market_cap = int(round(close_val * shares_outstanding)) if shares_outstanding else None
        prev_data = raw_df.loc[raw_df.index < dt]
        pred_val = round(float(predictions[i]), 2)
        if not prev_data.empty:

            prev_close = prev_data.iloc[-1]["Close"].item()
            actual_pct = round(((close_val - prev_close) / prev_close) * 100, 2)

            pred_pct = round(((pred_val - prev_close) / prev_close) * 100, 2)

        else:
            actual_pct = None
            pred_pct = None
        json_data["data"][date] = {
            "prediction": pred_val,
            "actual_percentage": actual_pct,
            "predict_percentage": pred_pct,
            "market_capitalization": market_cap,
            "close": close_val
        }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, default=convert_to_serializable)
    print(f"JSON 파일 저장 완료: {filename}")


def save_stocks_file(filename, tickers):
    json_data = {"stocks": tickers}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)
    print(f"Stocks JSON 파일 저장 완료: {filename}")

###############################
# 전체 파이프라인 (티커별 처리)
###############################

def process_ticker(ticker, start_date, end_date, mode="single"):
    print(f"\n========== Processing {ticker} ==========")
    raw_data = fetch_data(ticker, start_date, end_date)
    if raw_data.empty:
        print(f"{ticker} 데이터가 없습니다.")
        return

    # 1) 원본 데이터 전처리
    processed_data = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()
    if mode == "multi":
        close_series = processed_data['Close'].squeeze()
        processed_data['RSI'] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
        macd = ta.trend.MACD(close_series)
        processed_data['MACD'] = macd.macd()
        processed_data['High_Close_Ratio'] = processed_data['High'] / processed_data['Close']
        processed_data['Low_Close_Ratio'] = processed_data['Low'] / processed_data['Close']
        processed_data.fillna(method='bfill', inplace=True)
        features = ['Close', 'RSI', 'MACD', 'High_Close_Ratio', 'Low_Close_Ratio']
    else:
        features = ['Close']

    # 2) 학습/검증용 데이터 분리 (2024-12-31까지만)
    limit_date = pd.to_datetime("2024-12-31")
    train_val_data = processed_data.loc[processed_data.index <= limit_date]

    # 3) 전체 데이터 스케일링 (train/val + test)
    X_full = processed_data[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled_full = scaler.fit_transform(X_full)

    # 종가 역변환용 scaler
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(processed_data[['Close']].values)

    # 4) train/val 전처리
    X_scaled_train_val = X_scaled_full[:len(train_val_data)]
    dates_train_val = train_val_data.index

    # 5) 슬라이딩 윈도우 생성
    overall_start = pd.to_datetime(start_date)
    overall_end = train_val_data.index[-1]
    train_window = timedelta(days=60)
    val_window = timedelta(days=15)
    step = timedelta(days=30)
    windows = generate_sliding_windows(overall_start, overall_end, train_window, val_window, step)
    print(f"총 {len(windows)}개의 sliding window가 생성되었습니다.")

    # 6) 모델 및 학습 설정
    seq_len = 5
    pred_len = 1
    batch_size = 32
    num_epochs = 50
    kernel_size = 7
    a = 0.7
    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = Configs(seq_len, pred_len, individual=False, enc_in=len(features), kernel_size=kernel_size)
    model = Model(configs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    prev_state = copy.deepcopy(model.state_dict())
    mse_list = []

    # 7) 슬라이딩 윈도우 학습/검증 루프
    for i, w in enumerate(windows):
        print(f"\n--- Sliding Window Step {i+1} ---")
        print(f"Train: {w['train_start'].date()} ~ {w['train_end'].date()}")
        print(f"Val  : {w['val_start'].date()} ~ {w['val_end'].date()}")

        train_mask = (dates_train_val >= w['train_start']) & (dates_train_val <= w['train_end'])
        val_mask   = (dates_train_val >= w['val_start']) & (dates_train_val <= w['val_end'])
        train_data = X_scaled_train_val[train_mask]
        val_data   = X_scaled_train_val[val_mask]

        if len(train_data) < seq_len+pred_len or len(val_data) < seq_len+pred_len:
            print("데이터 부족으로 이 윈도우는 건너뜁니다.")
            continue

        train_loader = DataLoader(StockDataset(train_data, seq_len, pred_len), batch_size=batch_size, shuffle=False, pin_memory=True)
        val_loader   = DataLoader(StockDataset(val_data, seq_len, pred_len), batch_size=batch_size, shuffle=False, pin_memory=True)

        # 학습
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Step {i+1} Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

        # 검증
        model.eval()
        val_loss, cnt = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item()
                cnt += 1
        val_mse = val_loss/cnt if cnt else None
        mse_list.append(val_mse)
        print(f"Step {i+1} Validation MSE: {val_mse:.4f}")

        # 파라미터 스무딩
        curr_state = model.state_dict()
        for k in curr_state:
            curr_state[k] = a*curr_state[k] + (1-a)*prev_state[k]
        model.load_state_dict(curr_state)
        prev_state = copy.deepcopy(curr_state)

    print("\nSliding Window Validation 완료. MSE List:", mse_list)

    # 8) 예측 단계 (2025-01-01 이후)
    full_dates = processed_data.index
    target_date = pd.to_datetime("2025-01-01")
    if target_date in full_dates:
        idx = full_dates.get_loc(target_date)
    else:
        idx = full_dates.get_indexer([target_date], method='nearest')[0]

    test_data = X_scaled_full[idx-seq_len:]
    test_dates = full_dates[idx-seq_len:]
    test_loader = DataLoader(StockDataset(test_data, seq_len, pred_len), batch_size=batch_size, shuffle=False, pin_memory=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
    preds = np.concatenate(preds, axis=0).squeeze()
    preds_orig = close_scaler.inverse_transform(preds.reshape(-1,1)).squeeze()

    # 9) 날짜 및 결과 필터링
    pred_dates = test_dates[seq_len: seq_len + len(preds_orig)].strftime("%Y-%m-%d").tolist()
    yesterday = pd.to_datetime(datetime.today().strftime("%Y-%m-%d")) - pd.Timedelta(days=1)
    filt_dates, filt_preds = [], []
    for d, p in zip(pred_dates, preds_orig):
        if pd.to_datetime(d) <= yesterday:
            filt_dates.append(d)
            filt_preds.append(p)

    # 10) 결과 저장 (results/ 폴더 안에)
    save_json_file1(os.path.join('results', f"{ticker}_data.json"),
                    processed_data, ticker)
    save_json_file2(os.path.join('results', f"{ticker}_prediction.json"),
                    filt_dates, np.array(filt_preds),
                    processed_data, yf.Ticker(ticker).info.get("sharesOutstanding",0),
                    ticker)

# 실행 예시
if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "GOOGL", "TSLA", "AMZN", "META"]
    for t in tickers:
        process_ticker(t, "2020-01-01", datetime.today().strftime("%Y-%m-%d"), mode="multi")
    save_stocks_file(os.path.join('results',"stocks.json"), tickers)
