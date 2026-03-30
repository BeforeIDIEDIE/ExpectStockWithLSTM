import os
import json
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler

#미완 리스트
# 1. 날짜비례 epoch 추가학습방안 : 현재는 추가데이터를 고정 3번 학습함, 다만 기본 학습 epoch가 300임을 감안하면 이에맞추어 추가적인 epoch조정이 필요할 것으로 보임
# 3. Symbol 입력시 별도의 폴더 생성 -> 해당 폴더에 각각 모델, scaler, info파일 생성 관리
# 4. 한글 입력시 해당 한글을 Symbol로 변경.
# 5. 이거 국장도 되나?
# 7. ETF | 레버리지 | 일반 구분해서 LOOK BACK혹은 학습 EPOCH 조정 필요할듯
# 8. 현재 추가적인 정보 있을시 3번 더 학습함, 다만 이러할 경우 최신의 정보에 너무 OVERFITTING된 정보가 들어올 가능성이 있음 따라서 대안으로 매일, 주간 마다 추가적인 학습을 수행하는 것이 좋아보임
# 8-1. 문제는 추가적인 학습을 특정 시간에 수해하는것은 번거롭고, 추후 다른 종목도 추가할 것을 고려하면 좋은 대안은 아님 따라서 추후 GITHUB서버를 빌려 해당 작업을 수행하도록 할 생각(자동화도)


#완성 리스트
# 2. 현재는 종가만을 입력함, 다만 데이터가 너무 부족해보임, 적어도 근 10일의 그래프 추세에 따른 예측을 수행하도록 하는것이 좋을
# 6. scaler요인이 너무 부족 -> RSI와 MACD를 고려?

# --- [설정 및 경로] ---
BASE_DIR = "./models"
TICKER_MAP_FILE = "managed_tickers_map.json"  # 딕셔너리 형태 저장 파일
os.makedirs(BASE_DIR, exist_ok=True)

# --- [1. 티커 맵 관리 기능] ---

def load_ticker_map():
    if os.path.exists(TICKER_MAP_FILE):
        with open(TICKER_MAP_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {} # 초기값 {입력명: 티커}

def save_ticker_map(ticker_map):
    with open(TICKER_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(ticker_map, f, ensure_ascii=False, indent=4)

def get_ticker_symbol(input_name):
    input_name = input_name.strip()
    if input_name.upper() == 'BACK': return 'BACK'
    
    # [보정] yfinance Search를 먼저 수행하여 실제 티커를 우선적으로 가져옵니다.
    print(f"🔍 '{input_name}' 검색 중...")
    try:
        search = yf.Search(input_name, max_results=1).quotes
        if search:
            best_ticker = search[0]['symbol']
            print(f"✅ 검색 결과 발견: {best_ticker}")
            return best_ticker
    except:
        pass

    #영어에도 없다면 국장 ticker
    try:
        df_krx = fdr.StockListing('KRX')
        match = df_krx[df_krx['Name'].str.contains(input_name, na=False)]
        if not match.empty:
            row = match.iloc[0]
            suffix = ".KS" if row['Market'] == 'KOSPI' else ".KQ"
            return row['Code'] + suffix
    except:
        pass
        
    return None

# --- [2. 모델 정의] ---
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

# --- [3. 핵심 실행 파이프라인] ---

def run_prediction_pipeline(SYMBOL_INPUT,is_weekly=False):
    SAVE_DIR = f"./models/{SYMBOL_INPUT}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    MODEL_FILE = os.path.join(SAVE_DIR, 'model.pth')
    SCALER_FILE = os.path.join(SAVE_DIR, 'scaler.pkl')
    INFO_FILE = os.path.join(SAVE_DIR, 'info.json')

    # 자산 유형 및 파라미터 설정
    ticker_obj = yf.Ticker(SYMBOL_INPUT)
    is_etf = ticker_obj.info.get('quoteType') == 'ETF'
    LOOK_BACK = 20 if is_etf else 10
    FEATURES = ['Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD']
    INPUT_SIZE = len(FEATURES)

    # 데이터 로드
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    df = yf.download(SYMBOL_INPUT, start="2015-01-01", end=today)
    if df.empty or len(df) < LOOK_BACK: 
        print(f"❌ {SYMBOL_INPUT}의 데이터가 부족합니다.")
        return

    # 지표 계산
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss + 1e-9))) # 0 나누기 방지
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df.dropna(inplace=True)

    # 모델 선언 (오류 방지를 위해 여기서 미리 선언)
    model = StockLSTM(input_size=INPUT_SIZE)

    # 학습 날짜 로드
    last_date = "2020-01-01"
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE, 'r') as f: last_date = json.load(f).get('last_train_date', last_date)
    
    train_df = df[df.index >= last_date][FEATURES]

    # [중요] 학습 데이터가 있을 때만 학습 진행
    if len(train_df) > 5:
        if os.path.exists(SCALER_FILE):
            scaler = joblib.load(SCALER_FILE)
            full_scaled = scaler.transform(df[FEATURES])
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            full_scaled = scaler.fit_transform(df[FEATURES])
            joblib.dump(scaler, SCALER_FILE)

        X, y = [], []
        for i in range(LOOK_BACK, len(full_scaled)):
            X.append(full_scaled[i-LOOK_BACK:i, :])
            y.append(full_scaled[i, 0])
        
        X_train = torch.FloatTensor(np.array(X))
        y_train = torch.FloatTensor(np.array(y)).unsqueeze(-1)

        if os.path.exists(MODEL_FILE):
            model.load_state_dict(torch.load(MODEL_FILE))
            if is_weekly:
                epochs = 50  # 주간 학습: 50번 학습
                lr = 0.0005
                print(f"주간 정기 재학습: {epochs} Epochs")
            else:
                epochs = 3   # 일일 업데이트: 3번
                lr = 0.0003  # 일일 학습은 낮은 학습률으로 (overfitting방지)
                print(f"일일 최신화: {epochs} Epochs")
        else:
            epochs = 300
            lr = 0.001
            print(f"신규 모델 학습 (300 Epochs)")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.6f}")
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), MODEL_FILE)
        with open(INFO_FILE, 'w') as f: json.dump({"last_train_date": today}, f)
    else:
        # 학습할 데이터가 없으면 기존 모델 로드
        if os.path.exists(MODEL_FILE):
            model.load_state_dict(torch.load(MODEL_FILE))
            scaler = joblib.load(SCALER_FILE)
            print("📅 최신 상태입니다. 기존 모델을 사용합니다.")
        else:
            print("❌ 학습된 모델이 없고 새로 학습할 데이터도 부족합니다.")
            return

    # 예측 단계
    model.eval()
    recent_data = df[FEATURES].tail(LOOK_BACK).values
    scaled_recent = scaler.transform(recent_data)
    scaled_recent = torch.FloatTensor(scaled_recent).unsqueeze(0)

    with torch.no_grad():
        pred_val = model(scaled_recent).item()
        dummy = np.zeros((1, INPUT_SIZE))
        dummy[0, 0] = pred_val
        pred_price = scaler.inverse_transform(dummy)[0, 0]

    current_price = df['Close'].iloc[-1].item()
    diff_pct = ((pred_price - current_price) / current_price) * 100
    weather = "☀️ 맑음" if diff_pct > 1.0 else "🌧️ 비" if diff_pct < -1.0 else "☁️ 흐림"

    print("-" * 45)
    print(f"📈 종목: {SYMBOL_INPUT} | 현재가: {current_price:.2f} | 예상가: {pred_price:.2f}")
    print(f"📊 예측: {weather} ({diff_pct:+.2f}%)")
    print("-" * 45)

# --- [4. 메인 메뉴] ---
def main_menu():
    ticker_map = load_ticker_map() 
    
    while True:
        ticker_keys = list(ticker_map.keys())
        print("\n" + "="*15 + " MENU " + "="*15)
        for i, ticker in enumerate(ticker_keys):
            print(f"{i+1}. {ticker_map[ticker]} ({ticker})")
        
        print(f"{len(ticker_keys)+1}. 새 종목 추가")
        print(f"{len(ticker_keys)+2}. 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(ticker_keys):
            run_prediction_pipeline(ticker_keys[int(choice)-1])
            
        elif choice == str(len(ticker_keys)+1):
            user_input = input("종목명/티커 입력 (back: 이전): ").strip()
            if user_input.upper() == 'BACK': continue
            
            sym = get_ticker_symbol(user_input)
            
            if sym:
                # [중요] 데이터가 실제로 존재하는지 한 번 더 체크 (404 방지)
                test_df = yf.download(sym, period="1d", progress=False)
                if test_df.empty:
                    print(f"❌ '{sym}'은(는) 유효한 주식 데이터가 없습니다.")
                    continue

                if sym in ticker_map:
                    print(f"💡 '{ticker_map[sym]}' 종목으로 이미 등록되어 있습니다.")
                else:
                    # 맵에 저장 (티커를 키로, 검색된 이름을 값으로)
                    ticker_map[sym] = user_input 
                    save_ticker_map(ticker_map)
                    print(f"✅ '{user_input}'({sym}) 등록 완료!")
                
                run_prediction_pipeline(sym)
            else:
                print("❌ 종목을 찾을 수 없습니다.")
                
        elif choice == str(len(ticker_keys)+2):
            break

if __name__ == "__main__":
    main_menu()
