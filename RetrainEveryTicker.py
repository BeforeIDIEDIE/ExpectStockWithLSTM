from ForeWeatherStyledExpectedStock import load_ticker_map, run_prediction_pipeline

def retrain_all():#토요일마다 매일 학습 제외 주간학습 실
    ticker_map = load_ticker_map()
    if not ticker_map:
        print("학습할 티커가 없습니다.")
        return
    
    for ticker in ticker_map.keys():
        print(f"🚀 자동 재학습 시작: {ticker}")
        try:
            run_prediction_pipeline(ticker)
        except Exception as e:
            print(f"❌ {ticker} 학습 중 오류 발생: {e}")

if __name__ == "__main__":
    retrain_all()
