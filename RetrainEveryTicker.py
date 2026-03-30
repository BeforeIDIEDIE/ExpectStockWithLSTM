import sys# system argv사용해서.....
from ForeWeatherStyledExpectedStock import load_ticker_map, run_prediction_pipeline

def retrain_all():#토요일마다 매일 학습 제외 주간학습 실
    ticker_map = load_ticker_map()
    if not ticker_map:
        print("학습할 티커가 없습니다.")
        return
    
    is_weekly = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'weekly':
        is_weekly = True
        print("주간 정기 학습 실행 ")

    for ticker in ticker_map.keys():
        print(f"\n{'='*30}")
        print(f"대상 종목: {ticker}")
        try:
            # 파라미터 전달
            run_prediction_pipeline(ticker, is_weekly=is_weekly)
        except Exception as e:
            print(f"❌ {ticker} 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    retrain_all()
