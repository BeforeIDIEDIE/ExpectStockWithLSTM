import sys# system argv사용해서.....
import os

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ForeWeatherStyledExpectedStock import load_ticker_map, run_prediction_pipeline

#이메일 송신용(비밀이얌!!!)
SENDER_EMAIL = os.getenv("SENDER_EMAIL") 
SENDER_PASSWORD = os.getenv("EMAIL_PASS") 
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

def send_email_report(subject, body):
    if not SENDER_PASSWORD:
        print("이메일 비밀번호가 다른뎁")
        return

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        # Gmail SMTP 서버 설정
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("리포트 발송!")
    except Exception as e:
        print(f"메일 발송 실패: {e}")

def retrain_all():#토요일마다 매일 학습 제외 주간학습 실
    ticker_map = load_ticker_map()
    if not ticker_map:
        print("학습할 티커가 없습니다.")
        return
    
    is_weekly = False
    mode = "일일"
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'weekly':
        is_weekly = True
        mode = "주간"
        print("주간 정기 학습 실행 ")

    report_content = f"📊 AI 주식 예측 [{mode} 업데이트]\n"
    report_content += "="*40 + "\n"

    for ticker, name in ticker_map.items():
        print(f"\n{'='*30}")
        print(f"대상 종목: {ticker}")
        try:
            # 파라미터 전달
            run_prediction_pipeline(ticker, is_weekly=is_weekly)
            report_content += f"{name} ({ticker}): 해당 업데이트 완료\n"
        except Exception as e:
            print(f"❌ {ticker} 실행 중 오류 발생: {e}\n")
            report_content += f"{name} ({ticker}): 오류 발생 ({e})\n"
    send_email_report(f"[{mode} 업데이트] 예측 결과", report_content)

if __name__ == "__main__":
    retrain_all()
