"""
미국 주식 데이터 다운로드 (TQQQ)

사용법:
    python download_us_stock_data.py

기능:
    - yfinance API로 미국 주식 데이터 다운로드
    - 일봉 데이터 수집 (최대 10년)
    - CSV 파일로 저장 (TQQQ_1d.csv)
    
주의:
    - yfinance는 무료 API (제한 있음)
    - 일봉 기준으로 충분한 과거 데이터 수집
    - 전체 소요시간: 약 1~2분
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# UTF-8 출력 설정 (Windows 콘솔)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 설정
TICKER = "TQQQ"  # ProShares UltraPro QQQ (3배 레버리지 QQQ)
TIMEFRAME = '1d'  # 일봉
TARGET_DAYS = 3650  # 목표: 최근 10년치 데이터 (10년 × 365일)
OUTPUT_FILE = "TQQQ_1d.csv"  # 출력 파일명

# API 제한
SLEEP_BETWEEN_CALLS = 0.5  # API 호출 간격 (초)


def download_yfinance_historical_data(ticker, target_days):
    """yfinance에서 과거 데이터 다운로드"""
    
    print("=" * 80)
    print("📊 미국 주식 데이터 다운로드")
    print("=" * 80)
    print(f"종목: {ticker}")
    print("봉 간격: 1일")
    print(f"목표 기간: 최근 {target_days}일 ({target_days/365:.1f}년)")
    print("=" * 80)
    
    # 시작 날짜 계산 (10년 전)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=target_days)
    
    print(f"\n📌 다운로드 기간: {start_date.strftime('%Y-%m-%d')} ~ "
          f"{end_date.strftime('%Y-%m-%d')}")
    print("📌 예상 소요시간: 약 1~2분")
    print("\n🚀 데이터 다운로드 시작...\n")
    
    try:
        # yfinance로 데이터 다운로드
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d',
            auto_adjust=True,  # 자동 조정 (분할/배당 반영)
            prepost=True       # 장전/장후 거래 포함
        )
        
        if df is None or df.empty:
            print("❌ 데이터 다운로드 실패")
            return None
        
        print(f"✅ 다운로드 완료: {len(df)}개 봉")
        print(f"   기간: {df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')}")
        days_diff = (df.index[-1] - df.index[0]).days
        print(f"   일수: {days_diff}일 ({days_diff / 365:.1f}년)")
        
        return df
        
    except Exception as e:
        print(f"❌ 다운로드 오류: {e}")
        return None


def preprocess_data(df):
    """데이터 품질 검증 및 전처리 (고품질 데이터 생성)"""
    if df.empty:
        return df
    
    print(f"\n🔧 데이터 전처리 시작... (원본: {len(df)}개 행)")
    
    original_count = len(df)
    
    # 1. OHLC 무결성 검증
    print("   1️⃣ OHLC 무결성 검증 중...")
    invalid_hlc = df[df['High'] < df['Low']]
    invalid_hoc = df[(df['High'] < df['Open']) | 
                     (df['High'] < df['Close'])]
    invalid_loc = df[(df['Low'] > df['Open']) | 
                     (df['Low'] > df['Close'])]
    
    invalid_count = len(invalid_hlc) + len(invalid_hoc) + len(invalid_loc)
    if invalid_count > 0:
        print(f"      ⚠️ OHLC 관계 이상: {invalid_count}개 행 제거")
    
    df = df[df['High'] >= df['Low']]
    df = df[(df['High'] >= df['Open']) & (df['High'] >= df['Close'])]
    df = df[(df['Low'] <= df['Open']) & (df['Low'] <= df['Close'])]
    
    # 2. 결손치 확인 및 처리
    print("   2️⃣ 결손치 확인 중...")
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"      ⚠️ 결손치 발견: {null_count}개 → 보간 처리")
        df = df.interpolate(method='linear').ffill().bfill()
    else:
        print("      ✅ 결손치 없음")
    
    # 3. 0 이하 값 확인
    print("   3️⃣ 가격/거래량 검증 중...")
    zero_price = df[(df['Open'] <= 0) | (df['High'] <= 0) | 
                    (df['Low'] <= 0) | (df['Close'] <= 0)]
    if len(zero_price) > 0:
        print(f"      ⚠️ 0 이하 가격: {len(zero_price)}개 행 제거")
        df = df[(df['Open'] > 0) & (df['High'] > 0) & 
                (df['Low'] > 0) & (df['Close'] > 0)]
    else:
        print("      ✅ 가격 정상")
    
    # 음수 거래량 제거
    negative_volume = df[df['Volume'] < 0]
    if len(negative_volume) > 0:
        print(f"      ⚠️ 음수 거래량: {len(negative_volume)}개 행 제거")
        df = df[df['Volume'] >= 0]
    
    # 4. 극단적 이상치 제거 (IQR 방식)
    print("   4️⃣ 극단적 이상치 제거 중... (IQR 방식)")
    outliers_removed = 0
    
    for col in ['Open', 'High', 'Low', 'Close']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3.0 * iqr  # 3배 IQR (보수적)
        upper_bound = q3 + 3.0 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            outliers_removed += len(outliers)
            df.loc[(df[col] < lower_bound) | 
                   (df[col] > upper_bound), col] = np.nan
    
    if outliers_removed > 0:
        print(f"      ⚠️ 극단적 이상치: {outliers_removed}개 값 보간 처리")
        df = df.interpolate(method='linear').ffill().bfill()
    else:
        print("      ✅ 이상치 없음")
    
    # 5. 시간 연속성 확인 (1일 간격)
    print("   5️⃣ 시간 연속성 확인 중...")
    time_diff = df.index.to_series().diff()
    expected_interval = pd.Timedelta(days=1)
    
    gaps = df[time_diff > expected_interval * 1.5]  # 1.5일 이상 갭
    if len(gaps) > 0:
        print(f"      ⚠️ 시간 갭 발견: {len(gaps)}개 위치")
        # 갭 위치 샘플 출력 (처음 3개만)
        for idx in gaps.index[:3]:
            prev_idx = df.index[df.index.get_loc(idx) - 1]
            gap_days = (idx - prev_idx).total_seconds() / 86400
            print(f"         • {prev_idx} → {idx} "
                  f"(갭: {gap_days:.1f}일)")
        if len(gaps) > 3:
            print(f"         ... 외 {len(gaps) - 3}개")
    else:
        print("      ✅ 시간 연속성 정상")
    
    # 최종 NaN 제거
    df = df.dropna()
    
    final_count = len(df)
    removed_count = original_count - final_count
    
    print("\n✅ 전처리 완료:")
    print(f"   원본: {original_count}개 행")
    print(f"   제거: {removed_count}개 행 "
          f"({removed_count/original_count*100:.2f}%)")
    print(f"   최종: {final_count}개 행 (고품질 데이터)")
    
    return df


def save_to_csv(df, filename):
    """DataFrame을 CSV로 저장"""
    try:
        print(f"\n💾 CSV 파일 저장 중... ({filename})")
        
        # 컬럼 이름 변경 (백테스트와 호환)
        df_copy = df.copy()
        
        # 실제 컬럼 개수에 맞춰 동적으로 컬럼명 할당
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits': 'stock_splits'
        }
        
        # 기존 컬럼명을 새로운 컬럼명으로 변경
        df_copy = df_copy.rename(columns=column_mapping)
        df_copy.index.name = 'timestamp'
        
        # 백테스트에 필요한 컬럼만 선택
        df_final = df_copy[['open', 'high', 'low', 'close', 
                           'volume']].copy()
        
        # CSV 저장
        df_final.to_csv(filename)
        
        print(f"✅ 저장 완료: {filename}")
        print(f"   파일 크기: {len(df_final)}개 행")
        print(f"   컬럼: {list(df_final.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 저장 오류: {e}")
        return False


def main():
    print("\n" + "=" * 80)
    print("🔄 미국 주식 데이터 다운로드")
    print("=" * 80)
    print(f"📌 목표: {OUTPUT_FILE}")
    print(f"📌 종목: {TICKER}")
    print(f"📌 봉 타입: 일봉")
    print("=" * 80)
    
    # 데이터 다운로드
    df = download_yfinance_historical_data(TICKER, TARGET_DAYS)
    
    if df is None or df.empty:
        print("\n❌ 데이터 다운로드 실패")
        return
    
    # 데이터 전처리 (고품질 데이터 생성)
    df_processed = preprocess_data(df)
    
    if df_processed.empty:
        print("\n❌ 전처리 후 데이터가 비어있습니다")
        return
    
    # CSV 저장
    if save_to_csv(df_processed, OUTPUT_FILE):
        print("\n" + "=" * 80)
        print("✅ 미국 주식 고품질 데이터 생성 완료!")
        print("=" * 80)
        print(f"📄 파일: {OUTPUT_FILE}")
        print(f"📊 데이터: {len(df_processed)}개 봉 (품질 검증 완료)")
        print(f"📅 기간: {df_processed.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df_processed.index[-1].strftime('%Y-%m-%d')}")
        print("✅ 검증 항목:")
        print("   • OHLC 무결성")
        print("   • 결손치 보간")
        print("   • 이상치 제거")
        print("   • 시간 연속성")
        print("🎯 이제 TQQQ 백테스트를 실행하세요!")
        print("=" * 80)
    else:
        print("\n❌ CSV 저장 실패")


if __name__ == '__main__':
    main()
