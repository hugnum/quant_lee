# =============================================================================
# MOM Strategy2 단순 백테스트 (고정 파라미터) - 다중 종목 지원
# =============================================================================
# 
# 기능:
# - MOM Strategy2 (모멘텀 + MFI) 전략 백테스트
# - 대화형 백테스트 기간 설정
# - 고정 파라미터 사용 (최적화 없음)
# - 여러 종목 동시 백테스트 및 비교
# - 상세한 성과 분석 및 시각화 (CAGR 포함)
#
# 작성자: AI Assistant
# 날짜: 2025-01-14
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import time
from datetime import datetime

# =============================================================================
# 전략 함수
# =============================================================================

def mom_strategy2(df, p1, p2, ml, sl, verbose=True):
    """
    MOM Strategy2: 모멘텀 + MFI 전략
    
    Parameters:
    -----------
    df : pandas.DataFrame
        주가 데이터 (OHLCV)
    p1 : int
        모멘텀 계산 기간
    p2 : int
        MFI 계산 기간
    ml : float
        MFI 임계값
    sl : float
        손절매 비율
    verbose : bool
        결과 출력 여부
        
    Returns:
    --------
    data : pandas.DataFrame
        백테스트 결과 데이터
    final_cum_return : float
        최종 누적 수익률
    """
    # 거래 수수료 설정
    fee_rate = 0.001
    allocation_ratio = 0.5
    data = df.copy()

    # 파라미터 설정
    period1 = p1      # 모멘텀 계산 기간
    period2 = p2      # MFI 계산 기간
    mfi_level = ml    # MFI 임계값
    stop_loss = sl    # 손절매 비율
    
    # 기술적 지표 계산
    data['Mom'] = data['Close'].pct_change(periods=period1)
    data['MFI'] = ta.volume.money_flow_index(data.High, data.Low, 
                                           data.Close, data.Volume, period2) 
    data.dropna(inplace=True)

    # 모멘텀 신호 생성 (모멘텀 > 0 → 1, 그 외 → 0)
    mom_pos = pd.Series(np.where(data['Mom'] > 0, 1, 0), index=data.index)
    mom_signals = mom_pos.diff().fillna(0)

    # 백테스트 초기 설정
    cash_init = 10000  # 초기 자본
    cash = cash_init
    asset = np.zeros(len(data))
    asset[0] = cash

    # 백테스트 실행을 위한 변수들
    prices = data['Close'].values
    signals = mom_signals.values
    mfi = data['MFI'].values
    positions = np.zeros(len(data))
    pos = 0  # 현재 포지션 (0: 없음, 1: 보유)
    num = 0  # 보유 주수
    
    # 백테스트 루프
    for i in range(1, len(data)):
        if pos == 0:  # 포지션 없음
            # 매수 조건: 모멘텀 양전 AND MFI > 임계값
            if signals[i] == 1 and mfi[i] > mfi_level:
                entry_price = prices[i]
                investable_cash = cash * allocation_ratio
                potential_shares = int(investable_cash / (entry_price * (1 + fee_rate)))
                if potential_shares <= 0:
                    continue
                pos = 1
                positions[i] = 1
                num = potential_shares
                cash -= entry_price * num * (1 + fee_rate)
                stop_loss_price = entry_price * (1 - stop_loss)
                
        elif pos == 1:  # 포지션 보유 중
            if prices[i] < stop_loss_price:  # 손절 발생
                pos = 0
                cash += prices[i] * num * (1 - fee_rate)  
            else:  # 손절가 갱신 (상승 시에만)
                positions[i] = 1
                stop_loss_price = max(stop_loss_price, prices[i] * (1 - stop_loss))

        # 자산 가치 갱신
        if pos == 0:
            asset[i] = cash
        elif pos == 1:
            asset[i] = cash + prices[i] * num    

    # 결과 데이터 정리
    data['Position'] = positions
    data['Signal'] = data['Position'].diff().fillna(0)
    
    # 매수/매도 가격 기록
    data['Buy_Price'] = np.where(data['Signal'] == 1, data['Close'], np.nan)
    data['Sell_Price'] = np.where(data['Signal'] == -1, data['Close'], np.nan)   
    
    # 누적 수익률 계산
    data['Cumulative_Return'] = asset / cash_init
    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1
    
    if verbose:
        print(f'Final cumulative return of the strategy: {100*final_cum_return:.2f}%')
    
    return data, final_cum_return

# =============================================================================
# 성과 분석 함수
# =============================================================================

def tear_sheet1(data):
    """
    백테스트 결과 상세 분석 및 출력
    
    Parameters:
    -----------
    data : pandas.DataFrame
        백테스트 결과 데이터 (Position, Signal, Cumulative_Return 포함)
    """
    fee_rate = 0.001
    
    # =================================================================
    # 1. 기본 정보
    # =================================================================
    trading_period = len(data) / 252  # 투자 기간 (년)
    
    # =================================================================
    # 2. 수익률 분석
    # =================================================================
    # Buy & Hold 수익률 (수수료 포함)
    buy_and_hold = (data['Close'].iloc[-1] * (1 - fee_rate) / 
                    (data['Close'].iloc[0] * (1 + fee_rate))) - 1
    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1
    
    # =================================================================
    # 3. CAGR (연평균 성장률) 계산
    # =================================================================
    CAGR_strategy = (data['Cumulative_Return'].iloc[-1]) ** (1/trading_period) - 1
    CAGR_benchmark = (buy_and_hold + 1) ** (1/trading_period) - 1
    
    # =================================================================
    # 4. 샤프 비율 및 Sortino 비율 계산
    # =================================================================
    risk_free_rate = 0.003  # 무위험 수익률 (0.3%)
    strategy_daily_return = data['Cumulative_Return'].pct_change().fillna(0)
    mean_return = strategy_daily_return.mean() * 252
    std_return = strategy_daily_return.std() * np.sqrt(252)
    sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
    
    # Sortino Ratio (하방 변동성만 고려)
    downside_returns = strategy_daily_return[strategy_daily_return < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    # =================================================================
    # 5. 최대 낙폭 (MDD) 계산
    # =================================================================
    # 전략 MDD
    data['Cumulative_Max'] = data['Cumulative_Return'].cummax()
    data['Drawdown'] = data['Cumulative_Return'] / data['Cumulative_Max'] - 1
    max_drawdown = data['Drawdown'].min()
    
    # MDD 발생 기간 찾기
    mdd_idx = data['Drawdown'].idxmin()
    mdd_start_idx = data.loc[:mdd_idx, 'Cumulative_Return'].idxmax()
    mdd_duration = (mdd_idx - mdd_start_idx).days if mdd_idx != mdd_start_idx else 0
    
    # 벤치마크 MDD
    cumulative_returns = (1 + data['Close'].pct_change()).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    mdd_benchmark = drawdown.min()
    
    # =================================================================
    # 6. Calmar Ratio 계산
    # =================================================================
    calmar_ratio = CAGR_strategy / abs(max_drawdown) if max_drawdown != 0 else 0
    calmar_benchmark = CAGR_benchmark / abs(mdd_benchmark) if mdd_benchmark != 0 else 0
    
    # =================================================================
    # 7. 거래 통계 분석
    # =================================================================
    buy_signals = data[data['Signal'] == 1].index
    sell_signals = data[data['Signal'] == -1].index
    returns = []
    holding_periods = []
    
    # 각 거래의 수익률과 보유 기간 계산
    for buy_date in buy_signals:
        sell_dates = sell_signals[sell_signals > buy_date]
        if not sell_dates.empty:
            sell_date = sell_dates[0]
            buy_price = data.loc[buy_date, 'Close']
            sell_price = data.loc[sell_date, 'Close']
            return_pct = (sell_price * (1 - fee_rate) / 
                         (buy_price * (1 + fee_rate))) - 1
            returns.append(return_pct)          
            holding_period = np.busday_count(buy_date.date(), sell_date.date())
            holding_periods.append(holding_period)
    
    # 거래 통계
    profitable_trades = len([r for r in returns if r > 0])
    loss_trades = len([r for r in returns if r <= 0])
    total_trades = len(returns)
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    # 평균 보유 기간
    average_holding_period = np.mean(holding_periods) if holding_periods else 0
    median_holding_period = np.median(holding_periods) if holding_periods else 0
    
    # 평균 수익/손실
    average_profit = np.mean([r for r in returns if r > 0]) if profitable_trades > 0 else 0
    average_loss = np.mean([r for r in returns if r <= 0]) if loss_trades > 0 else 0
    max_profit = max(returns) if returns else 0
    max_loss = min(returns) if returns else 0
    
    # 수익/손실 비율
    profit_loss_ratio = (average_profit / abs(average_loss) 
                        if average_loss != 0 else np.inf)
    
    # =================================================================
    # 8. 결과 출력
    # =================================================================
    print("\n" + "=" * 70)
    print("📊 백테스트 상세 분석 결과")
    print("=" * 70)
    
    print(f"\n📅 기본 정보:")
    print(f"   • 투자 기간: {trading_period:.2f}년 ({len(data)} 거래일)")
    
    print(f"\n💰 수익률 분석:")
    print(f"   • 전략 누적 수익률: {100*final_cum_return:.2f}%")
    print(f"   • Buy & Hold 수익률: {100*buy_and_hold:.2f}%")
    print(f"   • 초과 수익: {100*(final_cum_return - buy_and_hold):.2f}%p")
    
    print(f"\n📈 연평균 수익률 (CAGR):")
    print(f"   • 전략 CAGR: {100*CAGR_strategy:.2f}%")
    print(f"   • Buy & Hold CAGR: {100*CAGR_benchmark:.2f}%")
    
    print(f"\n⚡ 위험 조정 수익률 지표:")
    print(f"   • 샤프 비율 (Sharpe Ratio): {sharpe_ratio:.3f}")
    print(f"   • 소르티노 비율 (Sortino Ratio): {sortino_ratio:.3f}")
    print(f"   • 칼마 비율 (Calmar Ratio): {calmar_ratio:.3f}")
    print(f"     (Buy & Hold Calmar: {calmar_benchmark:.3f})")
    
    print(f"\n📉 최대 낙폭 (Maximum Drawdown):")
    print(f"   • 전략 MDD: {100*max_drawdown:.2f}%")
    print(f"   • Buy & Hold MDD: {100*mdd_benchmark:.2f}%")
    if mdd_duration > 0:
        print(f"   • MDD 지속 기간: {mdd_duration}일 ({mdd_duration/365:.1f}년)")
        print(f"   • MDD 시작: {mdd_start_idx.strftime('%Y-%m-%d')}")
        print(f"   • MDD 최저점: {mdd_idx.strftime('%Y-%m-%d')}")
    
    print(f"\n📊 거래 통계:")
    print(f"   • 총 거래 횟수: {total_trades}회")
    print(f"   • 수익 거래: {profitable_trades}회")
    print(f"   • 손실 거래: {loss_trades}회")
    print(f"   • 승률: {100*win_rate:.2f}%")
    print(f"   • 평균 보유 기간: {average_holding_period:.1f}일")
    print(f"   • 중앙값 보유 기간: {median_holding_period:.1f}일")
    
    print(f"\n💵 거래별 수익/손실 분석:")
    print(f"   • 평균 수익률 (승리 거래): {100*average_profit:.3f}%")
    print(f"   • 평균 손실률 (손실 거래): {100*average_loss:.3f}%")
    print(f"   • 최대 수익률: {100*max_profit:.2f}%")
    print(f"   • 최대 손실률: {100*max_loss:.2f}%")
    print(f"   • 수익/손실 비율: {profit_loss_ratio:.2f}")
    
    print("=" * 70)
    
    # 결과 딕셔너리 반환 (최종 요약에서 사용)
    return {
        'trading_period': trading_period,
        'strategy_return': final_cum_return,
        'buy_hold_return': buy_and_hold,
        'cagr_strategy': CAGR_strategy,
        'cagr_benchmark': CAGR_benchmark,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'mdd_benchmark': mdd_benchmark,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio
    }

# =============================================================================
# 백테스트 기간 설정 함수
# =============================================================================

def get_stock_selection():
    """
    대화형으로 백테스트할 종목 선택
    
    Returns:
    --------
    list : 선택된 종목 리스트
    """
    available_stocks = ['AAPL', 'NVDA', 'TSLA', 'TQQQ', 'UPRO', 'QQQ']
    
    print("\n" + "=" * 60)
    print("백테스트 종목 선택")
    print("=" * 60)
    print("\n사용 가능한 종목:")
    for i, stock in enumerate(available_stocks, 1):
        print(f"  {i}. {stock}")
    print(f"  7. 전체 종목 백테스트")
    print()
    
    try:
        choice = input("선택 (번호 입력, 여러 개는 쉼표로 구분, 예: 1,2,3 또는 7) [기본값: 7]: ").strip() or "7"
    except EOFError:
        print("자동으로 전체 종목을 사용합니다.")
        choice = "7"
    
    selected_stocks = []
    
    if choice == "7":
        selected_stocks = available_stocks
    else:
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            for idx in indices:
                if 1 <= idx <= 6:
                    selected_stocks.append(available_stocks[idx - 1])
                else:
                    print(f"잘못된 번호: {idx}")
            
            if not selected_stocks:
                print("선택된 종목이 없습니다. 전체 종목을 사용합니다.")
                selected_stocks = available_stocks
        except Exception:
            print("잘못된 입력입니다. 전체 종목을 사용합니다.")
            selected_stocks = available_stocks
    
    print(f"\n선택된 종목: {', '.join(selected_stocks)}")
    return selected_stocks

def get_backtest_period():
    """
    대화형으로 백테스트 기간 설정
    
    Returns:
    --------
    dict : 기간 설정 정보
        - mode: 'full', 'range', 'recent'
        - start_date: 시작일 (str)
        - end_date: 종료일 (str)  
        - days: 일수 (int, recent 모드에서만)
    """
    print("\n" + "=" * 60)
    print("MOM Strategy2 백테스트 기간 설정")
    print("=" * 60)
    
    # 현재 날짜 계산
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    print("\n백테스트 기간을 선택하세요:")
    print(f"1. 전체 데이터 사용 (2015-01-01 ~ {current_date}, 약 10년)")
    print("2. 특정 기간 지정 (예: 2020-01-01 ~ 2024-12-31)")
    print("3. 최근 N일 사용 (예: 1000일)")
    print()
    
    try:
        choice = input("선택 (1/2/3) [기본값: 1]: ").strip() or "1"
    except EOFError:
        print("자동으로 전체 데이터를 사용합니다.")
        choice = "1"
    
    if choice == "1":
        return {
            'mode': 'full',
            'start_date': '2015-01-01',
            'end_date': current_date,
            'days': None
        }
    
    elif choice == "2":
        print("\n특정 기간을 입력하세요 (YYYY-MM-DD 형식):")
        print("   예시: 2020년만 테스트 → 2020-01-01 ~ 2020-12-31")
        try:
            start = input("시작일 [기본값: 2020-01-01]: ").strip() or "2020-01-01"
            end = input(f"종료일 [기본값: {current_date}]: ").strip() or current_date
        except EOFError:
            print("자동으로 전체 데이터를 사용합니다.")
            return {
                'mode': 'full',
                'start_date': '2015-01-01',
                'end_date': current_date,
                'days': None
            }
        
        try:
            # 날짜 검증
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            if end_dt <= start_dt:
                print("종료일이 시작일보다 이전입니다. 전체 데이터로 진행합니다.")
                return {
                    'mode': 'full',
                    'start_date': '2015-01-01',
                    'end_date': current_date,
                    'days': None
                }
            
            return {
                'mode': 'range',
                'start_date': start,
                'end_date': end,
                'days': None
            }
        except Exception:
            print("잘못된 날짜 형식입니다. 전체 데이터로 진행합니다.")
            return {
                'mode': 'full',
                'start_date': '2015-01-01',
                'end_date': current_date,
                'days': None
            }
    
    elif choice == "3":
        print("\n최근 N일 데이터 사용:")
        print("   추천: 1000일(약 4년), 1500일(약 6년), 2000일(약 8년)")
        print("   엔터만 치면 1000일 사용")
        try:
            try:
                days_input = input("일수 입력 [기본값: 1000]: ").strip()
            except EOFError:
                print("자동으로 전체 데이터를 사용합니다.")
                return {
                    'mode': 'full',
                    'start_date': '2015-01-01',
                    'end_date': current_date,
                    'days': None
                }
            if not days_input:  # 엔터만 친 경우
                days = 1000
            else:
                days = int(days_input)
            
            if days <= 0:
                print("일수는 양수여야 합니다. 전체 데이터로 진행합니다.")
                return {
                    'mode': 'full',
                    'start_date': '2015-01-01',
                    'end_date': current_date,
                    'days': None
                }
            
            return {
                'mode': 'recent',
                'start_date': None,
                'end_date': None,
                'days': days
            }
        except Exception:
            print("잘못된 입력입니다. 전체 데이터로 진행합니다.")
            return {
                'mode': 'full',
                'start_date': '2015-01-01',
                'end_date': current_date,
                'days': None
            }
    
    else:
        print("잘못된 선택입니다. 전체 데이터로 진행합니다.")
        return {
            'mode': 'full',
            'start_date': '2015-01-01',
            'end_date': current_date,
            'days': None
        }

# =============================================================================
# 데이터 로드 함수
# =============================================================================

def load_stock_data_with_period(ticker, period_config):
    """
    기간 설정에 따라 주식 데이터 로드
    
    Parameters:
    -----------
    ticker : str
        종목 심볼
    period_config : dict
        백테스트 기간 설정 정보
        
    Returns:
    --------
    pandas.DataFrame : 필터링된 주식 데이터
    """
    print(f"\n{ticker} 데이터 다운로드 중...")
    
    try:
        # 전체 데이터 다운로드 (최대 범위) - 현재 날짜까지
        current_date = datetime.now().strftime('%Y-%m-%d')
        df = yf.download(ticker, start='2015-01-01', end=current_date, progress=False)
        
        # MultiIndex 컬럼을 단일 레벨로 변환
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        original_start = df.index[0] if len(df) > 0 else None
        original_end = df.index[-1] if len(df) > 0 else None
        
        # 백테스트 기간 필터링
        if period_config['mode'] == 'range':
            # 특정 기간 사용
            start_dt = pd.to_datetime(period_config['start_date'])
            end_dt = pd.to_datetime(period_config['end_date'])
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            print(f"   기간 필터: {period_config['start_date']} ~ {period_config['end_date']}")
        
        elif period_config['mode'] == 'recent':
            # 최근 N일 사용
            days = period_config['days']
            df = df.tail(days)
            print(f"   최근 {days}일 데이터 사용")
        
        else:  # 'full'
            print(f"   전체 데이터 사용")
        
        # 시간 정렬 보장
        df = df.sort_index()
        
        # 빈 데이터 체크
        if df.empty:
            print(f"   ⚠️ {ticker}: 기간 필터 결과 데이터가 비어 있습니다.")
            return pd.DataFrame()
        
        # 기간 정보
        start_date = df.index[0]
        end_date = df.index[-1]
        total_days = (end_date - start_date).days
        
        print(f"   ✅ {ticker} 데이터 로드 완료: {len(df)}개 거래일")
        if original_start and original_end:
            print(f"      전체 데이터 기간: {original_start.strftime('%Y-%m-%d')} ~ {original_end.strftime('%Y-%m-%d')}")
        print(f"      백테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({total_days}일, {total_days/365:.1f}년)")
        
        return df
        
    except Exception as e:
        print(f"   ❌ {ticker} 데이터 로드 오류: {e}")
        return pd.DataFrame()

# =============================================================================
# 메인 실행 부분
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MOM Strategy2 백테스트 (고정 파라미터) - 다중 종목")
    print("=" * 60)

    # =================================================================
    # 1. 종목 선택
    # =================================================================
    selected_stocks = get_stock_selection()

    # =================================================================
    # 2. 백테스트 기간 설정
    # =================================================================
    period_config = get_backtest_period()

    # 백테스트 기간 정보 출력
    print(f"\n백테스트 설정:")
    if period_config['mode'] == 'full':
        print(f"백테스트 기간: 전체 데이터 (2015-01-01 ~ {period_config['end_date']})")
    elif period_config['mode'] == 'range':
        period_days = (pd.to_datetime(period_config['end_date']) - 
                      pd.to_datetime(period_config['start_date'])).days
        period_years = period_days / 365
        print(f"백테스트 기간: {period_config['start_date']} ~ {period_config['end_date']} "
              f"({period_days}일, {period_years:.1f}년)")
    else:
        days = period_config['days']
        print(f"백테스트 기간: 최근 {days}일 ({days/365:.1f}년)")

    print(f"전략: MOM Strategy2 (모멘텀 + MFI)")
    print("=" * 60)

    # =================================================================
    # 3. 실행 확인
    # =================================================================
    print()
    try:
        input("백테스트를 시작하려면 Enter를 누르세요...")
    except EOFError:
        print("자동으로 백테스트를 시작합니다...")

    # =================================================================
    # 4. 고정 파라미터 설정
    # =================================================================
    period1 = 5      # 모멘텀 계산 기간
    period2 = 14     # MFI 계산 기간  
    mfi_level = 45.0 # MFI 임계값
    stop_loss = 0.10 # 손절매 비율

    print(f"\n사용된 파라미터:")
    print(f"- 모멘텀 기간: {period1}일")
    print(f"- MFI 기간: {period2}일")
    print(f"- MFI 임계값: {mfi_level}")
    print(f"- 손절매 비율: {stop_loss*100:.1f}%")

    # =================================================================
    # 5. 각 종목별 백테스트 실행
    # =================================================================
    results = {}  # 종목별 결과 저장
    
    print("\n" + "=" * 70)
    print("백테스트 실행 중...")
    print("=" * 70)
    
    for ticker in selected_stocks:
        print(f"\n[{ticker}] 백테스트 진행 중...")
        
        # 데이터 로드
        df = load_stock_data_with_period(ticker, period_config)
        if df.empty:
            print(f"   ⚠️ {ticker}: 데이터 로드 실패, 건너뜁니다.")
            continue
        
        # 기본 정보 출력
        print(f"\n   {ticker} 기본 정보:")
        print(f"   시작가: ${df['Close'].iloc[0]:.2f}")
        print(f"   종료가: ${df['Close'].iloc[-1]:.2f}")
        print(f"   기간 수익률: {((df['Close'].iloc[-1]/df['Close'].iloc[0])-1)*100:.2f}%")
        
        # 백테스트 실행
        t1 = time.time()
        try:
            data, ret = mom_strategy2(df, period1, period2, mfi_level, stop_loss, verbose=False)
            t2 = time.time()
            print(f"   백테스트 완료 (소요시간: {(t2-t1):.2f}초)")
            
            # 성과 분석
            stats = tear_sheet1(data)
            
            # 결과 저장
            results[ticker] = {
                'data': data,
                'stats': stats,
                'df': df
            }
            
        except Exception as e:
            print(f"   ❌ {ticker} 백테스트 오류: {e}")
            continue
    
    # =================================================================
    # 6. 종목별 결과 비교 테이블
    # =================================================================
    if not results:
        print("\n❌ 백테스트 결과가 없습니다.")
        exit()
    
    print("\n" + "=" * 100)
    print("📊 종목별 성과 비교 (CAGR 포함)")
    print("=" * 100)
    
    # 비교 테이블 생성
    comparison_data = []
    for ticker, result in results.items():
        stats = result['stats']
        comparison_data.append({
            '종목': ticker,
            '전략 수익률 (%)': f"{stats['strategy_return']*100:.2f}",
            'Buy&Hold 수익률 (%)': f"{stats['buy_hold_return']*100:.2f}",
            '초과 수익 (%p)': f"{(stats['strategy_return'] - stats['buy_hold_return'])*100:.2f}",
            '전략 CAGR (%)': f"{stats['cagr_strategy']*100:.2f}",
            'Buy&Hold CAGR (%)': f"{stats['cagr_benchmark']*100:.2f}",
            '샤프 비율': f"{stats['sharpe_ratio']:.3f}",
            '소르티노 비율': f"{stats['sortino_ratio']:.3f}",
            '칼마 비율': f"{stats['calmar_ratio']:.3f}",
            'MDD (%)': f"{stats['max_drawdown']*100:.2f}",
            '총 거래': f"{stats['total_trades']}",
            '승률 (%)': f"{stats['win_rate']*100:.2f}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print("=" * 100)
    
    # =================================================================
    # 7. 결과 시각화 - 여러 종목 비교
    # =================================================================
    print("\n" + "=" * 50)
    print("결과 시각화")
    print("=" * 50)
    
    # 여러 종목 누적 수익률 비교 그래프
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (ticker, result), color in zip(results.items(), colors):
        data = result['data']
        data['Cumulative_Return'].plot(ax=ax, label=f'{ticker} Strategy', 
                                      linewidth=2, color=color, alpha=0.8)
        
        # Buy & Hold도 함께 표시
        df = result['df']
        buy_hold = df['Close'] / df['Close'].iloc[0]
        buy_hold.plot(ax=ax, label=f'{ticker} Buy&Hold', 
                     linewidth=1.5, color=color, linestyle='--', alpha=0.6)
    
    ax.set_title('MOM Strategy2 백테스트 결과 비교 (다중 종목)', fontsize=18)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Returns', fontsize=12)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # CAGR 비교 바 차트
    fig, ax = plt.subplots(figsize=(12, 6))
    
    tickers = list(results.keys())
    strategy_cagrs = [results[t]['stats']['cagr_strategy']*100 for t in tickers]
    buyhold_cagrs = [results[t]['stats']['cagr_benchmark']*100 for t in tickers]
    
    x = np.arange(len(tickers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, strategy_cagrs, width, label='Strategy CAGR', alpha=0.8)
    bars2 = ax.bar(x + width/2, buyhold_cagrs, width, label='Buy&Hold CAGR', alpha=0.8)
    
    ax.set_xlabel('종목', fontsize=12)
    ax.set_ylabel('CAGR (%)', fontsize=12)
    ax.set_title('종목별 CAGR 비교', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # =================================================================
    # 8. 최종 결과 요약
    # =================================================================
    print("\n" + "=" * 100)
    print("🎯 최종 결과 요약")
    print("=" * 100)
    
    # 최고 성과 종목 찾기
    best_strategy_return = max(results.items(), key=lambda x: x[1]['stats']['strategy_return'])
    best_cagr = max(results.items(), key=lambda x: x[1]['stats']['cagr_strategy'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['stats']['sharpe_ratio'])
    
    print(f"\n🏆 최고 성과 종목:")
    print(f"   • 최고 수익률: {best_strategy_return[0]} ({best_strategy_return[1]['stats']['strategy_return']*100:.2f}%)")
    print(f"   • 최고 CAGR: {best_cagr[0]} ({best_cagr[1]['stats']['cagr_strategy']*100:.2f}%)")
    print(f"   • 최고 샤프 비율: {best_sharpe[0]} ({best_sharpe[1]['stats']['sharpe_ratio']:.3f})")
    
    print(f"\n📊 종목별 상세 결과는 위의 비교 테이블을 참조하세요.")
    print(f"   (CAGR, 샤프 비율, MDD 등 모든 지표 포함)")
    
    print("\n" + "=" * 100)
    print("✅ 백테스트 완료!")
    print("=" * 100)
