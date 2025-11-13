# =============================================================================
# TQQQ MOM Strategy2 + 2% R값 시스템 백테스트 (V2)
# =============================================================================
# 
# 기능:
# - MOM Strategy2 (모멘텀 + MFI) 전략 + 2% R값 시스템
# - ATR 기반 동적 손절 및 5단계 트레일링 스탑
# - 대화형 백테스트 기간 설정
# - 상세한 성과 분석 및 시각화
# - R-배수 기반 성과 분석
#
# 작성자: AI Assistant
# 날짜: 2025-01-14
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import time
from datetime import datetime

# =============================================================================
# ATR 및 R값 계산 함수들
# =============================================================================

def calculate_atr_indicators(df, atr_length=14):
    """ATR 및 관련 지표 계산"""
    # 컬럼명 확인 및 통일 (소문자)
    if 'High' in df.columns:
        df.rename(columns={'High': 'high', 'Low': 'low', 
                          'Close': 'close', 'Volume': 'volume'}, inplace=True)
    
    df['ATR'] = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'], window=atr_length)
    df['ATR_21'] = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'], window=21)
    return df

def calculate_position_size_2pct_risk(cash, entry_price, atr_value, 
                                       atr_mult=2.0):
    """전체 자본의 2% 리스크로 포지션 사이즈 계산"""
    # 2% 리스크 금액
    risk_amount = cash * 0.02
    
    # R값 거리 계산 (ATR × 배수)
    r_distance = atr_value * atr_mult
    
    # 1코인당 손실 계산
    per_coin_loss = r_distance
    
    # 포지션 사이즈 계산
    position_size = risk_amount / per_coin_loss
    
    return position_size, r_distance

def calculate_profit_multiple(current_price, entry_price, r_distance):
    """R-배수 계산"""
    if r_distance <= 0:
        return 0.0
    return (current_price - entry_price) / r_distance

# =============================================================================
# 5단계 트레일링 스탑 시스템
# =============================================================================

# 트레일링 파라미터
BE_LOCK_AT_R = 1.0      # 1R에서 손익분기점 락인
TIER_1_R = 2.0          # 2R부터 트레일링 시작
TIER_1_FACTOR = 1.00    # 1R 뒤에서 트레일링
TIER_2_R = 3.0          # 3R에서 더 타이트하게
TIER_2_FACTOR = 0.80    # 0.8R 뒤에서 트레일링
TIER_3_R = 5.0          # 5R에서 더 타이트하게
TIER_3_FACTOR = 0.60    # 0.6R 뒤에서 트레일링
TIER_4_R = 7.0          # 7R에서 최종 단계
TIER_4_FACTOR = 0.40    # 0.4R 뒤에서 트레일링
TIER_MIN_FACTOR = 0.30  # 최소 트레일링 팩터

def update_trailing_stop(entry_price, current_price, peak_price, r_distance):
    """5단계 트레일링 스탑 업데이트"""
    if r_distance <= 0:
        return 0.0, "OFF"
    
    # R-peak 계산
    r_peak = (peak_price - entry_price) / r_distance
    
    # 1) 손익분기점 락인
    be_stop = 0.0
    if r_peak >= BE_LOCK_AT_R:
        be_stop = entry_price
    
    # 2) 5단계 트레일링
    trailing_stop = 0.0
    trail_mode = "OFF"
    
    if r_peak < TIER_1_R:
        trail_mode = "OFF"
    elif r_peak < TIER_2_R:
        trail_factor = TIER_1_FACTOR
        trailing_stop = peak_price - (r_distance * trail_factor)
        trail_mode = "Tier1(1.00R)"
    elif r_peak < TIER_3_R:
        trail_factor = TIER_2_FACTOR
        trailing_stop = peak_price - (r_distance * trail_factor)
        trail_mode = "Tier2(0.80R)"
    elif r_peak < TIER_4_R:
        trail_factor = TIER_3_FACTOR
        trailing_stop = peak_price - (r_distance * trail_factor)
        trail_mode = "Tier3(0.60R)"
    else:
        trail_factor = max(TIER_4_FACTOR, TIER_MIN_FACTOR)
        trailing_stop = peak_price - (r_distance * trail_factor)
        trail_mode = f"Tier4({trail_factor:.2f}R)"
    
    # 3) BE락인과 트레일링 중 높은 것
    if be_stop > 0 and trailing_stop > 0:
        final_stop = max(be_stop, trailing_stop)
    elif be_stop > 0:
        final_stop = be_stop
        trail_mode = "BE"  # 손익분기 전용
    else:
        final_stop = trailing_stop
    
    return final_stop, trail_mode

# =============================================================================
# 전략 함수 (2% R값 시스템 적용)
# =============================================================================

def mom_strategy2_2pct_risk(df, p1, p2, ml, atr_mult, verbose=True):
    """
    MOM Strategy2: 모멘텀 + MFI + 2% R값 시스템
    
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
    atr_mult : float
        ATR 배수 (손절거리)
    verbose : bool
        결과 출력 여부
        
    Returns:
    --------
    data : pandas.DataFrame
        백테스트 결과 데이터
    final_cum_return : float
        최종 누적 수익률
    trades : list
        거래 기록 리스트
    """
    fee_rate = 0.001
    data = df.copy()
    
    # ATR 지표 계산 (내부에서 컬럼명 통일 처리)
    data = calculate_atr_indicators(data)
    
    # 기술적 지표 계산
    data['Mom'] = data['close'].pct_change(periods=p1)
    data['MFI'] = ta.volume.money_flow_index(
        data['high'], data['low'], data['close'], data['volume'], p2)
    data.dropna(inplace=True)

    # 모멘텀 신호 생성 (모멘텀 > 0 → 1, 그 외 → 0)
    mom_pos = pd.Series(np.where(data['Mom'] > 0, 1, 0), index=data.index)
    mom_signals = mom_pos.diff().fillna(0)

    # 백테스트 초기 설정
    cash_init = 10000
    cash = cash_init
    asset = np.zeros(len(data))
    asset[0] = cash

    prices = data['close'].values
    signals = mom_signals.values
    mfi = data['MFI'].values
    atr_values = data['ATR'].values
    positions = np.zeros(len(data))
    
    # 포지션 관리 변수
    pos = 0
    entry_price = 0.0
    r_distance = 0.0
    peak_price = 0.0
    position_size = 0.0
    
    # 거래 통계
    trades = []
    current_trade = None
    
    # 백테스트 루프
    for i in range(1, len(data)):
        current_price = prices[i]
        current_atr = atr_values[i]
        
        # 포지션 없음 - 진입 확인
        if pos == 0:
            # 매수 조건: 모멘텀 양전 AND MFI > 임계값
            if signals[i] == 1 and mfi[i] > ml:
                # 2% 리스크 기반 포지션 사이즈 계산
                position_size, r_distance = calculate_position_size_2pct_risk(
                    cash, current_price, current_atr, atr_mult)
                
                if position_size > 0:
                    pos = 1
                    positions[i] = 1
                    entry_price = current_price
                    peak_price = current_price
                    
                    # 실제 매수 (수수료 포함)
                    total_cost = position_size * current_price * (1 + fee_rate)
                    if total_cost <= cash:
                        cash -= total_cost
                        
                        # 거래 기록 시작
                        current_trade = {
                            'entry_date': data.index[i],
                            'entry_price': entry_price,
                            'position_size': position_size,
                            'r_distance': r_distance,
                            'peak_price': peak_price
                        }
                    else:
                        # 자금 부족
                        pos = 0
                        positions[i] = 0
                        position_size = 0.0
        
        # 포지션 있음 - 청산 확인
        else:
            # Peak 업데이트
            if current_price > peak_price:
                peak_price = current_price
            
            # 손절가 계산
            stop_loss_price = entry_price - r_distance
            
            # 트레일링 스탑 계산
            trailing_stop, trail_mode = update_trailing_stop(
                entry_price, current_price, peak_price, r_distance)
            
            # 청산 조건 체크
            should_close = False
            close_reason = ""
            
            if current_price <= stop_loss_price:
                should_close = True
                close_reason = "손절"
            elif trailing_stop > 0 and current_price <= trailing_stop:
                should_close = True
                close_reason = f"트레일링_{trail_mode}"
            
            if should_close:
                # 포지션 청산
                pos = 0
                cash += position_size * current_price * (1 - fee_rate)
                
                # 거래 기록 완료
                if current_trade:
                    current_trade.update({
                        'exit_date': data.index[i],
                        'exit_price': current_price,
                        'exit_reason': close_reason,
                        'r_multiple': calculate_profit_multiple(
                            current_price, entry_price, r_distance)
                    })
                    trades.append(current_trade)
                    current_trade = None
                
                # 변수 초기화
                entry_price = 0.0
                r_distance = 0.0
                peak_price = 0.0
                position_size = 0.0
            else:
                # 포지션 유지
                positions[i] = 1

        # 자산 가치 갱신
        if pos == 0:
            asset[i] = cash
        else:
            asset[i] = cash + position_size * current_price

    # 결과 데이터 정리
    data['Position'] = positions
    data['Signal'] = data['Position'].diff().fillna(0)
    
    # 매수/매도 가격 기록
    data['Buy_Price'] = np.where(data['Signal'] == 1, data['close'], np.nan)
    data['Sell_Price'] = np.where(data['Signal'] == -1, data['close'], np.nan)
    
    # 누적 수익률 계산
    data['Cumulative_Return'] = asset / cash_init
    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1
    
    # 컬럼명 복원 (tear_sheet1 호환성)
    data.rename(columns={'close': 'Close'}, inplace=True)
    
    if verbose:
        print(f'Final cumulative return of the strategy: '
              f'{100*final_cum_return:.2f}%')
    
    return data, final_cum_return, trades

# =============================================================================
# R-배수 기반 성과 분석
# =============================================================================

def analyze_r_performance(trades):
    """R-배수 기반 성과 분석"""
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_winning_r': 0.0,
            'avg_losing_r': 0.0,
            'avg_r_multiple': 0.0,
            'expectancy': 0.0,
            'max_winning_r': 0.0,
            'max_losing_r': 0.0
        }
    
    winning_trades = [t for t in trades if t['r_multiple'] > 0]
    losing_trades = [t for t in trades if t['r_multiple'] <= 0]
    
    total_trades = len(trades)
    winning_count = len(winning_trades)
    losing_count = len(losing_trades)
    win_rate = winning_count / total_trades if total_trades > 0 else 0
    
    avg_winning_r = (np.mean([t['r_multiple'] for t in winning_trades]) 
                     if winning_trades else 0)
    avg_losing_r = (np.mean([t['r_multiple'] for t in losing_trades]) 
                   if losing_trades else 0)
    avg_r_multiple = np.mean([t['r_multiple'] for t in trades])
    
    expectancy = ((win_rate * avg_winning_r) + 
                 ((1 - win_rate) * avg_losing_r))
    
    max_winning_r = (max([t['r_multiple'] for t in winning_trades]) 
                    if winning_trades else 0)
    max_losing_r = (min([t['r_multiple'] for t in losing_trades]) 
                   if losing_trades else 0)
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_count,
        'losing_trades': losing_count,
        'win_rate': win_rate,
        'avg_winning_r': avg_winning_r,
        'avg_losing_r': avg_losing_r,
        'avg_r_multiple': avg_r_multiple,
        'expectancy': expectancy,
        'max_winning_r': max_winning_r,
        'max_losing_r': max_losing_r
    }

def print_r_analysis(r_stats):
    """R-배수 분석 결과 출력"""
    print(f"\n{'='*70}")
    print("R-배수 기반 성과 분석")
    print(f"{'='*70}")
    
    print(f"\n거래 통계:")
    print(f"   • 총 거래: {r_stats['total_trades']}회")
    print(f"   • 수익 거래: {r_stats['winning_trades']}회")
    print(f"   • 손실 거래: {r_stats['losing_trades']}회")
    print(f"   • 승률: {r_stats['win_rate']*100:.2f}%")
    
    print(f"\nR-배수 분석:")
    print(f"   • 평균 수익 R: {r_stats['avg_winning_r']:.2f}R")
    print(f"   • 평균 손실 R: {r_stats['avg_losing_r']:.2f}R")
    print(f"   • 평균 R-배수: {r_stats['avg_r_multiple']:.2f}R")
    print(f"   • 기대값: {r_stats['expectancy']:.2f}R")
    
    print(f"\n극값:")
    print(f"   • 최대 수익: {r_stats['max_winning_r']:.2f}R")
    print(f"   • 최대 손실: {r_stats['max_losing_r']:.2f}R")
    
    print("=" * 70)

# =============================================================================
# 성과 분석 함수
# =============================================================================

def tear_sheet1(data, cash_init=10000):
    """
    백테스트 결과 상세 분석 및 출력
    
    Parameters:
    -----------
    data : pandas.DataFrame
        백테스트 결과 데이터 (Position, Signal, Cumulative_Return 포함)
    cash_init : float
        초기 자본
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
    buy_and_hold = ((data['Close'].iloc[-1] * (1 - fee_rate) / 
                    (data['Close'].iloc[0] * (1 + fee_rate))) - 1)
    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1
    
    # 최종 자산 계산
    final_asset_strategy = cash_init * (1 + final_cum_return)
    final_asset_buyhold = cash_init * (1 + buy_and_hold)
    
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
    sharpe_ratio = ((mean_return - risk_free_rate) / std_return 
                    if std_return > 0 else 0)
    
    # Sortino Ratio (하방 변동성만 고려)
    downside_returns = strategy_daily_return[strategy_daily_return < 0]
    downside_std = (downside_returns.std() * np.sqrt(252) 
                   if len(downside_returns) > 0 else 0)
    sortino_ratio = ((mean_return - risk_free_rate) / downside_std 
                    if downside_std > 0 else 0)
    
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
    mdd_duration = ((mdd_idx - mdd_start_idx).days 
                   if mdd_idx != mdd_start_idx else 0)
    
    # 벤치마크 MDD
    cumulative_returns = (1 + data['Close'].pct_change()).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    mdd_benchmark = drawdown.min()
    
    # =================================================================
    # 6. Calmar Ratio 계산
    # =================================================================
    calmar_ratio = (CAGR_strategy / abs(max_drawdown) 
                   if max_drawdown != 0 else 0)
    calmar_benchmark = (CAGR_benchmark / abs(mdd_benchmark) 
                       if mdd_benchmark != 0 else 0)
    
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
            return_pct = ((sell_price * (1 - fee_rate) / 
                          (buy_price * (1 + fee_rate))) - 1)
            returns.append(return_pct)          
            holding_period = np.busday_count(buy_date.date(), sell_date.date())
            holding_periods.append(holding_period)
    
    # 거래 통계
    profitable_trades = len([r for r in returns if r > 0])
    loss_trades = len([r for r in returns if r <= 0])
    total_trades = len(returns)
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    # 평균 보유 기간
    average_holding_period = (np.mean(holding_periods) 
                             if holding_periods else 0)
    median_holding_period = (np.median(holding_periods) 
                            if holding_periods else 0)
    
    # 평균 수익/손실
    average_profit = (np.mean([r for r in returns if r > 0]) 
                     if profitable_trades > 0 else 0)
    average_loss = (np.mean([r for r in returns if r <= 0]) 
                   if loss_trades > 0 else 0)
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
    
    print(f"\n💰 초기 자본: ${cash_init:,.2f}")
    print(f"\n📈 최종 자산:")
    print(f"   • 전략 최종 자산: ${final_asset_strategy:,.2f}")
    print(f"   • Buy & Hold 최종 자산: ${final_asset_buyhold:,.2f}")
    
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
        'profit_loss_ratio': profit_loss_ratio,
        'final_asset_strategy': final_asset_strategy,
        'final_asset_buyhold': final_asset_buyhold
    }

# =============================================================================
# 백테스트 기간 설정 함수
# =============================================================================

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
    print("TQQQ MOM Strategy2 + 2% R값 시스템 백테스트 기간 설정")
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
        except:
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
        except:
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

def load_tqqq_data_with_period(period_config):
    """
    기간 설정에 따라 TQQQ 데이터 로드 (CSV 파일 사용)
    
    Parameters:
    -----------
    period_config : dict
        백테스트 기간 설정 정보
        
    Returns:
    --------
    pandas.DataFrame : 필터링된 TQQQ 데이터
    """
    print(f"\nTQQQ 데이터 로드 중 (CSV 파일 사용)...")
    
    try:
        # CSV 파일에서 데이터 로드
        df = pd.read_csv('TQQQ_1d.csv', index_col='timestamp', parse_dates=True)
        
        # 컬럼명을 대문자로 변환 (기존 코드 호환성)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        original_len = len(df)
        original_start = df.index[0]
        original_end = df.index[-1]
        
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
            print("기간 필터 결과: 데이터가 비어 있습니다.")
            return pd.DataFrame()
        
        # 기간 정보
        start_date = df.index[0]
        end_date = df.index[-1]
        total_days = (end_date - start_date).days
        
        print(f"TQQQ 데이터 로드 완료: {len(df)}개 거래일 (원본: {original_len}개)")
        print(f"   전체 데이터 기간: {original_start.strftime('%Y-%m-%d')} ~ {original_end.strftime('%Y-%m-%d')}")
        print(f"   백테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({total_days}일, {total_days/365:.1f}년)")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ TQQQ_1d.csv 파일을 찾을 수 없습니다.")
        print(f"   파일이 현재 디렉토리에 있는지 확인하세요.")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ TQQQ 데이터 로드 오류: {e}")
        return pd.DataFrame()

# =============================================================================
# 메인 실행 부분
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TQQQ MOM Strategy2 + 2% R값 시스템 백테스트 (V2)")
    print("=" * 70)

    # =================================================================
    # 1. 백테스트 기간 설정
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

    print(f"전략: MOM Strategy2 (모멘텀 + MFI) + 2% R값 시스템")
    print("=" * 70)

    # =================================================================
    # 2. 실행 확인 및 데이터 로드
    # =================================================================
    print()
    try:
        input("백테스트를 시작하려면 Enter를 누르세요...")
    except EOFError:
        print("자동으로 백테스트를 시작합니다...")

    # TQQQ 데이터 로드
    df = load_tqqq_data_with_period(period_config)
    if df.empty:
        print("데이터 로드 실패")
        exit()

    # =================================================================
    # 3. 기본 정보 출력
    # =================================================================
    print(f"\nTQQQ 기본 정보:")
    print(f"시작가: ${df['Close'].iloc[0]:.2f}")
    print(f"종료가: ${df['Close'].iloc[-1]:.2f}")
    print(f"기간 수익률: {((df['Close'].iloc[-1]/df['Close'].iloc[0])-1)*100:.2f}%")

    # =================================================================
    # 4. 고정 파라미터로 Strategy2 백테스트 (2% R값 시스템)
    # =================================================================
    print("\n" + "=" * 70)
    print("MOM Strategy2 + 2% R값 시스템 백테스트")
    print("=" * 70)

    # 고정 파라미터 설정
    period1 = 6      # 모멘텀 계산 기간
    period2 = 6      # MFI 계산 기간  
    mfi_level = 50   # MFI 임계값
    atr_mult = 2.0   # ATR 배수 (손절거리)

    print(f"\n사용된 파라미터:")
    print(f"   • 모멘텀 기간: {period1}일")
    print(f"   • MFI 기간: {period2}일")
    print(f"   • MFI 임계값: {mfi_level}")
    print(f"   • ATR 배수: {atr_mult}")
    print(f"   • 리스크: 2% 고정")

    # 백테스트 실행
    print(f"\n백테스트 실행 중...")
    t1 = time.time()
    data, ret, trades = mom_strategy2_2pct_risk(
        df, period1, period2, mfi_level, atr_mult)
    t2 = time.time()
    print(f'백테스트 완료 (소요시간: {(t2-t1):.2f}초)')

    print(f"\nMOM Strategy2 + 2% R값 시스템 결과:")
    cash_init = 10000
    stats = tear_sheet1(data, cash_init)

    # R-배수 분석
    r_stats = analyze_r_performance(trades)
    print_r_analysis(r_stats)

    # =================================================================
    # 5. 결과 시각화
    # =================================================================
    print("\n" + "=" * 70)
    print("결과 시각화")
    print("=" * 70)

    # Buy & Hold 수익률 계산
    buy_and_hold = df['Close'] / df['Close'].iloc[0]

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 8))

    # 수익률 플롯
    buy_and_hold.plot(ax=ax, label='Buy & Hold', linewidth=2)
    data['Cumulative_Return'].plot(ax=ax, label='MOM Strategy2 + 2% R값', linewidth=2)

    # 매수/매도 포인트 표시
    buy_price = data['Buy_Price'] / data['Close'].iloc[0]
    sell_price = data['Sell_Price'] / data['Close'].iloc[0]

    buy_price.plot(ax=ax, label='Buy', marker='^', color='green', 
                   markersize=6, alpha=0.7)
    sell_price.plot(ax=ax, label='Sell', marker='v', color='red', 
                    markersize=6, alpha=0.7)

    ax.set_title('TQQQ MOM Strategy2 + 2% R값 시스템 백테스트 결과', fontsize=18)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Returns', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # =================================================================
    # 6. 상세 분석 그래프
    # =================================================================
    # 모멘텀과 MFI 지표 시각화
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True, 
                          height_ratios=(5, 2.5, 2.5))

    # 수익률 비교
    buy_and_hold.plot(ax=ax[0], label='Buy & Hold', linewidth=2)
    data['Cumulative_Return'].plot(ax=ax[0], label='MOM Strategy2 + 2% R값', linewidth=2)
    ax[0].set_ylabel('Cumulative Returns', fontsize=12)
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # 모멘텀 지표
    data['Mom'].plot(ax=ax[1], label='Momentum', color='orange', linewidth=1)
    ax[1].axhline(y=0, color='red', linestyle='-', alpha=0.7)
    ax[1].set_ylabel('Momentum', fontsize=12)
    ax[1].grid(alpha=0.3)

    # MFI 지표
    data['MFI'].plot(ax=ax[2], label='MFI', color='purple', linewidth=1)
    ax[2].axhline(y=mfi_level, color='red', linestyle='-', alpha=0.7)
    ax[2].set_xlabel('Date', fontsize=12)
    ax[2].set_ylabel('MFI', fontsize=12)
    ax[2].grid(alpha=0.3)

    plt.suptitle('TQQQ MOM Strategy2 + 2% R값 시스템 상세 분석', fontsize=16)
    plt.tight_layout()
    plt.show()

    # =================================================================
    # 7. 최종 결과 요약
    # =================================================================
    print("\n" + "=" * 70)
    print("🎯 TQQQ Strategy2 + 2% R값 시스템 최종 결과 요약")
    print("=" * 70)

    # 백테스트 기간 정보
    start_date = df.index[0].strftime('%Y-%m-%d')
    end_date = df.index[-1].strftime('%Y-%m-%d')
    print(f"\n📅 백테스트 기간: {start_date} ~ {end_date}")
    print(f"📊 거래일 수: {len(df)}일 ({stats['trading_period']:.2f}년)")

    print(f"\n💰 수익률 비교:")
    print(f"   • Buy & Hold: {stats['buy_hold_return']*100:.2f}%")
    print(f"   • MOM Strategy2 + 2% R값: {stats['strategy_return']*100:.2f}%")
    excess = (stats['strategy_return'] - stats['buy_hold_return']) * 100
    if excess > 0:
        print(f"   • 초과 수익: +{excess:.2f}%p ✅")
    else:
        print(f"   • 초과 수익: {excess:.2f}%p")

    print(f"\n📈 성과 지표 요약:")
    print(f"   • CAGR: {stats['cagr_strategy']*100:.2f}% (Buy & Hold: {stats['cagr_benchmark']*100:.2f}%)")
    print(f"   • 샤프 비율: {stats['sharpe_ratio']:.3f}")
    print(f"   • 소르티노 비율: {stats['sortino_ratio']:.3f}")
    print(f"   • 칼마 비율: {stats['calmar_ratio']:.3f}")
    print(f"   • 최대 낙폭 (MDD): {stats['max_drawdown']*100:.2f}% (Buy & Hold: {stats['mdd_benchmark']*100:.2f}%)")

    print(f"\n🔧 사용된 파라미터:")
    print(f"   • 모멘텀 기간: {period1}일")
    print(f"   • MFI 기간: {period2}일")
    print(f"   • MFI 임계값: {mfi_level}")
    print(f"   • ATR 배수: {atr_mult}")
    print(f"   • 리스크: 2% 고정")

    print(f"\n📊 거래 통계 요약:")
    print(f"   • 총 거래: {stats['total_trades']}회")
    print(f"   • 승률: {stats['win_rate']*100:.2f}%")
    print(f"   • 수익/손실 비율: {stats['profit_loss_ratio']:.2f}")

    print(f"\n📊 R-배수 통계 요약:")
    print(f"   • 평균 R-배수: {r_stats['avg_r_multiple']:.2f}R")
    print(f"   • 기대값: {r_stats['expectancy']:.2f}R")

    print("\n" + "=" * 70)
    print("✅ 백테스트 완료!")
    print("=" * 70)

