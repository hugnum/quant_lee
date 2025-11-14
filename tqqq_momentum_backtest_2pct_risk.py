# TQQQ 모멘텀 전략 + 2% R값 시스템 백테스트
# 참조: tqqq_momentum_backtest_R.py + backtest_3ema_single.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import time

# =============================================================================
# ATR 및 R값 계산 함수들
# =============================================================================

def calculate_atr_indicators(df, atr_length=14):
    """ATR 및 관련 지표 계산"""
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_length)
    df['ATR_21'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=21)
    return df

def calculate_position_size_2pct_risk(cash, entry_price, atr_value, 
                                       atr_mult=2.0):
    """전체 자본의 2% 리스크로 포지션 사이즈 계산"""
    # 2% 리스크 금액
    risk_amount = cash * 0.02
    
    # R값 거리 계산 (ATR × 배수)
    r_distance = atr_value * atr_mult
    
    # ATR이 0이거나 NaN인 경우 거래 건너뛰기
    if not np.isfinite(r_distance) or r_distance <= 0:
        return 0.0, 0.0
    
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

# 트레일링 파라미터 (3-EMA 시스템과 동일)
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
# 2% R값 기반 모멘텀 전략
# =============================================================================

def mom_strategy_2pct_risk(df, p1, p2, p3, ml, atr_mult, verbose=True):
    """
    모멘텀 + MFI + 2% R값 시스템 (가속도 제거)
    - p1: 모멘텀 계산 기간
    - p2: MFI 계산 기간  
    - p3: 가속도 계산 기간 (사용 안함)
    - ml: MFI 레벨
    - atr_mult: ATR 배수 (손절거리)
    """
    fee_rate = 0.001
    data = df.copy()
    
    # ATR 지표 계산
    data = calculate_atr_indicators(data)
    
    # 기존 모멘텀 지표 계산
    data['Mom'] = data['close'].pct_change(periods=p1)
    data['MFI'] = ta.volume.money_flow_index(data['high'], 
                data['low'], data['close'], data['volume'], p2)
    data['Mom_Acceleration'] = data['Mom'].pct_change(periods=p3)
    data.dropna(inplace=True)

    # 모멘텀 + MFI 포지션 결정 (가속도 제거)
    mom_pos = pd.Series(np.where((data['Mom'] > 0) & 
                                     (data['MFI'] > ml), 1, 0), 
                        index=data.index)
    mom_signals = mom_pos.diff().fillna(0)

    # 백테스트 엔진 초기화
    cash_init = 10000
    cash = cash_init
    asset = np.zeros(len(data))
    asset[0] = cash

    prices = data['close'].values
    signals = mom_signals.values
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
    
    for i in range(1, len(data)):
        current_price = prices[i]
        current_atr = atr_values[i]
        
        # 포지션 없음 - 진입 확인
        if pos == 0:
            if signals[i] == 1:  # 모멘텀 신호 발생
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

        # 자산 가치 계산
        if pos == 0:
            asset[i] = cash
        else:
            asset[i] = cash + position_size * current_price

    # 결과 데이터프레임 생성
    data['Position'] = positions
    data['Signal'] = data['Position'].diff().fillna(0)
    data['Buy_Price'] = np.where(data['Signal'] == 1, data['close'], np.nan)
    data['Sell_Price'] = np.where(data['Signal'] == -1, data['close'], np.nan)
    data['Cumulative_Return'] = asset / cash_init
    
    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1
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
    
    avg_winning_r = np.mean([t['r_multiple'] for t in winning_trades]) if winning_trades else 0
    avg_losing_r = np.mean([t['r_multiple'] for t in losing_trades]) if losing_trades else 0
    avg_r_multiple = np.mean([t['r_multiple'] for t in trades])
    
    expectancy = (win_rate * avg_winning_r) + ((1 - win_rate) * avg_losing_r)
    
    max_winning_r = max([t['r_multiple'] for t in winning_trades]) if winning_trades else 0
    max_losing_r = min([t['r_multiple'] for t in losing_trades]) if losing_trades else 0
    
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
    print(f"\n{'='*60}")
    print("R-배수 기반 성과 분석")
    print(f"{'='*60}")
    
    print(f"거래 통계:")
    print(f"   - 총 거래: {r_stats['total_trades']}회")
    print(f"   - 수익 거래: {r_stats['winning_trades']}회")
    print(f"   - 손실 거래: {r_stats['losing_trades']}회")
    print(f"   - 승률: {r_stats['win_rate']*100:.2f}%")
    
    print(f"\nR-배수 분석:")
    print(f"   - 평균 수익 R: {r_stats['avg_winning_r']:.2f}R")
    print(f"   - 평균 손실 R: {r_stats['avg_losing_r']:.2f}R")
    print(f"   - 평균 R-배수: {r_stats['avg_r_multiple']:.2f}R")
    print(f"   - 기대값: {r_stats['expectancy']:.2f}R")
    
    print(f"\n극값:")
    print(f"   - 최대 수익: {r_stats['max_winning_r']:.2f}R")
    print(f"   - 최대 손실: {r_stats['max_losing_r']:.2f}R")

# =============================================================================
# 백테스트 실행
# =============================================================================

def load_tqqq_data():
    """TQQQ 데이터 로드"""
    try:
        df = pd.read_csv('TQQQ_1d.csv', index_col='timestamp', parse_dates=True)
        print(f"TQQQ 데이터 로드 완료: {len(df)}개 거래일")
        print(f"   기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        return df
    except Exception as e:
        print(f"❌ 데이터 로드 오류: {e}")
        return pd.DataFrame()

def main():
    print("=" * 80)
    print("TQQQ 모멘텀 전략 + 2% R값 시스템 백테스트")
    print("=" * 80)
    
    # 데이터 로드
    df = load_tqqq_data()
    if df.empty:
        print("데이터 로드 실패")
        return
    
    # 전략 파라미터 (최적화 결과)
    p1, p2, p3 = 14, 10, 3  # 모멘텀, MFI, 가속도 기간
    mfi_level = 47.5  # MFI 레벨
    atr_mult = 2.0  # ATR 배수
    
    print(f"\n전략 파라미터:")
    print(f"   - 모멘텀 기간: {p1}일")
    print(f"   - MFI 기간: {p2}일, 레벨: {mfi_level}")
    print(f"   - 가속도 기간: {p3}일 (사용 안함)")
    print(f"   - ATR 배수: {atr_mult}")
    print(f"   - 리스크: 2% 고정")
    
    # 백테스트 실행
    print(f"\n백테스트 실행 중...")
    start_time = time.time()
    
    data, final_return, trades = mom_strategy_2pct_risk(
        df, p1, p2, p3, mfi_level, atr_mult)
    
    end_time = time.time()
    print(f"백테스트 완료 (소요시간: {end_time - start_time:.2f}초)")
    
    # 초기 자본 설정
    cash_init = 10000
    
    # 기본 성과 분석
    print("\n" + "=" * 70)
    print("📊 백테스트 성과 분석")
    print("=" * 70)
    
    # 투자 기간 계산
    trading_period = len(data) / 252
    
    # Buy & Hold 수익률 (수수료 포함)
    fee_rate = 0.001
    buy_hold_return = ((df['close'].iloc[-1] * (1 - fee_rate)) / 
                       (df['close'].iloc[0] * (1 + fee_rate))) - 1
    
    # 최종 자산 계산
    final_asset_strategy = cash_init * (1 + final_return)
    final_asset_buyhold = cash_init * (1 + buy_hold_return)
    
    print(f"\n💰 초기 자본: ${cash_init:,.2f}")
    print(f"\n📈 최종 자산:")
    print(f"   • 전략 최종 자산: ${final_asset_strategy:,.2f}")
    print(f"   • Buy & Hold 최종 자산: ${final_asset_buyhold:,.2f}")
    
    print(f"\n📊 수익률 비교:")
    print(f"   • 전략 누적 수익률: {final_return*100:.2f}%")
    print(f"   • Buy & Hold 수익률: {buy_hold_return*100:.2f}%")
    excess_return = (final_return - buy_hold_return) * 100
    if excess_return > 0:
        print(f"   • 초과 수익: +{excess_return:.2f}%p ✅")
    else:
        print(f"   • 초과 수익: {excess_return:.2f}%p")
    
    # CAGR 계산
    CAGR_strategy = (data['Cumulative_Return'].iloc[-1]) ** (1/trading_period) - 1
    CAGR_benchmark = (buy_hold_return + 1) ** (1/trading_period) - 1
    
    print(f"\n📈 연평균 수익률 (CAGR):")
    print(f"   • 전략 CAGR: {CAGR_strategy*100:.2f}%")
    print(f"   • Buy & Hold CAGR: {CAGR_benchmark*100:.2f}%")
    
    # MDD 계산
    # 전략 MDD
    data['Cumulative_Max'] = data['Cumulative_Return'].cummax()
    data['Drawdown'] = data['Cumulative_Return'] / data['Cumulative_Max'] - 1
    max_drawdown = data['Drawdown'].min()
    
    # MDD 발생 기간 찾기
    mdd_idx = data['Drawdown'].idxmin()
    mdd_start_idx = data.loc[:mdd_idx, 'Cumulative_Return'].idxmax()
    mdd_duration = (mdd_idx - mdd_start_idx).days if mdd_idx != mdd_start_idx else 0
    
    # 벤치마크 MDD
    cumulative_returns = (1 + df['close'].pct_change()).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    mdd_benchmark = drawdown.min()
    
    print(f"\n📉 최대 낙폭 (Maximum Drawdown):")
    print(f"   • 전략 MDD: {max_drawdown*100:.2f}%")
    print(f"   • Buy & Hold MDD: {mdd_benchmark*100:.2f}%")
    if mdd_duration > 0:
        print(f"   • MDD 지속 기간: {mdd_duration}일 ({mdd_duration/365:.1f}년)")
        print(f"   • MDD 시작: {mdd_start_idx.strftime('%Y-%m-%d')}")
        print(f"   • MDD 최저점: {mdd_idx.strftime('%Y-%m-%d')}")
    
    # 샤프 비율 계산
    risk_free_rate = 0.003
    strategy_daily_return = data['Cumulative_Return'].pct_change().fillna(0)
    mean_return = strategy_daily_return.mean() * 252
    std_return = strategy_daily_return.std() * np.sqrt(252)
    sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
    
    # Calmar Ratio 계산
    calmar_ratio = CAGR_strategy / abs(max_drawdown) if max_drawdown != 0 else 0
    calmar_benchmark = CAGR_benchmark / abs(mdd_benchmark) if mdd_benchmark != 0 else 0
    
    print(f"\n⚡ 위험 조정 수익률 지표:")
    print(f"   • 샤프 비율 (Sharpe Ratio): {sharpe_ratio:.3f}")
    print(f"   • 칼마 비율 (Calmar Ratio): {calmar_ratio:.3f}")
    print(f"     (Buy & Hold Calmar: {calmar_benchmark:.3f})")
    
    print(f"\n📅 백테스트 기간:")
    print(f"   • 시작일: {data.index[0].strftime('%Y-%m-%d')}")
    print(f"   • 종료일: {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   • 거래일 수: {len(data)}일")
    print(f"   • 투자 기간: {trading_period:.2f}년")
    
    print("=" * 70)
    
    # R-배수 분석
    r_stats = analyze_r_performance(trades)
    print_r_analysis(r_stats)
    
    # 시각화
    print("\n결과 시각화...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 수익률 비교
    buy_hold_cum = df['close'] / df['close'].iloc[0]
    buy_hold_cum.plot(ax=ax1, label='Buy & Hold', linewidth=2)
    data['Cumulative_Return'].plot(ax=ax1, label='2% R값 전략', linewidth=2)
    ax1.set_title('TQQQ 2% R값 모멘텀 전략 vs Buy & Hold')
    ax1.set_ylabel('누적 수익률')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # R-배수 분포
    if trades:
        r_multiples = [t['r_multiple'] for t in trades]
        ax2.hist(r_multiples, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('R-배수 분포')
        ax2.set_xlabel('R-배수')
        ax2.set_ylabel('거래 횟수')
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n백테스트 완료!")

if __name__ == '__main__':
    main()

