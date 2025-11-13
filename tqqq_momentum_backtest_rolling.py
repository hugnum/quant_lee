
# TQQQ 모멘텀 전략 백테스트
# 참조: ref/ch_08_momentum_strategy_annotated.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import time
from datetime import datetime

# 필요한 함수들 (ref 파일에서 가져옴)
def mom_strategy1(df, p, sl, verbose=True):
    fee_rate = 0.001
    data = df.copy()
    period = p
    stop_loss = sl
    
    data['Mom'] = data['Close'].pct_change(periods=period)
    data.dropna(inplace=True)

    mom_pos = pd.Series(np.where(data['Mom']>0, 1, 0), \
                        index=data.index)
    mom_signals = mom_pos.diff().fillna(0)

    cash_init = 10000
    cash = cash_init
    asset = np.zeros(len(data))
    asset[0] = cash

    prices = data['Close'].values
    signals = mom_signals.values
    positions = np.zeros(len(data))
    pos = 0    
    
    for i in range(1, len(data)):
        if pos == 0:
            if signals[i] == 1: # 모멘텀 양전 -> 매수
                pos = 1
                positions[i] = 1
                entry_price = prices[i]
                num = int(cash/(entry_price*(1+fee_rate)))
                cash -= entry_price*num*(1+fee_rate)
                stop_loss_price = entry_price*(1 - stop_loss)
        elif pos == 1:
            if prices[i] < stop_loss_price: # 손절 발생
                pos = 0
                cash += prices[i]*num*(1-fee_rate)  
            else: # 손절가 갱신
                positions[i] = 1
                stop_loss_price =\
                max(stop_loss_price, prices[i]*(1 - stop_loss))

        # asset 갱신
        if pos == 0:
            asset[i] = cash
        elif pos == 1:
            asset[i] = cash + prices[i]*num    

    data['Position'] = positions
    data['Signal'] = data['Position'].diff().fillna(0)
    
    data['Buy_Price'] = \
    np.where(data['Signal'] == 1, data['Close'], np.nan)
    data['Sell_Price'] = \
    np.where(data['Signal'] == -1, data['Close'], np.nan)   
    
    data['Cumulative_Return'] = asset/cash_init
    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1
    if verbose:
        print(f'Final cumulative return of the strategy: '
          f'{100*final_cum_return:.2f}%')
    return data, final_cum_return

def mom_strategy2(df, p1, p2, ml, sl, verbose=True):
    fee_rate = 0.001
    data = df.copy()

    period1 = p1
    period2 = p2
    mfi_level = ml
    stop_loss = sl
    
    data['Mom'] = data['Close'].pct_change(periods=period1)
    data['MFI'] = ta.volume.money_flow_index(data.High, \
                data.Low, data.Close, data.Volume, period2) 
    data.dropna(inplace=True)

    mom_pos = pd.Series(np.where(data['Mom']>0, 1, 0), \
                        index=data.index)
    mom_signals = mom_pos.diff().fillna(0)

    cash_init = 10000
    cash = cash_init
    asset = np.zeros(len(data))
    asset[0] = cash

    prices = data['Close'].values
    signals = mom_signals.values
    mfi = data['MFI'].values
    positions = np.zeros(len(data))
    pos = 0    
    
    for i in range(1, len(data)):
        if pos == 0:
            if signals[i] == 1 and mfi[i] > mfi_level: # 매수
                pos = 1
                positions[i] = 1
                entry_price = prices[i]
                num = int(cash/(entry_price*(1+fee_rate)))
                cash -= entry_price*num*(1+fee_rate)
                stop_loss_price = entry_price*(1 - stop_loss)
        elif pos == 1:
            if prices[i] < stop_loss_price: # 손절 발생
                pos = 0
                cash += prices[i]*num*(1-fee_rate)  
            else: # 손절가 갱신
                positions[i] = 1
                stop_loss_price =\
                max(stop_loss_price, prices[i]*(1 - stop_loss))

        # asset 갱신
        if pos == 0:
            asset[i] = cash
        elif pos == 1:
            asset[i] = cash + prices[i]*num    

    data['Position'] = positions
    data['Signal'] = data['Position'].diff().fillna(0)
    
    data['Buy_Price'] = \
    np.where(data['Signal'] == 1, data['Close'], np.nan)
    data['Sell_Price'] = \
    np.where(data['Signal'] == -1, data['Close'], np.nan)   
    
    data['Cumulative_Return'] = asset/cash_init
    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1
    if verbose:
        print(f'Final cumulative return of the strategy: '
          f'{100*final_cum_return:.2f}%')
    return data, final_cum_return

def mom_strategy3(df, p1, p2, p3, ml, sl, verbose=True):
    """
    모멘텀 + MFI + 가속도 모멘텀 전략
    - p1: 모멘텀 계산 기간
    - p2: MFI 계산 기간
    - p3: 가속도 계산 기간
    - ml: MFI 레벨
    - sl: 손절 비율
    """
    fee_rate = 0.001
    data = df.copy()
    period1 = p1
    period2 = p2
    period3 = p3
    mfi_level = ml
    stop_loss = sl
    
    # 1차 모멘텀 계산
    data['Mom'] = data['Close'].pct_change(periods=period1)
    # MFI 계산
    data['MFI'] = ta.volume.money_flow_index(data.High, 
                data.Low, data.Close, data.Volume, period2)
    # 2차 모멘텀 (가속도) 계산
    data['Mom_Acceleration'] = data['Mom'].pct_change(periods=period3)
    data.dropna(inplace=True)

    # 모멘텀 + MFI + 가속도 모멘텀 포지션 결정
    # 모멘텀 > 0 AND MFI > 설정값 AND 가속도 > 0 → 매수 (1)
    # 그 외 → 매도 (0)
    mom_pos = pd.Series(np.where((data['Mom'] > 0) & (data['MFI'] > mfi_level) & (data['Mom_Acceleration'] > 0), 1, 0), 
                        index=data.index)
    mom_signals = mom_pos.diff().fillna(0)

    cash_init = 10000
    cash = cash_init
    asset = np.zeros(len(data))
    asset[0] = cash

    prices = data['Close'].values
    signals = mom_signals.values
    positions = np.zeros(len(data))
    pos = 0    
    
    for i in range(1, len(data)):
        if pos == 0:
            if signals[i] == 1: # 가속도 모멘텀 양전 -> 매수
                pos = 1
                positions[i] = 1
                entry_price = prices[i]
                num = int(cash/(entry_price*(1+fee_rate)))
                cash -= entry_price*num*(1+fee_rate)
                stop_loss_price = entry_price*(1 - stop_loss)
        elif pos == 1:
            if prices[i] < stop_loss_price: # 손절 발생
                pos = 0
                cash += prices[i]*num*(1-fee_rate)  
            else: # 손절가 갱신
                positions[i] = 1
                stop_loss_price = \
                max(stop_loss_price, prices[i]*(1 - stop_loss))

        # asset 갱신
        if pos == 0:
            asset[i] = cash
        elif pos == 1:
            asset[i] = cash + prices[i]*num    

    data['Position'] = positions
    data['Signal'] = data['Position'].diff().fillna(0)
    
    data['Buy_Price'] = \
    np.where(data['Signal'] == 1, data['Close'], np.nan)
    data['Sell_Price'] = \
    np.where(data['Signal'] == -1, data['Close'], np.nan)   
    
    data['Cumulative_Return'] = asset/cash_init
    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1
    if verbose:
        print(f'Final cumulative return of the strategy: '
          f'{100*final_cum_return:.2f}%')
    return data, final_cum_return

def tear_sheet1(data, strategy_name="Strategy"):
    """개선된 백테스트 결과 출력 함수"""
    fee_rate = 0.001
    
    # 투자기간 계산
    trading_period = len(data)/252
    
    # 수익률 계산
    buy_and_hold = data['Close'].iloc[-1]*(1-fee_rate)/(data['Close'].iloc[0]*(1+fee_rate)) - 1
    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1
    
    # CAGR 계산
    CAGR_strategy = (data['Cumulative_Return'].iloc[-1])**(1/trading_period)-1
    CAGR_benchmark = (buy_and_hold+1)**(1/trading_period)-1
    
    # 샤프 지수 계산
    risk_free_rate = 0.003
    strategy_daily_return = data['Cumulative_Return'].pct_change().fillna(0)
    mean_return = strategy_daily_return.mean()*252
    std_return = strategy_daily_return.std()*np.sqrt(252)
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    
    # 최대 낙폭 계산
    data['Cumulative_Max'] = data['Cumulative_Return'].cummax()
    data['Drawdown'] = data['Cumulative_Return'] / data['Cumulative_Max'] - 1
    max_drawdown = data['Drawdown'].min()
    cumulative_returns = (1 + data['Close'].pct_change()).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns/running_max - 1
    mdd_benchmark = drawdown.min()
    
    # 거래 통계 계산
    buy_signals = data[data['Signal'] == 1].index
    sell_signals = data[data['Signal'] == -1].index
    returns = []
    holding_periods = []
    
    for buy_date in buy_signals:
        sell_dates = sell_signals[sell_signals > buy_date]
        if not sell_dates.empty:
            sell_date = sell_dates[0]
            buy_price = data.loc[buy_date, 'Close']
            sell_price = data.loc[sell_date, 'Close']
            return_pct = sell_price*(1-fee_rate)/(buy_price*(1+fee_rate)) - 1
            returns.append(return_pct)          
            holding_period = np.busday_count(buy_date.date(), sell_date.date())
            holding_periods.append(holding_period)
    
    profitable_trades = len([r for r in returns if r > 0])
    loss_trades = len([r for r in returns if r <= 0])
    total_trades = len(returns)
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    # 평균 보유 기간
    average_holding_period = np.mean(holding_periods) if holding_periods else 0
    
    # 평균 이익/손실
    average_profit = np.mean([r for r in returns if r > 0]) if profitable_trades > 0 else 0
    average_loss = np.mean([r for r in returns if r <= 0]) if loss_trades > 0 else 0
    profit_loss_ratio = average_profit / abs(average_loss) if average_loss != 0 else np.inf
    
    # 결과 딕셔너리 반환
    results = {
        'strategy_name': strategy_name,
        'trading_period': trading_period,
        'strategy_return': final_cum_return,
        'benchmark_return': buy_and_hold,
        'strategy_cagr': CAGR_strategy,
        'benchmark_cagr': CAGR_benchmark,
        'sharpe_ratio': sharpe_ratio,
        'strategy_mdd': max_drawdown,
        'benchmark_mdd': mdd_benchmark,
        'profitable_trades': profitable_trades,
        'loss_trades': loss_trades,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_holding_period': average_holding_period,
        'avg_profit': average_profit,
        'avg_loss': average_loss,
        'profit_loss_ratio': profit_loss_ratio
    }
    
    return results

def print_strategy_results(results):
    """전략 결과를 가독성 좋게 출력"""
    print(f"\n{'='*60}")
    print(f"{results['strategy_name']} 백테스트 결과")
    print(f"{'='*60}")
    
    print(f"📊 투자 기간: {results['trading_period']:.1f}년")
    print(f"💰 수익률: {results['strategy_return']*100:.2f}% (벤치마크: {results['benchmark_return']*100:.2f}%)")
    print(f"📈 연평균 수익률: {results['strategy_cagr']*100:.2f}% (벤치마크: {results['benchmark_cagr']*100:.2f}%)")
    print(f"⚡ 샤프 비율: {results['sharpe_ratio']:.2f}")
    print(f"📉 최대 낙폭: {results['strategy_mdd']*100:.2f}% (벤치마크: {results['benchmark_mdd']*100:.2f}%)")
    
    print(f"\n📈 거래 통계:")
    print(f"   • 총 거래 횟수: {results['total_trades']}회")
    print(f"   • 수익 거래: {results['profitable_trades']}회")
    print(f"   • 손실 거래: {results['loss_trades']}회")
    print(f"   • 승률: {results['win_rate']*100:.2f}%")
    print(f"   • 평균 보유 기간: {results['avg_holding_period']:.1f}일")
    print(f"   • 평균 수익률: {results['avg_profit']*100:.3f}%")
    print(f"   • 평균 손실률: {results['avg_loss']*100:.3f}%")
    print(f"   • 수익/손실 비율: {results['profit_loss_ratio']:.2f}")

def create_strategy_comparison_table(results_list):
    """전략 비교 표 생성"""
    print(f"\n{'='*100}")
    print("📊 전략 비교 표")
    print(f"{'='*100}")
    
    # 표 헤더
    print(f"{'전략':<20} {'수익률(%)':<12} {'CAGR(%)':<12} {'샤프비율':<10} {'MDD(%)':<12} {'승률(%)':<10} {'거래수':<8}")
    print(f"{'-'*100}")
    
    # 각 전략 결과 출력
    for results in results_list:
        print(f"{results['strategy_name']:<20} "
              f"{results['strategy_return']*100:<12.2f} "
              f"{results['strategy_cagr']*100:<12.2f} "
              f"{results['sharpe_ratio']:<10.2f} "
              f"{results['strategy_mdd']*100:<12.2f} "
              f"{results['win_rate']*100:<10.2f} "
              f"{results['total_trades']:<8}")
    
    print(f"{'-'*100}")
    
    # 벤치마크 정보
    benchmark = results_list[0]  # 모든 전략이 같은 벤치마크 사용
    print(f"{'Buy & Hold':<20} "
          f"{benchmark['benchmark_return']*100:<12.2f} "
          f"{benchmark['benchmark_cagr']*100:<12.2f} "
          f"{'N/A':<10} "
          f"{benchmark['benchmark_mdd']*100:<12.2f} "
          f"{'N/A':<10} "
          f"{'N/A':<8}")
    
    print(f"{'='*100}")

def mom_parameter_optimizer1(input_df):
    period = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,\
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    stop_loss = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,\
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2] 
    ret_list = []  

    for x1, x2, in [(a,b) for a in period for b in stop_loss]:
        df = input_df.copy()
        data, ror = mom_strategy1(df, x1, x2, verbose=False)
        ret_list.append((x1, x2, ror))

    max_ror = max(ret_list, key=lambda x:x[2])[2]
    max_tups = [tup for tup in ret_list if tup[2] == max_ror]
    params1 = [tup[0] for tup in max_tups]
    params2 = [tup[1] for tup in max_tups]
    opt_param1 = int(np.median(params1))
    opt_param2 = round(np.median(params2),4)

    optimal_df = pd.DataFrame(ret_list, \
                columns=['period','stop_loss','ror'])
    print(f'Max Tuples:{max_tups}')
    print(f'Optimal Parameters:{opt_param1, opt_param2}, '
    f'Optimized Return:{100*max_ror:.2f}%')

    return (opt_param1, opt_param2), optimal_df

def mom_parameter_optimizer2(input_df):
    period1 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    period2 = [3, 4, 5, 6, 7, 8, 9, 10]
    mfi_level = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,\
                 56, 57, 58]
    stop_loss = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12,\
                 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]  
    ret_list = []  

    for x1, x2, x3, x4 in [(a,b,c,d) for a in period1 for b in \
                period2 for c in mfi_level for d in stop_loss]:
        df = input_df.copy()
        data, ror = mom_strategy2(df, x1, x2, x3, x4, verbose=False)
        ret_list.append((x1, x2, x3, x4, ror))

    max_ror = max(ret_list, key=lambda x:x[4])[4]
    max_tups = [tup for tup in ret_list if tup[4] == max_ror]
    params1 = [tup[0] for tup in max_tups]
    params2 = [tup[1] for tup in max_tups]
    params3 = [tup[2] for tup in max_tups]
    params4 = [tup[3] for tup in max_tups]    
    opt_param1 = int(np.median(params1))
    opt_param2 = int(np.median(params2))
    opt_param3 = round(np.median(params3),1)
    opt_param4 = round(np.median(params4),4)    

    optimal_df = pd.DataFrame(ret_list, columns=
                ['period1','period2', 'mfi_level', 'stop_loss','ror'])
    print(f'Max Tuples:{max_tups}')
    print(f'Optimal Parameters:'
    f'{opt_param1, opt_param2, opt_param3, opt_param4}, '
    f'Optimized Return:{100*max_ror:.2f}%')

    return (opt_param1, opt_param2, opt_param3, opt_param4), optimal_df

def mom_parameter_optimizer3(input_df):
    """
    Strategy3 파라미터 최적화 (모멘텀 + MFI + 가속도 모멘텀)
    - period1: 모멘텀 계산 기간
    - period2: MFI 계산 기간
    - period3: 가속도 계산 기간
    - mfi_level: MFI 레벨
    - stop_loss: 손절 비율
    """
    period1 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    period2 = [3, 4, 5, 6, 7, 8, 9, 10]
    period3 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    mfi_level = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                 56, 57, 58]
    stop_loss = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12,
                 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]  
    ret_list = []  

    for x1, x2, x3, x4, x5 in [(a,b,c,d,e) for a in period1 for b in period2 for c in period3 for d in mfi_level for e in stop_loss]:
        df = input_df.copy()
        data, ror = mom_strategy3(df, x1, x2, x3, x4, x5, verbose=False)
        ret_list.append((x1, x2, x3, x4, x5, ror))

    max_ror = max(ret_list, key=lambda x:x[5])[5]
    max_tups = [tup for tup in ret_list if tup[5] == max_ror]
    params1 = [tup[0] for tup in max_tups]
    params2 = [tup[1] for tup in max_tups]
    params3 = [tup[2] for tup in max_tups]
    params4 = [tup[3] for tup in max_tups]
    params5 = [tup[4] for tup in max_tups]
    opt_param1 = int(np.median(params1))
    opt_param2 = int(np.median(params2))
    opt_param3 = int(np.median(params3))
    opt_param4 = round(np.median(params4),1)
    opt_param5 = round(np.median(params5),4)    

    optimal_df = pd.DataFrame(ret_list, columns=
                ['period1','period2', 'period3', 'mfi_level', 'stop_loss','ror'])
    print(f'Max Tuples:{max_tups}')
    print(f'Optimal Parameters:'
    f'{opt_param1, opt_param2, opt_param3, opt_param4, opt_param5}, '
    f'Optimized Return:{100*max_ror:.2f}%')

    return (opt_param1, opt_param2, opt_param3, opt_param4, opt_param5), optimal_df

# =============================================================================
# 대화형 백테스트 기간 설정
# =============================================================================

def get_backtest_period():
    """대화형으로 백테스트 기간 설정"""
    print("\n" + "=" * 80)
    print("📅 TQQQ 모멘텀 전략 백테스트 기간 설정")
    print("=" * 80)
    # 현재 날짜 계산
    from datetime import datetime
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    print("\n백테스트 기간을 선택하세요:")
    print(f"1️⃣  전체 데이터 사용 (2015-01-01 ~ {current_date}, 약 10년)")
    print("2️⃣  특정 기간 지정 (예: 2020-01-01 ~ 2024-12-31)")
    print("3️⃣  최근 N일 사용 (예: 1000일)")
    print()
    
    choice = input("선택 (1/2/3) [기본값: 1]: ").strip() or "1"
    
    if choice == "1":
        return {
            'mode': 'full',
            'start_date': '2015-01-01',
            'end_date': current_date,
            'days': None
        }
    
    elif choice == "2":
        print("\n📅 특정 기간을 입력하세요 (YYYY-MM-DD 형식):")
        print("   예시: 2020년만 테스트 → 2020-01-01 ~ 2020-12-31")
        start = input("시작일 [기본값: 2020-01-01]: ").strip() or "2020-01-01"
        end = input(f"종료일 [기본값: {current_date}]: ").strip() or current_date
        
        try:
            # 날짜 검증
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            if end_dt <= start_dt:
                print("⚠️ 종료일이 시작일보다 이전입니다. 전체 데이터로 진행합니다.")
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
            print("⚠️ 잘못된 날짜 형식입니다. 전체 데이터로 진행합니다.")
            return {
                'mode': 'full',
                'start_date': '2015-01-01',
                'end_date': current_date,
                'days': None
            }
    
    elif choice == "3":
        print("\n📅 최근 N일 데이터 사용:")
        print("   추천: 1000일(약 4년), 1500일(약 6년), 2000일(약 8년)")
        print("   엔터만 치면 1000일 사용")
        try:
            days_input = input("일수 입력 [기본값: 1000]: ").strip()
            if not days_input:  # 엔터만 친 경우
                days = 1000
            else:
                days = int(days_input)
            
            if days <= 0:
                print("⚠️ 일수는 양수여야 합니다. 전체 데이터로 진행합니다.")
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
            print("⚠️ 잘못된 입력입니다. 전체 데이터로 진행합니다.")
            return {
                'mode': 'full',
                'start_date': '2015-01-01',
                'end_date': current_date,
                'days': None
            }
    
    else:
        print("⚠️ 잘못된 선택입니다. 전체 데이터로 진행합니다.")
        return {
            'mode': 'full',
            'start_date': '2015-01-01',
            'end_date': current_date,
            'days': None
        }

def load_tqqq_data_with_period(period_config):
    """기간 설정에 따라 TQQQ 데이터 로드"""
    ticker = 'TQQQ'
    
    print(f"\n📊 TQQQ 데이터 다운로드 중...")
    
    try:
        # 전체 데이터 다운로드 (최대 범위) - 현재 날짜까지
        from datetime import datetime
        current_date = datetime.now().strftime('%Y-%m-%d')
        df = yf.download(ticker, start='2015-01-01', end=current_date)
        
        # MultiIndex 컬럼을 단일 레벨로 변환
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        original_len = len(df)
        original_start = df.index[0]
        original_end = df.index[-1]
        
        # 백테스트 기간 필터링
        if period_config['mode'] == 'range':
            # 특정 기간 사용
            start_dt = pd.to_datetime(period_config['start_date'])
            end_dt = pd.to_datetime(period_config['end_date'])
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            print(f"   📅 기간 필터: {period_config['start_date']} ~ {period_config['end_date']}")
        
        elif period_config['mode'] == 'recent':
            # 최근 N일 사용
            days = period_config['days']
            df = df.tail(days)
            print(f"   📅 최근 {days}일 데이터 사용")
        
        else:  # 'full'
            print(f"   📅 전체 데이터 사용")
        
        # 시간 정렬 보장
        df = df.sort_index()
        
        # 빈 데이터 체크
        if df.empty:
            print("❌ 기간 필터 결과: 데이터가 비어 있습니다.")
            return pd.DataFrame()
        
        # 기간 정보
        start_date = df.index[0]
        end_date = df.index[-1]
        total_days = (end_date - start_date).days
        
        print(f"✅ TQQQ 데이터 로드 완료: {len(df)}개 거래일 (원본: {original_len}개)")
        print(f"   전체 데이터 기간: {original_start.strftime('%Y-%m-%d')} ~ {original_end.strftime('%Y-%m-%d')}")
        print(f"   백테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({total_days}일, {total_days/365:.1f}년)")
        
        return df
        
    except Exception as e:
        print(f"❌ TQQQ 데이터 로드 오류: {e}")
        return pd.DataFrame()

# =============================================================================
# TQQQ MOM Strategy2 롤링 백테스트 실행
# =============================================================================

print("=" * 80)
print("🚀 TQQQ MOM Strategy2 롤링 백테스트")
print("=" * 80)

print(f"📌 전략: MOM Strategy2 (모멘텀 + MFI)")
print(f"📌 롤링 테스트: 6개월 간격 10개 시점")
print(f"📌 각 시점: 훈련(5년) → 최적화 → 테스트(2년)")
print("=" * 80)

# 실행 확인
print()
input("⏎ 롤링 백테스트를 시작하려면 Enter를 누르세요...")

# =============================================================================
# MOM Strategy2 롤링 백테스트
# =============================================================================

def rolling_test_mom2(ticker, date):
    """
    MOM Strategy2 롤링 테스트 함수
    - 훈련 기간: 기준일 5년 전 ~ 기준일
    - 테스트 기간: 기준일 ~ 기준일 2년 후
    """
    from datetime import datetime
    
    # 데이터 다운로드
    middle_date = date
    middle_date_dt = datetime.strptime(middle_date, '%Y-%m-%d')
    start_date_dt = middle_date_dt.replace(year=middle_date_dt.year - 5)
    start_date = start_date_dt.strftime('%Y-%m-%d')   
    end_date_dt = middle_date_dt.replace(year=middle_date_dt.year + 2)
    end_date = end_date_dt.strftime('%Y-%m-%d')
    df = yf.download(ticker, start_date, end_date)
    
    # MultiIndex 컬럼을 단일 레벨로 변환
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # 훈련 데이터로 파라미터 최적화
    df_train = df.loc[start_date:middle_date].copy()
    optimal_params, optimal_df = mom_parameter_optimizer2(df_train)   
    
    # 테스트 데이터로 백테스트
    df_test = df.loc[middle_date:].copy()
    data, ret = mom_strategy2(df_test, optimal_params[0], optimal_params[1], 
                              optimal_params[2], optimal_params[3])

    # 연평균 성장률 CAGR
    fee_rate = 0.001
    trading_period = len(data)/252 # in years   
    buy_and_hold = data['Close'].iloc[-1]*(1-fee_rate)/(data['Close'].iloc[0]*(1+fee_rate))
    CAGR_strategy = (data['Cumulative_Return'].iloc[-1])**(1/trading_period)-1
    CAGR_benchmark = (buy_and_hold)**(1/trading_period)-1

    # 최대 낙폭 Maximum Drawdown
    data['Cumulative_Max'] = data['Cumulative_Return'].cummax()
    data['Drawdown'] = data['Cumulative_Return'] / data['Cumulative_Max'] - 1
    mdd_strategy = data['Drawdown'].min()

    cumulative_returns = (1 + data['Close'].pct_change()).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns/running_max - 1
    mdd_benchmark = drawdown.min()  

    return CAGR_strategy, mdd_strategy, CAGR_benchmark, mdd_benchmark

# 롤링 테스트 실행
print("\n" + "=" * 80)
print("🔄 MOM Strategy2 롤링 백테스트")
print("=" * 80)

# 롤링 테스트 날짜 설정 (6개월 단위)
dates = ['2018-01-01','2018-07-01','2019-01-01','2019-07-01','2020-01-01',
         '2020-07-01','2021-01-01','2021-07-01','2022-01-01','2022-07-01']

results = {
    ('Strategy','CAGR'):[],
    ('Strategy','MDD'):[],
    ('Benchmark','CAGR'):[],
    ('Benchmark','MDD'):[]
}

ticker = 'TQQQ'

print(f"📊 {ticker} MOM Strategy2 롤링 테스트 시작...")
print(f"📅 테스트 기간: {len(dates)}개 시점 (6개월 간격)")
print(f"🔧 각 시점마다: 훈련(5년) → 최적화 → 테스트(2년)")

for i, date in enumerate(dates, 1):
    print(f"\n[{i}/{len(dates)}] {date} 시점 테스트 중...")
    
    try:
        CAGR_strategy, mdd_strategy, CAGR_benchmark, mdd_benchmark = \
        rolling_test_mom2(ticker, date)
        
        results[('Strategy','CAGR')].append(CAGR_strategy)
        results[('Strategy','MDD')].append(mdd_strategy)
        results[('Benchmark','CAGR')].append(CAGR_benchmark)
        results[('Benchmark','MDD')].append(mdd_benchmark)
        
        print(f'✅ 완료 - CAGR_Strategy:{100*CAGR_strategy:.2f}%, '
              f'MDD_Strategy:{100*mdd_strategy:.2f}%')
        print(f'   CAGR_Benchmark:{100*CAGR_benchmark:.2f}%, '
              f'MDD_Benchmark:{100*mdd_benchmark:.2f}%')
              
    except Exception as e:
        print(f'❌ 오류 발생: {e}')
        # 오류 발생 시 NaN 값 추가
        results[('Strategy','CAGR')].append(np.nan)
        results[('Strategy','MDD')].append(np.nan)
        results[('Benchmark','CAGR')].append(np.nan)
        results[('Benchmark','MDD')].append(np.nan)

# 결과 DataFrame 생성
results_df = pd.DataFrame(results, index=dates)

print(f"\n📊 롤링 테스트 결과 요약:")
print(f"{'='*60}")
print(f"{'시점':<12} {'전략CAGR(%)':<12} {'벤치CAGR(%)':<12} {'전략MDD(%)':<12} {'벤치MDD(%)':<12}")
print(f"{'-'*60}")

for date in dates:
    idx = dates.index(date)
    strategy_cagr = results_df[('Strategy','CAGR')].iloc[idx]
    benchmark_cagr = results_df[('Benchmark','CAGR')].iloc[idx]
    strategy_mdd = results_df[('Strategy','MDD')].iloc[idx]
    benchmark_mdd = results_df[('Benchmark','MDD')].iloc[idx]
    
    if not np.isnan(strategy_cagr):
        print(f"{date:<12} {strategy_cagr*100:<12.2f} {benchmark_cagr*100:<12.2f} "
              f"{strategy_mdd*100:<12.2f} {benchmark_mdd*100:<12.2f}")
    else:
        print(f"{date:<12} {'오류':<12} {'오류':<12} {'오류':<12} {'오류':<12}")

# 롤링 테스트 결과 시각화
print(f"\n📈 롤링 테스트 결과 시각화...")

# 데이터 준비
values1 = results_df[('Strategy','CAGR')].values
values2 = results_df[('Benchmark','CAGR')].values
values3 = results_df[('Strategy','MDD')].values
values4 = results_df[('Benchmark','MDD')].values

# NaN 값 처리
values1 = np.nan_to_num(values1, nan=0)
values2 = np.nan_to_num(values2, nan=0)
values3 = np.nan_to_num(values3, nan=0)
values4 = np.nan_to_num(values4, nan=0)

# 그래프 생성
bar_width = 0.3
index = np.arange(len(dates))

fig, ax = plt.subplots(2,1, figsize=(12, 10), sharex=True)

# CAGR 비교
ax[0].bar(index, values1*100, bar_width, label='CAGR_Strategy', color='blue', alpha=0.7)
ax[0].bar(index + bar_width, values2*100, bar_width, label='CAGR_Benchmark', color='red', alpha=0.7)

# MDD 비교
ax[1].bar(index, values3*100, bar_width, label='MDD_Strategy', color='blue', alpha=0.7)
ax[1].bar(index + bar_width, values4*100, bar_width, label='MDD_Benchmark', color='red', alpha=0.7)

# 그래프 설정
ax[0].set_ylabel('CAGR(%)', fontsize=15)
ax[0].set_title(f'{ticker} MOM Strategy2 롤링 테스트 결과', fontsize=20)
ax[0].legend(fontsize=13)
ax[0].grid(alpha=0.3)

ax[1].set_ylabel('MDD(%)', fontsize=15)
ax[1].set_xlabel('테스트 시점', fontsize=15)
ax[1].set_xticks(index + bar_width / 2)
ax[1].set_xticklabels(dates, rotation=45)
ax[1].legend(fontsize=13)
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 통계 요약
print(f"\n📊 롤링 테스트 통계 요약:")
print(f"{'='*50}")

# 유효한 결과만 필터링
valid_strategy_cagr = [x for x in results_df[('Strategy','CAGR')] if not np.isnan(x)]
valid_benchmark_cagr = [x for x in results_df[('Benchmark','CAGR')] if not np.isnan(x)]
valid_strategy_mdd = [x for x in results_df[('Strategy','MDD')] if not np.isnan(x)]
valid_benchmark_mdd = [x for x in results_df[('Benchmark','MDD')] if not np.isnan(x)]

if valid_strategy_cagr:
    print(f"📈 전략 CAGR:")
    print(f"   • 평균: {np.mean(valid_strategy_cagr)*100:.2f}%")
    print(f"   • 최고: {np.max(valid_strategy_cagr)*100:.2f}%")
    print(f"   • 최저: {np.min(valid_strategy_cagr)*100:.2f}%")
    print(f"   • 표준편차: {np.std(valid_strategy_cagr)*100:.2f}%")

if valid_benchmark_cagr:
    print(f"\n📈 벤치마크 CAGR:")
    print(f"   • 평균: {np.mean(valid_benchmark_cagr)*100:.2f}%")
    print(f"   • 최고: {np.max(valid_benchmark_cagr)*100:.2f}%")
    print(f"   • 최저: {np.min(valid_benchmark_cagr)*100:.2f}%")
    print(f"   • 표준편차: {np.std(valid_benchmark_cagr)*100:.2f}%")

if valid_strategy_mdd:
    print(f"\n📉 전략 MDD:")
    print(f"   • 평균: {np.mean(valid_strategy_mdd)*100:.2f}%")
    print(f"   • 최악: {np.min(valid_strategy_mdd)*100:.2f}%")
    print(f"   • 최고: {np.max(valid_strategy_mdd)*100:.2f}%")

if valid_benchmark_mdd:
    print(f"\n📉 벤치마크 MDD:")
    print(f"   • 평균: {np.mean(valid_benchmark_mdd)*100:.2f}%")
    print(f"   • 최악: {np.min(valid_benchmark_mdd)*100:.2f}%")
    print(f"   • 최고: {np.max(valid_benchmark_mdd)*100:.2f}%")

# 성과 비교
if valid_strategy_cagr and valid_benchmark_cagr:
    outperformance_count = sum(1 for s, b in zip(valid_strategy_cagr, valid_benchmark_cagr) if s > b)
    total_tests = len(valid_strategy_cagr)
    print(f"\n🏆 성과 비교:")
    print(f"   • 전략이 벤치마크를 상회한 횟수: {outperformance_count}/{total_tests} ({outperformance_count/total_tests*100:.1f}%)")

print(f"\n✅ 롤링 백테스트 완료!")

# =============================================================================
# 최종 결과 요약
# =============================================================================

print(f"\n{'='*80}")
print("🎯 TQQQ MOM Strategy2 롤링 백테스트 최종 결과")
print(f"{'='*80}")

print(f"📅 롤링 테스트 기간: 2018-01-01 ~ 2022-07-01 (6개월 간격)")
print(f"📊 테스트 시점: {len(dates)}개")
print(f"🔧 각 시점마다: 훈련(5년) → 최적화 → 테스트(2년)")

print(f"\n✅ 롤링 백테스트 완료!")
