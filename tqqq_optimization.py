# TQQQ 2% R값 시스템 파라미터 최적화
# 다양한 파라미터 조합으로 최적 성과 찾기

import pandas as pd
import numpy as np
from datetime import datetime
from tqqq_momentum_backtest_2pct_risk import (
    mom_strategy_2pct_risk,
    load_tqqq_data,
)


def calculate_performance_summary(
    data,
    trades,
    cash_init=10000,
    fee_rate=0.001,
):
    """최적 파라미터 성과 요약 계산"""
    summary = {}

    strategy_cum = data['Cumulative_Return']
    summary['start_date'] = data.index[0]
    summary['end_date'] = data.index[-1]
    final_cum_return = strategy_cum.iloc[-1] - 1
    summary['strategy_return'] = final_cum_return
    summary['final_asset_strategy'] = cash_init * strategy_cum.iloc[-1]

    price_series = data['close']
    buy_hold_return = (
        (price_series.iloc[-1] * (1 - fee_rate))
        / (price_series.iloc[0] * (1 + fee_rate))
    ) - 1
    summary['buy_hold_return'] = buy_hold_return
    summary['final_asset_buyhold'] = cash_init * (1 + buy_hold_return)

    trading_period = len(data) / 252
    summary['trading_period_years'] = trading_period
    summary['trading_days'] = len(data)

    summary['cagr_strategy'] = (
        strategy_cum.iloc[-1]
    ) ** (1 / trading_period) - 1
    summary['cagr_benchmark'] = (
        buy_hold_return + 1
    ) ** (1 / trading_period) - 1

    risk_free_rate = 0.003
    strategy_daily_return = strategy_cum.pct_change().fillna(0)
    mean_return = strategy_daily_return.mean() * 252
    std_return = strategy_daily_return.std() * np.sqrt(252)
    summary['sharpe_ratio'] = ((mean_return - risk_free_rate) / std_return
                               if std_return > 0 else 0)

    downside_returns = strategy_daily_return[strategy_daily_return < 0]
    downside_std = (downside_returns.std() * np.sqrt(252)
                    if len(downside_returns) > 0 else 0)
    summary['sortino_ratio'] = ((mean_return - risk_free_rate) / downside_std
                                if downside_std > 0 else 0)

    strategy_cummax = strategy_cum.cummax()
    drawdown = strategy_cum / strategy_cummax - 1
    summary['max_drawdown'] = drawdown.min()

    mdd_idx = drawdown.idxmin()
    mdd_start_idx = data.loc[:mdd_idx, 'Cumulative_Return'].idxmax()
    summary['mdd_duration_days'] = ((mdd_idx - mdd_start_idx).days
                                    if mdd_idx != mdd_start_idx else 0)
    summary['mdd_start'] = mdd_start_idx
    summary['mdd_end'] = mdd_idx

    price_cum = (1 + price_series.pct_change()).cumprod()
    price_cummax = price_cum.cummax()
    summary['mdd_benchmark'] = (price_cum / price_cummax - 1).min()

    summary['calmar_ratio'] = (
        summary['cagr_strategy'] / abs(summary['max_drawdown'])
        if summary['max_drawdown'] != 0 else 0
    )
    summary['calmar_benchmark'] = (
        summary['cagr_benchmark'] / abs(summary['mdd_benchmark'])
        if summary['mdd_benchmark'] != 0 else 0
    )

    total_trades = len(trades)
    winning_trades = [t for t in trades if t['r_multiple'] > 0]
    losing_trades = [t for t in trades if t['r_multiple'] <= 0]

    summary['total_trades'] = total_trades
    summary['winning_trades'] = len(winning_trades)
    summary['losing_trades'] = len(losing_trades)
    summary['win_rate'] = (len(winning_trades) / total_trades
                           if total_trades > 0 else 0)

    winning_rs = [t['r_multiple'] for t in winning_trades]
    losing_rs = [t['r_multiple'] for t in losing_trades]
    avg_winning_r = np.mean(winning_rs) if winning_rs else 0
    avg_losing_r = np.mean(losing_rs) if losing_rs else 0
    summary['avg_r_multiple'] = (
        np.mean([t['r_multiple'] for t in trades]) if trades else 0
    )
    summary['expectancy'] = (
        (summary['win_rate'] * avg_winning_r)
        + ((1 - summary['win_rate']) * avg_losing_r)
    )
    summary['profit_loss_ratio'] = (
        avg_winning_r / abs(avg_losing_r) if avg_losing_r < 0 else np.inf
    )

    return summary


def print_best_summary(best_params, summary):
    """최적 파라미터 상세 요약 출력"""
    print("\n" + "=" * 70)
    print("🎯 TQQQ Strategy2 + 2% R값 시스템 최종 결과 요약")
    print("=" * 70)

    print(
        f"\n📅 백테스트 기간: "
        f"{summary['start_date'].strftime('%Y-%m-%d')} ~ "
        f"{summary['end_date'].strftime('%Y-%m-%d')}"
    )
    print(
        f"📊 거래일 수: {summary['trading_days']}일 "
        f"({summary['trading_period_years']:.2f}년)"
    )

    print("\n💰 수익률 비교:")
    print(f"   • Buy & Hold: {summary['buy_hold_return']*100:.2f}%")
    print(f"   • MOM Strategy2 + 2% R값: {summary['strategy_return']*100:.2f}%")
    excess = (summary['strategy_return'] - summary['buy_hold_return']) * 100
    if excess > 0:
        print(f"   • 초과 수익: +{excess:.2f}%p ✅")
    else:
        print(f"   • 초과 수익: {excess:.2f}%p")

    print("\n📈 성과 지표 요약:")
    print(
        f"   • CAGR: {summary['cagr_strategy']*100:.2f}% "
        f"(Buy & Hold: {summary['cagr_benchmark']*100:.2f}%)"
    )
    print(f"   • 샤프 비율: {summary['sharpe_ratio']:.3f}")
    print(f"   • 소르티노 비율: {summary['sortino_ratio']:.3f}")
    print(f"   • 칼마 비율: {summary['calmar_ratio']:.3f}")
    print(
        f"   • 최대 낙폭 (MDD): {summary['max_drawdown']*100:.2f}% "
        f"(Buy & Hold: {summary['mdd_benchmark']*100:.2f}%)"
    )

    print("\n🔧 사용된 파라미터:")
    print(f"   • 모멘텀 기간: {best_params['p1']}일")
    print(f"   • MFI 기간: {best_params['p2']}일")
    print(f"   • MFI 레벨: {best_params['mfi_level']}")
    print(f"   • ATR 배수: {best_params['atr_mult']}")
    print("   • 리스크: 2% 고정")

    print("\n📊 거래 통계 요약:")
    print(f"   • 총 거래: {summary['total_trades']}회")
    print(f"   • 승률: {summary['win_rate']*100:.2f}%")
    print(f"   • 수익/손실 비율: {summary['profit_loss_ratio']:.2f}")

    print("\n📊 R-배수 통계 요약:")
    print(f"   • 평균 R-배수: {summary['avg_r_multiple']:.2f}R")
    print(f"   • 기대값: {summary['expectancy']:.2f}R")


def get_backtest_period():
    """대화형으로 백테스트 기간 설정"""
    print("\n" + "=" * 60)
    print("TQQQ 2% R값 시스템 최적화 기간 설정")
    print("=" * 60)

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
            'days': None,
        }

    if choice == "2":
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
                'days': None,
            }

        try:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            if end_dt <= start_dt:
                print("종료일이 시작일보다 이전입니다. 전체 데이터로 진행합니다.")
                return {
                    'mode': 'full',
                    'start_date': '2015-01-01',
                    'end_date': current_date,
                    'days': None,
                }

            return {
                'mode': 'range',
                'start_date': start,
                'end_date': end,
                'days': None,
            }
        except Exception:
            print("잘못된 날짜 형식입니다. 전체 데이터로 진행합니다.")
            return {
                'mode': 'full',
                'start_date': '2015-01-01',
                'end_date': current_date,
                'days': None,
            }

    if choice == "3":
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
                    'days': None,
                }
            days = int(days_input) if days_input else 1000

            if days <= 0:
                print("일수는 양수여야 합니다. 전체 데이터로 진행합니다.")
                return {
                    'mode': 'full',
                    'start_date': '2015-01-01',
                    'end_date': current_date,
                    'days': None,
                }

            return {
                'mode': 'recent',
                'start_date': None,
                'end_date': None,
                'days': days,
            }
        except Exception:
            print("잘못된 입력입니다. 전체 데이터로 진행합니다.")
            return {
                'mode': 'full',
                'start_date': '2015-01-01',
                'end_date': current_date,
                'days': None,
            }

    print("잘못된 선택입니다. 전체 데이터로 진행합니다.")
    return {
        'mode': 'full',
        'start_date': '2015-01-01',
        'end_date': current_date,
        'days': None,
    }


def load_tqqq_data_for_period(period_config):
    """기간 설정에 따라 TQQQ 데이터 로드 및 필터링"""
    df = load_tqqq_data()
    if df.empty:
        return df

    df = df.sort_index()

    if period_config['mode'] == 'range':
        start_dt = pd.to_datetime(period_config['start_date'])
        end_dt = pd.to_datetime(period_config['end_date'])
        df = df.loc[start_dt:end_dt]
        print(
            f"기간 필터 적용: {period_config['start_date']} ~ "
            f"{period_config['end_date']}"
        )

    elif period_config['mode'] == 'recent':
        days = period_config['days']
        df = df.tail(days)
        print(f"최근 {days}일 데이터 사용")

    else:
        print("전체 데이터 사용")

    if df.empty:
        print("기간 필터 결과: 데이터가 비어 있습니다.")
        return df

    start_date = df.index[0]
    end_date = df.index[-1]
    total_days = (end_date - start_date).days

    print(
        f"선택된 백테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ "
        f"{end_date.strftime('%Y-%m-%d')} ({total_days}일, "
        f"{total_days/365:.1f}년)"
    )

    return df


def run_optimization():
    """파라미터 최적화 실행"""
    print("=" * 80)
    print("TQQQ 2% R값 시스템 파라미터 최적화")
    print("=" * 80)

    period_config = get_backtest_period()

    print("\n백테스트 설정:")
    if period_config['mode'] == 'full':
        print(
            f"백테스트 기간: 전체 데이터 "
            f"(2015-01-01 ~ {period_config['end_date']})"
        )
    elif period_config['mode'] == 'range':
        print(
            f"백테스트 기간: {period_config['start_date']} ~ "
            f"{period_config['end_date']}"
        )
    else:
        days = period_config['days']
        print(f"백테스트 기간: 최근 {days}일 ({days/365:.1f}년)")

    df = load_tqqq_data_for_period(period_config)
    if df.empty:
        print("데이터 로드 실패")
        return

    # 최적화 파라미터 범위
    p1_range = [3, 5, 7, 10, 14]  # 모멘텀 기간
    p2_range = [5, 6, 8, 10, 14]  # MFI 기간
    mfi_levels = [45, 47.5, 50, 52.5, 55]  # MFI 레벨
    atr_mults = [1.5, 2.0, 2.5, 3.0]  # ATR 배수

    print("최적화 범위:")
    print(f"   - 모멘텀 기간: {p1_range}")
    print(f"   - MFI 기간: {p2_range}")
    print(f"   - MFI 레벨: {mfi_levels}")
    print(f"   - ATR 배수: {atr_mults}")
    total_grid = (
        len(p1_range) * len(p2_range) * len(mfi_levels) * len(atr_mults)
    )
    print(f"   - 총 조합: {total_grid}개")

    # 결과 저장
    results = []
    total_combinations = (
        len(p1_range) * len(p2_range) * len(mfi_levels) * len(atr_mults)
    )
    current = 0

    print("\n최적화 시작...")

    for p1 in p1_range:
        for p2 in p2_range:
            for mfi_level in mfi_levels:
                for atr_mult in atr_mults:
                    current += 1

                    try:
                        # 백테스트 실행
                        data, final_return, trades = mom_strategy_2pct_risk(
                            df,
                            p1,
                            p2,
                            3,
                            mfi_level,
                            atr_mult,
                            verbose=False,
                        )

                        # Buy & Hold 비교
                        buy_hold_return = (
                            df['close'].iloc[-1] / df['close'].iloc[0]
                        ) - 1
                        excess_return = final_return - buy_hold_return

                        # 거래 통계
                        total_trades = len(trades)
                        winning_trades = len(
                            [t for t in trades if t['r_multiple'] > 0]
                        )
                        win_rate = (
                            winning_trades / total_trades
                            if total_trades > 0
                            else 0
                        )

                        # R-배수 통계
                        r_multiples = [t['r_multiple'] for t in trades]
                        avg_r = np.mean(r_multiples) if r_multiples else 0

                        # 결과 저장
                        result = {
                            'p1': p1,
                            'p2': p2,
                            'mfi_level': mfi_level,
                            'atr_mult': atr_mult,
                            'strategy_return': final_return,
                            'buy_hold_return': buy_hold_return,
                            'excess_return': excess_return,
                            'total_trades': total_trades,
                            'win_rate': win_rate,
                            'avg_r_multiple': avg_r
                        }
                        results.append(result)

                        # 진행 상황 출력
                        if current % 50 == 0 or current == total_combinations:
                            progress = current / total_combinations * 100
                            print(
                                f"진행률: {current}/{total_combinations} "
                                f"({progress:.1f}%)"
                            )

                    except Exception as e:
                        print(
                            "오류 발생: "
                            f"p1={p1}, p2={p2}, mfi={mfi_level}, "
                            f"atr={atr_mult}: {e}"
                        )
                        continue

    # 결과 분석
    if not results:
        print("최적화 결과가 없습니다.")
        return

    results_df = pd.DataFrame(results)

    print(f"\n최적화 완료! 총 {len(results)}개 조합 테스트")

    # 상위 결과 출력
    print(f"\n{'='*80}")
    print("TOP 10 결과 (수익률 기준)")
    print(f"{'='*80}")

    top_results = results_df.nlargest(10, 'strategy_return')
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"{i:2d}. 수익률: {row['strategy_return']*100:6.2f}% | "
              f"초과수익: {row['excess_return']*100:6.2f}%p | "
              f"거래: {int(row['total_trades']):3d}회 | "
              f"승률: {row['win_rate']*100:5.1f}% | "
              f"R배수: {row['avg_r_multiple']:5.2f}R | "
              f"파라미터: p1={int(row['p1']):2d}, p2={int(row['p2']):2d}, "
              f"mfi={row['mfi_level']:4.1f}, atr={row['atr_mult']:3.1f}")

    # 최고 성과 파라미터
    best_result = results_df.loc[results_df['strategy_return'].idxmax()]

    print(f"\n{'='*80}")
    print("최적 파라미터")
    print(f"{'='*80}")
    print(f"모멘텀 기간: {int(best_result['p1'])}일")
    print(f"MFI 기간: {int(best_result['p2'])}일")
    print(f"MFI 레벨: {best_result['mfi_level']}")
    print(f"ATR 배수: {best_result['atr_mult']}")
    print()
    print("성과:")
    print(f"  - 전략 수익률: {best_result['strategy_return']*100:.2f}%")
    print(f"  - Buy & Hold: {best_result['buy_hold_return']*100:.2f}%")
    print(f"  - 초과 수익: {best_result['excess_return']*100:.2f}%p")
    print(f"  - 총 거래: {int(best_result['total_trades'])}회")
    print(f"  - 승률: {best_result['win_rate']*100:.2f}%")
    print(f"  - 평균 R배수: {best_result['avg_r_multiple']:.2f}R")

    # 통계 분석
    print(f"\n{'='*80}")
    print("통계 분석")
    print(f"{'='*80}")
    print("전체 조합 수익률 분포:")
    print(f"  - 최고: {results_df['strategy_return'].max()*100:.2f}%")
    print(f"  - 최저: {results_df['strategy_return'].min()*100:.2f}%")
    print(f"  - 평균: {results_df['strategy_return'].mean()*100:.2f}%")
    print(f"  - 중앙값: {results_df['strategy_return'].median()*100:.2f}%")

    # 양수 수익률 비율
    positive_returns = (results_df['strategy_return'] > 0).sum()
    positive_ratio = positive_returns / len(results_df) * 100
    print(
        f"  - 양수 수익률 비율: {positive_returns}/{len(results_df)} "
        f"({positive_ratio:.1f}%)"
    )

    # Buy & Hold 초과 비율
    beat_buyhold = (results_df['excess_return'] > 0).sum()
    beat_ratio = beat_buyhold / len(results_df) * 100
    print(
        f"  - Buy & Hold 초과 비율: {beat_buyhold}/{len(results_df)} "
        f"({beat_ratio:.1f}%)"
    )

    # 최적 파라미터 상세 요약
    best_params = {
        'p1': int(best_result['p1']),
        'p2': int(best_result['p2']),
        'mfi_level': best_result['mfi_level'],
        'atr_mult': best_result['atr_mult'],
    }
    best_data, _, best_trades = mom_strategy_2pct_risk(
        df,
        best_params['p1'],
        best_params['p2'],
        3,
        best_params['mfi_level'],
        best_params['atr_mult'],
        verbose=False
    )
    best_summary = calculate_performance_summary(best_data, best_trades)
    print_best_summary(best_params, best_summary)

    return results_df, best_result


if __name__ == '__main__':
    results_df, best_result = run_optimization()
