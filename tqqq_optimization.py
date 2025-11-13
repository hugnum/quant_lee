# TQQQ 2% R값 시스템 파라미터 최적화
# 다양한 파라미터 조합으로 최적 성과 찾기

import pandas as pd
import numpy as np
import itertools
from tqqq_momentum_backtest_2pct_risk import mom_strategy_2pct_risk, load_tqqq_data

def run_optimization():
    """파라미터 최적화 실행"""
    print("=" * 80)
    print("TQQQ 2% R값 시스템 파라미터 최적화")
    print("=" * 80)
    
    # 데이터 로드
    df = load_tqqq_data()
    if df.empty:
        print("데이터 로드 실패")
        return
    
    # 최적화 파라미터 범위
    p1_range = [3, 5, 7, 10, 14]  # 모멘텀 기간
    p2_range = [5, 6, 8, 10, 14]  # MFI 기간
    mfi_levels = [45, 47.5, 50, 52.5, 55]  # MFI 레벨
    atr_mults = [1.5, 2.0, 2.5, 3.0]  # ATR 배수
    
    print(f"최적화 범위:")
    print(f"   - 모멘텀 기간: {p1_range}")
    print(f"   - MFI 기간: {p2_range}")
    print(f"   - MFI 레벨: {mfi_levels}")
    print(f"   - ATR 배수: {atr_mults}")
    print(f"   - 총 조합: {len(p1_range) * len(p2_range) * len(mfi_levels) * len(atr_mults)}개")
    
    # 결과 저장
    results = []
    total_combinations = len(p1_range) * len(p2_range) * len(mfi_levels) * len(atr_mults)
    current = 0
    
    print(f"\n최적화 시작...")
    
    for p1 in p1_range:
        for p2 in p2_range:
            for mfi_level in mfi_levels:
                for atr_mult in atr_mults:
                    current += 1
                    
                    try:
                        # 백테스트 실행
                        data, final_return, trades = mom_strategy_2pct_risk(
                            df, p1, p2, 3, mfi_level, atr_mult, verbose=False)
                        
                        # Buy & Hold 비교
                        buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
                        excess_return = final_return - buy_hold_return
                        
                        # 거래 통계
                        total_trades = len(trades)
                        winning_trades = len([t for t in trades if t['r_multiple'] > 0])
                        win_rate = winning_trades / total_trades if total_trades > 0 else 0
                        
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
                            print(f"진행률: {current}/{total_combinations} "
                                  f"({current/total_combinations*100:.1f}%)")
                        
                    except Exception as e:
                        print(f"오류 발생: p1={p1}, p2={p2}, mfi={mfi_level}, atr={atr_mult}: {e}")
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
    print(f"")
    print(f"성과:")
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
    print(f"전체 조합 수익률 분포:")
    print(f"  - 최고: {results_df['strategy_return'].max()*100:.2f}%")
    print(f"  - 최저: {results_df['strategy_return'].min()*100:.2f}%")
    print(f"  - 평균: {results_df['strategy_return'].mean()*100:.2f}%")
    print(f"  - 중앙값: {results_df['strategy_return'].median()*100:.2f}%")
    
    # 양수 수익률 비율
    positive_returns = (results_df['strategy_return'] > 0).sum()
    print(f"  - 양수 수익률 비율: {positive_returns}/{len(results_df)} "
          f"({positive_returns/len(results_df)*100:.1f}%)")
    
    # Buy & Hold 초과 비율
    beat_buyhold = (results_df['excess_return'] > 0).sum()
    print(f"  - Buy & Hold 초과 비율: {beat_buyhold}/{len(results_df)} "
          f"({beat_buyhold/len(results_df)*100:.1f}%)")
    
    return results_df, best_result

if __name__ == '__main__':
    results_df, best_result = run_optimization()
