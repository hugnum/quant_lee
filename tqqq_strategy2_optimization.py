"""
TQQQ MOM Strategy2 파라미터 최적화 스크립트
================================================
- 모멘텀 기간, MFI 기간, MFI 임계값, 손절 비율, 투자 비율 그리드 서치
- `tqqq_strategy2_simple`의 기간 선택 및 데이터 로드 유틸리티 재사용
- 결과는 수익률, 샤프비율, MDD 기준으로 정렬
"""

import itertools
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import ta

from tqqq_strategy2_simple import (
    get_backtest_period,
    load_tqqq_data_with_period,
)


# =============================================================================
# 전략 로직 (투자 비율/수수료 조정 가능 버전)
# =============================================================================

def mom_strategy2_configurable(
    df: pd.DataFrame,
    period1: int,
    period2: int,
    mfi_level: float,
    stop_loss: float,
    allocation_ratio: float,
    fee_rate: float = 0.001,
) -> Tuple[pd.DataFrame, float]:
    """
    MOM Strategy2: 모멘텀 + MFI 전략 (투자 비율/수수료 조정)
    """
    data = df.copy()

    # 기술적 지표 계산
    data["Mom"] = data["Close"].pct_change(periods=period1)
    data["MFI"] = ta.volume.money_flow_index(
        data.High, data.Low, data.Close, data.Volume, period2
    )
    data.dropna(inplace=True)

    mom_pos = pd.Series(np.where(data["Mom"] > 0, 1, 0), index=data.index)
    mom_signals = mom_pos.diff().fillna(0)

    cash_init = 10000
    cash = cash_init
    asset = np.zeros(len(data))
    asset[0] = cash

    prices = data["Close"].values
    signals = mom_signals.values
    mfi = data["MFI"].values
    positions = np.zeros(len(data))
    pos = 0
    num = 0
    stop_loss_price = 0.0

    for i in range(1, len(data)):
        if pos == 0:
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

        elif pos == 1:
            if prices[i] < stop_loss_price:
                pos = 0
                cash += prices[i] * num * (1 - fee_rate)
                num = 0
            else:
                positions[i] = 1
                stop_loss_price = max(stop_loss_price, prices[i] * (1 - stop_loss))

        if pos == 0:
            asset[i] = cash
        else:
            asset[i] = cash + prices[i] * num

    data["Position"] = positions
    data["Signal"] = data["Position"].diff().fillna(0)
    data["Buy_Price"] = np.where(data["Signal"] == 1, data["Close"], np.nan)
    data["Sell_Price"] = np.where(data["Signal"] == -1, data["Close"], np.nan)
    data["Cumulative_Return"] = asset / cash_init
    final_cum_return = data["Cumulative_Return"].iloc[-1] - 1

    return data, final_cum_return


# =============================================================================
# 성과 지표 계산 헬퍼
# =============================================================================

@dataclass
class StrategyMetrics:
    final_return: float
    sharpe_ratio: float
    max_drawdown: float
    cagr: float


def calculate_metrics(
    cumulative_returns: pd.Series,
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.003,
) -> StrategyMetrics:
    final_return = cumulative_returns.iloc[-1] - 1

    strategy_daily_return = cumulative_returns.pct_change().fillna(0)
    mean_return = strategy_daily_return.mean() * trading_days_per_year
    std_return = strategy_daily_return.std() * np.sqrt(trading_days_per_year)
    sharpe_ratio = (
        (mean_return - risk_free_rate) / std_return if std_return > 0 else np.nan
    )

    cumulative_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / cumulative_max - 1
    max_drawdown = drawdown.min()

    trading_period_years = len(cumulative_returns) / trading_days_per_year
    cagr = cumulative_returns.iloc[-1] ** (1 / trading_period_years) - 1

    return StrategyMetrics(
        final_return=final_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        cagr=cagr,
    )


# =============================================================================
# 그리드 서치 로직
# =============================================================================

def iterate_param_grid(param_grid: Dict[str, Iterable]) -> Iterable[Dict[str, float]]:
    keys = list(param_grid.keys())
    for values in itertools.product(*(param_grid[key] for key in keys)):
        yield dict(zip(keys, values))


def run_grid_search(
    df: pd.DataFrame,
    param_grid: Dict[str, Iterable],
    top_n: int = 5,
    risk_free_rate: float = 0.003,
) -> pd.DataFrame:
    results: List[Dict[str, float]] = []
    combinations = list(iterate_param_grid(param_grid))
    total_jobs = len(combinations)

    print(f"\n총 {total_jobs}개 조합 평가를 시작합니다...")
    start_time = time.time()

    for idx, params in enumerate(combinations, start=1):
        if params["period1"] <= 0 or params["period2"] <= 0:
            continue
        if not 0 < params["allocation_ratio"] <= 1:
            continue
        if not 0 < params["stop_loss"] < 1:
            continue

        data, _ = mom_strategy2_configurable(
            df,
            period1=int(params["period1"]),
            period2=int(params["period2"]),
            mfi_level=float(params["mfi_level"]),
            stop_loss=float(params["stop_loss"]),
            allocation_ratio=float(params["allocation_ratio"]),
        )

        metrics = calculate_metrics(
            data["Cumulative_Return"], risk_free_rate=risk_free_rate
        )

        results.append(
            {
                **params,
                "final_return": metrics.final_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "cagr": metrics.cagr,
            }
        )

        if idx % 10 == 0 or idx == total_jobs:
            elapsed = time.time() - start_time
            print(
                f"  - 진행 상황: {idx}/{total_jobs} ({idx / total_jobs:.1%}) | "
                f"경과 시간: {elapsed:.1f}초"
            )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df

    results_df.sort_values(
        by=["final_return", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
        inplace=True,
    )

    print("\n상위 조합 미리보기:")
    print(
        results_df.head(top_n)[
            [
                "period1",
                "period2",
                "mfi_level",
                "stop_loss",
                "allocation_ratio",
                "final_return",
                "sharpe_ratio",
                "max_drawdown",
            ]
        ]
    )

    best = results_df.iloc[0]
    print("\n최적 조합 요약:")
    print(
        f"  • 모멘텀 기간: {int(best['period1'])}일 | "
        f"MFI 기간: {int(best['period2'])}일 | "
        f"MFI 임계값: {best['mfi_level']:.1f}"
    )
    print(
        f"  • 손절 비율: {best['stop_loss']*100:.1f}% | "
        f"투자 비율: {best['allocation_ratio']*100:.1f}%"
    )
    print(
        f"  • 최종 수익률: {best['final_return']*100:.2f}% | "
        f"샤프 비율: {best['sharpe_ratio']:.3f} | "
        f"MDD: {best['max_drawdown']*100:.2f}%"
    )

    return results_df


# =============================================================================
# 실행부
# =============================================================================

def main():
    print("=" * 70)
    print("TQQQ MOM Strategy2 파라미터 최적화")
    print("=" * 70)

    period_config = get_backtest_period()

    print("\n데이터 로드 중...")
    df = load_tqqq_data_with_period(period_config)
    if df.empty:
        print("데이터 로드 실패 – 스크립트를 종료합니다.")
        return

    print("\nTQQQ 데이터 요약:")
    print(f"  • 시작가: ${df['Close'].iloc[0]:.2f}")
    print(f"  • 종료가: ${df['Close'].iloc[-1]:.2f}")
    print(
        f"  • 전체 수익률: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:.2f}%"
    )

    param_grid = {
        "period1": [4, 6, 8, 10, 12],
        "period2": [4, 6, 8, 10, 12],
        "mfi_level": [45, 50, 55],
        "stop_loss": [0.08, 0.10, 0.12],
        "allocation_ratio": [0.3, 0.5, 0.7],
    }

    print("\n최적화 파라미터 그리드:")
    for key, values in param_grid.items():
        print(f"  • {key}: {values}")

    results_df = run_grid_search(df, param_grid, top_n=5)
    if results_df.empty:
        print("\n평가 결과가 없습니다. 파라미터 범위를 확인하세요.")
        return

    save_choice = input("\n결과를 CSV로 저장할까요? (y/N): ").strip().lower()
    if save_choice == "y":
        filename = "tqqq_strategy2_optimization_results.csv"
        results_df.to_csv(filename, index=False)
        print(f"결과가 '{filename}' 파일로 저장되었습니다.")

    print("\n✅ 최적화 완료!")


if __name__ == "__main__":
    main()



