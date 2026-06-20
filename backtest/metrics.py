"""
backtest/metrics.py
-------------------
Performance & risk metrics computed from a backtest's results.

These turn an equity curve / return series into the standard numbers used
to evaluate a strategy:

    - total_return     : overall growth (endpoint of the equity curve)
    - annualized_return: geometric mean return, scaled to a year
    - sharpe_ratio     : return per unit of risk (the headline risk-adjusted number)
    - max_drawdown     : worst peak-to-trough decline (the key risk number)

Convention: returns are daily. We annualize with 252 trading days/year.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    """
    Overall return across the whole period.

    TODO:
      - return equity.iloc[-1] - 1.0
        (equity starts at 1.0, so endpoint - 1 is the total fractional return)
    """
    
    return equity.iloc[-1] - 1.0

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.

    Sharpe = (mean daily excess return / std of daily returns) * sqrt(252)

    Args:
        returns:   daily strategy returns (net of costs).
        risk_free: daily risk-free rate (leave 0.0 for a first pass; note it).

    TODO:
      1. excess = returns - risk_free
      2. If excess.std() == 0 -> return 0.0 (avoid divide-by-zero on a
         flat/never-traded series).
      3. return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)

    Note: .std() defaults to sample std (ddof=1), which is fine here.
    """

    excess = returns - risk_free

    if excess.std() == 0:
        return 0.0

    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)
    


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown: the largest peak-to-trough drop in the equity curve.
    Returned as a NEGATIVE number (e.g. -0.18 means a 18% worst decline).

    TODO:
      1. running_max = equity.cummax()
            The highest equity seen up to each point.
      2. drawdown = equity / running_max - 1.0
            How far below the prior peak we are, each day (<= 0).
      3. return drawdown.min()
            The deepest point.
    """
    
    running_max = equity.cummax()

    drawdown = equity / running_max - 1.0

    return drawdown.min()

def summarize(results: pd.DataFrame, return_col: str = "net_return",
              equity_col: str = "equity") -> dict:
    """
    Convenience: compute all metrics from a backtest results DataFrame.

    Assumes `results` has an equity column and a (net) return column.
    Adjust the column names to match what your engine actually produces.

    TODO:
      - Build and return a dict like:
          {
            "total_return": total_return(results[equity_col]),
            "sharpe":       sharpe_ratio(results[return_col]),
            "max_drawdown": max_drawdown(results[equity_col]),
          }
    """
    
    summary = {
            "total_return" : total_return(results[equity_col]),
            "sharpe": sharpe_ratio(results[return_col]),
            "max_drawdown": max_drawdown(results[equity_col])
            }

    return summary

if __name__ == "__main__":
    from data.loaders import get_prices
    from strategies.sma_crossover import SMACrossover
    from backtest.engine import run_backtest
    from backtest.costs import apply_costs

    prices = get_prices("AAPL", "2023-01-01", "2024-01-01")
    strat = SMACrossover(fast=20, slow=50)
    results = run_backtest(prices, strat)

    buy_hold = prices["close"].iloc[-1] / prices["close"].iloc[0] - 1.0

    stats = summarize(results)
    print("Strategy performance")
    print(f"  Total return : {stats['total_return']:+.2%}")
    print(f"  Sharpe       : {stats['sharpe']:.2f}")
    print(f"  Max drawdown : {stats['max_drawdown']:.2%}")
    print(f"  Buy & Hold return : {buy_hold:.2%}")
    
