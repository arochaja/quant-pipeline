"""
backtest/costs.py
-----------------
Transaction cost modeling.

A backtest without costs overstates performance, sometimes wildly, for
strategies that trade frequently. This module applies a simple, honest
cost model: every change in position incurs a cost proportional to the
size of the trade.

Cost model (simple "proportional" / per-trade-bps model):
    cost_per_trade is expressed in basis points (bps).
        1 bp  = 0.01% = 0.0001
        10 bps = 0.10% = 0.0010
    On each day, trade_size = |position_today - position_yesterday|.
    cost_today = trade_size * (cost_per_trade_bps / 10_000)
    net_return = gross_return - cost_today

This captures spread + slippage + commission lumped into one number.
More sophisticated models (volume-dependent, nonlinear impact) can
replace this later WITHOUT changing the engine, as long as the function
signature stays the same.
"""

import pandas as pd


def apply_costs(
    position: pd.Series,
    gross_return: pd.Series,
    cost_per_trade_bps: float = 10.0,
) -> pd.Series:
    """
    Subtract transaction costs from gross strategy returns.

    Args:
        position:           the position actually held each day
                            (already shifted for look-ahead in the engine).
        gross_return:       the strategy's per-day return BEFORE costs
                            (position * asset_return).
        cost_per_trade_bps: cost of a full position change, in basis points.

    Returns:
        net_return: gross_return minus per-day transaction costs,
                    indexed identically to the inputs.

    Steps:
      1. trade_size = position.diff().abs()
            Day-over-day change in position. The first row is NaN
            (no prior day) -> fill with the absolute first position,
            since going from 0 (assumed flat pre-history) to position[0]
            IS a trade. Simple approach: position.diff().abs().fillna(
                position.abs().iloc[0] if len(position) else 0.0)
            (Or just .fillna(0.0) and note the tiny first-row simplification
             in your decisions log — your call.)

      2. cost = trade_size * (cost_per_trade_bps / 10_000)
            Convert bps to a fraction, scale by how much you traded.

      3. net_return = gross_return - cost

      4. Return net_return.
    """
    
    trade_size = position.diff().abs()

    cost = trade_size * (cost_per_trade_bps / 10000)

    net_return = gross_return - cost

    return net_return

if __name__ == "__main__":
    from data.loaders import get_prices
    from strategies.sma_crossover import SMACrossover
    from backtest.engine import run_backtest

    prices = get_prices("AAPL", "2023-01-01", "2024-01-01")
    strat = SMACrossover(fast=20, slow=50)
    results = run_backtest(prices, strat)

    pos = results["position"]
    gross = results["gross_return"]
    net = apply_costs(pos, gross, cost_per_trade_bps=10.0)

    n_trades = (pos.diff().abs() > 0).sum()
    total_cost_drag = (gross - net).sum()
    print(f"trades: {n_trades},  total cost drag: {total_cost_drag:.4f}")
