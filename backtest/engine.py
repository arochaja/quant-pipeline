"""
backtest/engine.py
------------------
The backtest engine: turn a strategy's signals into a simulated P&L.

Given price data and a Strategy, simulate holding the strategy's target
positions over history and produce per-day returns (gross and net of
transaction costs) and an equity curve.

Two pieces of methodology are centralized here, so every strategy
inherits them uniformly:
    - One-day execution lag to avoid look-ahead bias: a signal computed
      from day t's close is acted on from day t+1 onward (signal.shift(1)).
    - Transaction costs, applied via backtest.costs, so the reported
      equity curve is net of trading frictions.
"""

import pandas as pd
from strategies.base import Strategy
from backtest.costs import apply_costs


def run_backtest(data: pd.DataFrame, strategy: Strategy,
                 cost_per_trade_bps: float = 10.0) -> pd.DataFrame:
    """
    Run `strategy` over `data` and return a results DataFrame.

    Args:
        data:               cleaned price DataFrame (indexed by date, has 'close').
        strategy:           any Strategy instance.
        cost_per_trade_bps: per-trade transaction cost in basis points.

    Returns:
        DataFrame indexed by date with these columns:
            'position'      position held each day (the shifted signal)
            'asset_return'  daily return of the underlying asset
            'gross_return'  strategy return before costs (position * asset_return)
            'net_return'    strategy return after transaction costs
            'equity'        cumulative equity curve from net returns, starting at 1.0

    Steps:
      1. signals: call the PUBLIC get_signals() so the base class validates output.
      2. asset_return: daily fractional return of the asset (pct_change).
      3. position: shift the signal forward one day (the look-ahead fix), so a
         signal based on day t's close is only acted on from day t+1. Row 0 has
         no prior signal -> fill with 0.0 (flat).
      4. gross_return: position * asset_return. While held long the position
         stays 1 and earns each day's return; while flat it is 0 and earns
         nothing. Costs are applied separately, not here.
      5. net_return: gross_return minus transaction costs (charged only when the
         position changes). NaNs (first row from pct_change) -> 0.0.
      6. equity: compound the NET returns into a curve starting at 1.0.
    """

    signals = strategy.get_signals(data)
    asset_return = data["close"].pct_change()
    position = signals.shift(1).fillna(0.0)

    gross_return = position * asset_return
    net_return = apply_costs(position, gross_return, cost_per_trade_bps).fillna(0.0)
    equity = (1 + net_return).cumprod()          # equity is net of costs

    return pd.DataFrame({
        "position": position,
        "asset_return": asset_return,
        "gross_return": gross_return,
        "net_return": net_return,
        "equity": equity,
    })


if __name__ == "__main__":
    from data.loaders import get_prices
    from strategies.sma_crossover import SMACrossover

    prices = get_prices("AAPL", "2023-01-01", "2024-01-01")
    strat = SMACrossover(fast=20, slow=50)

    results = run_backtest(prices, strat, cost_per_trade_bps=10.0)

    print(results.tail())
    final = results["equity"].iloc[-1]
    print(f"\nFinal equity: {final:.4f}  ->  {(final - 1) * 100:+.2f}% total return (net of costs)")

    # Sanity baseline: buy-and-hold over the same window.
    bh = prices["close"].iloc[-1] / prices["close"].iloc[0]
    print(f"Buy & hold:    {bh:.4f}  ->  {(bh - 1) * 100:+.2f}% total return")
