"""
run.py
------
Entry point for the backtesting pipeline.

This is the single 'front door' to the project. It wires the layers
together in order and reports the result:

    get_prices  ->  Strategy  ->  run_backtest  ->  summarize  ->  report

Run it from the project root:
    python run.py

To change what gets tested, edit the CONFIG block below.
"""

from data.loaders import get_prices
from strategies.sma_crossover import SMACrossover
from strategies.ema_crossover import EMACrossover
from strategies.ts_momentum import TSMomentum
from backtest.engine import run_backtest
from backtest.metrics import summarize



# --------------------------------------------------------------------------
# CONFIG — edit these to run a different backtest.
# --------------------------------------------------------------------------
TICKER = "BTC-USD"
START = "2026-01-01"
END = "2026-06-20"

LOOKBACK = 1
COST_BPS = 10.0
# --------------------------------------------------------------------------


def main() -> None:
    """Orchestrate one backtest and print a scorecard. Stays thin by design."""
    # 1-3. Run the pipeline.
    prices = get_prices(TICKER, START, END)
    strat = TSMomentum(lookback = LOOKBACK)

    results = run_backtest(prices, strat, cost_per_trade_bps=COST_BPS)

    # 4. Performance & risk metrics.
    stats = summarize(results)

    # 5. Buy & hold baseline for context.
    buy_hold = prices["close"].iloc[-1] / prices["close"].iloc[0] - 1.0

    # 6. Report.
    print("=" * 52)
    print("  BACKTEST REPORT")
    print("=" * 52)
    print(f"  Ticker        : {TICKER}")
    print(f"  Period        : {START}  ->  {END}")
    print(f"  Strategy      : Time Series Momentum (Lookback: {LOOKBACK})")
    print(f"  Cost per trade: {COST_BPS:.1f} bps")
    print("-" * 52)
    print(f"  Total return  : {stats['total_return']:+.2%}")
    print(f"  Sharpe ratio  : {stats['sharpe']:.2f}")
    print(f"  Max drawdown  : {stats['max_drawdown']:.2%}")
    print("-" * 52)
    print(f"  Buy & hold    : {buy_hold:+.2%}")
    print("=" * 52)


if __name__ == "__main__":
    main()
