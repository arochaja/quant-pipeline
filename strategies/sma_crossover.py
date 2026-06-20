"""
strategies/sma_crossover.py
---------------------------
Simple Moving Average (SMA) crossover — the pipeline's first concrete strategy.

Idea:
    - Compute a fast SMA and a slow SMA of the close price.
    - Go long (+1) when fast SMA > slow SMA (uptrend).
    - Go flat (0) otherwise.

This implements the Strategy contract: it only defines generate_signals().
The base class's get_signals() will validate the output for us.
"""

import pandas as pd
from strategies.base import Strategy


class SMACrossover(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50):
        """
        Args:
            fast: window (days) for the fast moving average.
            slow: window (days) for the slow moving average.

        TODO:
          - Store fast and slow on self.
          - Sanity check: fast should be < slow. Raise ValueError if not
            (a fast window >= slow window is almost certainly a mistake).
        """
        self.fast = fast
        self.slow = slow

        if fast >= slow:
            raise ValueError("Fast window is capturing more than small window")
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Produce target positions from price data.

        Steps:
          1. fast_ma = rolling mean of data['close'] over self.fast
                 -> data['close'].rolling(self.fast).mean()
          2. slow_ma = rolling mean over self.slow (same pattern)
          3. Build the raw signal: 1.0 where fast_ma > slow_ma, else 0.0
                 Hint: (fast_ma > slow_ma) gives a boolean Series;
                       .astype(float) turns True/False into 1.0/0.0
          4. The first `slow` rows have NaN moving averages (not enough
             history yet). After the boolean comparison those become
             False -> 0.0, which is what we want (flat until warmed up).
          5. Return the signal as a pd.Series indexed exactly like `data`.

        NOTE ON LOOK-AHEAD (important, but handled later):
          The signal at day t is computed from prices through day t.
          To trade honestly you act on it the NEXT day. We'll apply that
          one-day shift in the backtest engine, in ONE place, so every
          strategy gets it consistently. So here: do NOT shift. Just
          return the signal aligned to the day it was computed.

        Return:
            pd.Series of 0.0/1.0 values, indexed identically to `data`.
        """

        fast_ma = data["close"].rolling(self.fast).mean()
        slow_ma = data["close"].rolling(self.slow).mean()
        signal = (fast_ma > slow_ma).astype(float)
        
        return signal

if __name__ == "__main__":
    # Smoke test against real cached data.
    from data.loaders import get_prices

    prices = get_prices("AAPL", "2023-01-01", "2024-01-01")
    strat = SMACrossover(fast=20, slow=50)

    # Call the PUBLIC method so validation runs.
    signals = strat.get_signals(prices)

    print(signals.value_counts())          # how many long vs flat days
    print(signals.tail())                  # peek at the most recent signals
