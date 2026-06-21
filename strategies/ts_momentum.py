import pandas as pd
import numpy as np
from strategies.base import Strategy

class TSMomentum(Strategy):
    def __init__(self, lookback: int = 256):
        # 256 trading days ≈ 1 calendar year
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns:
            raise ValueError(f"Expected 'close' column, got: {list(data.columns)}")

        cumulative_return = data["close"].pct_change(periods=self.lookback)
        signal = np.sign(cumulative_return).fillna(0)  # +1, -1, or 0

        return signal
