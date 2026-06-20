import pandas as pd
from strategies.base import Strategy

class EMACrossover(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

        if fast >= slow:
            raise ValueError("Fast window is capturing more than small window")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast_ma = data["close"].ewm(span=self.fast).mean()
        slow_ma = data["close"].ewm(span=self.slow).mean()
        signal = (fast_ma > slow_ma).astype(float)

        return signal

if __name__ == "__main__":
    from data.loaders import get_prices

    prices = get_prices("AAPL", "2023-01-01", "2024-01-01")
    strat = EMACrossover(fast=20, slow=50)

    signals = strat.get_signals(prices)

    print(signals.value_counts)
    print(signals.tail())
