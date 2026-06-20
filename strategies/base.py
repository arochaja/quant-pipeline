"""
strategies/base.py
------------------
The Strategy interface — the architectural keystone of the pipeline.

Every concrete strategy (SMA crossover, momentum, vol-targeting, ...)
subclasses Strategy and implements generate_signals(). The backtest
engine consumes any Strategy through this identical interface and never
needs to know which concrete strategy it received.

Signal convention:
    generate_signals() returns a pd.Series indexed by the same dates as
    the input price data. Each value is a TARGET POSITION:
        +1.0  = fully long
         0.0  = flat (in cash)
        -1.0  = fully short
    Fractional values (e.g. 0.5) are allowed for partial sizing.
"""

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """Abstract base class. All strategies inherit from this."""

    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """Public entry point: calls the strategy, then validates."""
        signals = self.generate_signals(data)
        if not signals.between(-1, 1).all():
            raise ValueError("Signals must be in [-1, 1]")
        if not signals.index.equals(data.index):
            raise ValueError("Signal index must match data index")
        return signals

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Map price data to target positions.

        Args:
            data: cleaned price DataFrame from data.loaders.get_prices,
                  indexed by date with at least a 'close' column.

        Returns:
            pd.Series of target positions (-1..1), indexed identically
            to `data`. One value per row in `data`.

        Concrete strategies implement this. The base class only defines
        the contract — it intentionally has no implementation.
        """
        ...
