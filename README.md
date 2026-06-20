# Quant Pipeline

An extensible backtesting pipeline for systematic equity trading strategies, built
from scratch in Python. The project prioritizes clean architecture, correct
backtesting methodology (no look-ahead bias, honest transaction costs, and
benchmark-relative evaluation), and reproducibility over execution speed.

A simple moving-average (SMA) crossover strategy is included as the first concrete
implementation and as an end-to-end validation of the framework. New strategies can be
added by implementing a single method, without modifying any other layer.

## Features

- **Pluggable strategy interface** — every strategy implements one method and is
  consumed identically by the engine.
- **Cache-first data loading** — historical prices are fetched from Yahoo Finance and
  cached locally, so repeated runs are fast and reproducible.
- **Look-ahead-safe simulation** — a one-day execution lag is applied centrally, so no
  strategy can accidentally trade on information it would not have had in time.
- **Transaction-cost modeling** — a configurable per-trade cost is charged whenever the
  position changes, so reported results are net of trading frictions.
- **Risk-aware metrics** — total return, annualized Sharpe ratio, and maximum drawdown,
  reported alongside a buy-and-hold benchmark.

## Project structure

```
quant-pipeline/
├── data/
│   ├── __init__.py
│   └── loaders.py          # fetch, clean, and cache price data
├── strategies/
│   ├── __init__.py
│   ├── base.py             # abstract Strategy interface (the keystone)
│   └── sma_crossover.py    # first concrete strategy
├── backtest/
│   ├── __init__.py
│   ├── engine.py           # simulate P&L from signals
│   ├── costs.py            # transaction-cost model
│   └── metrics.py          # performance & risk statistics
├── run.py                  # configuration + single entry point
├── requirements.txt        # direct dependencies
└── requirements.lock.txt   # exact pinned environment
```

## Installation

Requires Python 3.9 or newer.

```bash
# Clone the repository
git clone https://github.com/<your-username>/quant-pipeline.git
cd quant-pipeline

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

For an exact reproduction of the development environment, install from the lock file
instead: `pip install -r requirements.lock.txt`.

## Usage

Run a backtest from the project root:

```bash
python run.py
```

To change what gets tested, edit the configuration block at the top of `run.py`:

```python
TICKER = "AAPL"
START  = "2023-01-01"
END    = "2024-01-01"
FAST   = 20
SLOW   = 50
COST_BPS = 10.0
```

Example output:

```
====================================================
  BACKTEST REPORT
====================================================
  Ticker        : AAPL
  Period        : 2023-01-01  ->  2024-01-01
  Strategy      : SMA crossover (20/50)
  Cost per trade: 10.0 bps
----------------------------------------------------
  Total return  : +18.52%
  Sharpe ratio  : 1.39
  Max drawdown  : -10.09%
----------------------------------------------------
  Buy & hold    : +54.80%
====================================================
```

## Adding a new strategy

Subclass `Strategy` and implement `generate_signals`. The method receives a price
DataFrame and returns a Series of target positions in `[-1, 1]` (`+1` long, `0` flat,
`-1` short), indexed identically to the input. The base class validates the output and
the engine handles execution, costs, and accounting automatically.

```python
from strategies.base import Strategy
import pandas as pd

class MyStrategy(Strategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # ... your logic here ...
        return signals
```

## Methodology notes

- Prices are split- and dividend-adjusted; the adjusted close is the traded price.
- Signals express a target position (a desired end state), not buy/sell orders — trades
  are inferred from changes in position, so a strategy holds continuously while its
  signal is unchanged.
- A one-day execution lag (`signal.shift(1)`) prevents look-ahead bias.
- Results are net of a configurable per-trade transaction cost.
- The Sharpe ratio is annualized with 252 trading days and assumes a zero risk-free rate
  by default.

> **Disclaimer.** This project is for educational and research purposes only. It is not
> financial advice and is not intended for live trading.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
