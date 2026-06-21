"""
data/loaders.py
---------------
Fetch and cache historical price data.

Public interface (the only thing downstream code should call):
    get_prices(ticker, start, end) -> pd.DataFrame

The returned DataFrame is indexed by date and has clean OHLCV columns.
Whether the data came from the network or a local cache is an
implementation detail that callers never need to know about.
"""

from pathlib import Path
import pandas as pd
import yfinance as yf


# Where cached price files live. Keep this out of git (add to .gitignore).
CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    """
    Build the cache filename for a given request.

    Each unique (ticker, start, end) combination maps to one file.

    TODO:
      - Return a Path inside CACHE_DIR.
      - Suggested name: f"{ticker}_{start}_{end}.parquet"
    """

    return CACHE_DIR / f"{ticker.upper()}_{start}_{end}.parquet"  


def _fetch_from_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download raw data from Yahoo Finance.

    TODO:
      - Call yf.download(ticker, start=start, end=end).
        Consider auto_adjust=True so prices are adjusted for splits/dividends
        (record WHY in your decisions log — it changes your returns).
      - yfinance can return an empty DataFrame on a bad ticker or no data.
        Decide how to handle that (raise a clear error?).
      - Return the DataFrame. Cleaning happens in _clean(), not here.
    """
    
    df = yf.download(tickers=ticker, start=start, end=end, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker!r} between {start} and {end}")

    return df
         
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw Yahoo output into a tidy, predictable DataFrame.
 
    TODO:
      - Ensure the index is a sorted DatetimeIndex.
      - Drop rows with missing prices (or decide on another policy — note it).
      - Standardize column names to lowercase: open/high/low/close/volume.
      - Return the cleaned frame. 
    """

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()                      

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")

    df = df.rename(columns={
        "Adj Close": "close",
        "Open": "open", 
        "High": "high", 
        "Low": "low", 
        "Volume": "volume",
        })
    df = df.drop(columns=["Close"])           # drop raw close; trade on adjusted
    df = df.dropna(subset=["close"])          # this line is only drops na values in "close" col.

    return df[["open", "high", "low", "close", "volume"]]

def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Get historical prices for `ticker` between `start` and `end`.

    Cache-first: if we've fetched this exact request before, read from disk.
    Otherwise fetch from Yahoo, clean it, cache it, and return it.

    Args:
        ticker: e.g. "AAPL"
        start:  "YYYY-MM-DD"
        end:    "YYYY-MM-DD"

    Returns:
        Cleaned DataFrame indexed by date.

    TODO: 
      1. Make sure CACHE_DIR exists (Path.mkdir(parents=True, exist_ok=True)).
      2. Compute the cache path via _cache_path(...).
      3. If that file exists -> read and return it (pd.read_parquet).
      4. Otherwise:
           - raw = _fetch_from_yahoo(...)
           - df  = _clean(raw)
           - write df to the cache path (df.to_parquet(...))
           - return df
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)        # 1. ensure dir exists
    cache_path = _cache_path(ticker, start, end)        # 2. compute path

    if cache_path.exists():                             # 3. cache hit -> read & return
        return pd.read_parquet(cache_path)

    raw = _fetch_from_yahoo(ticker, start, end)         # 4. miss -> fetch, clean, cache

    df = _clean(raw)
    df.to_parquet(cache_path)

    return df

if __name__ == "__main__":
    # Quick manual test.
    df = get_prices("AAPL", "2023-01-01", "2024-01-01")
    print(df.head())
    print(f"\n{len(df)} rows, columns: {list(df.columns)}")
