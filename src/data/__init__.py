"""
Data module - Market data and processing
"""
from .futu_quote import (
    FutuQuoteClient,
    AsyncFutuQuoteClient,
    QuoteData,
    KLineData,
)
from .data_processor import (
    DataProcessor,
    MarketSnapshot,
    TechnicalIndicators,
)
from .options_data import (
    OptionContract,
    OptionChain,
    OptionSide,
    OptionStrategy,
    OptionsDataClient,
    OptionsStrategyBuilder,
)
from .persistence import (
    TradeDatabase,
    TradeRecord,
    DailyPerformanceRecord,
    get_trade_database,
)

__all__ = [
    "FutuQuoteClient",
    "AsyncFutuQuoteClient",
    "QuoteData",
    "KLineData",
    "DataProcessor",
    "MarketSnapshot",
    "TechnicalIndicators",
    "OptionContract",
    "OptionChain",
    "OptionSide",
    "OptionStrategy",
    "OptionsDataClient",
    "OptionsStrategyBuilder",
    "TradeDatabase",
    "TradeRecord",
    "DailyPerformanceRecord",
    "get_trade_database",
]
