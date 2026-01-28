"""
Symbol Registry for unified symbol management across all trading instruments.
Supports US stocks, ETFs, and options with zero-change deployment to new symbols.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
from datetime import time


class Market(Enum):
    """Supported markets"""
    US = "US"  # US stocks
    HK = "HK"  # Hong Kong stocks
    CN = "CN"  # China A-shares (via Stock Connect)
    SG = "SG"  # Singapore stocks


class InstrumentType(Enum):
    """Types of trading instruments"""
    STOCK = "stock"
    ETF = "etf"
    LEVERAGED_ETF = "leveraged_etf"
    INVERSE_ETF = "inverse_etf"
    OPTION_CALL = "option_call"
    OPTION_PUT = "option_put"
    FUTURE = "future"


class TradingSession(Enum):
    """Trading sessions for US market"""
    PRE_MARKET = "pre_market"      # 04:00 - 09:30 ET
    REGULAR = "regular"             # 09:30 - 16:00 ET
    AFTER_HOURS = "after_hours"     # 16:00 - 20:00 ET
    CLOSED = "closed"


@dataclass
class TradingSymbol:
    """
    Represents a trading symbol with all necessary metadata.
    Designed for zero-change deployment to new instruments.
    """
    # Core identifiers
    symbol: str                     # e.g., "TQQQ"
    market: Market                  # e.g., Market.US
    futu_code: str                  # e.g., "US.TQQQ"

    # Instrument metadata
    instrument_type: InstrumentType = InstrumentType.ETF
    name: str = ""
    sector: str = ""

    # Trading parameters
    lot_size: int = 1               # Minimum trading lot
    tick_size: float = 0.01         # Minimum price increment

    # Position limits
    max_position_usd: float = 50000.0
    max_position_shares: int = 10000

    # Risk parameters
    volatility_factor: float = 1.0  # Multiplier for position sizing (higher for leveraged ETFs)
    margin_requirement: float = 0.25  # Initial margin requirement

    # Trading restrictions
    can_short: bool = True
    has_options: bool = True
    supports_extended_hours: bool = True

    # Option-specific (if applicable)
    underlying_symbol: Optional[str] = None
    strike_price: Optional[float] = None
    expiry_date: Optional[str] = None
    option_type: Optional[str] = None

    # Calculated fields
    _last_price: float = field(default=0.0, repr=False)
    _bid: float = field(default=0.0, repr=False)
    _ask: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if not self.futu_code:
            self.futu_code = f"{self.market.value}.{self.symbol}"

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self._bid > 0 and self._ask > 0:
            return (self._ask - self._bid) / ((self._ask + self._bid) / 2)
        return 0.0

    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        if self._bid > 0 and self._ask > 0:
            return (self._ask + self._bid) / 2
        return self._last_price

    def update_quote(self, last: float, bid: float, ask: float):
        """Update latest quote data"""
        self._last_price = last
        self._bid = bid
        self._ask = ask

    def max_shares_for_usd(self, usd_amount: float) -> int:
        """Calculate maximum shares purchasable for given USD amount"""
        if self._last_price <= 0:
            return 0
        shares = int(usd_amount / self._last_price)
        return min(shares, self.max_position_shares)

    def is_leveraged(self) -> bool:
        """Check if this is a leveraged instrument"""
        return self.instrument_type in (
            InstrumentType.LEVERAGED_ETF,
            InstrumentType.INVERSE_ETF
        )

    def is_option(self) -> bool:
        """Check if this is an option"""
        return self.instrument_type in (
            InstrumentType.OPTION_CALL,
            InstrumentType.OPTION_PUT
        )


class SymbolRegistry:
    """
    Central registry for all trading symbols.
    Supports dynamic symbol addition for zero-change deployment to new instruments.
    """

    # Pre-configured symbols
    DEFAULT_SYMBOLS: Dict[str, dict] = {
        # Leveraged ETFs
        "US.TQQQ": {
            "symbol": "TQQQ",
            "market": Market.US,
            "name": "ProShares UltraPro QQQ",
            "instrument_type": InstrumentType.LEVERAGED_ETF,
            "volatility_factor": 3.0,
            "sector": "Technology",
            "max_position_usd": 50000.0,
        },
        "US.QQQ": {
            "symbol": "QQQ",
            "market": Market.US,
            "name": "Invesco QQQ Trust",
            "instrument_type": InstrumentType.ETF,
            "volatility_factor": 1.0,
            "sector": "Technology",
            "max_position_usd": 100000.0,
        },
        "US.SQQQ": {
            "symbol": "SQQQ",
            "market": Market.US,
            "name": "ProShares UltraPro Short QQQ",
            "instrument_type": InstrumentType.INVERSE_ETF,
            "volatility_factor": 3.0,
            "sector": "Technology",
            "max_position_usd": 30000.0,
        },
        "US.SPXL": {
            "symbol": "SPXL",
            "market": Market.US,
            "name": "Direxion Daily S&P 500 Bull 3X",
            "instrument_type": InstrumentType.LEVERAGED_ETF,
            "volatility_factor": 3.0,
            "sector": "Broad Market",
            "max_position_usd": 50000.0,
        },
        "US.SOXL": {
            "symbol": "SOXL",
            "market": Market.US,
            "name": "Direxion Daily Semiconductor Bull 3X",
            "instrument_type": InstrumentType.LEVERAGED_ETF,
            "volatility_factor": 3.0,
            "sector": "Semiconductors",
            "max_position_usd": 50000.0,
        },
        # Individual stocks
        "US.AAPL": {
            "symbol": "AAPL",
            "market": Market.US,
            "name": "Apple Inc.",
            "instrument_type": InstrumentType.STOCK,
            "volatility_factor": 1.0,
            "sector": "Technology",
            "max_position_usd": 100000.0,
        },
        "US.NVDA": {
            "symbol": "NVDA",
            "market": Market.US,
            "name": "NVIDIA Corporation",
            "instrument_type": InstrumentType.STOCK,
            "volatility_factor": 1.5,
            "sector": "Semiconductors",
            "max_position_usd": 75000.0,
        },
        "US.TSLA": {
            "symbol": "TSLA",
            "market": Market.US,
            "name": "Tesla Inc.",
            "instrument_type": InstrumentType.STOCK,
            "volatility_factor": 2.0,
            "sector": "Automotive",
            "max_position_usd": 50000.0,
        },
        "US.SPY": {
            "symbol": "SPY",
            "market": Market.US,
            "name": "SPDR S&P 500 ETF Trust",
            "instrument_type": InstrumentType.ETF,
            "volatility_factor": 1.0,
            "sector": "Broad Market",
            "max_position_usd": 200000.0,
        },
    }

    def __init__(self):
        self._symbols: Dict[str, TradingSymbol] = {}
        self._active_symbols: Set[str] = set()
        self._load_default_symbols()

    def _load_default_symbols(self):
        """Load pre-configured default symbols"""
        for futu_code, config in self.DEFAULT_SYMBOLS.items():
            self.register(
                futu_code=futu_code,
                **config
            )

    def register(
        self,
        futu_code: str,
        symbol: str = None,
        market: Market = Market.US,
        **kwargs
    ) -> TradingSymbol:
        """
        Register a new trading symbol.

        Args:
            futu_code: Futu API code (e.g., "US.TQQQ")
            symbol: Ticker symbol (extracted from futu_code if not provided)
            market: Trading market
            **kwargs: Additional TradingSymbol fields

        Returns:
            Registered TradingSymbol instance
        """
        if symbol is None:
            # Extract symbol from futu_code
            parts = futu_code.split(".")
            symbol = parts[-1] if len(parts) > 1 else futu_code

        trading_symbol = TradingSymbol(
            symbol=symbol,
            market=market,
            futu_code=futu_code,
            **kwargs
        )

        self._symbols[futu_code] = trading_symbol
        return trading_symbol

    def register_option(
        self,
        underlying: str,
        strike: float,
        expiry: str,
        option_type: str = "call",
        market: Market = Market.US,
    ) -> TradingSymbol:
        """
        Register an option contract.

        Args:
            underlying: Underlying symbol (e.g., "AAPL")
            strike: Strike price
            expiry: Expiry date (YYYYMMDD format)
            option_type: "call" or "put"
            market: Trading market

        Returns:
            Registered TradingSymbol for the option
        """
        # Construct option symbol
        opt_suffix = "C" if option_type.lower() == "call" else "P"
        strike_str = f"{int(strike * 1000):08d}"
        option_symbol = f"{underlying}{expiry}{opt_suffix}{strike_str}"
        futu_code = f"{market.value}.{option_symbol}"

        instrument_type = (
            InstrumentType.OPTION_CALL if option_type.lower() == "call"
            else InstrumentType.OPTION_PUT
        )

        return self.register(
            futu_code=futu_code,
            symbol=option_symbol,
            market=market,
            instrument_type=instrument_type,
            underlying_symbol=underlying,
            strike_price=strike,
            expiry_date=expiry,
            option_type=option_type,
            volatility_factor=5.0,  # Options are highly volatile
            max_position_usd=20000.0,
            supports_extended_hours=False,
        )

    def get(self, futu_code: str) -> Optional[TradingSymbol]:
        """Get a trading symbol by Futu code"""
        return self._symbols.get(futu_code)

    def get_or_create(self, futu_code: str) -> TradingSymbol:
        """Get existing symbol or create new one with defaults"""
        if futu_code in self._symbols:
            return self._symbols[futu_code]

        # Auto-register with defaults
        return self.register(futu_code=futu_code)

    def activate(self, *futu_codes: str):
        """Activate symbols for trading"""
        for code in futu_codes:
            if code in self._symbols:
                self._active_symbols.add(code)
            else:
                # Auto-register and activate
                self.register(futu_code=code)
                self._active_symbols.add(code)

    def deactivate(self, *futu_codes: str):
        """Deactivate symbols from trading"""
        for code in futu_codes:
            self._active_symbols.discard(code)

    @property
    def active_symbols(self) -> List[TradingSymbol]:
        """Get list of active trading symbols"""
        return [self._symbols[code] for code in self._active_symbols]

    @property
    def active_futu_codes(self) -> List[str]:
        """Get list of active Futu codes for subscription"""
        return list(self._active_symbols)

    def all_symbols(self) -> List[TradingSymbol]:
        """Get all registered symbols"""
        return list(self._symbols.values())

    def symbols_by_type(self, instrument_type: InstrumentType) -> List[TradingSymbol]:
        """Get symbols by instrument type"""
        return [s for s in self._symbols.values() if s.instrument_type == instrument_type]

    def symbols_by_sector(self, sector: str) -> List[TradingSymbol]:
        """Get symbols by sector"""
        return [s for s in self._symbols.values() if s.sector.lower() == sector.lower()]

    @staticmethod
    def get_current_session() -> TradingSession:
        """
        Get current US market trading session based on Eastern Time.
        Note: Simplified logic - production should use proper timezone handling.
        """
        from datetime import datetime
        import pytz

        try:
            et = pytz.timezone('US/Eastern')
            now = datetime.now(et)
            current_time = now.time()
            weekday = now.weekday()

            # Weekend check
            if weekday >= 5:  # Saturday=5, Sunday=6
                return TradingSession.CLOSED

            pre_market_start = time(4, 0)
            regular_start = time(9, 30)
            regular_end = time(16, 0)
            after_hours_end = time(20, 0)

            if current_time < pre_market_start:
                return TradingSession.CLOSED
            elif current_time < regular_start:
                return TradingSession.PRE_MARKET
            elif current_time < regular_end:
                return TradingSession.REGULAR
            elif current_time < after_hours_end:
                return TradingSession.AFTER_HOURS
            else:
                return TradingSession.CLOSED

        except ImportError:
            # If pytz not available, assume regular hours
            return TradingSession.REGULAR


# Singleton registry instance
_registry: Optional[SymbolRegistry] = None


def get_symbol_registry() -> SymbolRegistry:
    """Get the global symbol registry instance"""
    global _registry
    if _registry is None:
        _registry = SymbolRegistry()
    return _registry
