"""
Futu Quote Client - Real-time market data subscription via Futu OpenD
Provides 1-min K-line, real-time quotes, and market state detection
"""
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from threading import Thread, Lock
import queue

from futu import (
    OpenQuoteContext,
    SubType,
    KLType,
    RET_OK,
    RET_ERROR,
    KL_FIELD,
    SysNotifyHandlerBase,
    StockQuoteHandlerBase,
    CurKlineHandlerBase,
    OrderBookHandlerBase,
)

from src.core.logger import get_logger, PerformanceLogger
from src.core.config import get_settings

logger = get_logger(__name__)
perf_logger = PerformanceLogger()


@dataclass
class QuoteData:
    """Real-time quote snapshot"""
    symbol: str
    futu_code: str
    last_price: float
    bid_price: float
    ask_price: float
    bid_volume: int
    ask_volume: int
    volume: int
    turnover: float
    timestamp: datetime

    # Calculated fields
    spread: float = field(init=False)
    mid_price: float = field(init=False)

    def __post_init__(self):
        if self.bid_price > 0 and self.ask_price > 0:
            self.spread = (self.ask_price - self.bid_price) / self.mid_price
            self.mid_price = (self.ask_price + self.bid_price) / 2
        else:
            self.spread = 0.0
            self.mid_price = self.last_price


@dataclass
class KLineData:
    """K-Line (candlestick) data"""
    symbol: str
    futu_code: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    turnover: float
    timestamp: datetime
    kl_type: str = "K_1M"

    @property
    def range_pct(self) -> float:
        """Price range as percentage"""
        if self.low > 0:
            return (self.high - self.low) / self.low
        return 0.0

    @property
    def body_pct(self) -> float:
        """Candle body as percentage"""
        if self.open > 0:
            return abs(self.close - self.open) / self.open
        return 0.0

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish"""
        return self.close > self.open


class QuoteHandler(StockQuoteHandlerBase):
    """Handler for real-time quote updates"""

    def __init__(self, callback: Callable[[str, QuoteData], None]):
        super().__init__()
        self.callback = callback
        self._lock = Lock()

    def on_recv_rsp(self, rsp_pb):
        """Handle quote response"""
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret == RET_OK and data is not None:
            for _, row in data.iterrows():
                try:
                    quote = QuoteData(
                        symbol=row['code'].split('.')[-1],
                        futu_code=row['code'],
                        last_price=float(row.get('last_price', 0)),
                        bid_price=float(row.get('bid_price', 0)),
                        ask_price=float(row.get('ask_price', 0)),
                        bid_volume=int(row.get('bid_vol', 0)),
                        ask_volume=int(row.get('ask_vol', 0)),
                        volume=int(row.get('volume', 0)),
                        turnover=float(row.get('turnover', 0)),
                        timestamp=datetime.now()
                    )
                    self.callback(row['code'], quote)
                except Exception as e:
                    logger.error(f"Error processing quote: {e}")
        return ret, data


class KLineHandler(CurKlineHandlerBase):
    """Handler for real-time K-line updates"""

    def __init__(self, callback: Callable[[str, KLineData], None]):
        super().__init__()
        self.callback = callback

    def on_recv_rsp(self, rsp_pb):
        """Handle K-line response"""
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret == RET_OK and data is not None:
            for _, row in data.iterrows():
                try:
                    kline = KLineData(
                        symbol=row['code'].split('.')[-1],
                        futu_code=row['code'],
                        open=float(row.get('open', 0)),
                        high=float(row.get('high', 0)),
                        low=float(row.get('low', 0)),
                        close=float(row.get('close', 0)),
                        volume=int(row.get('volume', 0)),
                        turnover=float(row.get('turnover', 0)),
                        timestamp=datetime.strptime(
                            row.get('time_key', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                            '%Y-%m-%d %H:%M:%S'
                        ),
                        kl_type=row.get('k_type', 'K_1M')
                    )
                    self.callback(row['code'], kline)
                except Exception as e:
                    logger.error(f"Error processing kline: {e}")
        return ret, data


class FutuQuoteClient:
    """
    Futu OpenD quote client for real-time market data.
    Provides async interface for 1-min K-line and quote subscription.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        rsa_path: str = None
    ):
        settings = get_settings()
        self.host = host or settings.futu_host
        self.port = port or settings.futu_port
        self.rsa_path = rsa_path or settings.futu_rsa_path

        self._context: Optional[OpenQuoteContext] = None
        self._subscribed_symbols: Dict[str, set] = {}  # symbol -> {SubType}

        # Data storage
        self._quotes: Dict[str, QuoteData] = {}
        self._klines: Dict[str, deque] = {}  # symbol -> deque of KLineData
        self._kline_history_size = 100

        # Callbacks
        self._quote_callbacks: List[Callable[[str, QuoteData], None]] = []
        self._kline_callbacks: List[Callable[[str, KLineData], None]] = []

        # Threading
        self._lock = Lock()
        self._event_queue: queue.Queue = queue.Queue()
        self._running = False

    def connect(self) -> bool:
        """Establish connection to Futu OpenD"""
        try:
            start_time = time.time()

            if self.rsa_path:
                self._context = OpenQuoteContext(
                    host=self.host,
                    port=self.port,
                    security_firm=None,
                    is_encrypt=True,
                    rsa_file=self.rsa_path
                )
            else:
                self._context = OpenQuoteContext(
                    host=self.host,
                    port=self.port
                )

            # Set handlers
            self._context.set_handler(QuoteHandler(self._on_quote_update))
            self._context.set_handler(KLineHandler(self._on_kline_update))

            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Connected to Futu OpenD at {self.host}:{self.port} in {latency_ms:.2f}ms")
            perf_logger.latency_record("quote_connect", latency_ms)

            self._running = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Futu OpenD: {e}")
            return False

    def disconnect(self):
        """Disconnect from Futu OpenD"""
        self._running = False
        if self._context:
            self._context.close()
            self._context = None
        logger.info("Disconnected from Futu OpenD")

    def subscribe(
        self,
        symbols: List[str],
        sub_types: List[SubType] = None,
        extended_hours: bool = True
    ) -> bool:
        """
        Subscribe to market data for given symbols.

        Args:
            symbols: List of Futu codes (e.g., ["US.TQQQ", "US.QQQ"])
            sub_types: Subscription types (default: quote + 1min kline)
            extended_hours: Whether to subscribe to extended hours data

        Returns:
            True if subscription successful
        """
        if not self._context:
            logger.error("Not connected to Futu OpenD")
            return False

        if sub_types is None:
            sub_types = [SubType.QUOTE, SubType.K_1M]

        try:
            start_time = time.time()

            # Subscribe to all types for all symbols
            ret, err = self._context.subscribe(
                symbols,
                sub_types,
                subscribe_push=True,
                is_first_push=True,
                extended_time=extended_hours
            )

            if ret != RET_OK:
                logger.error(f"Subscription failed: {err}")
                return False

            # Track subscriptions
            for symbol in symbols:
                if symbol not in self._subscribed_symbols:
                    self._subscribed_symbols[symbol] = set()
                self._subscribed_symbols[symbol].update(sub_types)

                # Initialize storage
                if symbol not in self._klines:
                    self._klines[symbol] = deque(maxlen=self._kline_history_size)

            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Subscribed to {len(symbols)} symbols in {latency_ms:.2f}ms")
            perf_logger.latency_record("quote_subscribe", latency_ms)

            return True

        except Exception as e:
            logger.error(f"Subscription error: {e}")
            return False

    def unsubscribe(self, symbols: List[str], sub_types: List[SubType] = None) -> bool:
        """Unsubscribe from market data"""
        if not self._context:
            return False

        if sub_types is None:
            sub_types = [SubType.QUOTE, SubType.K_1M]

        try:
            ret, err = self._context.unsubscribe(symbols, sub_types)
            if ret != RET_OK:
                logger.error(f"Unsubscribe failed: {err}")
                return False

            for symbol in symbols:
                if symbol in self._subscribed_symbols:
                    self._subscribed_symbols[symbol] -= set(sub_types)

            return True
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")
            return False

    def get_history_kline(
        self,
        symbol: str,
        kl_type: KLType = KLType.K_1M,
        count: int = 100
    ) -> List[KLineData]:
        """
        Get historical K-line data.

        Args:
            symbol: Futu code
            kl_type: K-line type (default 1-min)
            count: Number of bars to fetch

        Returns:
            List of KLineData
        """
        if not self._context:
            return []

        try:
            start_time = time.time()

            ret, data, _ = self._context.get_cur_kline(
                symbol,
                count,
                kl_type=kl_type,
                autype=None
            )

            if ret != RET_OK:
                logger.error(f"Failed to get kline: {data}")
                return []

            klines = []
            for _, row in data.iterrows():
                kline = KLineData(
                    symbol=symbol.split('.')[-1],
                    futu_code=symbol,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume']),
                    turnover=float(row.get('turnover', 0)),
                    timestamp=datetime.strptime(row['time_key'], '%Y-%m-%d %H:%M:%S'),
                    kl_type=str(kl_type)
                )
                klines.append(kline)

            latency_ms = (time.time() - start_time) * 1000
            perf_logger.latency_record("get_history_kline", latency_ms)

            return klines

        except Exception as e:
            logger.error(f"Error getting history kline: {e}")
            return []

    def get_snapshot(self, symbols: List[str]) -> Dict[str, QuoteData]:
        """
        Get current quote snapshot for symbols.

        Args:
            symbols: List of Futu codes

        Returns:
            Dict mapping symbol to QuoteData
        """
        if not self._context:
            return {}

        try:
            start_time = time.time()

            ret, data = self._context.get_stock_quote(symbols)

            if ret != RET_OK:
                logger.error(f"Failed to get quote: {data}")
                return {}

            result = {}
            for _, row in data.iterrows():
                code = row['code']
                quote = QuoteData(
                    symbol=code.split('.')[-1],
                    futu_code=code,
                    last_price=float(row.get('last_price', 0)),
                    bid_price=float(row.get('bid_price', 0)),
                    ask_price=float(row.get('ask_price', 0)),
                    bid_volume=int(row.get('bid_vol', 0)),
                    ask_volume=int(row.get('ask_vol', 0)),
                    volume=int(row.get('volume', 0)),
                    turnover=float(row.get('turnover', 0)),
                    timestamp=datetime.now()
                )
                result[code] = quote
                self._quotes[code] = quote

            latency_ms = (time.time() - start_time) * 1000
            perf_logger.latency_record("get_snapshot", latency_ms)

            return result

        except Exception as e:
            logger.error(f"Error getting snapshot: {e}")
            return {}

    def get_market_state(self, symbols: List[str]) -> Dict[str, str]:
        """
        Get market state for symbols.

        Returns dict mapping symbol to state:
        - MORNING: Pre-market
        - REST: Lunch break (HK)
        - AFTERNOON: Regular session
        - AFTERNOON_END: After hours
        - NIGHT_OPEN: Night session
        """
        if not self._context:
            return {}

        try:
            ret, data = self._context.get_global_state()
            if ret != RET_OK:
                return {}

            # Parse market state per symbol
            states = {}
            for symbol in symbols:
                market = symbol.split('.')[0]
                if market == 'US':
                    states[symbol] = data.get('market_us', 'UNKNOWN')
                elif market == 'HK':
                    states[symbol] = data.get('market_hk', 'UNKNOWN')
                else:
                    states[symbol] = 'UNKNOWN'

            return states

        except Exception as e:
            logger.error(f"Error getting market state: {e}")
            return {}

    def _on_quote_update(self, futu_code: str, quote: QuoteData):
        """Internal callback for quote updates"""
        with self._lock:
            self._quotes[futu_code] = quote

        # Notify registered callbacks
        for callback in self._quote_callbacks:
            try:
                callback(futu_code, quote)
            except Exception as e:
                logger.error(f"Quote callback error: {e}")

    def _on_kline_update(self, futu_code: str, kline: KLineData):
        """Internal callback for kline updates"""
        with self._lock:
            if futu_code not in self._klines:
                self._klines[futu_code] = deque(maxlen=self._kline_history_size)
            self._klines[futu_code].append(kline)

        # Notify registered callbacks
        for callback in self._kline_callbacks:
            try:
                callback(futu_code, kline)
            except Exception as e:
                logger.error(f"KLine callback error: {e}")

    def on_quote(self, callback: Callable[[str, QuoteData], None]):
        """Register callback for quote updates"""
        self._quote_callbacks.append(callback)

    def on_kline(self, callback: Callable[[str, KLineData], None]):
        """Register callback for kline updates"""
        self._kline_callbacks.append(callback)

    def get_latest_quote(self, futu_code: str) -> Optional[QuoteData]:
        """Get latest cached quote"""
        return self._quotes.get(futu_code)

    def get_latest_klines(self, futu_code: str, count: int = None) -> List[KLineData]:
        """Get latest cached klines"""
        with self._lock:
            if futu_code not in self._klines:
                return []
            klines = list(self._klines[futu_code])
            if count:
                return klines[-count:]
            return klines

    @property
    def is_connected(self) -> bool:
        """Check if connected to OpenD"""
        return self._context is not None and self._running

    @property
    def subscribed_symbols(self) -> List[str]:
        """Get list of subscribed symbols"""
        return list(self._subscribed_symbols.keys())

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class AsyncFutuQuoteClient:
    """
    Async wrapper for FutuQuoteClient.
    Provides async/await interface for use in async trading loops.
    """

    def __init__(self, *args, **kwargs):
        self._client = FutuQuoteClient(*args, **kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self) -> bool:
        """Async connect"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._client.connect
        )

    async def disconnect(self):
        """Async disconnect"""
        await asyncio.get_event_loop().run_in_executor(
            None, self._client.disconnect
        )

    async def subscribe(self, *args, **kwargs) -> bool:
        """Async subscribe"""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._client.subscribe(*args, **kwargs)
        )

    async def get_snapshot(self, symbols: List[str]) -> Dict[str, QuoteData]:
        """Async get snapshot"""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._client.get_snapshot(symbols)
        )

    async def get_history_kline(self, *args, **kwargs) -> List[KLineData]:
        """Async get history kline"""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._client.get_history_kline(*args, **kwargs)
        )

    def get_latest_quote(self, futu_code: str) -> Optional[QuoteData]:
        """Get cached quote (no async needed)"""
        return self._client.get_latest_quote(futu_code)

    def get_latest_klines(self, futu_code: str, count: int = None) -> List[KLineData]:
        """Get cached klines (no async needed)"""
        return self._client.get_latest_klines(futu_code, count)

    def on_quote(self, callback):
        """Register quote callback"""
        self._client.on_quote(callback)

    def on_kline(self, callback):
        """Register kline callback"""
        self._client.on_kline(callback)

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
