"""
Data Processor - Technical indicators and market snapshot generation
Provides processed data for LLM decision making
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

from .futu_quote import QuoteData, KLineData
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TechnicalIndicators:
    """Technical indicators for a symbol"""
    # Price-based
    sma_5: float = 0.0
    sma_10: float = 0.0
    sma_20: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0

    # Momentum
    rsi_14: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # Volatility
    bollinger_upper: float = 0.0
    bollinger_middle: float = 0.0
    bollinger_lower: float = 0.0
    atr_14: float = 0.0

    # Volume
    volume_sma_20: float = 0.0
    volume_ratio: float = 1.0  # Current volume / SMA volume

    # Trend
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0

    @property
    def is_overbought(self) -> bool:
        return self.rsi_14 > 70

    @property
    def is_oversold(self) -> bool:
        return self.rsi_14 < 30

    @property
    def macd_bullish(self) -> bool:
        return self.macd > self.macd_signal

    @property
    def macd_bearish(self) -> bool:
        return self.macd < self.macd_signal

    @property
    def above_bollinger_upper(self) -> bool:
        return self.bollinger_upper > 0  # Need current price to compare

    @property
    def trend_strength(self) -> str:
        """ADX-based trend strength"""
        if self.adx < 20:
            return "weak"
        elif self.adx < 40:
            return "moderate"
        else:
            return "strong"

    def to_dict(self) -> dict:
        """Convert to dictionary for LLM prompt"""
        return {
            "sma_5": round(self.sma_5, 4),
            "sma_10": round(self.sma_10, 4),
            "sma_20": round(self.sma_20, 4),
            "ema_12": round(self.ema_12, 4),
            "ema_26": round(self.ema_26, 4),
            "rsi_14": round(self.rsi_14, 2),
            "macd": round(self.macd, 4),
            "macd_signal": round(self.macd_signal, 4),
            "macd_histogram": round(self.macd_histogram, 4),
            "bollinger_upper": round(self.bollinger_upper, 4),
            "bollinger_middle": round(self.bollinger_middle, 4),
            "bollinger_lower": round(self.bollinger_lower, 4),
            "atr_14": round(self.atr_14, 4),
            "adx": round(self.adx, 2),
            "volume_ratio": round(self.volume_ratio, 2),
            "trend_strength": self.trend_strength,
            "is_overbought": self.is_overbought,
            "is_oversold": self.is_oversold,
            "macd_signal_bullish": self.macd_bullish,
        }


@dataclass
class MarketSnapshot:
    """
    Complete market snapshot for a symbol at a point in time.
    Used as input for LLM decision making.
    """
    # Identity
    symbol: str
    futu_code: str
    timestamp: datetime

    # Current price data
    last_price: float
    bid_price: float
    ask_price: float
    spread_pct: float

    # Recent OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: int

    # Price changes
    change_1m: float = 0.0  # 1-minute change
    change_5m: float = 0.0  # 5-minute change
    change_15m: float = 0.0  # 15-minute change
    change_1h: float = 0.0  # 1-hour change
    change_day: float = 0.0  # Daily change

    # Technical indicators
    indicators: TechnicalIndicators = field(default_factory=TechnicalIndicators)

    # Market context
    market_session: str = "regular"  # pre_market, regular, after_hours
    seconds_to_close: int = 0  # Seconds until market close

    # Recent klines for context
    recent_klines: List[Dict] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """
        Generate formatted context string for LLM prompt.
        """
        indicators = self.indicators.to_dict()

        context = f"""
=== Market Snapshot for {self.symbol} ===
Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Session: {self.market_session}

PRICE DATA:
- Current: ${self.last_price:.4f}
- Bid/Ask: ${self.bid_price:.4f} / ${self.ask_price:.4f} (spread: {self.spread_pct:.4f}%)
- OHLC: O={self.open:.4f}, H={self.high:.4f}, L={self.low:.4f}, C={self.close:.4f}
- Volume: {self.volume:,}

PRICE CHANGES:
- 1 min: {self.change_1m:+.2%}
- 5 min: {self.change_5m:+.2%}
- 15 min: {self.change_15m:+.2%}
- 1 hour: {self.change_1h:+.2%}
- Today: {self.change_day:+.2%}

TECHNICAL INDICATORS:
- SMA: 5={indicators['sma_5']:.4f}, 10={indicators['sma_10']:.4f}, 20={indicators['sma_20']:.4f}
- EMA: 12={indicators['ema_12']:.4f}, 26={indicators['ema_26']:.4f}
- RSI(14): {indicators['rsi_14']:.1f} {'(OVERBOUGHT)' if indicators['is_overbought'] else '(OVERSOLD)' if indicators['is_oversold'] else ''}
- MACD: {indicators['macd']:.4f}, Signal: {indicators['macd_signal']:.4f}, Histogram: {indicators['macd_histogram']:.4f}
- Bollinger: Upper={indicators['bollinger_upper']:.4f}, Mid={indicators['bollinger_middle']:.4f}, Lower={indicators['bollinger_lower']:.4f}
- ATR(14): {indicators['atr_14']:.4f}
- ADX: {indicators['adx']:.1f} (Trend: {indicators['trend_strength']})
- Volume Ratio: {indicators['volume_ratio']:.2f}x average

RECENT CANDLES (last 5):
"""
        for i, kl in enumerate(self.recent_klines[-5:]):
            direction = "▲" if kl.get('close', 0) > kl.get('open', 0) else "▼"
            context += f"  [{i+1}] {direction} O={kl.get('open', 0):.4f} H={kl.get('high', 0):.4f} L={kl.get('low', 0):.4f} C={kl.get('close', 0):.4f} V={kl.get('volume', 0):,}\n"

        return context

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "futu_code": self.futu_code,
            "timestamp": self.timestamp.isoformat(),
            "last_price": self.last_price,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "spread_pct": self.spread_pct,
            "ohlcv": {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume
            },
            "changes": {
                "1m": self.change_1m,
                "5m": self.change_5m,
                "15m": self.change_15m,
                "1h": self.change_1h,
                "day": self.change_day
            },
            "indicators": self.indicators.to_dict(),
            "market_session": self.market_session,
            "seconds_to_close": self.seconds_to_close
        }


class DataProcessor:
    """
    Processes raw market data into technical indicators and snapshots.
    Maintains rolling buffers for efficient calculation.
    """

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self._kline_buffers: Dict[str, deque] = {}  # symbol -> deque of dicts
        self._quote_cache: Dict[str, QuoteData] = {}
        self._indicator_cache: Dict[str, TechnicalIndicators] = {}

    def update_kline(self, futu_code: str, kline: KLineData):
        """Update kline buffer for symbol"""
        if futu_code not in self._kline_buffers:
            self._kline_buffers[futu_code] = deque(maxlen=self.history_size)

        self._kline_buffers[futu_code].append({
            'timestamp': kline.timestamp,
            'open': kline.open,
            'high': kline.high,
            'low': kline.low,
            'close': kline.close,
            'volume': kline.volume
        })

        # Invalidate indicator cache
        if futu_code in self._indicator_cache:
            del self._indicator_cache[futu_code]

    def update_quote(self, futu_code: str, quote: QuoteData):
        """Update quote cache for symbol"""
        self._quote_cache[futu_code] = quote

    def load_history(self, futu_code: str, klines: List[KLineData]):
        """Load historical klines into buffer"""
        self._kline_buffers[futu_code] = deque(maxlen=self.history_size)
        for kline in klines:
            self._kline_buffers[futu_code].append({
                'timestamp': kline.timestamp,
                'open': kline.open,
                'high': kline.high,
                'low': kline.low,
                'close': kline.close,
                'volume': kline.volume
            })

    def get_latest_klines(self, futu_code: str, count: int = None) -> List[dict]:
        """Get latest klines from buffer"""
        if futu_code not in self._kline_buffers:
            return []
        klines = list(self._kline_buffers[futu_code])
        if count:
            return klines[-count:]
        return klines

    def calculate_indicators(self, futu_code: str) -> TechnicalIndicators:
        """
        Calculate technical indicators for symbol.
        Uses ta library if available, otherwise falls back to manual calculation.
        """
        if futu_code in self._indicator_cache:
            return self._indicator_cache[futu_code]

        if futu_code not in self._kline_buffers or len(self._kline_buffers[futu_code]) < 5:
            return TechnicalIndicators()

        # Convert to pandas DataFrame
        klines = list(self._kline_buffers[futu_code])
        df = pd.DataFrame(klines)

        indicators = TechnicalIndicators()

        try:
            if HAS_TA:
                indicators = self._calculate_with_ta(df)
            else:
                indicators = self._calculate_manual(df)
        except Exception as e:
            logger.warning(f"Error calculating indicators for {futu_code}: {e}")

        self._indicator_cache[futu_code] = indicators
        return indicators

    def _calculate_with_ta(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate indicators using ta library"""
        indicators = TechnicalIndicators()

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Moving averages
        indicators.sma_5 = ta.trend.sma_indicator(close, window=5).iloc[-1]
        indicators.sma_10 = ta.trend.sma_indicator(close, window=10).iloc[-1]
        indicators.sma_20 = ta.trend.sma_indicator(close, window=20).iloc[-1] if len(df) >= 20 else close.mean()
        indicators.ema_12 = ta.trend.ema_indicator(close, window=12).iloc[-1]
        indicators.ema_26 = ta.trend.ema_indicator(close, window=26).iloc[-1] if len(df) >= 26 else close.ewm(span=12).mean().iloc[-1]

        # RSI
        if len(df) >= 14:
            indicators.rsi_14 = ta.momentum.rsi(close, window=14).iloc[-1]

        # MACD
        if len(df) >= 26:
            macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
            indicators.macd = macd.macd().iloc[-1]
            indicators.macd_signal = macd.macd_signal().iloc[-1]
            indicators.macd_histogram = macd.macd_diff().iloc[-1]

        # Bollinger Bands
        if len(df) >= 20:
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            indicators.bollinger_upper = bb.bollinger_hband().iloc[-1]
            indicators.bollinger_middle = bb.bollinger_mavg().iloc[-1]
            indicators.bollinger_lower = bb.bollinger_lband().iloc[-1]

        # ATR
        if len(df) >= 14:
            indicators.atr_14 = ta.volatility.average_true_range(high, low, close, window=14).iloc[-1]

        # ADX
        if len(df) >= 14:
            adx = ta.trend.ADXIndicator(high, low, close, window=14)
            indicators.adx = adx.adx().iloc[-1]
            indicators.plus_di = adx.adx_pos().iloc[-1]
            indicators.minus_di = adx.adx_neg().iloc[-1]

        # Volume
        if len(df) >= 20:
            indicators.volume_sma_20 = volume.rolling(window=20).mean().iloc[-1]
            if indicators.volume_sma_20 > 0:
                indicators.volume_ratio = volume.iloc[-1] / indicators.volume_sma_20

        return indicators

    def _calculate_manual(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Fallback manual calculation without ta library"""
        indicators = TechnicalIndicators()

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Simple moving averages
        indicators.sma_5 = close.tail(5).mean()
        indicators.sma_10 = close.tail(10).mean()
        indicators.sma_20 = close.tail(20).mean() if len(df) >= 20 else close.mean()

        # EMA
        indicators.ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        indicators.ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1] if len(df) >= 26 else indicators.ema_12

        # RSI (simplified)
        if len(df) >= 14:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 0.0001)
            indicators.rsi_14 = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD (simplified)
        indicators.macd = indicators.ema_12 - indicators.ema_26

        # Bollinger Bands
        if len(df) >= 20:
            sma = close.rolling(window=20).mean()
            std = close.rolling(window=20).std()
            indicators.bollinger_middle = sma.iloc[-1]
            indicators.bollinger_upper = (sma + 2 * std).iloc[-1]
            indicators.bollinger_lower = (sma - 2 * std).iloc[-1]

        # ATR (simplified)
        if len(df) >= 14:
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            indicators.atr_14 = tr.rolling(window=14).mean().iloc[-1]

        # Volume ratio
        if len(df) >= 20:
            vol_sma = volume.rolling(window=20).mean().iloc[-1]
            if vol_sma > 0:
                indicators.volume_ratio = volume.iloc[-1] / vol_sma

        return indicators

    def get_snapshot(
        self,
        futu_code: str,
        market_session: str = "regular"
    ) -> Optional[MarketSnapshot]:
        """
        Generate complete market snapshot for symbol.
        """
        if futu_code not in self._kline_buffers or len(self._kline_buffers[futu_code]) < 1:
            return None

        klines = list(self._kline_buffers[futu_code])
        quote = self._quote_cache.get(futu_code)

        # Get latest kline
        latest = klines[-1]

        # Calculate price changes
        change_1m = 0.0
        change_5m = 0.0
        change_15m = 0.0
        change_1h = 0.0

        if len(klines) >= 2:
            change_1m = (latest['close'] - klines[-2]['close']) / klines[-2]['close'] if klines[-2]['close'] > 0 else 0
        if len(klines) >= 5:
            change_5m = (latest['close'] - klines[-5]['close']) / klines[-5]['close'] if klines[-5]['close'] > 0 else 0
        if len(klines) >= 15:
            change_15m = (latest['close'] - klines[-15]['close']) / klines[-15]['close'] if klines[-15]['close'] > 0 else 0
        if len(klines) >= 60:
            change_1h = (latest['close'] - klines[-60]['close']) / klines[-60]['close'] if klines[-60]['close'] > 0 else 0

        # Day change (using first bar of the day as reference)
        day_open = klines[0]['open']
        change_day = (latest['close'] - day_open) / day_open if day_open > 0 else 0

        # Get technical indicators
        indicators = self.calculate_indicators(futu_code)

        # Build snapshot
        snapshot = MarketSnapshot(
            symbol=futu_code.split('.')[-1],
            futu_code=futu_code,
            timestamp=latest['timestamp'] if isinstance(latest['timestamp'], datetime) else datetime.now(),
            last_price=quote.last_price if quote else latest['close'],
            bid_price=quote.bid_price if quote else latest['close'],
            ask_price=quote.ask_price if quote else latest['close'],
            spread_pct=(quote.spread * 100) if quote else 0,
            open=latest['open'],
            high=latest['high'],
            low=latest['low'],
            close=latest['close'],
            volume=latest['volume'],
            change_1m=change_1m,
            change_5m=change_5m,
            change_15m=change_15m,
            change_1h=change_1h,
            change_day=change_day,
            indicators=indicators,
            market_session=market_session,
            recent_klines=klines[-10:]  # Last 10 candles
        )

        return snapshot

    def get_multi_snapshot(
        self,
        futu_codes: List[str],
        market_session: str = "regular"
    ) -> Dict[str, MarketSnapshot]:
        """Get snapshots for multiple symbols"""
        snapshots = {}
        for code in futu_codes:
            snapshot = self.get_snapshot(code, market_session)
            if snapshot:
                snapshots[code] = snapshot
        return snapshots
