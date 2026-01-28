"""
Unit tests for data processing module
"""
import pytest
from datetime import datetime


class TestQuoteData:
    """Test QuoteData class"""

    def test_quote_data_creation(self, sample_quote_data):
        """Test QuoteData creation and calculated fields"""
        assert sample_quote_data.symbol == "TQQQ"
        assert sample_quote_data.futu_code == "US.TQQQ"
        assert sample_quote_data.last_price == 50.00
        assert sample_quote_data.bid_price == 49.99
        assert sample_quote_data.ask_price == 50.01

        # Check calculated fields
        assert sample_quote_data.mid_price == pytest.approx(50.0, rel=0.001)
        assert sample_quote_data.spread > 0

    def test_quote_data_spread_calculation(self):
        """Test spread calculation"""
        from src.data.futu_quote import QuoteData

        quote = QuoteData(
            symbol="TEST",
            futu_code="US.TEST",
            last_price=100.0,
            bid_price=99.90,
            ask_price=100.10,
            bid_volume=1000,
            ask_volume=1000,
            volume=100000,
            turnover=10000000.0,
            timestamp=datetime.now()
        )

        # Spread = (ask - bid) / mid = 0.20 / 100 = 0.002 = 0.2%
        assert quote.spread == pytest.approx(0.002, rel=0.01)


class TestKLineData:
    """Test KLineData class"""

    def test_kline_data_creation(self, sample_kline_data):
        """Test KLineData creation"""
        assert sample_kline_data.symbol == "TQQQ"
        assert sample_kline_data.open == 49.50
        assert sample_kline_data.high == 50.50
        assert sample_kline_data.low == 49.00
        assert sample_kline_data.close == 50.00
        assert sample_kline_data.kl_type == "K_1M"

    def test_kline_bullish_detection(self):
        """Test bullish candle detection"""
        from src.data.futu_quote import KLineData

        # Bullish candle (close > open)
        bullish = KLineData(
            symbol="TEST",
            futu_code="US.TEST",
            open=49.0,
            high=51.0,
            low=48.5,
            close=50.0,
            volume=100000,
            turnover=5000000.0,
            timestamp=datetime.now()
        )
        assert bullish.is_bullish is True

        # Bearish candle (close < open)
        bearish = KLineData(
            symbol="TEST",
            futu_code="US.TEST",
            open=50.0,
            high=51.0,
            low=48.5,
            close=49.0,
            volume=100000,
            turnover=5000000.0,
            timestamp=datetime.now()
        )
        assert bearish.is_bullish is False

    def test_kline_range_calculation(self):
        """Test price range calculation"""
        from src.data.futu_quote import KLineData

        kline = KLineData(
            symbol="TEST",
            futu_code="US.TEST",
            open=50.0,
            high=52.0,
            low=48.0,
            close=51.0,
            volume=100000,
            turnover=5000000.0,
            timestamp=datetime.now()
        )

        # Range = (high - low) / low = 4 / 48 = 0.0833
        assert kline.range_pct == pytest.approx(0.0833, rel=0.01)


class TestDataProcessor:
    """Test DataProcessor class"""

    def test_kline_buffer_update(self, sample_kline_data):
        """Test kline buffer updates"""
        from src.data.data_processor import DataProcessor

        processor = DataProcessor()
        processor.update_kline("US.TQQQ", sample_kline_data)

        klines = processor.get_latest_klines("US.TQQQ")
        assert len(klines) > 0

    def test_history_loading(self, sample_klines):
        """Test historical kline loading"""
        from src.data.data_processor import DataProcessor

        processor = DataProcessor()
        processor.load_history("US.TQQQ", sample_klines)

        klines = processor.get_latest_klines("US.TQQQ")
        assert len(klines) == 100

    def test_indicator_calculation(self, sample_klines):
        """Test technical indicator calculation"""
        from src.data.data_processor import DataProcessor

        processor = DataProcessor()
        processor.load_history("US.TQQQ", sample_klines)

        indicators = processor.calculate_indicators("US.TQQQ")

        # Check indicators are calculated
        assert indicators.sma_5 > 0
        assert indicators.sma_10 > 0
        assert indicators.sma_20 > 0
        assert 0 <= indicators.rsi_14 <= 100

    def test_snapshot_generation(self, sample_klines, sample_quote_data):
        """Test market snapshot generation"""
        from src.data.data_processor import DataProcessor

        processor = DataProcessor()
        processor.load_history("US.TQQQ", sample_klines)
        processor.update_quote("US.TQQQ", sample_quote_data)

        snapshot = processor.get_snapshot("US.TQQQ")

        assert snapshot is not None
        assert snapshot.symbol == "TQQQ"
        assert snapshot.futu_code == "US.TQQQ"
        assert snapshot.last_price > 0
        assert snapshot.indicators is not None

    def test_snapshot_prompt_context(self, sample_market_snapshot):
        """Test snapshot prompt context generation"""
        context = sample_market_snapshot.to_prompt_context()

        assert "TQQQ" in context
        assert "PRICE DATA" in context
        assert "TECHNICAL INDICATORS" in context
        assert "RSI" in context
        assert "MACD" in context

    def test_snapshot_to_dict(self, sample_market_snapshot):
        """Test snapshot dictionary conversion"""
        data = sample_market_snapshot.to_dict()

        assert data["symbol"] == "TQQQ"
        assert data["futu_code"] == "US.TQQQ"
        assert "ohlcv" in data
        assert "changes" in data
        assert "indicators" in data


class TestTechnicalIndicators:
    """Test TechnicalIndicators class"""

    def test_overbought_detection(self):
        """Test overbought detection"""
        from src.data.data_processor import TechnicalIndicators

        overbought = TechnicalIndicators(rsi_14=75.0)
        assert overbought.is_overbought is True
        assert overbought.is_oversold is False

        oversold = TechnicalIndicators(rsi_14=25.0)
        assert oversold.is_overbought is False
        assert oversold.is_oversold is True

        neutral = TechnicalIndicators(rsi_14=50.0)
        assert neutral.is_overbought is False
        assert neutral.is_oversold is False

    def test_macd_signals(self):
        """Test MACD signal detection"""
        from src.data.data_processor import TechnicalIndicators

        bullish = TechnicalIndicators(macd=0.15, macd_signal=0.10)
        assert bullish.macd_bullish is True
        assert bullish.macd_bearish is False

        bearish = TechnicalIndicators(macd=0.05, macd_signal=0.10)
        assert bearish.macd_bullish is False
        assert bearish.macd_bearish is True

    def test_trend_strength(self):
        """Test trend strength classification"""
        from src.data.data_processor import TechnicalIndicators

        weak = TechnicalIndicators(adx=15.0)
        assert weak.trend_strength == "weak"

        moderate = TechnicalIndicators(adx=30.0)
        assert moderate.trend_strength == "moderate"

        strong = TechnicalIndicators(adx=50.0)
        assert strong.trend_strength == "strong"

    def test_to_dict(self):
        """Test dictionary conversion"""
        from src.data.data_processor import TechnicalIndicators

        indicators = TechnicalIndicators(
            sma_5=50.123456,
            rsi_14=55.5,
            macd=0.123456
        )

        data = indicators.to_dict()

        # Check rounding
        assert data["sma_5"] == 50.1235  # 4 decimal places
        assert data["rsi_14"] == 55.5    # 2 decimal places
