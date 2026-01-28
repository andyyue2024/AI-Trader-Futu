"""
Pytest configuration and shared fixtures
"""
import pytest
import asyncio
from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock application settings"""
    from src.core.config import Settings
    return Settings(
        futu_host="127.0.0.1",
        futu_port=11111,
        futu_trade_env="SIMULATE",
        futu_trade_password="test_password",
        llm_provider="openai",
        openai_api_key="sk-test-key",
        openai_model="gpt-4-turbo-preview",
        trading_symbols=["US.TQQQ", "US.QQQ"],
        default_position_size=10000.0,
        max_daily_drawdown=0.03,
        max_total_drawdown=0.15,
        slippage_tolerance=0.002,
        log_level="DEBUG",
    )


@pytest.fixture
def sample_quote_data():
    """Sample quote data for testing"""
    from src.data.futu_quote import QuoteData
    return QuoteData(
        symbol="TQQQ",
        futu_code="US.TQQQ",
        last_price=50.00,
        bid_price=49.99,
        ask_price=50.01,
        bid_volume=1000,
        ask_volume=1200,
        volume=5000000,
        turnover=250000000.0,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_kline_data():
    """Sample kline data for testing"""
    from src.data.futu_quote import KLineData
    return KLineData(
        symbol="TQQQ",
        futu_code="US.TQQQ",
        open=49.50,
        high=50.50,
        low=49.00,
        close=50.00,
        volume=1000000,
        turnover=50000000.0,
        timestamp=datetime.now(),
        kl_type="K_1M"
    )


@pytest.fixture
def sample_klines():
    """Generate sample kline history"""
    from src.data.futu_quote import KLineData
    import random

    klines = []
    base_price = 50.0

    for i in range(100):
        change = random.uniform(-0.5, 0.5)
        open_price = base_price + change
        high = open_price + random.uniform(0, 0.3)
        low = open_price - random.uniform(0, 0.3)
        close = open_price + random.uniform(-0.2, 0.2)
        base_price = close

        klines.append(KLineData(
            symbol="TQQQ",
            futu_code="US.TQQQ",
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=random.randint(100000, 1000000),
            turnover=close * random.randint(100000, 1000000),
            timestamp=datetime(2024, 1, 1, 9, 30 + i),
            kl_type="K_1M"
        ))

    return klines


@pytest.fixture
def sample_position():
    """Sample position for testing"""
    from src.action.futu_executor import Position
    return Position(
        symbol="TQQQ",
        futu_code="US.TQQQ",
        quantity=100,
        avg_cost=49.50,
        market_value=5000.0,
        unrealized_pnl=50.0,
        realized_pnl=0.0
    )


@pytest.fixture
def sample_market_snapshot(sample_klines):
    """Sample market snapshot for testing"""
    from src.data.data_processor import MarketSnapshot, TechnicalIndicators

    return MarketSnapshot(
        symbol="TQQQ",
        futu_code="US.TQQQ",
        timestamp=datetime.now(),
        last_price=50.00,
        bid_price=49.99,
        ask_price=50.01,
        spread_pct=0.04,
        open=49.50,
        high=50.50,
        low=49.00,
        close=50.00,
        volume=5000000,
        change_1m=0.002,
        change_5m=0.01,
        change_15m=0.015,
        change_1h=0.02,
        change_day=0.03,
        indicators=TechnicalIndicators(
            sma_5=49.80,
            sma_10=49.60,
            sma_20=49.40,
            ema_12=49.70,
            ema_26=49.50,
            rsi_14=55.0,
            macd=0.15,
            macd_signal=0.10,
            macd_histogram=0.05,
            bollinger_upper=51.00,
            bollinger_middle=50.00,
            bollinger_lower=49.00,
            atr_14=0.50,
            adx=25.0,
            volume_sma_20=4500000,
            volume_ratio=1.1
        ),
        market_session="regular",
        recent_klines=[
            {"open": 49.50, "high": 50.00, "low": 49.40, "close": 49.80, "volume": 100000},
            {"open": 49.80, "high": 50.10, "low": 49.70, "close": 50.00, "volume": 110000},
        ]
    )


@pytest.fixture
def mock_futu_context():
    """Mock Futu OpenQuoteContext"""
    mock = MagicMock()
    mock.subscribe.return_value = (0, None)  # RET_OK = 0
    mock.unsubscribe.return_value = (0, None)
    mock.get_stock_quote.return_value = (0, MagicMock())
    mock.get_cur_kline.return_value = (0, MagicMock(), None)
    mock.close.return_value = None
    return mock


@pytest.fixture
def mock_futu_trade_context():
    """Mock Futu OpenSecTradeContext"""
    import pandas as pd

    mock = MagicMock()
    mock.unlock_trade.return_value = (0, None)
    mock.get_acc_list.return_value = (0, pd.DataFrame([{"acc_id": "123"}]))
    mock.position_list_query.return_value = (0, pd.DataFrame())
    mock.place_order.return_value = (0, pd.DataFrame([{"order_id": "test-order-123"}]))
    mock.modify_order.return_value = (0, None)
    mock.close.return_value = None
    return mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '''
    {
        "action": "long",
        "confidence": 0.75,
        "position_size_pct": 5,
        "entry_price": 50.00,
        "stop_loss": 49.00,
        "take_profit": 52.00,
        "reasoning": "Strong bullish momentum with MACD crossover",
        "key_factors": ["MACD bullish", "RSI neutral", "Above SMA20"],
        "risk_level": "medium",
        "expected_return": 0.04,
        "risk_reward_ratio": 2.0
    }
    '''
    mock.chat.completions.create.return_value = mock_response
    return mock
