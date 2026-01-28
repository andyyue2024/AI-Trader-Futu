"""
Unit tests for core configuration module
"""
import pytest
import os
from unittest.mock import patch


class TestSettings:
    """Test Settings class"""

    def test_default_settings(self):
        """Test default settings values"""
        from src.core.config import Settings

        settings = Settings()

        assert settings.futu_host == "127.0.0.1"
        assert settings.futu_port == 11111
        assert settings.futu_trade_env == "SIMULATE"
        assert settings.llm_provider == "openai"
        assert settings.max_daily_drawdown == 0.03
        assert settings.max_total_drawdown == 0.15
        assert settings.slippage_tolerance == 0.002

    def test_settings_from_env(self):
        """Test settings loaded from environment variables"""
        from src.core.config import Settings

        with patch.dict(os.environ, {
            "FUTU_HOST": "192.168.1.100",
            "FUTU_PORT": "22222",
            "FUTU_TRADE_ENV": "REAL",
            "LLM_PROVIDER": "anthropic",
            "MAX_DAILY_DRAWDOWN": "0.05",
        }):
            settings = Settings()

            assert settings.futu_host == "192.168.1.100"
            assert settings.futu_port == 22222
            assert settings.futu_trade_env == "REAL"
            assert settings.llm_provider == "anthropic"
            assert settings.max_daily_drawdown == 0.05

    def test_trade_env_validation(self):
        """Test trade environment validation"""
        from src.core.config import Settings
        from pydantic import ValidationError

        # Valid values
        Settings(futu_trade_env="REAL")
        Settings(futu_trade_env="SIMULATE")
        Settings(futu_trade_env="real")  # Should be normalized to REAL

        # Invalid value
        with pytest.raises(ValidationError):
            Settings(futu_trade_env="INVALID")

    def test_llm_provider_validation(self):
        """Test LLM provider validation"""
        from src.core.config import Settings
        from pydantic import ValidationError

        # Valid values
        Settings(llm_provider="openai")
        Settings(llm_provider="anthropic")
        Settings(llm_provider="OpenAI")  # Should be normalized

        # Invalid value
        with pytest.raises(ValidationError):
            Settings(llm_provider="invalid_provider")

    def test_trading_symbols_parsing(self):
        """Test trading symbols parsing from string"""
        from src.core.config import Settings

        # From comma-separated string
        settings = Settings(trading_symbols="US.TQQQ,US.QQQ,US.SOXL")
        assert settings.trading_symbols == ["US.TQQQ", "US.QQQ", "US.SOXL"]

        # From list
        settings = Settings(trading_symbols=["US.AAPL", "US.TSLA"])
        assert settings.trading_symbols == ["US.AAPL", "US.TSLA"]

    def test_is_real_trading_property(self):
        """Test is_real_trading property"""
        from src.core.config import Settings

        settings = Settings(futu_trade_env="REAL")
        assert settings.is_real_trading is True
        assert settings.is_paper_trading is False

        settings = Settings(futu_trade_env="SIMULATE")
        assert settings.is_real_trading is False
        assert settings.is_paper_trading is True


class TestSymbolRegistry:
    """Test SymbolRegistry class"""

    def test_default_symbols_loaded(self):
        """Test that default symbols are loaded"""
        from src.core.symbols import SymbolRegistry

        registry = SymbolRegistry()

        # Check default symbols exist
        assert registry.get("US.TQQQ") is not None
        assert registry.get("US.QQQ") is not None
        assert registry.get("US.SOXL") is not None
        assert registry.get("US.AAPL") is not None

    def test_symbol_registration(self):
        """Test registering new symbols"""
        from src.core.symbols import SymbolRegistry, Market, InstrumentType

        registry = SymbolRegistry()

        # Register new symbol
        symbol = registry.register(
            futu_code="US.GOOGL",
            symbol="GOOGL",
            market=Market.US,
            instrument_type=InstrumentType.STOCK,
            name="Alphabet Inc."
        )

        assert symbol.symbol == "GOOGL"
        assert symbol.futu_code == "US.GOOGL"
        assert symbol.market == Market.US

        # Retrieve
        retrieved = registry.get("US.GOOGL")
        assert retrieved is not None
        assert retrieved.name == "Alphabet Inc."

    def test_symbol_activation(self):
        """Test symbol activation/deactivation"""
        from src.core.symbols import SymbolRegistry

        registry = SymbolRegistry()

        # Activate symbols
        registry.activate("US.TQQQ", "US.QQQ")

        assert "US.TQQQ" in registry.active_futu_codes
        assert "US.QQQ" in registry.active_futu_codes

        # Deactivate
        registry.deactivate("US.TQQQ")

        assert "US.TQQQ" not in registry.active_futu_codes
        assert "US.QQQ" in registry.active_futu_codes

    def test_option_registration(self):
        """Test option contract registration"""
        from src.core.symbols import SymbolRegistry, InstrumentType

        registry = SymbolRegistry()

        option = registry.register_option(
            underlying="AAPL",
            strike=150.0,
            expiry="20240315",
            option_type="call"
        )

        assert option.underlying_symbol == "AAPL"
        assert option.strike_price == 150.0
        assert option.instrument_type == InstrumentType.OPTION_CALL
        assert option.is_option()

    def test_leveraged_etf_detection(self):
        """Test leveraged ETF detection"""
        from src.core.symbols import SymbolRegistry, InstrumentType

        registry = SymbolRegistry()

        tqqq = registry.get("US.TQQQ")
        assert tqqq is not None
        assert tqqq.instrument_type == InstrumentType.LEVERAGED_ETF
        assert tqqq.is_leveraged()
        assert tqqq.volatility_factor == 3.0

        qqq = registry.get("US.QQQ")
        assert qqq is not None
        assert qqq.instrument_type == InstrumentType.ETF
        assert not qqq.is_leveraged()


class TestTradingSymbol:
    """Test TradingSymbol class"""

    def test_quote_update(self):
        """Test updating quote data"""
        from src.core.symbols import TradingSymbol, Market

        symbol = TradingSymbol(
            symbol="TEST",
            market=Market.US,
            futu_code="US.TEST"
        )

        symbol.update_quote(last=100.0, bid=99.95, ask=100.05)

        assert symbol._last_price == 100.0
        assert symbol._bid == 99.95
        assert symbol._ask == 100.05
        assert abs(symbol.mid_price - 100.0) < 0.001
        assert symbol.spread > 0

    def test_max_shares_calculation(self):
        """Test maximum shares calculation"""
        from src.core.symbols import TradingSymbol, Market

        symbol = TradingSymbol(
            symbol="TEST",
            market=Market.US,
            futu_code="US.TEST",
            max_position_shares=1000
        )

        symbol.update_quote(last=50.0, bid=49.95, ask=50.05)

        # For $10000, at $50/share = 200 shares
        shares = symbol.max_shares_for_usd(10000.0)
        assert shares == 200

        # For $100000, should be capped at max_position_shares
        shares = symbol.max_shares_for_usd(100000.0)
        assert shares == 1000
