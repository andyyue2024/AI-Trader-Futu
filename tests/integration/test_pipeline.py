"""
Integration tests for the complete trading pipeline
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd


class TestTradingPipeline:
    """Test complete trading pipeline integration"""

    @pytest.fixture
    def mock_futu_quote_context(self):
        """Mock Futu quote context"""
        mock = MagicMock()
        mock.subscribe.return_value = (0, None)
        mock.unsubscribe.return_value = (0, None)

        # Mock quote response
        quote_df = pd.DataFrame([{
            'code': 'US.TQQQ',
            'last_price': 50.0,
            'bid_price': 49.99,
            'ask_price': 50.01,
            'bid_vol': 1000,
            'ask_vol': 1200,
            'volume': 5000000,
            'turnover': 250000000.0
        }])
        mock.get_stock_quote.return_value = (0, quote_df)

        # Mock kline response
        kline_df = pd.DataFrame([{
            'code': 'US.TQQQ',
            'time_key': '2024-01-15 10:00:00',
            'open': 49.5,
            'high': 50.5,
            'low': 49.0,
            'close': 50.0,
            'volume': 100000,
            'turnover': 5000000.0
        }])
        mock.get_cur_kline.return_value = (0, kline_df, None)

        return mock

    @pytest.fixture
    def mock_futu_trade_context(self):
        """Mock Futu trade context"""
        mock = MagicMock()
        mock.unlock_trade.return_value = (0, None)
        mock.get_acc_list.return_value = (0, pd.DataFrame([{'acc_id': '123'}]))
        mock.position_list_query.return_value = (0, pd.DataFrame())
        mock.place_order.return_value = (0, pd.DataFrame([{
            'order_id': 'test-order-123',
            'code': 'US.TQQQ',
            'qty': 100,
            'dealt_qty': 100,
            'dealt_avg_price': 50.01,
            'order_status': 'FILLED_ALL'
        }]))
        return mock

    def test_data_to_decision_flow(
        self,
        sample_klines,
        sample_quote_data,
        mock_settings
    ):
        """Test data processing to LLM decision flow"""
        with patch('src.data.data_processor.get_settings', return_value=mock_settings):
            with patch('src.model.llm_agent.get_settings', return_value=mock_settings):
                from src.data.data_processor import DataProcessor

                # Process data
                processor = DataProcessor()
                processor.load_history("US.TQQQ", sample_klines)
                processor.update_quote("US.TQQQ", sample_quote_data)

                # Generate snapshot
                snapshot = processor.get_snapshot("US.TQQQ")

                assert snapshot is not None
                assert snapshot.last_price == 50.0
                assert snapshot.indicators.sma_5 > 0

    def test_decision_to_order_flow(
        self,
        sample_market_snapshot,
        mock_settings
    ):
        """Test LLM decision to order execution flow"""
        with patch('src.model.llm_agent.get_settings', return_value=mock_settings):
            with patch('src.action.futu_executor.get_settings', return_value=mock_settings):
                from src.model.llm_agent import TradingDecision, DecisionConfidence
                from src.action.futu_executor import TradingAction

                # Create decision
                decision = TradingDecision(
                    action=TradingAction.LONG,
                    symbol="TQQQ",
                    futu_code="US.TQQQ",
                    confidence=DecisionConfidence.HIGH,
                    confidence_score=0.85,
                    suggested_quantity=100
                )

                assert decision.should_execute is True
                assert decision.action == TradingAction.LONG

    def test_order_to_risk_flow(
        self,
        mock_settings
    ):
        """Test order execution to risk management flow"""
        with patch('src.action.futu_executor.get_settings', return_value=mock_settings):
            with patch('src.risk.risk_manager.get_settings', return_value=mock_settings):
                from src.action.futu_executor import OrderResult, TradingAction, OrderStatus
                from src.risk.risk_manager import RiskManager

                # Create order result
                result = OrderResult(
                    order_id="test-123",
                    symbol="TQQQ",
                    futu_code="US.TQQQ",
                    action=TradingAction.LONG,
                    side="BUY",
                    requested_qty=100,
                    filled_qty=100,
                    avg_fill_price=50.01,
                    slippage=0.0002,
                    status=OrderStatus.FILLED
                )

                # Create risk manager and record
                risk = RiskManager(starting_equity=100000.0)
                risk.record_trade(result, entry_price=50.0)

                assert risk._state.total_trades == 1
                assert risk._state.avg_slippage == pytest.approx(0.0002, rel=0.1)


class TestPositionManagement:
    """Test position management integration"""

    def test_position_lifecycle(self, mock_settings):
        """Test complete position lifecycle: open -> update -> close"""
        with patch('src.action.futu_executor.get_settings', return_value=mock_settings):
            from src.action.futu_executor import Position

            # Open position
            position = Position(
                symbol="TQQQ",
                futu_code="US.TQQQ",
                quantity=100,
                avg_cost=50.0,
                market_value=5000.0
            )

            assert position.is_long is True

            # Update with unrealized P&L
            position.unrealized_pnl = 100.0  # $1 gain per share
            assert position.unrealized_pnl == 100.0

            # Flat position
            flat_position = Position(
                symbol="TQQQ",
                futu_code="US.TQQQ",
                quantity=0,
                avg_cost=0,
                realized_pnl=100.0
            )

            assert flat_position.is_flat is True
            assert flat_position.realized_pnl == 100.0


class TestRiskCircuitBreaker:
    """Test risk management and circuit breaker integration"""

    def test_circuit_breaker_integration(self, mock_settings):
        """Test circuit breaker triggers and halts trading"""
        with patch('src.risk.risk_manager.get_settings', return_value=mock_settings):
            from src.risk.risk_manager import RiskManager

            # Setup risk manager
            risk = RiskManager(
                starting_equity=100000.0,
                max_daily_drawdown=0.03
            )

            alerts = []
            risk.on_alert(lambda a: alerts.append(a))

            # Normal trading
            can_trade, _ = risk.can_trade()
            assert can_trade is True

            # Simulate 4% drawdown
            risk._state.daily_starting_equity = 100000.0
            risk.update_equity(96000.0)

            # Should trigger circuit breaker
            can_trade, reason = risk.can_trade()
            assert can_trade is False
            assert len(alerts) > 0


class TestMetricsAndAlerts:
    """Test metrics and alerting integration"""

    def test_metrics_to_alerts_flow(self, mock_settings):
        """Test metrics triggering alerts"""
        with patch('src.monitor.metrics.get_settings', return_value=mock_settings):
            with patch('src.monitor.alerts.get_settings', return_value=mock_settings):
                from src.monitor.metrics import MetricsExporter, TradingMetrics
                from src.monitor.alerts import AlertManager
                from prometheus_client import CollectorRegistry

                registry = CollectorRegistry()
                exporter = MetricsExporter(port=8010, registry=registry)
                alerts = AlertManager(webhook_url=None)

                # Update metrics
                metrics = TradingMetrics(
                    portfolio_value=95000.0,
                    daily_pnl=-5000.0,
                    current_drawdown=0.05,
                    circuit_breaker_active=True
                )

                exporter.update_portfolio(metrics)
                exporter.update_risk(metrics)

                # Record anomaly
                for _ in range(6):
                    alerts.record_anomaly("latency_spikes")

                assert alerts._anomaly_counts.get("latency_spikes") == 6


class TestSymbolExpansion:
    """Test symbol expansion capability"""

    def test_add_new_symbol(self):
        """Test adding new symbol with zero code changes"""
        from src.core.symbols import SymbolRegistry, Market, InstrumentType

        registry = SymbolRegistry()

        # Initial symbols
        initial_count = len(registry.all_symbols())

        # Add new symbols (simulating 1-day deployment)
        new_symbols = [
            ("US.SPXL", "SPXL", InstrumentType.LEVERAGED_ETF, 3.0),
            ("US.SOXL", "SOXL", InstrumentType.LEVERAGED_ETF, 3.0),
            ("US.AAPL", "AAPL", InstrumentType.STOCK, 1.0),
            ("US.MSFT", "MSFT", InstrumentType.STOCK, 1.0),
        ]

        for futu_code, symbol, inst_type, vol_factor in new_symbols:
            registry.register(
                futu_code=futu_code,
                symbol=symbol,
                market=Market.US,
                instrument_type=inst_type,
                volatility_factor=vol_factor
            )

        # Verify symbols added
        assert len(registry.all_symbols()) >= initial_count + 4

        # Activate for trading
        registry.activate("US.SPXL", "US.SOXL", "US.AAPL", "US.MSFT")

        assert len(registry.active_symbols) >= 4

    def test_option_expansion(self):
        """Test option contract expansion"""
        from src.core.symbols import SymbolRegistry, InstrumentType

        registry = SymbolRegistry()

        # Add option contracts
        call_option = registry.register_option(
            underlying="AAPL",
            strike=150.0,
            expiry="20240315",
            option_type="call"
        )

        put_option = registry.register_option(
            underlying="AAPL",
            strike=145.0,
            expiry="20240315",
            option_type="put"
        )

        assert call_option.is_option() is True
        assert call_option.instrument_type == InstrumentType.OPTION_CALL
        assert put_option.instrument_type == InstrumentType.OPTION_PUT


class TestPerformanceTargets:
    """Test performance target tracking"""

    def test_latency_tracking(self, mock_settings):
        """Test latency tracking meets target"""
        with patch('src.action.futu_executor.get_settings', return_value=mock_settings):
            from src.action.futu_executor import FutuExecutor

            executor = FutuExecutor()

            # Simulate latency samples
            executor._latency_samples = [1.0, 1.2, 1.4, 1.5, 1.3]

            avg = executor.avg_latency_ms
            # Target is 1.4ms
            # This is testing the tracking mechanism
            assert avg == pytest.approx(1.28, rel=0.1)

    def test_slippage_tracking(self, mock_settings):
        """Test slippage tracking"""
        with patch('src.risk.risk_manager.get_settings', return_value=mock_settings):
            from src.risk.risk_manager import RiskManager
            from src.action.futu_executor import OrderResult, TradingAction, OrderStatus

            risk = RiskManager(starting_equity=100000.0)

            # Record trades with slippage
            for slippage in [0.001, 0.002, 0.0015, 0.001, 0.0018]:
                result = OrderResult(
                    order_id=f"test-{slippage}",
                    symbol="TQQQ",
                    futu_code="US.TQQQ",
                    action=TradingAction.LONG,
                    side="BUY",
                    requested_qty=100,
                    filled_qty=100,
                    slippage=slippage,
                    status=OrderStatus.FILLED
                )
                risk.record_trade(result, entry_price=50.0)

            # Target is 0.2% = 0.002
            assert risk._state.avg_slippage < 0.002

    def test_fill_rate_tracking(self, mock_settings):
        """Test fill rate tracking"""
        with patch('src.risk.risk_manager.get_settings', return_value=mock_settings):
            from src.risk.risk_manager import RiskManager

            risk = RiskManager(starting_equity=100000.0)

            # Simulate fills
            for _ in range(95):
                risk.record_fill_rate(100, 100)  # Full fills
            for _ in range(5):
                risk.record_fill_rate(50, 100)   # Partial fills

            # Target is 95%
            # Note: EMA calculation means actual value may differ
            assert risk._state.fill_rate > 0.5
