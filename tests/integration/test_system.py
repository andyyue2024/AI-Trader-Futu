"""
Integration tests for the complete system
End-to-end testing of all components
"""
import pytest
import asyncio
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile
import os


class TestSystemIntegration:
    """Test complete system integration"""

    def test_configuration_loading(self):
        """Test configuration loading from environment"""
        with patch.dict(os.environ, {
            'FUTU_HOST': '127.0.0.1',
            'FUTU_PORT': '11111',
            'LLM_PROVIDER': 'openai',
            'TRADING_SYMBOLS': 'US.TQQQ,US.QQQ'
        }):
            from src.core.config import Settings

            # Force reload
            settings = Settings()

            assert settings.futu_host == '127.0.0.1'
            assert settings.futu_port == 11111
            assert 'US.TQQQ' in settings.trading_symbols

    def test_symbol_registry_integration(self):
        """Test symbol registry with all modules"""
        from src.core.symbols import SymbolRegistry, Market, InstrumentType

        registry = SymbolRegistry()

        # Register various symbols
        registry.register(
            symbol="TQQQ",
            market=Market.US,
            instrument_type=InstrumentType.STOCK
        )

        # Register option
        option = registry.register_option(
            underlying="AAPL",
            strike=150.0,
            expiry="20240315",
            option_type="call"
        )

        # Activate
        registry.activate("US.TQQQ")
        registry.activate(option.futu_code)

        assert len(registry.active_symbols) == 2

    def test_session_manager_integration(self):
        """Test session manager with real time"""
        from src.core.session_manager import SessionManager, MarketSession

        manager = SessionManager()

        # Get session info
        info = manager.get_session_info()

        assert info.session in [
            MarketSession.CLOSED,
            MarketSession.PRE_MARKET,
            MarketSession.REGULAR,
            MarketSession.AFTER_HOURS
        ]
        assert info.progress_pct >= 0
        assert info.progress_pct <= 100

    def test_statistics_calculation(self):
        """Test trading statistics calculation"""
        from src.core.statistics import TradingStatistics

        stats = TradingStatistics(starting_equity=100000.0)

        # Record trades
        for i in range(20):
            entry_price = 50.0 + (i % 3) * 0.5
            exit_price = entry_price + (1.0 if i % 2 == 0 else -0.5)

            stats.record_entry(
                trade_id=f"test-{i}",
                symbol="TQQQ",
                futu_code="US.TQQQ",
                side="long",
                quantity=100,
                entry_price=entry_price
            )
            stats.record_exit("US.TQQQ", exit_price)

        metrics = stats.calculate_metrics()

        assert metrics.total_trades == 20
        assert metrics.win_rate >= 0.0
        assert metrics.win_rate <= 1.0

    def test_position_manager_integration(self):
        """Test position manager with risk controls"""
        from src.action.position_manager import PositionManager

        manager = PositionManager(starting_cash=100000.0)

        # Open position with risk
        position = manager.open_position(
            futu_code="US.TQQQ",
            quantity=100,
            price=50.0,
            order_id="test-001",
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )

        assert position.quantity == 100
        assert position.stop_loss_price is not None
        assert position.take_profit_price is not None

        # Check risk levels
        triggers = manager.check_risk_levels({"US.TQQQ": 48.0})
        assert len(triggers) == 1
        assert triggers[0][1] == "stop_loss"

    def test_order_optimizer_integration(self):
        """Test order optimizer with latency tracking"""
        from src.action.order_optimizer import OrderOptimizer, ExecutionTimer
        import time

        optimizer = OrderOptimizer(target_latency_ms=1.4)
        optimizer.warm_up()

        # Simulate order executions
        for _ in range(50):
            timer = ExecutionTimer()
            timer.start()

            # Simulate work
            time.sleep(0.0005)  # 0.5ms
            timer.checkpoint("process")

            time.sleep(0.0005)  # 0.5ms
            timer.checkpoint("execute")

            optimizer.latency_tracker.record_order_latency(timer.elapsed_ms())

        metrics = optimizer.get_latency_metrics()

        assert metrics.total_orders == 50
        assert metrics.avg_order_latency_ms > 0


class TestDatabaseIntegration:
    """Test database integration"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    def test_trade_persistence(self, temp_db):
        """Test trade record persistence"""
        from src.data.persistence import TradeDatabase, TradeRecord

        db = TradeDatabase(temp_db)

        # Insert trades
        for i in range(10):
            record = TradeRecord(
                trade_id=f"trade-{i}",
                symbol="TQQQ",
                futu_code="US.TQQQ",
                entry_time=datetime.now(),
                entry_price=50.0 + i * 0.1,
                entry_side="long",
                quantity=100,
                entry_order_id=f"order-{i}",
                pnl=10.0 if i % 2 == 0 else -5.0,
                status="closed"
            )
            db.insert_trade(record)

        # Query trades
        trades = db.get_recent_trades(20)
        assert len(trades) == 10

        # Get stats
        stats = db.get_trading_stats(30)
        assert stats["total_trades"] == 10

        db.close()

    def test_daily_performance_tracking(self, temp_db):
        """Test daily performance tracking"""
        from src.data.persistence import TradeDatabase, DailyPerformanceRecord

        db = TradeDatabase(temp_db)

        # Insert daily records
        for i in range(5):
            record = DailyPerformanceRecord(
                date=date.today() - timedelta(days=i),
                starting_equity=100000.0 + i * 1000,
                ending_equity=100000.0 + (i + 1) * 1000,
                realized_pnl=1000.0,
                total_trades=10,
                winning_trades=6
            )
            db.insert_daily_performance(record)

        # Query
        records = db.get_daily_performance(
            date.today() - timedelta(days=10),
            date.today()
        )

        assert len(records) == 5

        db.close()


class TestMonitoringIntegration:
    """Test monitoring integration"""

    def test_error_tracking(self):
        """Test error tracking integration"""
        from src.monitor.performance import ErrorTracker

        tracker = ErrorTracker()

        # Record various errors
        errors = [
            ValueError("Invalid value"),
            TypeError("Type mismatch"),
            RuntimeError("Runtime issue"),
        ]

        for error in errors:
            tracker.record_error(error, module="test")

        # Check summary
        summary = tracker.get_error_summary()
        assert len(summary) == 3

        # Check rate
        rate = tracker.get_error_rate(60)
        assert rate > 0

    def test_performance_monitoring(self):
        """Test performance monitor"""
        from src.monitor.performance import PerformanceMonitor

        monitor = PerformanceMonitor(sample_interval_seconds=1)

        # Get metrics
        metrics = monitor.get_current_metrics()

        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "timestamp" in metrics

        # Check health
        health = monitor.get_health_status()

        assert health["status"] in ["healthy", "warning", "critical"]

    def test_function_profiling(self):
        """Test function profiling"""
        from src.monitor.performance import FunctionProfiler
        import time

        profiler = FunctionProfiler()

        @profiler.profile("test_function")
        def slow_function():
            time.sleep(0.01)
            return 42

        # Call multiple times
        for _ in range(10):
            result = slow_function()
            assert result == 42

        stats = profiler.get_stats("test_function")

        assert stats["count"] == 10
        assert stats["avg_ms"] >= 10


class TestReportIntegration:
    """Test report generation integration"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_html_report_generation(self, temp_dir):
        """Test HTML report generation"""
        from src.report import ReportGenerator, ReportConfig

        config = ReportConfig(output_dir=temp_dir)
        generator = ReportGenerator(config)

        filepath = generator.generate_html(
            date.today() - timedelta(days=7),
            date.today()
        )

        assert os.path.exists(filepath)

        with open(filepath) as f:
            content = f.read()

        assert "AI Futu Trader" in content
        assert "html" in filepath


class TestBacktestIntegration:
    """Test backtest integration"""

    def test_complete_backtest(self):
        """Test complete backtest workflow"""
        from src.backtest import BacktestEngine, BacktestConfig, create_simple_strategy
        import pandas as pd
        import numpy as np

        # Generate data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1min')

        base_price = 50.0
        prices = [base_price]
        for _ in range(499):
            prices.append(prices[-1] * (1 + np.random.normal(0.0001, 0.005)))
        prices = np.array(prices)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(100000, 500000, 500)
        })

        # Create engine
        config = BacktestConfig(starting_capital=100000.0)
        engine = BacktestEngine(config)
        engine.load_data("US.TQQQ", data)

        # Run backtest
        strategy = create_simple_strategy()
        result = engine.run(strategy)

        # Verify results
        assert result.metrics is not None
        assert len(result.equity_curve) > 0
