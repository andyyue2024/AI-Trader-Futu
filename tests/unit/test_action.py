"""
Unit tests for action/execution module
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestOrderResult:
    """Test OrderResult class"""

    def test_order_result_creation(self):
        """Test OrderResult creation"""
        from src.action.futu_executor import OrderResult, TradingAction, OrderStatus

        result = OrderResult(
            order_id="test-123",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            action=TradingAction.LONG,
            side="BUY",
            requested_qty=100,
            filled_qty=100,
            avg_fill_price=50.00,
            status=OrderStatus.FILLED
        )

        assert result.order_id == "test-123"
        assert result.is_filled is True
        assert result.fill_rate == 1.0

    def test_order_partial_fill(self):
        """Test partial fill detection"""
        from src.action.futu_executor import OrderResult, TradingAction, OrderStatus

        result = OrderResult(
            order_id="test-123",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            action=TradingAction.LONG,
            side="BUY",
            requested_qty=100,
            filled_qty=50,
            status=OrderStatus.PARTIAL_FILLED
        )

        assert result.is_partial is True
        assert result.fill_rate == 0.5

    def test_slippage_calculation(self):
        """Test slippage calculation"""
        from src.action.futu_executor import OrderResult, TradingAction, OrderStatus

        result = OrderResult(
            order_id="test-123",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            action=TradingAction.LONG,
            side="BUY",
            requested_qty=100,
            filled_qty=100,
            requested_price=50.00,
            avg_fill_price=50.10,
            status=OrderStatus.FILLED
        )

        result.calculate_slippage(50.00)

        # Slippage = |50.10 - 50.00| / 50.00 = 0.002 = 0.2%
        assert result.slippage == pytest.approx(0.002, rel=0.01)
        assert result.slippage_pct == pytest.approx(0.2, rel=0.01)


class TestPosition:
    """Test Position class"""

    def test_position_long(self):
        """Test long position properties"""
        from src.action.futu_executor import Position

        position = Position(
            symbol="TQQQ",
            futu_code="US.TQQQ",
            quantity=100,
            avg_cost=50.00
        )

        assert position.is_long is True
        assert position.is_short is False
        assert position.is_flat is False
        assert position.abs_quantity == 100

    def test_position_short(self):
        """Test short position properties"""
        from src.action.futu_executor import Position

        position = Position(
            symbol="TQQQ",
            futu_code="US.TQQQ",
            quantity=-100,
            avg_cost=50.00
        )

        assert position.is_long is False
        assert position.is_short is True
        assert position.is_flat is False
        assert position.abs_quantity == 100

    def test_position_flat(self):
        """Test flat position properties"""
        from src.action.futu_executor import Position

        position = Position(
            symbol="TQQQ",
            futu_code="US.TQQQ",
            quantity=0,
            avg_cost=0
        )

        assert position.is_long is False
        assert position.is_short is False
        assert position.is_flat is True


class TestFutuExecutor:
    """Test FutuExecutor class"""

    def test_executor_creation(self, mock_settings):
        """Test executor creation with settings"""
        with patch('src.action.futu_executor.get_settings', return_value=mock_settings):
            from src.action.futu_executor import FutuExecutor

            executor = FutuExecutor()

            assert executor.host == "127.0.0.1"
            assert executor.port == 11111

    def test_long_order_without_connection(self, mock_settings):
        """Test long order fails gracefully without connection"""
        with patch('src.action.futu_executor.get_settings', return_value=mock_settings):
            from src.action.futu_executor import FutuExecutor, OrderStatus

            executor = FutuExecutor()
            # Don't connect

            result = executor.long("US.TQQQ", 100)

            assert result.status == OrderStatus.FAILED
            assert "Not connected" in result.error_message

    def test_short_order_without_connection(self, mock_settings):
        """Test short order fails gracefully without connection"""
        with patch('src.action.futu_executor.get_settings', return_value=mock_settings):
            from src.action.futu_executor import FutuExecutor, OrderStatus

            executor = FutuExecutor()

            result = executor.short("US.TQQQ", 100)

            assert result.status == OrderStatus.FAILED

    def test_get_position_empty(self, mock_settings):
        """Test getting position when no position exists"""
        with patch('src.action.futu_executor.get_settings', return_value=mock_settings):
            from src.action.futu_executor import FutuExecutor

            executor = FutuExecutor()

            position = executor.get_position("US.TQQQ")

            assert position.is_flat is True
            assert position.symbol == "TQQQ"

    def test_latency_tracking(self, mock_settings):
        """Test latency tracking"""
        with patch('src.action.futu_executor.get_settings', return_value=mock_settings):
            from src.action.futu_executor import FutuExecutor

            executor = FutuExecutor()
            executor._latency_samples = [1.0, 2.0, 3.0]

            assert executor.avg_latency_ms == 2.0

    def test_fill_rate_tracking(self, mock_settings):
        """Test fill rate tracking"""
        with patch('src.action.futu_executor.get_settings', return_value=mock_settings):
            from src.action.futu_executor import FutuExecutor

            executor = FutuExecutor()
            executor._fill_count = 95
            executor._total_orders = 100

            assert executor.fill_rate == 0.95


class TestTradingAction:
    """Test TradingAction enum"""

    def test_action_values(self):
        """Test action enum values"""
        from src.action.futu_executor import TradingAction

        assert TradingAction.LONG.value == "long"
        assert TradingAction.SHORT.value == "short"
        assert TradingAction.FLAT.value == "flat"
        assert TradingAction.HOLD.value == "hold"


class TestOrderStatus:
    """Test OrderStatus enum"""

    def test_status_values(self):
        """Test order status values"""
        from src.action.futu_executor import OrderStatus

        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
