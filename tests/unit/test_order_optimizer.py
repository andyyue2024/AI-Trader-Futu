"""
Unit tests for order optimizer
"""
import pytest
import time
from unittest.mock import MagicMock, patch


class TestLatencyMetrics:
    """Test LatencyMetrics class"""

    def test_metrics_creation(self):
        """Test metrics creation"""
        from src.action.order_optimizer import LatencyMetrics

        metrics = LatencyMetrics(
            avg_order_latency_ms=1.2,
            p50_order_latency_ms=1.0,
            p95_order_latency_ms=1.5,
            p99_order_latency_ms=2.0
        )

        assert metrics.avg_order_latency_ms == 1.2
        assert metrics.p95_order_latency_ms == 1.5


class TestLatencyTracker:
    """Test LatencyTracker class"""

    def test_tracker_creation(self):
        """Test tracker creation"""
        from src.action.order_optimizer import LatencyTracker

        tracker = LatencyTracker(max_samples=100)
        assert tracker.max_samples == 100

    def test_record_latency(self):
        """Test recording latency"""
        from src.action.order_optimizer import LatencyTracker

        tracker = LatencyTracker()

        for i in range(10):
            tracker.record_order_latency(1.0 + i * 0.1)

        metrics = tracker.get_metrics()

        assert metrics.total_orders == 10
        assert metrics.avg_order_latency_ms > 1.0

    def test_is_meeting_target(self):
        """Test target check"""
        from src.action.order_optimizer import LatencyTracker

        tracker = LatencyTracker()

        # All under target
        for _ in range(20):
            tracker.record_order_latency(1.0)

        assert tracker.is_meeting_target(1.4) is True

        # Add some over target
        for _ in range(10):
            tracker.record_order_latency(2.0)

        # P95 might now be over target


class TestExecutionTimer:
    """Test ExecutionTimer class"""

    def test_timer_basic(self):
        """Test basic timer functionality"""
        from src.action.order_optimizer import ExecutionTimer

        timer = ExecutionTimer()
        timer.start()

        time.sleep(0.01)  # 10ms

        elapsed = timer.elapsed_ms()
        assert elapsed >= 10

    def test_timer_checkpoints(self):
        """Test timer checkpoints"""
        from src.action.order_optimizer import ExecutionTimer

        timer = ExecutionTimer()
        timer.start()

        time.sleep(0.005)
        timer.checkpoint("quote")

        time.sleep(0.005)
        timer.checkpoint("model")

        time.sleep(0.005)
        timer.checkpoint("order")

        breakdown = timer.get_breakdown()

        assert "quote" in breakdown
        assert "model" in breakdown
        assert "order" in breakdown


class TestOrderOptimizer:
    """Test OrderOptimizer class"""

    def test_optimizer_creation(self):
        """Test optimizer creation"""
        from src.action.order_optimizer import OrderOptimizer

        optimizer = OrderOptimizer(target_latency_ms=1.4)

        assert optimizer.target_latency_ms == 1.4

    def test_warm_up(self):
        """Test optimizer warm-up"""
        from src.action.order_optimizer import OrderOptimizer

        optimizer = OrderOptimizer()
        optimizer.warm_up()

        assert optimizer._warmed_up is True
        assert "US.TQQQ" in optimizer._order_templates

    def test_optimize_order_params(self):
        """Test order parameter optimization"""
        from src.action.order_optimizer import OrderOptimizer

        optimizer = OrderOptimizer()
        optimizer.warm_up()

        params = optimizer.optimize_order_params(
            symbol="US.TQQQ",
            side="BUY",
            quantity=100,
            price=50.0
        )

        assert params["code"] == "US.TQQQ"
        assert params["qty"] == 100
        assert params["price"] == 50.0
