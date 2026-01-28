"""
Unit tests for monitoring module
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestTradingMetrics:
    """Test TradingMetrics class"""

    def test_metrics_creation(self):
        """Test TradingMetrics creation"""
        from src.monitor.metrics import TradingMetrics

        metrics = TradingMetrics(
            portfolio_value=100000.0,
            daily_pnl=500.0,
            sharpe_ratio=2.5,
            max_drawdown=0.05,
            win_rate=0.6,
            fill_rate=0.98
        )

        assert metrics.portfolio_value == 100000.0
        assert metrics.daily_pnl == 500.0
        assert metrics.sharpe_ratio == 2.5


class TestMetricsExporter:
    """Test MetricsExporter class"""

    def test_exporter_creation(self, mock_settings):
        """Test MetricsExporter creation"""
        with patch('src.monitor.metrics.get_settings', return_value=mock_settings):
            from src.monitor.metrics import MetricsExporter
            from prometheus_client import CollectorRegistry

            # Use separate registry to avoid conflicts
            registry = CollectorRegistry()
            exporter = MetricsExporter(port=8000, registry=registry)

            assert exporter.port == 8000

    def test_portfolio_update(self, mock_settings):
        """Test portfolio metrics update"""
        with patch('src.monitor.metrics.get_settings', return_value=mock_settings):
            from src.monitor.metrics import MetricsExporter, TradingMetrics
            from prometheus_client import CollectorRegistry

            registry = CollectorRegistry()
            exporter = MetricsExporter(port=8001, registry=registry)

            metrics = TradingMetrics(
                portfolio_value=100000.0,
                daily_pnl=500.0,
                total_pnl=1500.0,
                daily_trades=10,
                daily_volume=50000.0,
                open_positions=2
            )

            # Should not raise
            exporter.update_portfolio(metrics)

    def test_trade_recording(self, mock_settings):
        """Test trade recording"""
        with patch('src.monitor.metrics.get_settings', return_value=mock_settings):
            from src.monitor.metrics import MetricsExporter
            from prometheus_client import CollectorRegistry

            registry = CollectorRegistry()
            exporter = MetricsExporter(port=8002, registry=registry)

            # Should not raise
            exporter.record_trade(
                symbol="US.TQQQ",
                action="long",
                status="filled",
                latency_ms=1.5,
                slippage_pct=0.1
            )

    def test_connection_status(self, mock_settings):
        """Test connection status update"""
        with patch('src.monitor.metrics.get_settings', return_value=mock_settings):
            from src.monitor.metrics import MetricsExporter
            from prometheus_client import CollectorRegistry

            registry = CollectorRegistry()
            exporter = MetricsExporter(port=8003, registry=registry)

            exporter.update_connection_status("quote", True)
            exporter.update_connection_status("trade", False)


class TestFeishuMessage:
    """Test FeishuMessage class"""

    def test_message_creation(self):
        """Test FeishuMessage creation"""
        from src.monitor.alerts import FeishuMessage, AlertSeverity

        message = FeishuMessage(
            title="Test Alert",
            content="This is a test message",
            severity=AlertSeverity.WARNING,
            tags=["test", "alert"]
        )

        assert message.title == "Test Alert"
        assert message.severity == AlertSeverity.WARNING

    def test_message_to_card(self):
        """Test message to Feishu card conversion"""
        from src.monitor.alerts import FeishuMessage, AlertSeverity

        message = FeishuMessage(
            title="Test Alert",
            content="This is a test",
            severity=AlertSeverity.ERROR
        )

        card = message.to_card()

        assert card["msg_type"] == "interactive"
        assert "card" in card
        assert "header" in card["card"]

    def test_message_to_text(self):
        """Test message to text conversion"""
        from src.monitor.alerts import FeishuMessage, AlertSeverity

        message = FeishuMessage(
            title="Test Alert",
            content="This is a test"
        )

        text = message.to_text()

        assert text["msg_type"] == "text"
        assert "Test Alert" in text["content"]["text"]


class TestFeishuAlert:
    """Test FeishuAlert class"""

    def test_alert_creation(self):
        """Test FeishuAlert creation"""
        from src.monitor.alerts import FeishuAlert

        alert = FeishuAlert(
            webhook_url="https://example.com/webhook",
            cooldown_seconds=300
        )

        assert alert.webhook_url == "https://example.com/webhook"
        assert alert.cooldown_seconds == 300

    def test_rate_limiting(self):
        """Test alert rate limiting"""
        from src.monitor.alerts import FeishuAlert, FeishuMessage, AlertSeverity

        alert = FeishuAlert(
            webhook_url="https://example.com/webhook",
            cooldown_seconds=300,
            max_alerts_per_hour=2
        )

        message = FeishuMessage(
            title="Test",
            content="Test",
            severity=AlertSeverity.INFO
        )

        # First should pass
        assert alert._check_rate_limit(message) is True

        # Record sends
        alert._record_sent(message)
        alert._record_sent(message)

        # Third should be rate limited (hourly limit)
        assert alert._check_rate_limit(message) is False

    def test_deduplication(self):
        """Test alert deduplication"""
        from src.monitor.alerts import FeishuAlert, FeishuMessage

        alert = FeishuAlert(
            webhook_url="https://example.com/webhook"
        )

        message = FeishuMessage(
            title="Same Alert",
            content="Same content"
        )

        # First should pass
        assert alert._is_duplicate(message) is False

        # Second identical should be duplicate
        assert alert._is_duplicate(message) is True


class TestAlertManager:
    """Test AlertManager class"""

    def test_manager_creation(self, mock_settings):
        """Test AlertManager creation"""
        with patch('src.monitor.alerts.get_settings', return_value=mock_settings):
            from src.monitor.alerts import AlertManager

            manager = AlertManager(
                webhook_url="https://example.com/webhook",
                cooldown_minutes=5
            )

            assert manager.feishu.cooldown_seconds == 300

    def test_risk_alert_processing(self, mock_settings):
        """Test risk alert processing"""
        with patch('src.monitor.alerts.get_settings', return_value=mock_settings):
            from src.monitor.alerts import AlertManager
            from src.risk.risk_manager import RiskAlert, AlertType, RiskLevel

            manager = AlertManager(webhook_url=None)  # No actual webhook

            alert = RiskAlert(
                alert_type=AlertType.DRAWDOWN_WARNING,
                level=RiskLevel.HIGH,
                symbol="US.TQQQ",
                message="Test alert",
                value=-0.05,
                threshold=-0.03
            )

            # Should not raise even without webhook
            manager.process_risk_alert(alert)

    def test_anomaly_tracking(self, mock_settings):
        """Test anomaly tracking"""
        with patch('src.monitor.alerts.get_settings', return_value=mock_settings):
            from src.monitor.alerts import AlertManager

            manager = AlertManager(webhook_url=None)

            # Record anomalies
            manager.record_anomaly("latency_spikes")
            manager.record_anomaly("latency_spikes")

            assert manager._anomaly_counts.get("latency_spikes") == 2


class TestAlertSeverity:
    """Test AlertSeverity enum"""

    def test_severity_values(self):
        """Test severity enum values"""
        from src.monitor.alerts import AlertSeverity

        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"
