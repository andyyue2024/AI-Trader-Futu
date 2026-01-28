"""
Unit tests for enhanced Feishu alerting
"""
import pytest
from datetime import datetime


class TestAlertPriority:
    """Test AlertPriority enum"""

    def test_priority_values(self):
        """Test priority values and ordering"""
        from src.monitor.feishu_enhanced import AlertPriority

        assert AlertPriority.P0_CRITICAL.value == 0
        assert AlertPriority.P1_HIGH.value == 1
        assert AlertPriority.P2_MEDIUM.value == 2
        assert AlertPriority.P3_LOW.value == 3


class TestAlertCategory:
    """Test AlertCategory enum"""

    def test_category_values(self):
        """Test category values"""
        from src.monitor.feishu_enhanced import AlertCategory

        assert AlertCategory.CIRCUIT_BREAKER.value == "circuit_breaker"
        assert AlertCategory.RISK.value == "risk"
        assert AlertCategory.DAILY_REPORT.value == "daily_report"


class TestAlert:
    """Test Alert class"""

    def test_alert_creation(self):
        """Test alert creation"""
        from src.monitor.feishu_enhanced import Alert, AlertPriority, AlertCategory

        alert = Alert(
            title="Test Alert",
            content="Test content",
            priority=AlertPriority.P2_MEDIUM,
            category=AlertCategory.SYSTEM
        )

        assert alert.title == "Test Alert"
        assert alert.priority == AlertPriority.P2_MEDIUM
        assert len(alert.alert_id) > 0

    def test_alert_with_values(self):
        """Test alert with value/threshold"""
        from src.monitor.feishu_enhanced import Alert, AlertPriority, AlertCategory

        alert = Alert(
            title="Drawdown Warning",
            content="Portfolio drawdown exceeded threshold",
            priority=AlertPriority.P1_HIGH,
            category=AlertCategory.RISK,
            symbol="US.TQQQ",
            value=0.035,
            threshold=0.03
        )

        assert alert.symbol == "US.TQQQ"
        assert alert.value == 0.035
        assert alert.threshold == 0.03


class TestAlertConfig:
    """Test AlertConfig class"""

    def test_config_defaults(self):
        """Test config default values"""
        from src.monitor.feishu_enhanced import AlertConfig

        config = AlertConfig(webhook_url="https://example.com/webhook")

        assert config.cooldown_seconds == 300
        assert config.max_alerts_per_hour == 20
        assert config.p0_always_alert is True


class TestFeishuCardBuilder:
    """Test FeishuCardBuilder class"""

    def test_build_alert_card(self):
        """Test building alert card"""
        from src.monitor.feishu_enhanced import (
            FeishuCardBuilder, Alert, AlertPriority, AlertCategory
        )

        builder = FeishuCardBuilder()

        alert = Alert(
            title="Test Alert",
            content="Test content",
            priority=AlertPriority.P1_HIGH,
            category=AlertCategory.RISK
        )

        card = builder.build_alert_card(alert)

        assert card["msg_type"] == "interactive"
        assert "card" in card
        assert "header" in card["card"]
        assert card["card"]["header"]["template"] == "red"

    def test_build_daily_report_card(self):
        """Test building daily report card"""
        from src.monitor.feishu_enhanced import FeishuCardBuilder

        builder = FeishuCardBuilder()

        card = builder.build_daily_report_card(
            date="2024-01-15",
            pnl=1500.0,
            trades=25,
            win_rate=0.68,
            sharpe=2.3,
            max_drawdown=0.02,
            volume=75000.0,
            fill_rate=0.97,
            avg_latency=1.2
        )

        assert card["msg_type"] == "interactive"
        assert "card" in card
        assert card["card"]["header"]["template"] == "green"

    def test_build_daily_report_card_negative_pnl(self):
        """Test daily report card with negative P&L"""
        from src.monitor.feishu_enhanced import FeishuCardBuilder

        builder = FeishuCardBuilder()

        card = builder.build_daily_report_card(
            date="2024-01-15",
            pnl=-500.0,
            trades=25,
            win_rate=0.4,
            sharpe=0.5,
            max_drawdown=0.025,
            volume=50000.0,
            fill_rate=0.95,
            avg_latency=1.5
        )

        assert card["card"]["header"]["template"] == "red"

    def test_build_anomaly_report_card(self):
        """Test building anomaly report card"""
        from src.monitor.feishu_enhanced import FeishuCardBuilder

        builder = FeishuCardBuilder()

        anomalies = [
            {"type": "high_latency", "count": 10, "threshold": 5, "severity": "high"},
            {"type": "high_slippage", "count": 4, "threshold": 3, "severity": "medium"}
        ]

        card = builder.build_anomaly_report_card(anomalies)

        assert card["msg_type"] == "interactive"
        assert "2 个异常" in card["card"]["header"]["title"]["content"]


class TestEnhancedFeishuAlert:
    """Test EnhancedFeishuAlert class"""

    def test_alert_system_creation(self):
        """Test alert system creation"""
        from src.monitor.feishu_enhanced import (
            EnhancedFeishuAlert, AlertConfig
        )

        config = AlertConfig(webhook_url="https://example.com/webhook")
        alerter = EnhancedFeishuAlert(config)

        assert alerter.config == config

    def test_record_anomaly(self):
        """Test anomaly recording"""
        from src.monitor.feishu_enhanced import (
            EnhancedFeishuAlert, AlertConfig
        )

        config = AlertConfig(webhook_url="")
        alerter = EnhancedFeishuAlert(config)

        alerter.record_anomaly("high_latency")
        alerter.record_anomaly("high_latency")
        alerter.record_anomaly("high_slippage")

        assert alerter._anomaly_counts["high_latency"] == 2
        assert alerter._anomaly_counts["high_slippage"] == 1

    def test_quiet_hours_check(self):
        """Test quiet hours check"""
        from src.monitor.feishu_enhanced import (
            EnhancedFeishuAlert, AlertConfig, AlertPriority
        )

        config = AlertConfig(
            webhook_url="",
            quiet_hours_enabled=True,
            quiet_start_hour=22,
            quiet_end_hour=7
        )
        alerter = EnhancedFeishuAlert(config)

        # P0 should never be in quiet hours
        assert alerter._is_quiet_hours(AlertPriority.P0_CRITICAL) is False
