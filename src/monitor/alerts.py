"""
Alert Manager - Feishu (Lark) webhook integration for real-time alerts
Implements 5-minute anomaly detection and notification
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib

import requests
import aiohttp

from src.core.config import get_settings
from src.core.logger import get_logger
from src.risk.risk_manager import RiskAlert, RiskLevel, AlertType

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels for Feishu"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class FeishuMessage:
    """Feishu message structure"""
    title: str
    content: str
    severity: AlertSeverity = AlertSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    def to_card(self) -> dict:
        """Convert to Feishu interactive card format"""
        # Color based on severity
        colors = {
            AlertSeverity.INFO: "blue",
            AlertSeverity.WARNING: "orange",
            AlertSeverity.ERROR: "red",
            AlertSeverity.CRITICAL: "red"
        }

        header_template = {
            AlertSeverity.INFO: "blue",
            AlertSeverity.WARNING: "orange",
            AlertSeverity.ERROR: "red",
            AlertSeverity.CRITICAL: "red"
        }

        return {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"🤖 {self.title}"
                    },
                    "template": header_template[self.severity]
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": self.content
                        }
                    },
                    {
                        "tag": "hr"
                    },
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": f"⏰ {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Tags: {', '.join(self.tags) if self.tags else 'None'}"
                            }
                        ]
                    }
                ]
            }
        }

    def to_text(self) -> dict:
        """Convert to simple text format"""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }

        return {
            "msg_type": "text",
            "content": {
                "text": f"{severity_emoji[self.severity]} [{self.severity.value.upper()}] {self.title}\n\n{self.content}\n\n⏰ {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            }
        }


class FeishuAlert:
    """
    Feishu (Lark) webhook alert sender.
    Supports rate limiting and deduplication.
    """

    def __init__(
        self,
        webhook_url: str = None,
        cooldown_seconds: int = 300,  # 5 minutes
        max_alerts_per_hour: int = 20,
    ):
        settings = get_settings()
        self.webhook_url = webhook_url or settings.feishu_webhook_url
        self.cooldown_seconds = cooldown_seconds
        self.max_alerts_per_hour = max_alerts_per_hour

        # Rate limiting
        self._last_alert_times: Dict[str, datetime] = {}
        self._hourly_counts: List[datetime] = []

        # Deduplication
        self._recent_hashes: Dict[str, datetime] = {}
        self._hash_ttl = timedelta(minutes=30)

    def send(self, message: FeishuMessage, use_card: bool = True) -> bool:
        """
        Send alert to Feishu synchronously.

        Args:
            message: FeishuMessage to send
            use_card: Whether to use interactive card format

        Returns:
            True if sent successfully
        """
        if not self.webhook_url:
            logger.warning("Feishu webhook URL not configured")
            return False

        # Check rate limiting
        if not self._check_rate_limit(message):
            logger.debug(f"Alert rate limited: {message.title}")
            return False

        # Check deduplication
        if self._is_duplicate(message):
            logger.debug(f"Alert deduplicated: {message.title}")
            return False

        try:
            payload = message.to_card() if use_card else message.to_text()

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    logger.info(f"Feishu alert sent: {message.title}")
                    self._record_sent(message)
                    return True
                else:
                    logger.error(f"Feishu API error: {result}")
            else:
                logger.error(f"Feishu HTTP error: {response.status_code}")

            return False

        except Exception as e:
            logger.error(f"Failed to send Feishu alert: {e}")
            return False

    async def send_async(self, message: FeishuMessage, use_card: bool = True) -> bool:
        """Send alert to Feishu asynchronously"""
        if not self.webhook_url:
            return False

        if not self._check_rate_limit(message):
            return False

        if self._is_duplicate(message):
            return False

        try:
            payload = message.to_card() if use_card else message.to_text()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("code") == 0:
                            logger.info(f"Feishu alert sent: {message.title}")
                            self._record_sent(message)
                            return True

            return False

        except Exception as e:
            logger.error(f"Failed to send Feishu alert async: {e}")
            return False

    def _check_rate_limit(self, message: FeishuMessage) -> bool:
        """Check if alert passes rate limiting"""
        now = datetime.now()

        # Clean old hourly counts
        cutoff = now - timedelta(hours=1)
        self._hourly_counts = [t for t in self._hourly_counts if t > cutoff]

        # Check hourly limit
        if len(self._hourly_counts) >= self.max_alerts_per_hour:
            return False

        # Check per-type cooldown
        alert_key = f"{message.severity.value}:{message.title}"
        if alert_key in self._last_alert_times:
            elapsed = (now - self._last_alert_times[alert_key]).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False

        return True

    def _is_duplicate(self, message: FeishuMessage) -> bool:
        """Check if message is a duplicate"""
        now = datetime.now()

        # Clean old hashes
        self._recent_hashes = {
            h: t for h, t in self._recent_hashes.items()
            if now - t < self._hash_ttl
        }

        # Generate hash
        content_hash = hashlib.md5(
            f"{message.title}:{message.content}".encode()
        ).hexdigest()

        if content_hash in self._recent_hashes:
            return True

        self._recent_hashes[content_hash] = now
        return False

    def _record_sent(self, message: FeishuMessage):
        """Record that an alert was sent"""
        now = datetime.now()
        self._hourly_counts.append(now)

        alert_key = f"{message.severity.value}:{message.title}"
        self._last_alert_times[alert_key] = now


class AlertManager:
    """
    Central alert manager for trading system.
    Converts risk alerts to Feishu notifications with 5-minute anomaly detection.
    """

    def __init__(
        self,
        webhook_url: str = None,
        cooldown_minutes: int = 5,
    ):
        settings = get_settings()
        self.feishu = FeishuAlert(
            webhook_url=webhook_url or settings.feishu_webhook_url,
            cooldown_seconds=cooldown_minutes * 60
        )

        # Anomaly detection state
        self._anomaly_counts: Dict[str, int] = {}
        self._last_check: datetime = datetime.now()
        self._check_interval = timedelta(minutes=5)

        # Thresholds for anomaly detection
        self._thresholds = {
            "latency_spikes": 5,      # More than 5 high latency events
            "slippage_events": 3,     # More than 3 high slippage events
            "failed_orders": 2,       # More than 2 failed orders
            "risk_alerts": 3,         # More than 3 risk alerts
        }

    def process_risk_alert(self, alert: RiskAlert):
        """Process a risk alert and send to Feishu if needed"""
        # Map risk level to severity
        severity_map = {
            RiskLevel.LOW: AlertSeverity.INFO,
            RiskLevel.MEDIUM: AlertSeverity.WARNING,
            RiskLevel.HIGH: AlertSeverity.ERROR,
            RiskLevel.CRITICAL: AlertSeverity.CRITICAL
        }

        severity = severity_map.get(alert.level, AlertSeverity.WARNING)

        # Always send critical alerts immediately
        if alert.level == RiskLevel.CRITICAL:
            self._send_alert(
                title=f"🚨 {alert.alert_type.value.upper()}",
                content=self._format_risk_alert(alert),
                severity=severity,
                tags=["risk", alert.symbol, alert.alert_type.value]
            )
            return

        # Track for anomaly detection
        self._anomaly_counts["risk_alerts"] = self._anomaly_counts.get("risk_alerts", 0) + 1

        # Check if we should send
        if alert.level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            self._send_alert(
                title=f"Risk Alert: {alert.alert_type.value}",
                content=self._format_risk_alert(alert),
                severity=severity,
                tags=["risk", alert.symbol]
            )

    def record_anomaly(self, anomaly_type: str, value: float = 1.0):
        """Record an anomaly event for 5-minute detection"""
        self._anomaly_counts[anomaly_type] = self._anomaly_counts.get(anomaly_type, 0) + 1

        # Check if we should run anomaly detection
        now = datetime.now()
        if now - self._last_check >= self._check_interval:
            self._check_anomalies()
            self._last_check = now

    def _check_anomalies(self):
        """Check for anomalies in the past 5 minutes"""
        alerts_to_send = []

        for anomaly_type, count in self._anomaly_counts.items():
            threshold = self._thresholds.get(anomaly_type, 5)
            if count >= threshold:
                alerts_to_send.append({
                    "type": anomaly_type,
                    "count": count,
                    "threshold": threshold
                })

        if alerts_to_send:
            content = "**Anomalies detected in the last 5 minutes:**\n\n"
            for anomaly in alerts_to_send:
                content += f"• **{anomaly['type']}**: {anomaly['count']} events (threshold: {anomaly['threshold']})\n"

            self._send_alert(
                title="⚠️ Trading Anomaly Detected",
                content=content,
                severity=AlertSeverity.WARNING,
                tags=["anomaly", "5min-check"]
            )

        # Reset counts
        self._anomaly_counts.clear()

    def send_daily_summary(
        self,
        date: str,
        pnl: float,
        trades: int,
        win_rate: float,
        sharpe: float,
        max_drawdown: float,
        volume: float,
        fill_rate: float,
        avg_latency: float,
    ):
        """Send daily trading summary"""
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        content = f"""
**📊 Daily Trading Summary - {date}**

{pnl_emoji} **P&L**: ${pnl:,.2f}
📈 **Trades**: {trades}
🎯 **Win Rate**: {win_rate:.1%}
📐 **Sharpe Ratio**: {sharpe:.2f}
📉 **Max Drawdown**: {max_drawdown:.2%}
💰 **Volume**: ${volume:,.2f}
✅ **Fill Rate**: {fill_rate:.1%}
⏱️ **Avg Latency**: {avg_latency:.2f}ms

---
*Automated report from AI Trading System*
"""

        self._send_alert(
            title="📊 Daily Trading Summary",
            content=content,
            severity=AlertSeverity.INFO,
            tags=["daily-summary", date]
        )

    def send_circuit_breaker_alert(self, reason: str, drawdown: float):
        """Send circuit breaker activation alert"""
        content = f"""
**🚨 CIRCUIT BREAKER ACTIVATED**

**Reason**: {reason}
**Current Drawdown**: {drawdown:.2%}

⚠️ All trading has been halted. Manual intervention may be required.

The circuit breaker will automatically reset after the cooldown period, or you can manually reset it.
"""

        self._send_alert(
            title="🚨 CIRCUIT BREAKER ACTIVATED",
            content=content,
            severity=AlertSeverity.CRITICAL,
            tags=["circuit-breaker", "critical"]
        )

    def send_connection_alert(self, service: str, connected: bool, error: str = None):
        """Send connection status alert"""
        if connected:
            content = f"✅ **{service}** connection restored"
            severity = AlertSeverity.INFO
        else:
            content = f"❌ **{service}** connection lost"
            if error:
                content += f"\n\nError: {error}"
            severity = AlertSeverity.ERROR

        self._send_alert(
            title=f"Connection {'Restored' if connected else 'Lost'}: {service}",
            content=content,
            severity=severity,
            tags=["connection", service.lower()]
        )

    def _format_risk_alert(self, alert: RiskAlert) -> str:
        """Format risk alert for Feishu message"""
        return f"""
**Type**: {alert.alert_type.value}
**Symbol**: {alert.symbol}
**Level**: {alert.level.value.upper()}

**Details**: {alert.message}

**Value**: {alert.value:.4f}
**Threshold**: {alert.threshold:.4f}
**Time**: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""

    def _send_alert(
        self,
        title: str,
        content: str,
        severity: AlertSeverity,
        tags: List[str]
    ):
        """Internal method to send alert"""
        message = FeishuMessage(
            title=title,
            content=content,
            severity=severity,
            tags=tags
        )
        self.feishu.send(message)


# Singleton instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
