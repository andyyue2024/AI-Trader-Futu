"""
Enhanced Feishu Alerting - Rich alert content and smart notification
Implements 5-minute anomaly detection with detailed reporting
"""
import time
import hashlib
import hmac
import base64
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import threading
import json

import requests

from src.core.logger import get_logger

logger = get_logger(__name__)


class AlertPriority(Enum):
    """Alert priority levels"""
    P0_CRITICAL = 0  # Immediate action required
    P1_HIGH = 1      # Urgent attention needed
    P2_MEDIUM = 2    # Should be reviewed soon
    P3_LOW = 3       # Informational


class AlertCategory(Enum):
    """Alert categories"""
    CIRCUIT_BREAKER = "circuit_breaker"
    RISK = "risk"
    EXECUTION = "execution"
    CONNECTION = "connection"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    DAILY_REPORT = "daily_report"


@dataclass
class AlertConfig:
    """Alert configuration"""
    webhook_url: str
    secret: str = ""  # For signed webhooks

    # Rate limiting
    cooldown_seconds: int = 300  # 5 minutes
    max_alerts_per_hour: int = 20

    # Priority settings
    p0_always_alert: bool = True
    p1_cooldown_seconds: int = 60

    # Quiet hours
    quiet_hours_enabled: bool = False
    quiet_start_hour: int = 22  # 10 PM
    quiet_end_hour: int = 7    # 7 AM
    quiet_min_priority: AlertPriority = AlertPriority.P1_HIGH


@dataclass
class Alert:
    """Alert data structure"""
    title: str
    content: str
    priority: AlertPriority = AlertPriority.P2_MEDIUM
    category: AlertCategory = AlertCategory.SYSTEM

    # Additional data
    symbol: str = ""
    value: float = 0.0
    threshold: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    alert_id: str = field(default_factory=lambda: "")

    # Actions
    actions: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = hashlib.md5(
                f"{self.title}:{self.category.value}:{self.timestamp.timestamp()}".encode()
            ).hexdigest()[:8]


class FeishuCardBuilder:
    """
    Builds Feishu interactive card messages.
    Creates rich, actionable alert cards.
    """

    @staticmethod
    def build_alert_card(alert: Alert) -> dict:
        """Build alert card"""

        # Header color based on priority
        header_colors = {
            AlertPriority.P0_CRITICAL: "red",
            AlertPriority.P1_HIGH: "red",
            AlertPriority.P2_MEDIUM: "orange",
            AlertPriority.P3_LOW: "blue",
        }

        # Priority emoji
        priority_emoji = {
            AlertPriority.P0_CRITICAL: "🚨",
            AlertPriority.P1_HIGH: "❗",
            AlertPriority.P2_MEDIUM: "⚠️",
            AlertPriority.P3_LOW: "ℹ️",
        }

        # Category emoji
        category_emoji = {
            AlertCategory.CIRCUIT_BREAKER: "🔴",
            AlertCategory.RISK: "📉",
            AlertCategory.EXECUTION: "⚡",
            AlertCategory.CONNECTION: "🔌",
            AlertCategory.PERFORMANCE: "📊",
            AlertCategory.SYSTEM: "🖥️",
            AlertCategory.DAILY_REPORT: "📋",
        }

        elements = [
            # Main content
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": alert.content
                }
            },
        ]

        # Add value/threshold if present
        if alert.value != 0 or alert.threshold != 0:
            elements.append({
                "tag": "div",
                "fields": [
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**当前值**: {alert.value:.4f}"
                        }
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**阈值**: {alert.threshold:.4f}"
                        }
                    }
                ]
            })

        # Add symbol if present
        if alert.symbol:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**标的**: {alert.symbol}"
                }
            })

        # Separator
        elements.append({"tag": "hr"})

        # Footer
        elements.append({
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"⏰ {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | ID: {alert.alert_id} | {alert.category.value}"
                }
            ]
        })

        # Add action buttons if present
        if alert.actions:
            action_elements = []
            for action in alert.actions:
                action_elements.append({
                    "tag": "button",
                    "text": {
                        "tag": "plain_text",
                        "content": action.get("text", "Action")
                    },
                    "type": action.get("type", "default"),
                    "url": action.get("url", "")
                })

            elements.append({
                "tag": "action",
                "actions": action_elements
            })

        return {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"{priority_emoji.get(alert.priority, '📢')} {category_emoji.get(alert.category, '🔔')} {alert.title}"
                    },
                    "template": header_colors.get(alert.priority, "blue")
                },
                "elements": elements
            }
        }

    @staticmethod
    def build_daily_report_card(
        date: str,
        pnl: float,
        trades: int,
        win_rate: float,
        sharpe: float,
        max_drawdown: float,
        volume: float,
        fill_rate: float,
        avg_latency: float,
        top_trades: List[dict] = None
    ) -> dict:
        """Build daily report card"""

        pnl_emoji = "📈" if pnl >= 0 else "📉"
        pnl_color = "green" if pnl >= 0 else "red"

        elements = [
            # P&L Summary
            {
                "tag": "div",
                "fields": [
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"{pnl_emoji} **P&L**\n${pnl:+,.2f}"
                        }
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"📊 **交易次数**\n{trades}"
                        }
                    }
                ]
            },
            {
                "tag": "div",
                "fields": [
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"🎯 **胜率**\n{win_rate:.1%}"
                        }
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"📐 **夏普比率**\n{sharpe:.2f}"
                        }
                    }
                ]
            },
            {"tag": "hr"},
            # Risk Metrics
            {
                "tag": "div",
                "fields": [
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"📉 **最大回撤**\n{max_drawdown:.2%}"
                        }
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"💰 **成交额**\n${volume:,.2f}"
                        }
                    }
                ]
            },
            {"tag": "hr"},
            # Execution Quality
            {
                "tag": "div",
                "fields": [
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"✅ **成交率**\n{fill_rate:.1%}"
                        }
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"⏱️ **平均延迟**\n{avg_latency:.2f}ms"
                        }
                    }
                ]
            },
        ]

        # Add top trades if provided
        if top_trades:
            elements.append({"tag": "hr"})
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "**🏆 今日最佳交易**"
                }
            })

            for i, trade in enumerate(top_trades[:3], 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                elements.append({
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"{emoji} {trade.get('symbol', '')} | ${trade.get('pnl', 0):+,.2f}"
                    }
                })

        elements.append({"tag": "hr"})
        elements.append({
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"📅 {date} | AI Futu Trader Daily Report"
                }
            ]
        })

        return {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"📋 每日交易报告 - {date}"
                    },
                    "template": "green" if pnl >= 0 else "red"
                },
                "elements": elements
            }
        }

    @staticmethod
    def build_anomaly_report_card(
        anomalies: List[Dict],
        check_period_minutes: int = 5
    ) -> dict:
        """Build anomaly report card"""

        elements = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**过去 {check_period_minutes} 分钟检测到以下异常:**"
                }
            },
            {"tag": "hr"}
        ]

        for anomaly in anomalies:
            severity_emoji = {
                "critical": "🚨",
                "high": "❗",
                "medium": "⚠️",
                "low": "ℹ️"
            }.get(anomaly.get('severity', 'medium'), '⚠️')

            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"{severity_emoji} **{anomaly.get('type', 'Unknown')}**\n"
                               f"发生次数: {anomaly.get('count', 0)} | "
                               f"阈值: {anomaly.get('threshold', 0)}"
                }
            })

        elements.append({"tag": "hr"})
        elements.append({
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
            ]
        })

        return {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"⚠️ 异常检测报告 ({len(anomalies)} 个异常)"
                    },
                    "template": "orange"
                },
                "elements": elements
            }
        }


class EnhancedFeishuAlert:
    """
    Enhanced Feishu alert system with rich cards and smart notification.
    """

    def __init__(self, config: AlertConfig):
        self.config = config
        self._card_builder = FeishuCardBuilder()

        # Rate limiting
        self._last_alert_times: Dict[str, datetime] = {}
        self._hourly_counts: List[datetime] = []

        # Anomaly detection
        self._anomaly_counts: Dict[str, int] = {}
        self._last_anomaly_check: datetime = datetime.now()
        self._anomaly_thresholds = {
            "high_latency": 5,
            "high_slippage": 3,
            "order_failed": 2,
            "connection_error": 2,
            "risk_warning": 3,
        }

        # Lock for thread safety
        self._lock = threading.Lock()

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to Feishu"""
        if not self.config.webhook_url:
            logger.warning("Feishu webhook not configured")
            return False

        # Check rate limiting
        if not self._should_send(alert):
            logger.debug(f"Alert rate limited: {alert.title}")
            return False

        # Check quiet hours
        if self._is_quiet_hours(alert.priority):
            logger.debug(f"Alert suppressed during quiet hours: {alert.title}")
            return False

        try:
            # Build card
            payload = self._card_builder.build_alert_card(alert)

            # Sign if secret configured
            if self.config.secret:
                payload = self._sign_payload(payload)

            # Send
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    self._record_sent(alert)
                    logger.info(f"Alert sent: {alert.title}")
                    return True
                else:
                    logger.error(f"Feishu API error: {result}")
            else:
                logger.error(f"Feishu HTTP error: {response.status_code}")

            return False

        except Exception as e:
            logger.error(f"Failed to send Feishu alert: {e}")
            return False

    def send_daily_report(
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
        top_trades: List[dict] = None
    ) -> bool:
        """Send daily trading report"""
        if not self.config.webhook_url:
            return False

        try:
            payload = self._card_builder.build_daily_report_card(
                date, pnl, trades, win_rate, sharpe,
                max_drawdown, volume, fill_rate, avg_latency, top_trades
            )

            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            return response.status_code == 200 and response.json().get("code") == 0

        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")
            return False

    def record_anomaly(self, anomaly_type: str, severity: str = "medium"):
        """Record an anomaly event"""
        with self._lock:
            self._anomaly_counts[anomaly_type] = self._anomaly_counts.get(anomaly_type, 0) + 1

            # Check if we should send anomaly report
            now = datetime.now()
            if (now - self._last_anomaly_check).total_seconds() >= 300:  # 5 minutes
                self._check_and_send_anomalies()
                self._last_anomaly_check = now

    def _check_and_send_anomalies(self):
        """Check anomaly counts and send report if needed"""
        anomalies = []

        for anomaly_type, count in self._anomaly_counts.items():
            threshold = self._anomaly_thresholds.get(anomaly_type, 5)
            if count >= threshold:
                anomalies.append({
                    "type": anomaly_type,
                    "count": count,
                    "threshold": threshold,
                    "severity": "high" if count >= threshold * 2 else "medium"
                })

        if anomalies:
            self._send_anomaly_report(anomalies)

        # Reset counts
        self._anomaly_counts.clear()

    def _send_anomaly_report(self, anomalies: List[Dict]):
        """Send anomaly report"""
        if not self.config.webhook_url:
            return

        try:
            payload = self._card_builder.build_anomaly_report_card(anomalies)

            requests.post(
                self.config.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            logger.info(f"Anomaly report sent: {len(anomalies)} anomalies")

        except Exception as e:
            logger.error(f"Failed to send anomaly report: {e}")

    def _should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent based on rate limiting"""
        now = datetime.now()

        # P0 always sends
        if alert.priority == AlertPriority.P0_CRITICAL and self.config.p0_always_alert:
            return True

        # Check hourly limit
        with self._lock:
            cutoff = now - timedelta(hours=1)
            self._hourly_counts = [t for t in self._hourly_counts if t > cutoff]

            if len(self._hourly_counts) >= self.config.max_alerts_per_hour:
                return False

        # Check per-type cooldown
        cooldown = (
            self.config.p1_cooldown_seconds
            if alert.priority == AlertPriority.P1_HIGH
            else self.config.cooldown_seconds
        )

        alert_key = f"{alert.category.value}:{alert.title}"
        if alert_key in self._last_alert_times:
            elapsed = (now - self._last_alert_times[alert_key]).total_seconds()
            if elapsed < cooldown:
                return False

        return True

    def _record_sent(self, alert: Alert):
        """Record that an alert was sent"""
        with self._lock:
            self._hourly_counts.append(datetime.now())
            alert_key = f"{alert.category.value}:{alert.title}"
            self._last_alert_times[alert_key] = datetime.now()

    def _is_quiet_hours(self, priority: AlertPriority) -> bool:
        """Check if currently in quiet hours"""
        if not self.config.quiet_hours_enabled:
            return False

        if priority.value <= self.config.quiet_min_priority.value:
            return False

        hour = datetime.now().hour

        if self.config.quiet_start_hour > self.config.quiet_end_hour:
            # Crosses midnight
            return hour >= self.config.quiet_start_hour or hour < self.config.quiet_end_hour
        else:
            return self.config.quiet_start_hour <= hour < self.config.quiet_end_hour

    def _sign_payload(self, payload: dict) -> dict:
        """Sign payload for secure webhook"""
        timestamp = str(int(time.time()))
        string_to_sign = f"{timestamp}\n{self.config.secret}"

        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()

        sign = base64.b64encode(hmac_code).decode('utf-8')

        payload["timestamp"] = timestamp
        payload["sign"] = sign

        return payload
