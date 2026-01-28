"""
Prometheus Metrics Exporter for Trading System
Exports metrics for Grafana dashboards
"""
import time
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    start_http_server,
    REGISTRY,
    CollectorRegistry,
)

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradingMetrics:
    """Container for trading metrics values"""
    # Portfolio
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0

    # Performance
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    win_rate: float = 0.0

    # Trading activity
    total_trades: int = 0
    daily_trades: int = 0
    daily_volume: float = 0.0

    # Execution quality
    avg_latency_ms: float = 0.0
    fill_rate: float = 1.0
    avg_slippage: float = 0.0

    # Risk
    circuit_breaker_active: bool = False
    open_positions: int = 0


class MetricsExporter:
    """
    Prometheus metrics exporter for trading system.
    Provides real-time metrics for Grafana monitoring.
    """

    def __init__(self, port: int = None, registry: CollectorRegistry = None):
        settings = get_settings()
        self.port = port or settings.prometheus_port
        self._registry = registry or REGISTRY
        self._server_started = False

        # Initialize metrics
        self._init_metrics()

    def _init_metrics(self):
        """Initialize Prometheus metrics"""

        # =====================
        # Portfolio Metrics
        # =====================
        self.portfolio_value = Gauge(
            'trading_portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self._registry
        )

        self.daily_pnl = Gauge(
            'trading_daily_pnl_usd',
            'Daily profit/loss in USD',
            registry=self._registry
        )

        self.total_pnl = Gauge(
            'trading_total_pnl_usd',
            'Total profit/loss in USD',
            registry=self._registry
        )

        self.unrealized_pnl = Gauge(
            'trading_unrealized_pnl_usd',
            'Unrealized profit/loss in USD',
            ['symbol'],
            registry=self._registry
        )

        # =====================
        # Performance Metrics
        # =====================
        self.sharpe_ratio = Gauge(
            'trading_sharpe_ratio',
            'Annualized Sharpe ratio',
            registry=self._registry
        )

        self.max_drawdown = Gauge(
            'trading_max_drawdown_pct',
            'Maximum drawdown percentage',
            registry=self._registry
        )

        self.current_drawdown = Gauge(
            'trading_current_drawdown_pct',
            'Current drawdown percentage',
            registry=self._registry
        )

        self.daily_drawdown = Gauge(
            'trading_daily_drawdown_pct',
            'Daily drawdown percentage',
            registry=self._registry
        )

        self.win_rate = Gauge(
            'trading_win_rate_pct',
            'Win rate percentage',
            registry=self._registry
        )

        # =====================
        # Trading Activity
        # =====================
        self.trades_total = Counter(
            'trading_trades_total',
            'Total number of trades',
            ['symbol', 'action', 'status'],
            registry=self._registry
        )

        self.trades_daily = Gauge(
            'trading_trades_daily',
            'Number of trades today',
            registry=self._registry
        )

        self.volume_daily = Gauge(
            'trading_volume_daily_usd',
            'Daily trading volume in USD',
            registry=self._registry
        )

        self.position_size = Gauge(
            'trading_position_size',
            'Current position size in shares',
            ['symbol'],
            registry=self._registry
        )

        self.position_value = Gauge(
            'trading_position_value_usd',
            'Current position value in USD',
            ['symbol'],
            registry=self._registry
        )

        # =====================
        # Execution Quality
        # =====================
        self.order_latency = Histogram(
            'trading_order_latency_ms',
            'Order execution latency in milliseconds',
            ['symbol', 'action'],
            buckets=[0.5, 1, 1.4, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
            registry=self._registry
        )

        self.pipeline_latency = Histogram(
            'trading_pipeline_latency_ms',
            'Full pipeline latency (quote->model->order) in milliseconds',
            buckets=[100, 250, 500, 750, 1000, 1500, 2000, 5000],
            registry=self._registry
        )

        self.fill_rate = Gauge(
            'trading_fill_rate_pct',
            'Order fill rate percentage',
            registry=self._registry
        )

        self.slippage = Histogram(
            'trading_slippage_pct',
            'Order slippage percentage',
            ['symbol'],
            buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
            registry=self._registry
        )

        self.avg_slippage = Gauge(
            'trading_avg_slippage_pct',
            'Average slippage percentage',
            registry=self._registry
        )

        # =====================
        # Risk Metrics
        # =====================
        self.circuit_breaker_active = Gauge(
            'trading_circuit_breaker_active',
            'Circuit breaker active (1=active, 0=inactive)',
            registry=self._registry
        )

        self.open_positions = Gauge(
            'trading_open_positions',
            'Number of open positions',
            registry=self._registry
        )

        self.risk_alerts = Counter(
            'trading_risk_alerts_total',
            'Total number of risk alerts',
            ['alert_type', 'level'],
            registry=self._registry
        )

        # =====================
        # LLM Metrics
        # =====================
        self.llm_latency = Histogram(
            'trading_llm_latency_ms',
            'LLM inference latency in milliseconds',
            buckets=[100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000],
            registry=self._registry
        )

        self.llm_decisions = Counter(
            'trading_llm_decisions_total',
            'Total LLM trading decisions',
            ['action', 'confidence'],
            registry=self._registry
        )

        # =====================
        # System Metrics
        # =====================
        self.connection_status = Gauge(
            'trading_connection_status',
            'Connection status (1=connected, 0=disconnected)',
            ['service'],  # quote, trade, llm
            registry=self._registry
        )

        self.last_update = Gauge(
            'trading_last_update_timestamp',
            'Timestamp of last metrics update',
            registry=self._registry
        )

        self.system_info = Info(
            'trading_system',
            'Trading system information',
            registry=self._registry
        )

        # Set system info
        settings = get_settings()
        self.system_info.info({
            'version': '1.0.0',
            'environment': settings.futu_trade_env,
            'symbols': ','.join(settings.trading_symbols),
            'llm_provider': settings.llm_provider
        })

    def start_server(self):
        """Start the Prometheus HTTP server"""
        if not self._server_started:
            try:
                start_http_server(self.port, registry=self._registry)
                self._server_started = True
                logger.info(f"Prometheus metrics server started on port {self.port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")

    def update_portfolio(self, metrics: TradingMetrics):
        """Update portfolio-related metrics"""
        self.portfolio_value.set(metrics.portfolio_value)
        self.daily_pnl.set(metrics.daily_pnl)
        self.total_pnl.set(metrics.total_pnl)
        self.trades_daily.set(metrics.daily_trades)
        self.volume_daily.set(metrics.daily_volume)
        self.open_positions.set(metrics.open_positions)
        self.last_update.set(time.time())

    def update_performance(self, metrics: TradingMetrics):
        """Update performance metrics"""
        self.sharpe_ratio.set(metrics.sharpe_ratio)
        self.max_drawdown.set(metrics.max_drawdown * 100)
        self.current_drawdown.set(metrics.current_drawdown * 100)
        self.win_rate.set(metrics.win_rate * 100)
        self.fill_rate.set(metrics.fill_rate * 100)
        self.avg_slippage.set(metrics.avg_slippage * 100)

    def update_risk(self, metrics: TradingMetrics):
        """Update risk metrics"""
        self.circuit_breaker_active.set(1 if metrics.circuit_breaker_active else 0)

    def record_trade(
        self,
        symbol: str,
        action: str,
        status: str,
        latency_ms: float,
        slippage_pct: float = 0.0
    ):
        """Record a trade execution"""
        self.trades_total.labels(symbol=symbol, action=action, status=status).inc()
        self.order_latency.labels(symbol=symbol, action=action).observe(latency_ms)
        if slippage_pct > 0:
            self.slippage.labels(symbol=symbol).observe(slippage_pct)

    def record_pipeline_latency(self, latency_ms: float):
        """Record full pipeline latency"""
        self.pipeline_latency.observe(latency_ms)

    def record_llm_decision(
        self,
        action: str,
        confidence: str,
        latency_ms: float
    ):
        """Record LLM decision"""
        self.llm_decisions.labels(action=action, confidence=confidence).inc()
        self.llm_latency.observe(latency_ms)

    def record_risk_alert(self, alert_type: str, level: str):
        """Record a risk alert"""
        self.risk_alerts.labels(alert_type=alert_type, level=level).inc()

    def update_position(self, symbol: str, shares: int, value: float, unrealized_pnl: float):
        """Update position metrics for a symbol"""
        self.position_size.labels(symbol=symbol).set(shares)
        self.position_value.labels(symbol=symbol).set(value)
        self.unrealized_pnl.labels(symbol=symbol).set(unrealized_pnl)

    def update_connection_status(self, service: str, connected: bool):
        """Update connection status for a service"""
        self.connection_status.labels(service=service).set(1 if connected else 0)


# Singleton instance
_exporter: Optional[MetricsExporter] = None


def get_metrics_exporter() -> MetricsExporter:
    """Get the global metrics exporter instance"""
    global _exporter
    if _exporter is None:
        _exporter = MetricsExporter()
    return _exporter
