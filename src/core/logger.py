"""
Logging configuration for AI Futu Trader
Uses loguru for structured, rotated logging with console and file output
"""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = "logs/trading.log",
    rotation: str = "100 MB",
    retention: str = "7 days",
    enable_console: bool = True,
    enable_file: bool = True,
    json_format: bool = False
) -> None:
    """
    Configure the global logger with console and file handlers.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None to disable file logging)
        rotation: When to rotate log files (size or time)
        retention: How long to keep rotated log files
        enable_console: Whether to log to console
        enable_file: Whether to log to file
        json_format: Whether to use JSON format for structured logging
    """
    # Remove default handler
    logger.remove()

    # Console format with colors
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # File format (more detailed)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # JSON format for structured logging
    json_format_str = (
        '{{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
        '"level":"{level}",'
        '"module":"{name}",'
        '"function":"{function}",'
        '"line":{line},'
        '"message":"{message}"}}'
    )

    # Add console handler
    if enable_console:
        logger.add(
            sys.stderr,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )

    # Add file handler
    if enable_file and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=json_format_str if json_format else file_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="gz",
            backtrace=True,
            diagnose=True,
            enqueue=True  # Thread-safe async writing
        )

    logger.info(f"Logger initialized with level={log_level}")


def get_logger(name: str = None):
    """
    Get a logger instance with optional context binding.

    Args:
        name: Optional name to bind to the logger context

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class TradeLogger:
    """
    Specialized logger for trade-related events with structured data.
    Provides consistent logging format for orders, fills, and positions.
    """

    def __init__(self, symbol: str = None):
        self.symbol = symbol
        self._logger = logger.bind(component="trade", symbol=symbol)

    def order_placed(
        self,
        order_id: str,
        action: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
        **extra
    ):
        """Log order placement"""
        self._logger.info(
            f"ORDER_PLACED | {order_id} | {action} {quantity} @ {price} ({order_type})",
            order_id=order_id,
            action=action,
            quantity=quantity,
            price=price,
            order_type=order_type,
            **extra
        )

    def order_filled(
        self,
        order_id: str,
        fill_price: float,
        fill_qty: float,
        slippage: float,
        latency_ms: float,
        **extra
    ):
        """Log order fill"""
        slippage_pct = slippage * 100
        self._logger.info(
            f"ORDER_FILLED | {order_id} | {fill_qty} @ {fill_price} | "
            f"slippage={slippage_pct:.4f}% | latency={latency_ms:.2f}ms",
            order_id=order_id,
            fill_price=fill_price,
            fill_qty=fill_qty,
            slippage=slippage,
            latency_ms=latency_ms,
            **extra
        )

    def order_rejected(self, order_id: str, reason: str, **extra):
        """Log order rejection"""
        self._logger.warning(
            f"ORDER_REJECTED | {order_id} | {reason}",
            order_id=order_id,
            reason=reason,
            **extra
        )

    def order_cancelled(self, order_id: str, reason: str = None, **extra):
        """Log order cancellation"""
        msg = f"ORDER_CANCELLED | {order_id}"
        if reason:
            msg += f" | {reason}"
        self._logger.info(msg, order_id=order_id, reason=reason, **extra)

    def position_update(
        self,
        position: float,
        avg_price: float,
        unrealized_pnl: float,
        realized_pnl: float,
        **extra
    ):
        """Log position update"""
        self._logger.info(
            f"POSITION | {self.symbol} | qty={position} | avg={avg_price:.4f} | "
            f"unrealized_pnl={unrealized_pnl:.2f} | realized_pnl={realized_pnl:.2f}",
            position=position,
            avg_price=avg_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            **extra
        )

    def signal_generated(self, signal: str, confidence: float, reasoning: str = None, **extra):
        """Log trading signal from LLM"""
        self._logger.info(
            f"SIGNAL | {self.symbol} | {signal} | confidence={confidence:.2f}",
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            **extra
        )

    def risk_alert(self, alert_type: str, message: str, **extra):
        """Log risk management alert"""
        self._logger.warning(
            f"RISK_ALERT | {alert_type} | {message}",
            alert_type=alert_type,
            **extra
        )

    def circuit_breaker_triggered(self, reason: str, drawdown: float, **extra):
        """Log circuit breaker activation"""
        self._logger.error(
            f"CIRCUIT_BREAKER | {reason} | drawdown={drawdown:.2%}",
            reason=reason,
            drawdown=drawdown,
            **extra
        )


class PerformanceLogger:
    """
    Logger for performance metrics and latency tracking
    """

    def __init__(self):
        self._logger = logger.bind(component="performance")

    def latency_record(
        self,
        operation: str,
        latency_ms: float,
        success: bool = True,
        **extra
    ):
        """Record latency for an operation"""
        level = "info" if success else "warning"
        getattr(self._logger, level)(
            f"LATENCY | {operation} | {latency_ms:.3f}ms | success={success}",
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            **extra
        )

    def pipeline_timing(
        self,
        quote_ms: float,
        model_ms: float,
        order_ms: float,
        total_ms: float,
        **extra
    ):
        """Record full pipeline timing"""
        self._logger.info(
            f"PIPELINE | quote={quote_ms:.1f}ms | model={model_ms:.1f}ms | "
            f"order={order_ms:.1f}ms | total={total_ms:.1f}ms",
            quote_ms=quote_ms,
            model_ms=model_ms,
            order_ms=order_ms,
            total_ms=total_ms,
            **extra
        )

    def daily_stats(
        self,
        total_trades: int,
        win_rate: float,
        sharpe: float,
        max_drawdown: float,
        total_pnl: float,
        volume: float,
        fill_rate: float,
        avg_latency_ms: float,
        **extra
    ):
        """Log daily performance summary"""
        self._logger.info(
            f"DAILY_STATS | trades={total_trades} | win_rate={win_rate:.2%} | "
            f"sharpe={sharpe:.2f} | max_dd={max_drawdown:.2%} | pnl={total_pnl:.2f} | "
            f"volume={volume:.0f} | fill_rate={fill_rate:.2%} | avg_latency={avg_latency_ms:.2f}ms",
            total_trades=total_trades,
            win_rate=win_rate,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            total_pnl=total_pnl,
            volume=volume,
            fill_rate=fill_rate,
            avg_latency_ms=avg_latency_ms,
            **extra
        )
