"""
Risk Manager - Comprehensive risk management with circuit breakers
Implements:
- 3% daily drawdown circuit breaker
- Maximum 15% total drawdown limit
- Position sizing based on volatility
- Real-time Sharpe ratio tracking
"""
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Callable
from collections import deque
import math

from src.core.config import get_settings
from src.core.logger import get_logger, TradeLogger, PerformanceLogger
from src.action.futu_executor import Position, OrderResult

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of risk alerts"""
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_CRITICAL = "drawdown_critical"
    CIRCUIT_BREAKER = "circuit_breaker"
    POSITION_LIMIT = "position_limit"
    SLIPPAGE_HIGH = "slippage_high"
    LATENCY_HIGH = "latency_high"
    FILL_RATE_LOW = "fill_rate_low"
    SHARPE_LOW = "sharpe_low"


@dataclass
class RiskAlert:
    """Risk alert with details"""
    alert_type: AlertType
    level: RiskLevel
    symbol: str
    message: str
    value: float  # The metric value that triggered the alert
    threshold: float  # The threshold that was breached
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class RiskState:
    """Current risk state for a symbol or portfolio"""
    # P&L tracking
    starting_equity: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0

    # Daily tracking
    daily_starting_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_volume: float = 0.0

    # Drawdown
    current_drawdown: float = 0.0  # From peak
    daily_drawdown: float = 0.0  # Today's drawdown
    max_drawdown: float = 0.0

    # Performance metrics
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_slippage: float = 0.0
    fill_rate: float = 1.0

    # Circuit breaker state
    circuit_breaker_active: bool = False
    circuit_breaker_reason: str = ""
    circuit_breaker_until: Optional[datetime] = None

    # Last update
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed"""
        if self.circuit_breaker_active:
            if self.circuit_breaker_until and datetime.now() > self.circuit_breaker_until:
                return True
            return False
        return True

    @property
    def win_rate(self) -> float:
        """Calculate win rate"""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return self.winning_trades / total

    @property
    def risk_level(self) -> RiskLevel:
        """Calculate current risk level"""
        if self.circuit_breaker_active:
            return RiskLevel.CRITICAL
        if self.daily_drawdown >= 0.025 or self.current_drawdown >= 0.12:
            return RiskLevel.HIGH
        if self.daily_drawdown >= 0.015 or self.current_drawdown >= 0.08:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class CircuitBreaker:
    """
    Circuit breaker to halt trading when risk limits are breached.
    Implements 3% daily drawdown limit and 15% total drawdown limit.
    """

    def __init__(
        self,
        max_daily_drawdown: float = 0.03,
        max_total_drawdown: float = 0.15,
        cooldown_minutes: int = 30,
    ):
        self.max_daily_drawdown = max_daily_drawdown
        self.max_total_drawdown = max_total_drawdown
        self.cooldown_minutes = cooldown_minutes

        self._active = False
        self._reason = ""
        self._triggered_at: Optional[datetime] = None
        self._trade_logger = TradeLogger()

    def check(self, state: RiskState) -> Optional[RiskAlert]:
        """
        Check if circuit breaker should be triggered.

        Args:
            state: Current risk state

        Returns:
            RiskAlert if triggered, None otherwise
        """
        # Check daily drawdown
        if state.daily_drawdown >= self.max_daily_drawdown:
            self._trigger(
                reason=f"Daily drawdown limit breached: {state.daily_drawdown:.2%}",
                drawdown=state.daily_drawdown
            )
            return RiskAlert(
                alert_type=AlertType.CIRCUIT_BREAKER,
                level=RiskLevel.CRITICAL,
                symbol="PORTFOLIO",
                message=f"CIRCUIT BREAKER: Daily drawdown {state.daily_drawdown:.2%} >= {self.max_daily_drawdown:.2%}",
                value=state.daily_drawdown,
                threshold=self.max_daily_drawdown
            )

        # Check total drawdown
        if state.current_drawdown >= self.max_total_drawdown:
            self._trigger(
                reason=f"Total drawdown limit breached: {state.current_drawdown:.2%}",
                drawdown=state.current_drawdown
            )
            return RiskAlert(
                alert_type=AlertType.CIRCUIT_BREAKER,
                level=RiskLevel.CRITICAL,
                symbol="PORTFOLIO",
                message=f"CIRCUIT BREAKER: Total drawdown {state.current_drawdown:.2%} >= {self.max_total_drawdown:.2%}",
                value=state.current_drawdown,
                threshold=self.max_total_drawdown
            )

        return None

    def _trigger(self, reason: str, drawdown: float):
        """Trigger the circuit breaker"""
        self._active = True
        self._reason = reason
        self._triggered_at = datetime.now()

        self._trade_logger.circuit_breaker_triggered(reason, drawdown)
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")

    def reset(self):
        """Manually reset the circuit breaker"""
        if self._active:
            logger.info(f"Circuit breaker reset after {self._reason}")
        self._active = False
        self._reason = ""
        self._triggered_at = None

    @property
    def is_active(self) -> bool:
        """Check if circuit breaker is currently active"""
        if not self._active:
            return False

        # Auto-reset after cooldown
        if self._triggered_at:
            elapsed = (datetime.now() - self._triggered_at).total_seconds() / 60
            if elapsed >= self.cooldown_minutes:
                logger.info(f"Circuit breaker auto-reset after {self.cooldown_minutes} minutes cooldown")
                self.reset()
                return False

        return True

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def minutes_remaining(self) -> int:
        """Minutes remaining until auto-reset"""
        if not self._active or not self._triggered_at:
            return 0

        elapsed = (datetime.now() - self._triggered_at).total_seconds() / 60
        remaining = self.cooldown_minutes - elapsed
        return max(0, int(remaining))


class PositionSizer:
    """
    Position sizing based on volatility and risk budget.
    Implements Kelly criterion and volatility-adjusted sizing.
    """

    def __init__(
        self,
        default_risk_per_trade: float = 0.02,
        max_position_pct: float = 0.25,
        use_kelly: bool = True,
        kelly_fraction: float = 0.25,  # Quarter Kelly for safety
    ):
        self.default_risk_per_trade = default_risk_per_trade
        self.max_position_pct = max_position_pct
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(
        self,
        portfolio_value: float,
        current_price: float,
        atr: float,
        volatility_factor: float = 1.0,
        win_rate: float = 0.5,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
    ) -> int:
        """
        Calculate optimal position size in shares.

        Args:
            portfolio_value: Total portfolio value
            current_price: Current stock price
            atr: Average True Range for volatility
            volatility_factor: Multiplier for leveraged products
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return

        Returns:
            Optimal number of shares
        """
        if current_price <= 0 or portfolio_value <= 0:
            return 0

        # Base position size from risk budget
        risk_amount = portfolio_value * self.default_risk_per_trade

        # Volatility-adjusted position size
        if atr > 0:
            # Risk amount divided by ATR gives shares
            vol_adjusted_shares = risk_amount / (atr * volatility_factor * 2)
        else:
            vol_adjusted_shares = risk_amount / current_price

        # Kelly criterion adjustment
        if self.use_kelly and win_rate > 0 and avg_loss > 0:
            kelly_pct = self._calculate_kelly(win_rate, avg_win, avg_loss)
            kelly_amount = portfolio_value * kelly_pct * self.kelly_fraction
            kelly_shares = kelly_amount / current_price

            # Use minimum of volatility-adjusted and Kelly
            position_shares = min(vol_adjusted_shares, kelly_shares)
        else:
            position_shares = vol_adjusted_shares

        # Apply maximum position limit
        max_shares = (portfolio_value * self.max_position_pct) / current_price
        position_shares = min(position_shares, max_shares)

        # For leveraged ETFs, reduce position size
        if volatility_factor > 1:
            position_shares = position_shares / volatility_factor

        return max(1, int(position_shares))

    def _calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly criterion percentage.

        Kelly % = W - (1-W)/R
        where W = win probability, R = win/loss ratio
        """
        if avg_loss == 0:
            return self.max_position_pct

        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio

        # Clamp to reasonable range
        return max(0, min(kelly, self.max_position_pct))

    def adjust_for_correlation(
        self,
        position_size: int,
        correlated_exposure: float,
        max_correlated_exposure: float = 0.5
    ) -> int:
        """
        Reduce position size based on correlated exposure.

        Args:
            position_size: Calculated position size
            correlated_exposure: Current exposure to correlated assets
            max_correlated_exposure: Maximum allowed correlated exposure

        Returns:
            Adjusted position size
        """
        if correlated_exposure >= max_correlated_exposure:
            return 0

        adjustment = 1 - (correlated_exposure / max_correlated_exposure)
        return int(position_size * adjustment)


class RiskManager:
    """
    Comprehensive risk management system.
    Tracks P&L, drawdown, Sharpe ratio, and enforces risk limits.
    """

    def __init__(
        self,
        starting_equity: float = 100000.0,
        max_daily_drawdown: float = None,
        max_total_drawdown: float = None,
        min_sharpe_target: float = None,
        daily_volume_target: float = None,
        min_fill_rate: float = None,
    ):
        settings = get_settings()
        self.starting_equity = starting_equity
        self.max_daily_drawdown = max_daily_drawdown or settings.max_daily_drawdown
        self.max_total_drawdown = max_total_drawdown or settings.max_total_drawdown
        self.min_sharpe_target = min_sharpe_target or settings.min_sharpe_ratio
        self.daily_volume_target = daily_volume_target or settings.daily_volume_target
        self.min_fill_rate = min_fill_rate or settings.min_fill_rate

        # Core components
        self.circuit_breaker = CircuitBreaker(
            max_daily_drawdown=self.max_daily_drawdown,
            max_total_drawdown=self.max_total_drawdown
        )
        self.position_sizer = PositionSizer()

        # State tracking
        self._state = RiskState(
            starting_equity=starting_equity,
            current_equity=starting_equity,
            peak_equity=starting_equity,
            daily_starting_equity=starting_equity
        )

        # Returns history for Sharpe calculation
        self._daily_returns: deque = deque(maxlen=252)  # 1 year of trading days
        self._trade_returns: List[float] = []

        # Alert tracking
        self._alerts: List[RiskAlert] = []
        self._alert_callbacks: List[Callable[[RiskAlert], None]] = []

        # Slippage tracking
        self._slippage_samples: deque = deque(maxlen=100)

        self._trade_logger = TradeLogger()
        self._perf_logger = PerformanceLogger()

    def update_equity(self, current_equity: float):
        """Update current equity and calculate drawdown"""
        self._state.current_equity = current_equity
        self._state.last_update = datetime.now()

        # Update peak equity
        if current_equity > self._state.peak_equity:
            self._state.peak_equity = current_equity

        # Calculate drawdowns
        if self._state.peak_equity > 0:
            self._state.current_drawdown = (
                self._state.peak_equity - current_equity
            ) / self._state.peak_equity

        if self._state.daily_starting_equity > 0:
            self._state.daily_pnl = current_equity - self._state.daily_starting_equity
            self._state.daily_drawdown = max(0, -self._state.daily_pnl / self._state.daily_starting_equity)

        # Update max drawdown
        self._state.max_drawdown = max(self._state.max_drawdown, self._state.current_drawdown)

        # Check circuit breaker
        alert = self.circuit_breaker.check(self._state)
        if alert:
            self._raise_alert(alert)
            self._state.circuit_breaker_active = True
            self._state.circuit_breaker_reason = alert.message

    def record_trade(
        self,
        result: OrderResult,
        entry_price: float,
        exit_price: float = None,
        pnl: float = 0.0
    ):
        """Record a completed trade for statistics"""
        self._state.total_trades += 1
        self._state.daily_trades += 1

        if result.filled_qty > 0:
            self._state.daily_volume += result.filled_qty * result.avg_fill_price

        # Track slippage
        if result.slippage > 0:
            self._slippage_samples.append(result.slippage)
            self._state.avg_slippage = sum(self._slippage_samples) / len(self._slippage_samples)

        # Track P&L
        if exit_price is not None and exit_price > 0:
            trade_return = (exit_price - entry_price) / entry_price
            self._trade_returns.append(trade_return)

            if pnl > 0:
                self._state.winning_trades += 1
            else:
                self._state.losing_trades += 1

            self._state.realized_pnl += pnl

        # Check slippage alert
        if result.slippage > 0.002:  # > 0.2%
            self._raise_alert(RiskAlert(
                alert_type=AlertType.SLIPPAGE_HIGH,
                level=RiskLevel.MEDIUM,
                symbol=result.futu_code,
                message=f"High slippage: {result.slippage:.4%}",
                value=result.slippage,
                threshold=0.002
            ))

    def record_fill_rate(self, filled: int, requested: int):
        """Update fill rate statistics"""
        if requested > 0:
            rate = filled / requested
            # Exponential moving average
            alpha = 0.1
            self._state.fill_rate = alpha * rate + (1 - alpha) * self._state.fill_rate

            if self._state.fill_rate < self.min_fill_rate:
                self._raise_alert(RiskAlert(
                    alert_type=AlertType.FILL_RATE_LOW,
                    level=RiskLevel.MEDIUM,
                    symbol="PORTFOLIO",
                    message=f"Low fill rate: {self._state.fill_rate:.2%}",
                    value=self._state.fill_rate,
                    threshold=self.min_fill_rate
                ))

    def new_day(self):
        """Reset daily statistics for new trading day"""
        # Record previous day's return
        if self._state.daily_starting_equity > 0:
            daily_return = (
                self._state.current_equity - self._state.daily_starting_equity
            ) / self._state.daily_starting_equity
            self._daily_returns.append(daily_return)

        # Reset daily tracking
        self._state.daily_starting_equity = self._state.current_equity
        self._state.daily_pnl = 0.0
        self._state.daily_drawdown = 0.0
        self._state.daily_trades = 0
        self._state.daily_volume = 0.0

        # Reset circuit breaker for new day
        self.circuit_breaker.reset()
        self._state.circuit_breaker_active = False
        self._state.circuit_breaker_reason = ""

        # Recalculate Sharpe ratio
        self._calculate_sharpe()

        logger.info(f"New trading day started. Previous Sharpe: {self._state.sharpe_ratio:.2f}")

    def _calculate_sharpe(self):
        """Calculate Sharpe ratio from daily returns"""
        if len(self._daily_returns) < 20:
            return

        returns = list(self._daily_returns)
        avg_return = sum(returns) / len(returns)

        # Standard deviation
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0001

        # Annualized Sharpe (assuming 252 trading days, 0% risk-free rate)
        if std_dev > 0:
            self._state.sharpe_ratio = (avg_return / std_dev) * math.sqrt(252)

        # Check Sharpe target
        if self._state.sharpe_ratio < self.min_sharpe_target and len(self._daily_returns) >= 30:
            self._raise_alert(RiskAlert(
                alert_type=AlertType.SHARPE_LOW,
                level=RiskLevel.MEDIUM,
                symbol="PORTFOLIO",
                message=f"Sharpe ratio {self._state.sharpe_ratio:.2f} below target {self.min_sharpe_target}",
                value=self._state.sharpe_ratio,
                threshold=self.min_sharpe_target
            ))

    def _raise_alert(self, alert: RiskAlert):
        """Raise a risk alert"""
        self._alerts.append(alert)
        self._trade_logger.risk_alert(alert.alert_type.value, alert.message)

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def on_alert(self, callback: Callable[[RiskAlert], None]):
        """Register callback for risk alerts"""
        self._alert_callbacks.append(callback)

    def get_position_size(
        self,
        current_price: float,
        atr: float,
        volatility_factor: float = 1.0
    ) -> int:
        """Calculate position size using risk parameters"""
        return self.position_sizer.calculate_position_size(
            portfolio_value=self._state.current_equity,
            current_price=current_price,
            atr=atr,
            volatility_factor=volatility_factor,
            win_rate=self._state.win_rate,
            avg_win=self._avg_winning_trade(),
            avg_loss=self._avg_losing_trade()
        )

    def _avg_winning_trade(self) -> float:
        """Calculate average winning trade return"""
        winners = [r for r in self._trade_returns if r > 0]
        return sum(winners) / len(winners) if winners else 0.02

    def _avg_losing_trade(self) -> float:
        """Calculate average losing trade return"""
        losers = [abs(r) for r in self._trade_returns if r < 0]
        return sum(losers) / len(losers) if losers else 0.01

    def can_trade(self) -> tuple:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if self.circuit_breaker.is_active:
            return False, f"Circuit breaker active: {self.circuit_breaker.reason}"

        if self._state.current_drawdown >= self.max_total_drawdown * 0.9:
            return False, f"Approaching max drawdown: {self._state.current_drawdown:.2%}"

        return True, "OK"

    def check_position_risk(
        self,
        position: Position,
        current_price: float
    ) -> Optional[RiskAlert]:
        """Check if a position exceeds risk limits"""
        if position.is_flat:
            return None

        # Calculate position P&L percentage
        if position.avg_cost > 0:
            pnl_pct = (current_price - position.avg_cost) / position.avg_cost
            if position.is_short:
                pnl_pct = -pnl_pct

            # Check for excessive loss on single position
            if pnl_pct < -0.05:  # >5% loss on position
                return RiskAlert(
                    alert_type=AlertType.DRAWDOWN_WARNING,
                    level=RiskLevel.HIGH,
                    symbol=position.futu_code,
                    message=f"Position loss {pnl_pct:.2%} on {position.symbol}",
                    value=pnl_pct,
                    threshold=-0.05
                )

        return None

    @property
    def state(self) -> RiskState:
        """Get current risk state"""
        return self._state

    @property
    def is_trading_allowed(self) -> bool:
        """Quick check if trading is allowed"""
        return not self.circuit_breaker.is_active

    @property
    def recent_alerts(self) -> List[RiskAlert]:
        """Get recent alerts"""
        return self._alerts[-20:]

    def get_daily_stats(self) -> dict:
        """Get daily statistics summary"""
        return {
            "date": date.today().isoformat(),
            "starting_equity": self._state.daily_starting_equity,
            "current_equity": self._state.current_equity,
            "daily_pnl": self._state.daily_pnl,
            "daily_pnl_pct": self._state.daily_pnl / self._state.daily_starting_equity if self._state.daily_starting_equity > 0 else 0,
            "daily_trades": self._state.daily_trades,
            "daily_volume": self._state.daily_volume,
            "daily_drawdown": self._state.daily_drawdown,
            "current_drawdown": self._state.current_drawdown,
            "sharpe_ratio": self._state.sharpe_ratio,
            "win_rate": self._state.win_rate,
            "fill_rate": self._state.fill_rate,
            "avg_slippage": self._state.avg_slippage,
            "circuit_breaker_active": self._state.circuit_breaker_active
        }
