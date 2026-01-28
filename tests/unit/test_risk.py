"""
Unit tests for risk management module
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


class TestRiskState:
    """Test RiskState class"""

    def test_risk_state_creation(self):
        """Test RiskState creation"""
        from src.risk.risk_manager import RiskState

        state = RiskState(
            starting_equity=100000.0,
            current_equity=100000.0,
            peak_equity=100000.0
        )

        assert state.starting_equity == 100000.0
        assert state.is_trading_allowed is True

    def test_win_rate_calculation(self):
        """Test win rate calculation"""
        from src.risk.risk_manager import RiskState

        state = RiskState(
            winning_trades=60,
            losing_trades=40
        )

        assert state.win_rate == 0.6

    def test_risk_level_classification(self):
        """Test risk level classification"""
        from src.risk.risk_manager import RiskState, RiskLevel

        # Low risk
        low_risk = RiskState(daily_drawdown=0.01, current_drawdown=0.05)
        assert low_risk.risk_level == RiskLevel.LOW

        # Medium risk
        medium_risk = RiskState(daily_drawdown=0.02, current_drawdown=0.09)
        assert medium_risk.risk_level == RiskLevel.MEDIUM

        # High risk
        high_risk = RiskState(daily_drawdown=0.025, current_drawdown=0.13)
        assert high_risk.risk_level == RiskLevel.HIGH

        # Critical (circuit breaker active)
        critical = RiskState(circuit_breaker_active=True)
        assert critical.risk_level == RiskLevel.CRITICAL


class TestCircuitBreaker:
    """Test CircuitBreaker class"""

    def test_circuit_breaker_daily_drawdown(self):
        """Test circuit breaker triggers on daily drawdown"""
        from src.risk.risk_manager import CircuitBreaker, RiskState

        breaker = CircuitBreaker(max_daily_drawdown=0.03)

        # Below threshold - should not trigger
        state = RiskState(daily_drawdown=0.02)
        alert = breaker.check(state)
        assert alert is None
        assert breaker.is_active is False

        # At threshold - should trigger
        state = RiskState(daily_drawdown=0.03)
        alert = breaker.check(state)
        assert alert is not None
        assert breaker.is_active is True

    def test_circuit_breaker_total_drawdown(self):
        """Test circuit breaker triggers on total drawdown"""
        from src.risk.risk_manager import CircuitBreaker, RiskState

        breaker = CircuitBreaker(max_total_drawdown=0.15)

        # Below threshold
        state = RiskState(current_drawdown=0.10)
        alert = breaker.check(state)
        assert alert is None

        # At threshold
        state = RiskState(current_drawdown=0.15)
        alert = breaker.check(state)
        assert alert is not None

    def test_circuit_breaker_reset(self):
        """Test circuit breaker manual reset"""
        from src.risk.risk_manager import CircuitBreaker, RiskState

        breaker = CircuitBreaker(max_daily_drawdown=0.03)

        # Trigger breaker
        state = RiskState(daily_drawdown=0.05)
        breaker.check(state)
        assert breaker.is_active is True

        # Reset
        breaker.reset()
        assert breaker.is_active is False

    def test_circuit_breaker_cooldown(self):
        """Test circuit breaker auto-reset after cooldown"""
        from src.risk.risk_manager import CircuitBreaker, RiskState

        breaker = CircuitBreaker(
            max_daily_drawdown=0.03,
            cooldown_minutes=1  # Short cooldown for testing
        )

        # Trigger breaker
        state = RiskState(daily_drawdown=0.05)
        breaker.check(state)

        # Manually set triggered time to past
        breaker._triggered_at = datetime.now() - timedelta(minutes=2)

        # Should auto-reset
        assert breaker.is_active is False


class TestPositionSizer:
    """Test PositionSizer class"""

    def test_basic_position_sizing(self):
        """Test basic position size calculation"""
        from src.risk.risk_manager import PositionSizer

        sizer = PositionSizer(
            default_risk_per_trade=0.02,
            max_position_pct=0.25
        )

        shares = sizer.calculate_position_size(
            portfolio_value=100000.0,
            current_price=50.0,
            atr=1.0,
            volatility_factor=1.0
        )

        assert shares > 0
        # With 2% risk on $100k = $2000, at $50/share = max ~40 shares
        # but depends on ATR calculation
        assert shares <= 500  # Should be reasonable

    def test_volatility_adjusted_sizing(self):
        """Test volatility-adjusted position sizing"""
        from src.risk.risk_manager import PositionSizer

        sizer = PositionSizer()

        # Higher ATR = smaller position
        low_vol_shares = sizer.calculate_position_size(
            portfolio_value=100000.0,
            current_price=50.0,
            atr=0.5,
            volatility_factor=1.0
        )

        high_vol_shares = sizer.calculate_position_size(
            portfolio_value=100000.0,
            current_price=50.0,
            atr=2.0,
            volatility_factor=1.0
        )

        assert low_vol_shares > high_vol_shares

    def test_leveraged_etf_sizing(self):
        """Test position sizing for leveraged ETFs"""
        from src.risk.risk_manager import PositionSizer

        sizer = PositionSizer()

        # Standard ETF
        standard_shares = sizer.calculate_position_size(
            portfolio_value=100000.0,
            current_price=50.0,
            atr=1.0,
            volatility_factor=1.0
        )

        # 3x leveraged ETF
        leveraged_shares = sizer.calculate_position_size(
            portfolio_value=100000.0,
            current_price=50.0,
            atr=1.0,
            volatility_factor=3.0
        )

        # Leveraged should have smaller position
        assert leveraged_shares < standard_shares

    def test_kelly_criterion(self):
        """Test Kelly criterion calculation"""
        from src.risk.risk_manager import PositionSizer

        sizer = PositionSizer(use_kelly=True, kelly_fraction=1.0)

        # Favorable edge: 60% win rate, 2:1 reward/risk
        kelly = sizer._calculate_kelly(
            win_rate=0.6,
            avg_win=0.04,
            avg_loss=0.02
        )

        # Kelly = 0.6 - (0.4 / 2) = 0.6 - 0.2 = 0.4
        assert kelly == pytest.approx(0.4, rel=0.1)


class TestRiskManager:
    """Test RiskManager class"""

    def test_risk_manager_creation(self, mock_settings):
        """Test RiskManager creation"""
        with patch('src.risk.risk_manager.get_settings', return_value=mock_settings):
            from src.risk.risk_manager import RiskManager

            manager = RiskManager(starting_equity=100000.0)

            assert manager._state.starting_equity == 100000.0
            assert manager._state.current_equity == 100000.0

    def test_equity_update(self, mock_settings):
        """Test equity update and drawdown calculation"""
        with patch('src.risk.risk_manager.get_settings', return_value=mock_settings):
            from src.risk.risk_manager import RiskManager

            manager = RiskManager(starting_equity=100000.0)

            # Update equity
            manager.update_equity(98000.0)

            assert manager._state.current_equity == 98000.0
            # Drawdown = (100000 - 98000) / 100000 = 0.02
            assert manager._state.current_drawdown == pytest.approx(0.02, rel=0.01)

    def test_trade_recording(self, mock_settings):
        """Test trade recording"""
        with patch('src.risk.risk_manager.get_settings', return_value=mock_settings):
            from src.risk.risk_manager import RiskManager
            from src.action.futu_executor import OrderResult, TradingAction, OrderStatus

            manager = RiskManager(starting_equity=100000.0)

            result = OrderResult(
                order_id="test-123",
                symbol="TQQQ",
                futu_code="US.TQQQ",
                action=TradingAction.LONG,
                side="BUY",
                requested_qty=100,
                filled_qty=100,
                avg_fill_price=50.0,
                slippage=0.001,
                status=OrderStatus.FILLED
            )

            manager.record_trade(result, entry_price=50.0)

            assert manager._state.total_trades == 1
            assert manager._state.daily_trades == 1

    def test_can_trade(self, mock_settings):
        """Test trading permission check"""
        with patch('src.risk.risk_manager.get_settings', return_value=mock_settings):
            from src.risk.risk_manager import RiskManager

            manager = RiskManager(starting_equity=100000.0)

            # Initially should be able to trade
            can, reason = manager.can_trade()
            assert can is True

            # After circuit breaker triggers
            manager.circuit_breaker._active = True
            can, reason = manager.can_trade()
            assert can is False

    def test_new_day_reset(self, mock_settings):
        """Test new day reset"""
        with patch('src.risk.risk_manager.get_settings', return_value=mock_settings):
            from src.risk.risk_manager import RiskManager

            manager = RiskManager(starting_equity=100000.0)

            # Set some daily stats
            manager._state.daily_pnl = 500.0
            manager._state.daily_trades = 10
            manager._state.daily_drawdown = 0.01

            # New day
            manager.new_day()

            assert manager._state.daily_pnl == 0.0
            assert manager._state.daily_trades == 0
            assert manager._state.daily_drawdown == 0.0

    def test_alert_callback(self, mock_settings):
        """Test risk alert callbacks"""
        with patch('src.risk.risk_manager.get_settings', return_value=mock_settings):
            from src.risk.risk_manager import RiskManager

            manager = RiskManager(
                starting_equity=100000.0,
                max_daily_drawdown=0.03
            )

            alerts_received = []
            manager.on_alert(lambda alert: alerts_received.append(alert))

            # Trigger circuit breaker
            manager.update_equity(96000.0)  # 4% drawdown

            # Should have received an alert
            assert len(alerts_received) > 0


class TestRiskAlert:
    """Test RiskAlert class"""

    def test_alert_creation(self):
        """Test RiskAlert creation"""
        from src.risk.risk_manager import RiskAlert, AlertType, RiskLevel

        alert = RiskAlert(
            alert_type=AlertType.DRAWDOWN_WARNING,
            level=RiskLevel.HIGH,
            symbol="US.TQQQ",
            message="Position loss exceeds 5%",
            value=-0.06,
            threshold=-0.05
        )

        assert alert.alert_type == AlertType.DRAWDOWN_WARNING
        assert alert.level == RiskLevel.HIGH
        assert alert.acknowledged is False
