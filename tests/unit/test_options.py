"""
Unit tests for options data
"""
import pytest
from datetime import date, datetime


class TestOptionSide:
    """Test OptionSide enum"""

    def test_option_side_values(self):
        """Test option side values"""
        from src.data.options_data import OptionSide

        assert OptionSide.CALL.value == "call"
        assert OptionSide.PUT.value == "put"


class TestOptionContract:
    """Test OptionContract class"""

    def test_contract_creation(self):
        """Test contract creation"""
        from src.data.options_data import OptionContract, OptionSide

        contract = OptionContract(
            futu_code="US.AAPL240315C150000",
            underlying="AAPL",
            underlying_futu_code="US.AAPL",
            option_type=OptionSide.CALL,
            strike=150.0,
            expiry=date(2024, 3, 15)
        )

        assert contract.strike == 150.0
        assert contract.option_type == OptionSide.CALL

    def test_mid_price(self):
        """Test mid price calculation"""
        from src.data.options_data import OptionContract, OptionSide

        contract = OptionContract(
            futu_code="US.AAPL240315C150000",
            underlying="AAPL",
            underlying_futu_code="US.AAPL",
            option_type=OptionSide.CALL,
            strike=150.0,
            expiry=date(2024, 3, 15),
            bid=2.50,
            ask=2.60
        )

        assert contract.mid_price == pytest.approx(2.55, rel=0.01)

    def test_moneyness_call_itm(self):
        """Test in-the-money detection for calls"""
        from src.data.options_data import OptionContract, OptionSide

        contract = OptionContract(
            futu_code="US.AAPL240315C150000",
            underlying="AAPL",
            underlying_futu_code="US.AAPL",
            option_type=OptionSide.CALL,
            strike=150.0,
            expiry=date(2024, 3, 15),
            underlying_price=160.0
        )

        assert contract.moneyness == "itm"

    def test_moneyness_put_itm(self):
        """Test in-the-money detection for puts"""
        from src.data.options_data import OptionContract, OptionSide

        contract = OptionContract(
            futu_code="US.AAPL240315P150000",
            underlying="AAPL",
            underlying_futu_code="US.AAPL",
            option_type=OptionSide.PUT,
            strike=150.0,
            expiry=date(2024, 3, 15),
            underlying_price=140.0
        )

        assert contract.moneyness == "itm"

    def test_intrinsic_value_call(self):
        """Test intrinsic value for call"""
        from src.data.options_data import OptionContract, OptionSide

        contract = OptionContract(
            futu_code="US.AAPL240315C150000",
            underlying="AAPL",
            underlying_futu_code="US.AAPL",
            option_type=OptionSide.CALL,
            strike=150.0,
            expiry=date(2024, 3, 15),
            underlying_price=160.0
        )

        # Intrinsic = 160 - 150 = 10
        assert contract.intrinsic_value == 10.0

    def test_days_to_expiry(self):
        """Test days to expiry calculation"""
        from src.data.options_data import OptionContract, OptionSide
        from datetime import date, timedelta

        future_date = date.today() + timedelta(days=30)

        contract = OptionContract(
            futu_code="US.AAPL240315C150000",
            underlying="AAPL",
            underlying_futu_code="US.AAPL",
            option_type=OptionSide.CALL,
            strike=150.0,
            expiry=future_date
        )

        assert contract.days_to_expiry == 30


class TestOptionChain:
    """Test OptionChain class"""

    def test_chain_creation(self):
        """Test chain creation"""
        from src.data.options_data import OptionChain

        chain = OptionChain(
            underlying="AAPL",
            underlying_futu_code="US.AAPL",
            underlying_price=155.0,
            expiry_dates=[date(2024, 3, 15), date(2024, 4, 19)]
        )

        assert chain.underlying == "AAPL"
        assert chain.underlying_price == 155.0
        assert len(chain.expiry_dates) == 2

    def test_get_atm_strike(self):
        """Test ATM strike detection"""
        from src.data.options_data import OptionChain, OptionContract, OptionSide

        chain = OptionChain(
            underlying="AAPL",
            underlying_futu_code="US.AAPL",
            underlying_price=155.0
        )

        expiry = date(2024, 3, 15)
        chain.calls[expiry] = {}

        for strike in [140, 145, 150, 155, 160, 165, 170]:
            chain.calls[expiry][float(strike)] = OptionContract(
                futu_code=f"US.AAPL240315C{strike}000",
                underlying="AAPL",
                underlying_futu_code="US.AAPL",
                option_type=OptionSide.CALL,
                strike=float(strike),
                expiry=expiry
            )

        atm = chain.get_atm_strike(expiry)
        assert atm == 155.0


class TestOptionStrategy:
    """Test OptionStrategy enum"""

    def test_strategy_values(self):
        """Test strategy enum values"""
        from src.data.options_data import OptionStrategy

        assert OptionStrategy.LONG_CALL.value == "long_call"
        assert OptionStrategy.IRON_CONDOR.value == "iron_condor"
        assert OptionStrategy.COVERED_CALL.value == "covered_call"
