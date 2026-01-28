"""
Unit tests for data persistence
"""
import pytest
import tempfile
import os
from datetime import datetime, date, timedelta


class TestTradeRecord:
    """Test TradeRecord class"""

    def test_record_creation(self):
        """Test trade record creation"""
        from src.data.persistence import TradeRecord

        record = TradeRecord(
            trade_id="test-001",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            entry_time=datetime.now(),
            entry_price=50.0,
            entry_side="long",
            quantity=100,
            entry_order_id="order-001"
        )

        assert record.trade_id == "test-001"
        assert record.status == "open"

    def test_record_to_dict(self):
        """Test record dictionary conversion"""
        from src.data.persistence import TradeRecord

        record = TradeRecord(
            trade_id="test-001",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            entry_time=datetime.now(),
            entry_price=50.0,
            entry_side="long",
            quantity=100,
            entry_order_id="order-001"
        )

        data = record.to_dict()

        assert data["trade_id"] == "test-001"
        assert "entry_time" in data

    def test_record_from_dict(self):
        """Test creating record from dictionary"""
        from src.data.persistence import TradeRecord

        data = {
            "trade_id": "test-001",
            "symbol": "TQQQ",
            "futu_code": "US.TQQQ",
            "entry_time": datetime.now().isoformat(),
            "entry_price": 50.0,
            "entry_side": "long",
            "quantity": 100,
            "entry_order_id": "order-001"
        }

        record = TradeRecord.from_dict(data)

        assert record.trade_id == "test-001"
        assert isinstance(record.entry_time, datetime)


class TestDailyPerformanceRecord:
    """Test DailyPerformanceRecord class"""

    def test_record_creation(self):
        """Test daily performance record creation"""
        from src.data.persistence import DailyPerformanceRecord

        record = DailyPerformanceRecord(
            date=date.today(),
            starting_equity=100000.0,
            ending_equity=101000.0,
            realized_pnl=1000.0,
            total_trades=10
        )

        assert record.realized_pnl == 1000.0
        assert record.total_trades == 10

    def test_record_to_dict(self):
        """Test record dictionary conversion"""
        from src.data.persistence import DailyPerformanceRecord

        record = DailyPerformanceRecord(
            date=date.today(),
            starting_equity=100000.0,
            ending_equity=101000.0
        )

        data = record.to_dict()

        assert "date" in data
        assert data["starting_equity"] == 100000.0


class TestTradeDatabase:
    """Test TradeDatabase class"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_database_creation(self, temp_db):
        """Test database creation"""
        from src.data.persistence import TradeDatabase

        db = TradeDatabase(temp_db)

        assert os.path.exists(temp_db)
        db.close()

    def test_insert_and_get_trade(self, temp_db):
        """Test inserting and retrieving trade"""
        from src.data.persistence import TradeDatabase, TradeRecord

        db = TradeDatabase(temp_db)

        record = TradeRecord(
            trade_id="test-001",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            entry_time=datetime.now(),
            entry_price=50.0,
            entry_side="long",
            quantity=100,
            entry_order_id="order-001"
        )

        db.insert_trade(record)

        retrieved = db.get_trade("test-001")

        assert retrieved is not None
        assert retrieved.trade_id == "test-001"
        assert retrieved.symbol == "TQQQ"

        db.close()

    def test_update_trade(self, temp_db):
        """Test updating trade"""
        from src.data.persistence import TradeDatabase, TradeRecord

        db = TradeDatabase(temp_db)

        record = TradeRecord(
            trade_id="test-002",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            entry_time=datetime.now(),
            entry_price=50.0,
            entry_side="long",
            quantity=100,
            entry_order_id="order-001"
        )

        db.insert_trade(record)

        # Update with exit
        record.exit_time = datetime.now()
        record.exit_price = 52.0
        record.pnl = 200.0
        record.status = "closed"

        db.update_trade(record)

        retrieved = db.get_trade("test-002")

        assert retrieved.status == "closed"
        assert retrieved.pnl == 200.0

        db.close()

    def test_get_open_trades(self, temp_db):
        """Test getting open trades"""
        from src.data.persistence import TradeDatabase, TradeRecord

        db = TradeDatabase(temp_db)

        # Insert open trade
        db.insert_trade(TradeRecord(
            trade_id="open-001",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            entry_time=datetime.now(),
            entry_price=50.0,
            entry_side="long",
            quantity=100,
            entry_order_id="order-001",
            status="open"
        ))

        # Insert closed trade
        db.insert_trade(TradeRecord(
            trade_id="closed-001",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            entry_time=datetime.now(),
            entry_price=50.0,
            entry_side="long",
            quantity=100,
            entry_order_id="order-002",
            status="closed"
        ))

        open_trades = db.get_open_trades()

        assert len(open_trades) == 1
        assert open_trades[0].trade_id == "open-001"

        db.close()

    def test_insert_daily_performance(self, temp_db):
        """Test inserting daily performance"""
        from src.data.persistence import TradeDatabase, DailyPerformanceRecord

        db = TradeDatabase(temp_db)

        perf = DailyPerformanceRecord(
            date=date.today(),
            starting_equity=100000.0,
            ending_equity=101000.0,
            realized_pnl=1000.0,
            total_trades=10,
            winning_trades=7,
            sharpe_ratio=2.5
        )

        db.insert_daily_performance(perf)

        records = db.get_daily_performance(
            date.today() - timedelta(days=1),
            date.today() + timedelta(days=1)
        )

        assert len(records) == 1
        assert records[0].realized_pnl == 1000.0

        db.close()

    def test_get_trading_stats(self, temp_db):
        """Test getting aggregate trading stats"""
        from src.data.persistence import TradeDatabase, TradeRecord

        db = TradeDatabase(temp_db)

        # Insert some trades
        for i in range(5):
            db.insert_trade(TradeRecord(
                trade_id=f"trade-{i}",
                symbol="TQQQ",
                futu_code="US.TQQQ",
                entry_time=datetime.now(),
                entry_price=50.0,
                entry_side="long",
                quantity=100,
                entry_order_id=f"order-{i}",
                pnl=100.0 if i < 3 else -50.0,
                status="closed"
            ))

        stats = db.get_trading_stats(30)

        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3

        db.close()
