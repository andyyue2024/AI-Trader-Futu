#!/usr/bin/env python3
"""
AI Futu Trader - Demo Script
Demonstrates all features of the trading system

Usage:
    python demo.py --mode simulate    # Run in simulation mode
    python demo.py --mode backtest    # Run backtest
    python demo.py --mode status      # Show system status
"""
import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime, date, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def demo_configuration():
    """Demonstrate configuration system"""
    print("\n" + "=" * 60)
    print("📋 Configuration System Demo")
    print("=" * 60)

    from src.core.config import get_settings

    settings = get_settings()
    print(f"  Futu Host: {settings.futu_host}")
    print(f"  Futu Port: {settings.futu_port}")
    print(f"  Trade Env: {settings.futu_trade_env}")
    print(f"  LLM Provider: {settings.llm_provider}")
    print(f"  Trading Symbols: {settings.trading_symbols}")
    print(f"  Max Daily Drawdown: {settings.max_daily_drawdown:.1%}")
    print(f"  Max Total Drawdown: {settings.max_total_drawdown:.1%}")


def demo_symbols():
    """Demonstrate symbol registry"""
    print("\n" + "=" * 60)
    print("📊 Symbol Registry Demo")
    print("=" * 60)

    from src.core.symbols import get_symbol_registry, InstrumentType

    registry = get_symbol_registry()

    print("\n  Registered Symbols:")
    for symbol in registry.all_symbols()[:10]:
        print(f"    - {symbol.futu_code}: {symbol.name or symbol.symbol} ({symbol.instrument_type.value})")

    # Register an option
    print("\n  Registering option contract...")
    option = registry.register_option(
        underlying="AAPL",
        strike=150.0,
        expiry="20240315",
        option_type="call"
    )
    print(f"    Created: {option.futu_code}")
    print(f"    Is Option: {option.is_option()}")


def demo_session_manager():
    """Demonstrate session manager"""
    print("\n" + "=" * 60)
    print("⏰ Session Manager Demo")
    print("=" * 60)

    from src.core.session_manager import get_session_manager

    manager = get_session_manager()
    info = manager.get_session_info()

    print(f"\n  Current Session: {info.session.value}")
    print(f"  Trading Allowed: {info.is_trading_allowed}")
    print(f"  Session Progress: {info.progress_pct:.1f}%")
    print(f"  Seconds to Close: {info.seconds_to_close}")
    print(f"  Next Session: {info.next_session.value}")


def demo_statistics():
    """Demonstrate trading statistics"""
    print("\n" + "=" * 60)
    print("📈 Trading Statistics Demo")
    print("=" * 60)

    from src.core.statistics import TradingStatistics

    stats = TradingStatistics(starting_equity=100000.0)

    # Simulate some trades
    import random
    for i in range(20):
        entry_price = 50.0 + random.uniform(-2, 2)
        exit_price = entry_price + random.uniform(-1, 1.5)

        stats.record_entry(
            trade_id=f"demo-{i}",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            side="long",
            quantity=100,
            entry_price=entry_price
        )
        stats.record_exit("US.TQQQ", exit_price)

    metrics = stats.calculate_metrics()

    print(f"\n  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Total Return: ${metrics.total_return:,.2f}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")


def demo_strategy_config():
    """Demonstrate strategy configuration"""
    print("\n" + "=" * 60)
    print("⚙️ Strategy Configuration Demo")
    print("=" * 60)

    from src.core.strategy_config import StrategyConfig, list_strategies, get_strategy

    print("\n  Available Pre-built Strategies:")
    for name in list_strategies():
        strategy = get_strategy(name)
        print(f"    - {name}: {strategy.description}")

    # Create custom config
    config = StrategyConfig(
        name="demo_strategy",
        symbols=["US.TQQQ", "US.QQQ"],
    )

    print(f"\n  Custom Strategy: {config.name}")
    print(f"  Symbols: {config.symbols}")
    print(f"  Stop Loss: {config.risk.stop_loss_pct:.1%}")
    print(f"  Take Profit: {config.risk.take_profit_pct:.1%}")


def demo_position_manager():
    """Demonstrate position manager"""
    print("\n" + "=" * 60)
    print("💼 Position Manager Demo")
    print("=" * 60)

    from src.action.position_manager import PositionManager

    manager = PositionManager(starting_cash=100000.0)

    # Open position
    position = manager.open_position(
        futu_code="US.TQQQ",
        quantity=100,
        price=50.0,
        order_id="demo-001",
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )

    print(f"\n  Opened Position: {position.direction} {position.quantity} {position.symbol}")
    print(f"  Avg Cost: ${position.avg_cost:.2f}")
    print(f"  Stop Loss: ${position.stop_loss_price:.2f}")
    print(f"  Take Profit: ${position.take_profit_price:.2f}")

    # Update price
    manager.update_prices({"US.TQQQ": 51.0})

    print(f"\n  After Price Update to $51.00:")
    print(f"  Unrealized P&L: ${manager.total_unrealized_pnl:,.2f}")
    print(f"  Equity: ${manager.equity:,.2f}")

    # Close position
    pnl = manager.close_position("US.TQQQ", 100, 51.0, "demo-002")
    print(f"\n  Closed Position, Realized P&L: ${pnl:,.2f}")


def demo_order_optimizer():
    """Demonstrate order optimizer"""
    print("\n" + "=" * 60)
    print("⚡ Order Optimizer Demo")
    print("=" * 60)

    from src.action.order_optimizer import OrderOptimizer, ExecutionTimer

    optimizer = OrderOptimizer(target_latency_ms=1.4)
    optimizer.warm_up()

    print(f"\n  Target Latency: {optimizer.target_latency_ms}ms")
    print(f"  Warmed Up: {optimizer._warmed_up}")

    # Simulate latency measurements
    import random
    for _ in range(50):
        latency = random.uniform(0.8, 1.6)
        optimizer.latency_tracker.record_order_latency(latency)

    metrics = optimizer.get_latency_metrics()

    print(f"\n  Latency Metrics:")
    print(f"    P50: {metrics.p50_order_latency_ms:.2f}ms")
    print(f"    P95: {metrics.p95_order_latency_ms:.2f}ms")
    print(f"    P99: {metrics.p99_order_latency_ms:.2f}ms")
    print(f"    Meeting Target: {optimizer.is_meeting_target()}")

    # Demo execution timer
    timer = ExecutionTimer()
    timer.start()

    import time
    time.sleep(0.001)
    timer.checkpoint("quote")
    time.sleep(0.001)
    timer.checkpoint("model")
    time.sleep(0.001)
    timer.checkpoint("order")

    print(f"\n  Execution Timer Breakdown:")
    for name, elapsed in timer.get_breakdown().items():
        print(f"    {name}: {elapsed:.3f}ms")


def demo_persistence():
    """Demonstrate data persistence"""
    print("\n" + "=" * 60)
    print("💾 Data Persistence Demo")
    print("=" * 60)

    from src.data.persistence import TradeDatabase, TradeRecord
    import tempfile
    import os

    # Use temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = TradeDatabase(db_path)

        # Insert trade
        record = TradeRecord(
            trade_id="demo-001",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            entry_time=datetime.now(),
            entry_price=50.0,
            entry_side="long",
            quantity=100,
            entry_order_id="order-001",
            pnl=200.0,
            status="closed"
        )

        db.insert_trade(record)

        print(f"\n  Inserted trade: {record.trade_id}")

        # Retrieve trade
        retrieved = db.get_trade("demo-001")
        print(f"  Retrieved trade: {retrieved.trade_id}, P&L: ${retrieved.pnl:.2f}")

        # Get stats
        stats = db.get_trading_stats(30)
        print(f"\n  30-Day Stats:")
        print(f"    Total Trades: {stats['total_trades']}")
        print(f"    Total P&L: ${stats['total_pnl']:.2f}")

        db.close()

    finally:
        os.unlink(db_path)


def demo_enhanced_alerts():
    """Demonstrate enhanced Feishu alerts"""
    print("\n" + "=" * 60)
    print("🔔 Enhanced Feishu Alerts Demo")
    print("=" * 60)

    from src.monitor.feishu_enhanced import (
        FeishuCardBuilder, Alert, AlertPriority, AlertCategory
    )

    builder = FeishuCardBuilder()

    # Build alert card
    alert = Alert(
        title="Drawdown Warning",
        content="Portfolio drawdown exceeded 2.5%",
        priority=AlertPriority.P1_HIGH,
        category=AlertCategory.RISK,
        symbol="US.TQQQ",
        value=0.025,
        threshold=0.02
    )

    card = builder.build_alert_card(alert)

    print(f"\n  Alert Card Built:")
    print(f"    Type: {card['msg_type']}")
    print(f"    Template: {card['card']['header']['template']}")
    print(f"    Title: {card['card']['header']['title']['content'][:50]}...")

    # Build daily report card
    report = builder.build_daily_report_card(
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

    print(f"\n  Daily Report Card Built:")
    print(f"    Template: {report['card']['header']['template']}")


def run_backtest_demo():
    """Run backtest demonstration"""
    print("\n" + "=" * 60)
    print("🔄 Backtest Engine Demo")
    print("=" * 60)

    from src.backtest.engine import BacktestEngine, BacktestConfig, create_simple_strategy
    import pandas as pd
    import numpy as np

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1min')

    base_price = 50.0
    returns = np.random.normal(0.0001, 0.005, 500)
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    prices = np.array(prices)

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, 500)),
        'high': prices * (1 + np.random.uniform(0.001, 0.005, 500)),
        'low': prices * (1 - np.random.uniform(0.001, 0.005, 500)),
        'close': prices,
        'volume': np.random.randint(100000, 500000, 500)
    })

    # Create backtest engine
    config = BacktestConfig(
        starting_capital=100000.0,
        slippage_pct=0.001,
        max_daily_drawdown=0.03
    )

    engine = BacktestEngine(config)
    engine.load_data("US.TQQQ", data)

    print(f"\n  Loaded {len(data)} bars for US.TQQQ")

    # Create strategy
    strategy = create_simple_strategy(
        rsi_oversold=30,
        rsi_overbought=70,
        use_macd=True
    )

    print("  Running backtest...")

    # Run backtest
    result = engine.run(strategy)

    print(f"\n  Backtest Results:")
    print(f"    Total Trades: {result.metrics.total_trades}")
    print(f"    Win Rate: {result.metrics.win_rate:.1%}")
    print(f"    Total Return: {result.metrics.total_return_pct:.2%}")
    print(f"    Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"    Max Drawdown: {result.metrics.max_drawdown_pct:.2%}")


def show_system_status():
    """Show system status"""
    print("\n" + "=" * 60)
    print("🖥️ System Status")
    print("=" * 60)

    print("\n  Checking module imports...")

    modules_to_check = [
        ("src.core.config", "Configuration"),
        ("src.core.logger", "Logging"),
        ("src.core.symbols", "Symbol Registry"),
        ("src.core.session_manager", "Session Manager"),
        ("src.core.statistics", "Statistics"),
        ("src.core.strategy_config", "Strategy Config"),
        ("src.data.futu_quote", "Futu Quote"),
        ("src.data.data_processor", "Data Processor"),
        ("src.data.options_data", "Options Data"),
        ("src.data.persistence", "Persistence"),
        ("src.action.futu_executor", "Futu Executor"),
        ("src.action.position_manager", "Position Manager"),
        ("src.action.order_optimizer", "Order Optimizer"),
        ("src.model.llm_agent", "LLM Agent"),
        ("src.model.prompts", "Prompts"),
        ("src.risk.risk_manager", "Risk Manager"),
        ("src.monitor.metrics", "Metrics"),
        ("src.monitor.alerts", "Alerts"),
        ("src.monitor.feishu_enhanced", "Feishu Enhanced"),
        ("src.backtest.engine", "Backtest Engine"),
        ("src.engine", "Trading Engine"),
    ]

    all_ok = True
    for module, name in modules_to_check:
        try:
            __import__(module)
            print(f"    ✅ {name}")
        except Exception as e:
            print(f"    ❌ {name}: {e}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All modules loaded successfully!")
    else:
        print("⚠️ Some modules failed to load")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AI Futu Trader Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['all', 'config', 'symbols', 'session', 'stats', 'strategy',
                 'position', 'optimizer', 'persistence', 'alerts', 'backtest', 'status'],
        default='all',
        help='Demo mode to run'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🤖 AI Futu Trader - Feature Demo")
    print("=" * 60)

    if args.mode in ['all', 'status']:
        show_system_status()

    if args.mode in ['all', 'config']:
        demo_configuration()

    if args.mode in ['all', 'symbols']:
        demo_symbols()

    if args.mode in ['all', 'session']:
        demo_session_manager()

    if args.mode in ['all', 'stats']:
        demo_statistics()

    if args.mode in ['all', 'strategy']:
        demo_strategy_config()

    if args.mode in ['all', 'position']:
        demo_position_manager()

    if args.mode in ['all', 'optimizer']:
        demo_order_optimizer()

    if args.mode in ['all', 'persistence']:
        demo_persistence()

    if args.mode in ['all', 'alerts']:
        demo_enhanced_alerts()

    if args.mode in ['all', 'backtest']:
        run_backtest_demo()

    print("\n" + "=" * 60)
    print("✨ Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
