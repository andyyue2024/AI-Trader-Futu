#!/usr/bin/env python3
"""
AI Futu Trader - CLI Tools
Command-line utilities for system management

Usage:
    python -m src.cli health       # Health check
    python -m src.cli status       # System status
    python -m src.cli export       # Export data
    python -m src.cli symbols      # List symbols
    python -m src.cli backtest     # Run backtest
"""
import argparse
import sys
import json
from datetime import datetime, date, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_health(args):
    """Health check command"""
    print("\n🔍 AI Futu Trader Health Check")
    print("=" * 50)

    checks = []

    # Check Python version
    py_version = sys.version_info
    py_ok = py_version >= (3, 10)
    checks.append(("Python Version", f"{py_version.major}.{py_version.minor}", py_ok))

    # Check module imports
    modules_to_check = [
        "src.core.config",
        "src.core.logger",
        "src.data.futu_quote",
        "src.action.futu_executor",
        "src.model.llm_agent",
        "src.risk.risk_manager",
        "src.monitor.metrics",
        "src.engine",
    ]

    for module in modules_to_check:
        try:
            __import__(module)
            checks.append((module, "OK", True))
        except Exception as e:
            checks.append((module, str(e)[:30], False))

    # Check configuration
    try:
        from src.core.config import get_settings
        settings = get_settings()
        checks.append(("Configuration", "Loaded", True))

        # Check API keys
        if settings.llm_provider == "openai":
            has_key = bool(settings.openai_api_key)
            checks.append(("OpenAI API Key", "Set" if has_key else "Missing", has_key))
        elif settings.llm_provider == "anthropic":
            has_key = bool(settings.anthropic_api_key)
            checks.append(("Anthropic API Key", "Set" if has_key else "Missing", has_key))
    except Exception as e:
        checks.append(("Configuration", str(e)[:30], False))

    # Check database
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()
        stats = db.get_trading_stats(30)
        checks.append(("Database", f"{stats.get('total_trades', 0)} trades", True))
    except Exception as e:
        checks.append(("Database", str(e)[:30], False))

    # Print results
    all_ok = True
    for name, status, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}: {status}")
        if not ok:
            all_ok = False

    print("=" * 50)
    if all_ok:
        print("✅ All health checks passed!")
        return 0
    else:
        print("❌ Some health checks failed!")
        return 1


def cmd_status(args):
    """System status command"""
    print("\n📊 AI Futu Trader Status")
    print("=" * 50)

    try:
        from src.core.config import get_settings
        from src.core.session_manager import get_session_manager
        from src.core.symbols import get_symbol_registry

        settings = get_settings()
        session_mgr = get_session_manager()
        session_info = session_mgr.get_session_info()
        registry = get_symbol_registry()

        print(f"\n📡 Connection:")
        print(f"   Futu OpenD: {settings.futu_host}:{settings.futu_port}")
        print(f"   Trade Env: {settings.futu_trade_env}")

        print(f"\n⏰ Trading Session:")
        print(f"   Current: {session_info.session.value}")
        print(f"   Progress: {session_info.progress_pct:.1f}%")
        print(f"   Trading Allowed: {session_info.is_trading_allowed}")

        print(f"\n📈 Symbols:")
        active = list(registry.active_symbols)[:5]
        print(f"   Active: {', '.join(active)}")
        print(f"   Total Registered: {len(registry.all_symbols())}")

        print(f"\n🧠 LLM:")
        print(f"   Provider: {settings.llm_provider}")

        print(f"\n📉 Risk Parameters:")
        print(f"   Max Daily Drawdown: {settings.max_daily_drawdown:.1%}")
        print(f"   Max Total Drawdown: {settings.max_total_drawdown:.1%}")
        print(f"   Position Size: ${settings.default_position_size:,.2f}")

        # Performance metrics
        try:
            from src.monitor.performance import get_performance_monitor
            monitor = get_performance_monitor()
            metrics = monitor.get_current_metrics()

            print(f"\n💻 System:")
            print(f"   CPU: {metrics.get('cpu_percent', 0):.1f}%")
            print(f"   Memory: {metrics.get('memory_percent', 0):.1f}%")
            print(f"   Error Rate: {metrics.get('error_rate', 0):.2f}/min")
        except:
            pass

        print("=" * 50)
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_symbols(args):
    """List symbols command"""
    from src.core.symbols import get_symbol_registry

    registry = get_symbol_registry()
    symbols = registry.all_symbols()

    print("\n📊 Registered Symbols")
    print("=" * 70)
    print(f"{'Symbol':<15} {'Futu Code':<15} {'Type':<15} {'Active':<10}")
    print("-" * 70)

    for sym in symbols:
        is_active = "✅" if sym.futu_code in registry.active_symbols else ""
        print(f"{sym.symbol:<15} {sym.futu_code:<15} {sym.instrument_type.value:<15} {is_active:<10}")

    print("-" * 70)
    print(f"Total: {len(symbols)} symbols")

    if args.activate:
        for code in args.activate:
            registry.activate(code)
            print(f"✅ Activated: {code}")

    if args.deactivate:
        for code in args.deactivate:
            registry.deactivate(code)
            print(f"❌ Deactivated: {code}")

    return 0


def cmd_export(args):
    """Export data command"""
    from src.data.persistence import get_trade_database

    db = get_trade_database()

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    output_file = args.output or f"trades_export_{start_date}_{end_date}.json"

    print(f"\n📤 Exporting trades from {start_date} to {end_date}...")

    db.export_to_json(output_file, args.days)

    print(f"✅ Exported to: {output_file}")
    return 0


def cmd_backtest(args):
    """Run backtest command"""
    print("\n🔄 Running Backtest...")
    print("=" * 50)

    try:
        from src.backtest import BacktestEngine, BacktestConfig, create_simple_strategy
        import pandas as pd
        import numpy as np

        # Generate sample data if no file provided
        if args.data:
            data = pd.read_csv(args.data, parse_dates=['timestamp'])
            symbol = args.symbol or "US.TQQQ"
        else:
            print("Generating sample data for demo...")
            np.random.seed(42)
            dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')

            base_price = 50.0
            returns = np.random.normal(0.0001, 0.005, 1000)
            prices = [base_price]
            for r in returns[1:]:
                prices.append(prices[-1] * (1 + r))
            prices = np.array(prices)

            data = pd.DataFrame({
                'timestamp': dates,
                'open': prices * (1 + np.random.uniform(-0.001, 0.001, 1000)),
                'high': prices * (1 + np.random.uniform(0.001, 0.005, 1000)),
                'low': prices * (1 - np.random.uniform(0.001, 0.005, 1000)),
                'close': prices,
                'volume': np.random.randint(100000, 500000, 1000)
            })
            symbol = "US.TQQQ"

        # Create engine
        config = BacktestConfig(
            starting_capital=args.capital,
            slippage_pct=args.slippage,
            max_daily_drawdown=0.03,
            max_total_drawdown=0.15
        )

        engine = BacktestEngine(config)
        engine.load_data(symbol, data)

        print(f"Loaded {len(data)} bars for {symbol}")

        # Create strategy
        strategy = create_simple_strategy(
            rsi_oversold=args.rsi_low,
            rsi_overbought=args.rsi_high,
            use_macd=True
        )

        # Run backtest
        print("Running backtest...")
        result = engine.run(strategy)

        # Print results
        print("\n📊 Backtest Results:")
        print("-" * 50)
        print(f"   Total Trades: {result.metrics.total_trades}")
        print(f"   Win Rate: {result.metrics.win_rate:.1%}")
        print(f"   Total Return: ${result.metrics.total_return:+,.2f}")
        print(f"   Return %: {result.metrics.total_return_pct:.2%}")
        print(f"   Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {result.metrics.max_drawdown_pct:.2%}")
        print(f"   Profit Factor: {result.metrics.profit_factor:.2f}")
        print("-" * 50)

        # Save results
        if args.save:
            with open(args.save, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"✅ Results saved to: {args.save}")

        return 0

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_report(args):
    """Generate report command"""
    from src.report import ReportGenerator, ReportConfig

    config = ReportConfig(
        title=args.title or "AI Futu Trader Report",
        output_dir=args.output_dir or "reports"
    )

    generator = ReportGenerator(config)

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    print(f"\n📑 Generating {args.format.upper()} report...")
    print(f"   Period: {start_date} to {end_date}")

    if args.format == "pdf":
        filepath = generator.generate_pdf(start_date, end_date)
    elif args.format == "excel":
        filepath = generator.generate_excel(start_date, end_date)
    else:
        filepath = generator.generate_html(start_date, end_date)

    print(f"✅ Report generated: {filepath}")
    return 0


def cmd_web(args):
    """Start web server command"""
    from src.web.api import run_server

    print(f"\n🌐 Starting Web Server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   URL: http://{args.host}:{args.port}")

    run_server(host=args.host, port=args.port)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="AI Futu Trader CLI Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Health command
    health_parser = subparsers.add_parser("health", help="System health check")
    health_parser.set_defaults(func=cmd_health)

    # Status command
    status_parser = subparsers.add_parser("status", help="System status")
    status_parser.set_defaults(func=cmd_status)

    # Symbols command
    symbols_parser = subparsers.add_parser("symbols", help="List and manage symbols")
    symbols_parser.add_argument("--activate", nargs="+", help="Activate symbols")
    symbols_parser.add_argument("--deactivate", nargs="+", help="Deactivate symbols")
    symbols_parser.set_defaults(func=cmd_symbols)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export trade data")
    export_parser.add_argument("--days", type=int, default=30, help="Days to export")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.set_defaults(func=cmd_export)

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--data", help="CSV data file")
    backtest_parser.add_argument("--symbol", default="US.TQQQ", help="Symbol to backtest")
    backtest_parser.add_argument("--capital", type=float, default=100000, help="Starting capital")
    backtest_parser.add_argument("--slippage", type=float, default=0.001, help="Slippage")
    backtest_parser.add_argument("--rsi-low", type=float, default=30, help="RSI oversold")
    backtest_parser.add_argument("--rsi-high", type=float, default=70, help="RSI overbought")
    backtest_parser.add_argument("--save", help="Save results to file")
    backtest_parser.set_defaults(func=cmd_backtest)

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("--format", choices=["pdf", "excel", "html"], default="html")
    report_parser.add_argument("--days", type=int, default=30, help="Days to include")
    report_parser.add_argument("--title", help="Report title")
    report_parser.add_argument("--output-dir", help="Output directory")
    report_parser.set_defaults(func=cmd_report)

    # Web command
    web_parser = subparsers.add_parser("web", help="Start web server")
    web_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    web_parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    web_parser.set_defaults(func=cmd_web)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
