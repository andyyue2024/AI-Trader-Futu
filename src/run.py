#!/usr/bin/env python3
"""
AI Futu Trader - Main Entry Point
Ultra-low latency trading system based on Al-Trader architecture with Futu OpenD

Usage:
    python run.py                    # Run trading engine
    python run.py --simulate         # Run in simulation mode
    python run.py --backtest         # Run backtest mode
    python run.py --symbols TQQQ QQQ # Trade specific symbols
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path.parent))

from src.core.config import get_settings, Settings
from src.core.logger import setup_logger, get_logger
from src.engine import TradingEngine


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Futu Trader - Ultra-low latency trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings
    python run.py
    
    # Run in simulation mode
    python run.py --simulate
    
    # Trade specific symbols
    python run.py --symbols US.TQQQ US.QQQ US.SOXL
    
    # Use Anthropic Claude instead of OpenAI
    python run.py --llm anthropic
    
    # Adjust risk parameters
    python run.py --max-drawdown 0.02 --position-size 5000
"""
    )

    # Trading mode
    parser.add_argument(
        '--simulate', '-s',
        action='store_true',
        help='Run in simulation/paper trading mode'
    )

    parser.add_argument(
        '--real',
        action='store_true',
        help='Run in real trading mode (requires confirmation)'
    )

    # Symbols
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='List of symbols to trade (e.g., US.TQQQ US.QQQ)'
    )

    # LLM settings
    parser.add_argument(
        '--llm',
        choices=['openai', 'anthropic'],
        default=None,
        help='LLM provider to use'
    )

    parser.add_argument(
        '--llm-model',
        type=str,
        default=None,
        help='Specific LLM model to use'
    )

    # Risk parameters
    parser.add_argument(
        '--max-drawdown',
        type=float,
        default=None,
        help='Maximum daily drawdown (default: 0.03 = 3%%)'
    )

    parser.add_argument(
        '--position-size',
        type=float,
        default=None,
        help='Default position size in USD'
    )

    # Futu connection
    parser.add_argument(
        '--futu-host',
        type=str,
        default=None,
        help='Futu OpenD host address'
    )

    parser.add_argument(
        '--futu-port',
        type=int,
        default=None,
        help='Futu OpenD port'
    )

    # Monitoring
    parser.add_argument(
        '--metrics-port',
        type=int,
        default=None,
        help='Prometheus metrics port'
    )

    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=None,
        help='Logging level'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file path'
    )

    # Other
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze market but do not execute trades'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='AI Futu Trader v1.0.0'
    )

    return parser.parse_args()


def override_settings(args) -> Settings:
    """Override settings based on command line arguments"""
    import os

    # Set environment variables based on args
    if args.simulate:
        os.environ['FUTU_TRADE_ENV'] = 'SIMULATE'
    elif args.real:
        os.environ['FUTU_TRADE_ENV'] = 'REAL'

    if args.symbols:
        os.environ['TRADING_SYMBOLS'] = ','.join(args.symbols)

    if args.llm:
        os.environ['LLM_PROVIDER'] = args.llm

    if args.llm_model:
        if args.llm == 'anthropic':
            os.environ['ANTHROPIC_MODEL'] = args.llm_model
        else:
            os.environ['OPENAI_MODEL'] = args.llm_model

    if args.max_drawdown:
        os.environ['MAX_DAILY_DRAWDOWN'] = str(args.max_drawdown)

    if args.position_size:
        os.environ['DEFAULT_POSITION_SIZE'] = str(args.position_size)

    if args.futu_host:
        os.environ['FUTU_HOST'] = args.futu_host

    if args.futu_port:
        os.environ['FUTU_PORT'] = str(args.futu_port)

    if args.metrics_port:
        os.environ['PROMETHEUS_PORT'] = str(args.metrics_port)

    if args.log_level:
        os.environ['LOG_LEVEL'] = args.log_level

    if args.log_file:
        os.environ['LOG_FILE'] = args.log_file

    # Clear cached settings and reload
    from src.core.config import get_settings
    get_settings.cache_clear()

    return get_settings()


def confirm_real_trading() -> bool:
    """Confirm real trading mode with user"""
    print("\n" + "=" * 60)
    print("⚠️  WARNING: REAL TRADING MODE")
    print("=" * 60)
    print("\nYou are about to start REAL trading with actual money.")
    print("All trades will be executed on your live account.")
    print("\nPlease confirm by typing 'YES' (case-sensitive):")

    response = input("> ")
    return response == "YES"


def print_startup_banner(settings: Settings):
    """Print startup banner with configuration"""
    print("\n" + "=" * 60)
    print("🤖 AI Futu Trader v1.0.0")
    print("=" * 60)
    print(f"\n📊 Trading Mode: {settings.futu_trade_env}")
    print(f"📈 Symbols: {', '.join(settings.trading_symbols)}")
    print(f"🧠 LLM Provider: {settings.llm_provider}")
    print(f"💰 Position Size: ${settings.default_position_size:,.2f}")
    print(f"🎯 Max Daily Drawdown: {settings.max_daily_drawdown:.1%}")
    print(f"📉 Max Total Drawdown: {settings.max_total_drawdown:.1%}")
    print(f"⏱️  Target Order Latency: {settings.max_order_latency_ms}ms")
    print(f"📡 Futu OpenD: {settings.futu_host}:{settings.futu_port}")
    print(f"📊 Prometheus Port: {settings.prometheus_port}")
    print("=" * 60 + "\n")


async def run_trading(settings: Settings, dry_run: bool = False):
    """Run the trading engine"""
    engine = TradingEngine(settings=settings)

    if dry_run:
        print("\n🔍 DRY RUN MODE - No trades will be executed")
        # In dry run, we could run analysis only
        # For now, just run the engine without execution

    await engine.start()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()

    # Override settings
    settings = override_settings(args)

    # Setup logging
    setup_logger(
        log_level=settings.log_level,
        log_file=settings.log_file
    )

    logger = get_logger(__name__)

    # Confirm real trading
    if settings.is_real_trading and not args.dry_run:
        if not confirm_real_trading():
            print("\n❌ Real trading cancelled.")
            sys.exit(0)

    # Print banner
    print_startup_banner(settings)

    # Validate configuration
    if not settings.futu_host:
        logger.error("FUTU_HOST not configured")
        sys.exit(1)

    if settings.llm_provider == 'openai' and not settings.openai_api_key:
        logger.error("OPENAI_API_KEY not configured")
        sys.exit(1)

    if settings.llm_provider == 'anthropic' and not settings.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY not configured")
        sys.exit(1)

    # Run trading
    try:
        asyncio.run(run_trading(settings, dry_run=args.dry_run))
    except KeyboardInterrupt:
        print("\n\n👋 Trading stopped by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
