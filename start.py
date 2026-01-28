#!/usr/bin/env python3
"""
AI Futu Trader - Quick Start Script
Provides easy-to-use commands for common operations
"""
import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🤖 AI Futu Trader                                       ║
    ║   Ultra-Low Latency Trading System                        ║
    ║   Based on Futu OpenD + LLM                               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_requirements():
    """Check if requirements are installed"""
    try:
        import futu
        import openai
        import pandas
        import numpy
        import fastapi
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return False


def check_env():
    """Check environment configuration"""
    required_vars = []

    # Check Futu config
    if not os.getenv('FUTU_HOST'):
        required_vars.append('FUTU_HOST')

    # Check LLM config
    provider = os.getenv('LLM_PROVIDER', 'openai')
    if provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
        required_vars.append('OPENAI_API_KEY')
    elif provider == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
        required_vars.append('ANTHROPIC_API_KEY')

    if required_vars:
        print(f"⚠️ Missing environment variables: {', '.join(required_vars)}")
        print("\nPlease create a .env file or set environment variables.")
        print("See .env.example for reference.")
        return False

    return True


def run_command(cmd):
    """Run a command"""
    try:
        subprocess.run(cmd, check=True, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False


def cmd_setup():
    """Setup the project"""
    print("\n📦 Setting up AI Futu Trader...")

    # Create directories
    dirs = ['logs', 'data', 'reports', 'config']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"  ✅ Created {d}/")

    # Copy .env.example if .env doesn't exist
    if not Path('.env').exists() and Path('.env.example').exists():
        import shutil
        shutil.copy('.env.example', '.env')
        print("  ✅ Created .env from .env.example")

    # Install requirements
    print("\n📥 Installing requirements...")
    run_command('pip install -r requirements.txt')

    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("  1. Edit .env with your configuration")
    print("  2. Start Futu OpenD")
    print("  3. Run: python start.py trade --simulate")


def cmd_trade(simulate=True, symbols=None):
    """Start trading"""
    mode = "--simulate" if simulate else "--real"
    symbols_arg = f"--symbols {' '.join(symbols)}" if symbols else ""

    print(f"\n🚀 Starting trading ({mode})...")
    cmd = f"python run.py {mode} {symbols_arg}"
    run_command(cmd)


def cmd_backtest():
    """Run backtest"""
    print("\n🔄 Running backtest...")
    run_command("python -m src.cli backtest")


def cmd_web():
    """Start web interface"""
    print("\n🌐 Starting web interface...")
    run_command("python -m src.cli web")


def cmd_test():
    """Run tests"""
    print("\n🧪 Running tests...")
    run_command("pytest tests/ -v --cov=src")


def cmd_health():
    """Check system health"""
    print("\n🔍 Checking system health...")
    run_command("python -m src.cli health")


def cmd_demo():
    """Run demo"""
    print("\n🎮 Running demo...")
    run_command("python demo.py --mode all")


def cmd_docker():
    """Start with Docker"""
    print("\n🐳 Starting with Docker...")
    run_command("docker-compose up -d")


def print_help():
    """Print help"""
    print_banner()
    print("""
Usage: python start.py <command> [options]

Commands:
  setup              Setup the project
  trade              Start trading (use --real for real trading)
  backtest           Run backtest with sample data
  web                Start web interface
  test               Run tests
  health             Check system health
  demo               Run demo
  docker             Start with Docker

Trading Options:
  --simulate         Run in simulation mode (default)
  --real             Run in real trading mode
  --symbols SYMBOLS  Specify symbols (e.g., US.TQQQ US.QQQ)

Examples:
  python start.py setup
  python start.py trade --simulate
  python start.py trade --real --symbols US.TQQQ US.QQQ
  python start.py backtest
  python start.py web
    """)


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    if command == 'setup':
        cmd_setup()

    elif command == 'trade':
        simulate = '--real' not in sys.argv
        symbols = None
        if '--symbols' in sys.argv:
            idx = sys.argv.index('--symbols')
            symbols = []
            for i in range(idx + 1, len(sys.argv)):
                if sys.argv[i].startswith('--'):
                    break
                symbols.append(sys.argv[i])

        cmd_trade(simulate=simulate, symbols=symbols)

    elif command == 'backtest':
        cmd_backtest()

    elif command == 'web':
        cmd_web()

    elif command == 'test':
        cmd_test()

    elif command == 'health':
        cmd_health()

    elif command == 'demo':
        cmd_demo()

    elif command == 'docker':
        cmd_docker()

    elif command in ['help', '-h', '--help']:
        print_help()

    else:
        print(f"Unknown command: {command}")
        print_help()


if __name__ == "__main__":
    main()
