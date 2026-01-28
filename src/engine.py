"""
Main Trading Engine - Orchestrates the complete trading pipeline
Quote -> Model -> Order execution with sub-1s latency
"""
import asyncio
import signal
import time
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys

from src.core.config import get_settings, Settings
from src.core.logger import setup_logger, get_logger, PerformanceLogger
from src.core.symbols import SymbolRegistry, get_symbol_registry, TradingSession

from src.data.futu_quote import FutuQuoteClient, AsyncFutuQuoteClient, QuoteData, KLineData
from src.data.data_processor import DataProcessor, MarketSnapshot

from src.action.futu_executor import (
    FutuExecutor,
    AsyncFutuExecutor,
    TradingAction,
    OrderResult,
    Position
)

from src.model.llm_agent import LLMAgent, TradingDecision

from src.risk.risk_manager import RiskManager, RiskAlert

from src.monitor.metrics import MetricsExporter, TradingMetrics, get_metrics_exporter
from src.monitor.alerts import AlertManager, get_alert_manager

logger = get_logger(__name__)
perf_logger = PerformanceLogger()


@dataclass
class TradingState:
    """Current trading state"""
    is_running: bool = False
    last_trade_time: Optional[datetime] = None
    current_session: TradingSession = TradingSession.CLOSED
    symbols_subscribed: List[str] = None

    def __post_init__(self):
        if self.symbols_subscribed is None:
            self.symbols_subscribed = []


class TradingEngine:
    """
    Main trading engine orchestrating the complete pipeline:
    1. Market data subscription (Futu OpenD)
    2. Data processing and technical indicators
    3. LLM-based decision making
    4. Order execution with slippage control
    5. Risk management and circuit breakers
    6. Monitoring and alerting

    Target: Full pipeline (quote -> model -> order) <= 1 second
    """

    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()

        # Initialize components
        self._init_logging()
        self._init_components()

        # Trading state
        self.state = TradingState()
        self._shutdown_event = asyncio.Event()

        # Trading loop interval (1 minute for 1-min klines)
        self.trading_interval = 60  # seconds

        logger.info("Trading Engine initialized")

    def _init_logging(self):
        """Initialize logging"""
        setup_logger(
            log_level=self.settings.log_level,
            log_file=self.settings.log_file,
            enable_console=True,
            enable_file=True
        )

    def _init_components(self):
        """Initialize all trading components"""
        # Symbol registry
        self.symbol_registry = get_symbol_registry()
        for symbol in self.settings.trading_symbols:
            self.symbol_registry.activate(symbol)

        # Data components
        self.quote_client = FutuQuoteClient(
            host=self.settings.futu_host,
            port=self.settings.futu_port,
            rsa_path=self.settings.futu_rsa_path
        )
        self.data_processor = DataProcessor()

        # Execution
        self.executor = FutuExecutor(
            host=self.settings.futu_host,
            port=self.settings.futu_port,
            trade_env=self.settings.futu_trade_env,
            trade_password=self.settings.futu_trade_password,
            rsa_path=self.settings.futu_rsa_path
        )

        # LLM Agent
        self.llm_agent = LLMAgent(
            provider=self.settings.llm_provider,
            temperature=self.settings.llm_temperature
        )

        # Risk Manager
        self.risk_manager = RiskManager(
            starting_equity=self.settings.default_position_size * 10,  # 10x position size as starting equity
            max_daily_drawdown=self.settings.max_daily_drawdown,
            max_total_drawdown=self.settings.max_total_drawdown
        )

        # Monitoring
        self.metrics = get_metrics_exporter()
        self.alerts = get_alert_manager()

        # Register risk alert handler
        self.risk_manager.on_alert(self._handle_risk_alert)

    def connect(self) -> bool:
        """Connect to Futu OpenD"""
        logger.info("Connecting to Futu OpenD...")

        # Connect quote client
        if not self.quote_client.connect():
            logger.error("Failed to connect quote client")
            return False

        self.metrics.update_connection_status("quote", True)

        # Connect trade client
        if not self.executor.connect():
            logger.error("Failed to connect trade client")
            return False

        self.metrics.update_connection_status("trade", True)

        logger.info("Connected to Futu OpenD successfully")
        return True

    def disconnect(self):
        """Disconnect from all services"""
        logger.info("Disconnecting from services...")

        self.quote_client.disconnect()
        self.executor.disconnect()

        self.metrics.update_connection_status("quote", False)
        self.metrics.update_connection_status("trade", False)

        logger.info("Disconnected from all services")

    def subscribe_symbols(self) -> bool:
        """Subscribe to market data for trading symbols"""
        symbols = self.symbol_registry.active_futu_codes

        if not symbols:
            logger.warning("No symbols to subscribe")
            return False

        logger.info(f"Subscribing to symbols: {symbols}")

        # Subscribe to quote and 1-min klines
        success = self.quote_client.subscribe(
            symbols=symbols,
            extended_hours=True
        )

        if success:
            self.state.symbols_subscribed = symbols

            # Register callbacks
            self.quote_client.on_quote(self._on_quote_update)
            self.quote_client.on_kline(self._on_kline_update)

            # Load historical data
            for symbol in symbols:
                klines = self.quote_client.get_history_kline(symbol, count=100)
                self.data_processor.load_history(symbol, klines)

            logger.info(f"Subscribed to {len(symbols)} symbols")

        return success

    def _on_quote_update(self, futu_code: str, quote: QuoteData):
        """Handle real-time quote update"""
        self.data_processor.update_quote(futu_code, quote)

        # Update symbol registry
        symbol = self.symbol_registry.get(futu_code)
        if symbol:
            symbol.update_quote(quote.last_price, quote.bid_price, quote.ask_price)

    def _on_kline_update(self, futu_code: str, kline: KLineData):
        """Handle real-time kline update"""
        self.data_processor.update_kline(futu_code, kline)

    def _handle_risk_alert(self, alert: RiskAlert):
        """Handle risk alerts from risk manager"""
        self.alerts.process_risk_alert(alert)
        self.metrics.record_risk_alert(alert.alert_type.value, alert.level.value)

        if alert.alert_type.value == "circuit_breaker":
            self.alerts.send_circuit_breaker_alert(
                alert.message,
                alert.value
            )

    async def run_trading_loop(self):
        """
        Main async trading loop.
        Runs every minute to process new klines.
        """
        logger.info("Starting trading loop...")
        self.state.is_running = True

        while not self._shutdown_event.is_set():
            try:
                loop_start = time.time()

                # Check market session
                self.state.current_session = SymbolRegistry.get_current_session()

                if self.state.current_session == TradingSession.CLOSED:
                    logger.debug("Market closed, waiting...")
                    await asyncio.sleep(60)
                    continue

                # Check if trading is allowed
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    logger.warning(f"Trading not allowed: {reason}")
                    await asyncio.sleep(60)
                    continue

                # Process each symbol
                for futu_code in self.state.symbols_subscribed:
                    await self._process_symbol(futu_code)

                # Update metrics
                self._update_metrics()

                loop_duration = time.time() - loop_start
                perf_logger.latency_record("trading_loop", loop_duration * 1000)

                # Wait for next interval
                sleep_time = max(0, self.trading_interval - loop_duration)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

        self.state.is_running = False
        logger.info("Trading loop stopped")

    async def _process_symbol(self, futu_code: str):
        """
        Process a single symbol through the trading pipeline.
        Target latency: < 1 second
        """
        pipeline_start = time.time()

        try:
            # Step 1: Get market snapshot
            quote_start = time.time()
            snapshot = self.data_processor.get_snapshot(
                futu_code,
                market_session=self.state.current_session.value
            )
            quote_time = (time.time() - quote_start) * 1000

            if not snapshot:
                logger.warning(f"No snapshot available for {futu_code}")
                return

            # Step 2: Get current position
            position = self.executor.get_position(futu_code)

            # Step 3: LLM analysis
            model_start = time.time()
            decision = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_agent.analyze(
                    snapshot=snapshot,
                    current_position=position,
                    portfolio_value=self.risk_manager.state.current_equity,
                    risk_budget=0.02
                )
            )
            model_time = (time.time() - model_start) * 1000

            # Record LLM decision
            self.metrics.record_llm_decision(
                decision.action.value,
                decision.confidence.value,
                decision.model_latency_ms
            )

            # Step 4: Execute if decision warrants action
            order_time = 0.0
            if decision.should_execute:
                order_start = time.time()
                await self._execute_decision(decision, snapshot, position)
                order_time = (time.time() - order_start) * 1000

            # Record pipeline timing
            total_time = (time.time() - pipeline_start) * 1000
            perf_logger.pipeline_timing(
                quote_ms=quote_time,
                model_ms=model_time,
                order_ms=order_time,
                total_ms=total_time,
                symbol=futu_code
            )
            self.metrics.record_pipeline_latency(total_time)

            # Check pipeline latency target
            if total_time > self.settings.max_pipeline_latency_s * 1000:
                self.alerts.record_anomaly("latency_spikes")
                logger.warning(f"Pipeline latency {total_time:.0f}ms exceeds target for {futu_code}")

        except Exception as e:
            logger.error(f"Error processing {futu_code}: {e}")

    async def _execute_decision(
        self,
        decision: TradingDecision,
        snapshot: MarketSnapshot,
        position: Position
    ):
        """Execute a trading decision"""
        symbol_info = self.symbol_registry.get(decision.futu_code)

        # Calculate position size
        if decision.suggested_quantity > 0:
            quantity = decision.suggested_quantity
        else:
            quantity = self.risk_manager.get_position_size(
                current_price=snapshot.last_price,
                atr=snapshot.indicators.atr_14,
                volatility_factor=symbol_info.volatility_factor if symbol_info else 1.0
            )

        if quantity <= 0:
            logger.warning(f"Calculated quantity is 0 for {decision.futu_code}")
            return

        # Execute based on action
        result = None

        if decision.action == TradingAction.LONG:
            if position.is_short:
                # Close short first
                close_results = self.executor.flat(decision.futu_code)
                for r in close_results:
                    self._record_order_result(r)

            result = self.executor.long(decision.futu_code, quantity)

        elif decision.action == TradingAction.SHORT:
            if position.is_long:
                # Close long first
                close_results = self.executor.flat(decision.futu_code)
                for r in close_results:
                    self._record_order_result(r)

            result = self.executor.short(decision.futu_code, quantity)

        elif decision.action == TradingAction.FLAT:
            results = self.executor.flat(decision.futu_code)
            for r in results:
                self._record_order_result(r)
            return

        if result:
            # Wait for fill
            result = self.executor.wait_for_fill(result.order_id, timeout=5.0)
            self._record_order_result(result)

    def _record_order_result(self, result: OrderResult):
        """Record order result for metrics and risk management"""
        # Record trade metrics
        self.metrics.record_trade(
            symbol=result.futu_code,
            action=result.action.value,
            status=result.status.value,
            latency_ms=result.latency_ms,
            slippage_pct=result.slippage * 100
        )

        # Record for risk management
        self.risk_manager.record_trade(result, result.requested_price)

        # Check slippage
        if result.slippage > self.settings.slippage_tolerance:
            self.alerts.record_anomaly("slippage_events")

        # Check fill rate
        self.risk_manager.record_fill_rate(result.filled_qty, result.requested_qty)

        # Check order latency
        if result.latency_ms > self.settings.max_order_latency_ms:
            logger.warning(f"Order latency {result.latency_ms:.2f}ms exceeds target {self.settings.max_order_latency_ms}ms")

    def _update_metrics(self):
        """Update Prometheus metrics"""
        positions = self.executor.get_all_positions()
        risk_state = self.risk_manager.state

        metrics = TradingMetrics(
            portfolio_value=risk_state.current_equity,
            daily_pnl=risk_state.daily_pnl,
            total_pnl=risk_state.realized_pnl + risk_state.unrealized_pnl,
            sharpe_ratio=risk_state.sharpe_ratio,
            max_drawdown=risk_state.max_drawdown,
            current_drawdown=risk_state.current_drawdown,
            win_rate=risk_state.win_rate,
            total_trades=risk_state.total_trades,
            daily_trades=risk_state.daily_trades,
            daily_volume=risk_state.daily_volume,
            avg_latency_ms=self.executor.avg_latency_ms,
            fill_rate=risk_state.fill_rate,
            avg_slippage=risk_state.avg_slippage,
            circuit_breaker_active=risk_state.circuit_breaker_active,
            open_positions=len([p for p in positions.values() if not p.is_flat])
        )

        self.metrics.update_portfolio(metrics)
        self.metrics.update_performance(metrics)
        self.metrics.update_risk(metrics)

        # Update position metrics
        for code, pos in positions.items():
            if not pos.is_flat:
                self.metrics.update_position(
                    symbol=code,
                    shares=pos.quantity,
                    value=pos.market_value,
                    unrealized_pnl=pos.unrealized_pnl
                )

    async def start(self):
        """Start the trading engine"""
        logger.info("=" * 60)
        logger.info("AI Futu Trader Starting")
        logger.info(f"Environment: {self.settings.futu_trade_env}")
        logger.info(f"Symbols: {self.settings.trading_symbols}")
        logger.info(f"LLM Provider: {self.settings.llm_provider}")
        logger.info("=" * 60)

        # Start metrics server
        self.metrics.start_server()

        # Connect to Futu
        if not self.connect():
            logger.error("Failed to connect to Futu OpenD")
            return

        # Subscribe to symbols
        if not self.subscribe_symbols():
            logger.error("Failed to subscribe to symbols")
            self.disconnect()
            return

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        # Run trading loop
        try:
            await self.run_trading_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the trading engine"""
        logger.info("Stopping trading engine...")

        # Signal shutdown
        self._shutdown_event.set()

        # Close all positions if configured
        # self.executor.flat_all()

        # Send daily summary
        stats = self.risk_manager.get_daily_stats()
        self.alerts.send_daily_summary(
            date=stats['date'],
            pnl=stats['daily_pnl'],
            trades=stats['daily_trades'],
            win_rate=stats['win_rate'] if stats['win_rate'] else 0,
            sharpe=stats['sharpe_ratio'],
            max_drawdown=stats['current_drawdown'],
            volume=stats['daily_volume'],
            fill_rate=stats['fill_rate'],
            avg_latency=self.executor.avg_latency_ms
        )

        # Disconnect
        self.disconnect()

        logger.info("Trading engine stopped")

    def _handle_shutdown(self):
        """Handle shutdown signal"""
        logger.info("Shutdown signal received")
        self._shutdown_event.set()


async def main():
    """Main entry point"""
    engine = TradingEngine()
    await engine.start()


if __name__ == "__main__":
    asyncio.run(main())
