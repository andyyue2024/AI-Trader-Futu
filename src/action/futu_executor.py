"""
Futu Executor - Ultra-low latency order execution via Futu OpenD
Implements long/short/flat actions with slippage control and pre/after-hours support
Target latency: 0.0014s (1.4ms)
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from threading import Lock
import queue

from futu import (
    OpenSecTradeContext,
    TrdEnv,
    TrdSide,
    OrderType,
    OrderStatus as FutuOrderStatus,
    TrdMarket,
    SecurityFirm,
    RET_OK,
    RET_ERROR,
    TradeOrderHandlerBase,
    TradeDealHandlerBase,
)

from src.core.config import get_settings
from src.core.logger import get_logger, TradeLogger, PerformanceLogger

logger = get_logger(__name__)
perf_logger = PerformanceLogger()


class TradingAction(Enum):
    """Trading action types"""
    LONG = "long"      # Open/increase long position
    SHORT = "short"    # Open/increase short position
    FLAT = "flat"      # Close all positions
    HOLD = "hold"      # No action


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    symbol: str
    futu_code: str
    action: TradingAction
    side: str  # BUY or SELL

    # Order details
    requested_qty: int
    filled_qty: int = 0
    requested_price: float = 0.0
    avg_fill_price: float = 0.0

    # Status
    status: OrderStatus = OrderStatus.PENDING
    error_message: str = ""

    # Timing
    submit_time: datetime = field(default_factory=datetime.now)
    fill_time: Optional[datetime] = None
    latency_ms: float = 0.0

    # Slippage
    slippage: float = 0.0  # As decimal (0.001 = 0.1%)
    slippage_pct: float = 0.0

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_partial(self) -> bool:
        return self.status == OrderStatus.PARTIAL_FILLED

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.FAILED)

    @property
    def fill_rate(self) -> float:
        if self.requested_qty > 0:
            return self.filled_qty / self.requested_qty
        return 0.0

    def calculate_slippage(self, reference_price: float):
        """Calculate slippage from reference price"""
        if reference_price > 0 and self.avg_fill_price > 0:
            self.slippage = abs(self.avg_fill_price - reference_price) / reference_price
            self.slippage_pct = self.slippage * 100


@dataclass
class Position:
    """Current position for a symbol"""
    symbol: str
    futu_code: str
    quantity: int = 0  # Positive for long, negative for short
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def abs_quantity(self) -> int:
        return abs(self.quantity)


class OrderUpdateHandler(TradeOrderHandlerBase):
    """Handler for order status updates"""

    def __init__(self, callback: Callable[[str, dict], None]):
        super().__init__()
        self.callback = callback

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret == RET_OK and data is not None:
            for _, row in data.iterrows():
                order_info = {
                    'order_id': str(row.get('order_id', '')),
                    'code': row.get('code', ''),
                    'qty': int(row.get('qty', 0)),
                    'dealt_qty': int(row.get('dealt_qty', 0)),
                    'price': float(row.get('price', 0)),
                    'dealt_avg_price': float(row.get('dealt_avg_price', 0)),
                    'order_status': row.get('order_status', ''),
                    'trd_side': row.get('trd_side', ''),
                }
                self.callback(order_info['order_id'], order_info)
        return ret, data


class DealUpdateHandler(TradeDealHandlerBase):
    """Handler for trade deal updates (fills)"""

    def __init__(self, callback: Callable[[str, dict], None]):
        super().__init__()
        self.callback = callback

    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret == RET_OK and data is not None:
            for _, row in data.iterrows():
                deal_info = {
                    'order_id': str(row.get('order_id', '')),
                    'deal_id': str(row.get('deal_id', '')),
                    'code': row.get('code', ''),
                    'qty': int(row.get('qty', 0)),
                    'price': float(row.get('price', 0)),
                    'trd_side': row.get('trd_side', ''),
                    'create_time': row.get('create_time', ''),
                }
                self.callback(deal_info['order_id'], deal_info)
        return ret, data


class FutuExecutor:
    """
    Futu OpenD order executor with ultra-low latency focus.
    Implements long/short/flat actions with slippage monitoring.
    """

    # Map Futu order status to our status
    STATUS_MAP = {
        'NONE': OrderStatus.PENDING,
        'UNSUBMITTED': OrderStatus.PENDING,
        'WAITING_SUBMIT': OrderStatus.PENDING,
        'SUBMITTING': OrderStatus.SUBMITTED,
        'SUBMITTED': OrderStatus.SUBMITTED,
        'FILLED_PART': OrderStatus.PARTIAL_FILLED,
        'FILLED_ALL': OrderStatus.FILLED,
        'CANCELLED_PART': OrderStatus.CANCELLED,
        'CANCELLED_ALL': OrderStatus.CANCELLED,
        'FAILED': OrderStatus.FAILED,
        'DISABLED': OrderStatus.REJECTED,
        'DELETED': OrderStatus.CANCELLED,
    }

    def __init__(
        self,
        host: str = None,
        port: int = None,
        trade_env: str = None,
        trade_password: str = None,
        rsa_path: str = None,
        market: TrdMarket = TrdMarket.US,
    ):
        settings = get_settings()
        self.host = host or settings.futu_host
        self.port = port or settings.futu_port
        self.trade_env = self._parse_env(trade_env or settings.futu_trade_env)
        self.trade_password = trade_password or settings.futu_trade_password
        self.rsa_path = rsa_path or settings.futu_rsa_path
        self.market = market

        self._context: Optional[OpenSecTradeContext] = None
        self._trade_logger = TradeLogger()

        # Order tracking
        self._pending_orders: Dict[str, OrderResult] = {}
        self._completed_orders: Dict[str, OrderResult] = {}
        self._positions: Dict[str, Position] = {}

        # Threading
        self._lock = Lock()
        self._order_events: Dict[str, asyncio.Event] = {}

        # Performance tracking
        self._latency_samples: List[float] = []
        self._fill_count = 0
        self._total_orders = 0

        # Slippage tolerance
        self.max_slippage = settings.slippage_tolerance

    def _parse_env(self, env_str: str) -> TrdEnv:
        """Parse trading environment string"""
        if env_str.upper() == "REAL":
            return TrdEnv.REAL
        return TrdEnv.SIMULATE

    def connect(self) -> bool:
        """Establish connection to Futu OpenD trade context"""
        try:
            start_time = time.time()

            if self.rsa_path:
                self._context = OpenSecTradeContext(
                    host=self.host,
                    port=self.port,
                    filter_trdmarket=self.market,
                    security_firm=SecurityFirm.FUTUSECURITIES,
                    is_encrypt=True,
                    rsa_file=self.rsa_path
                )
            else:
                self._context = OpenSecTradeContext(
                    host=self.host,
                    port=self.port,
                    filter_trdmarket=self.market,
                    security_firm=SecurityFirm.FUTUSECURITIES,
                )

            # Set handlers for order updates
            self._context.set_handler(OrderUpdateHandler(self._on_order_update))
            self._context.set_handler(DealUpdateHandler(self._on_deal_update))

            # Unlock trade if password provided
            if self.trade_password and self.trade_env == TrdEnv.REAL:
                ret, data = self._context.unlock_trade(self.trade_password)
                if ret != RET_OK:
                    logger.error(f"Failed to unlock trade: {data}")
                    return False

            # Get account list
            ret, acc_list = self._context.get_acc_list()
            if ret != RET_OK:
                logger.error(f"Failed to get account list: {acc_list}")
                return False

            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Connected to Futu Trade API in {latency_ms:.2f}ms, env={self.trade_env}")
            perf_logger.latency_record("trade_connect", latency_ms)

            # Load current positions
            self._refresh_positions()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Futu Trade API: {e}")
            return False

    def disconnect(self):
        """Disconnect from Futu OpenD"""
        if self._context:
            self._context.close()
            self._context = None
        logger.info("Disconnected from Futu Trade API")

    def _refresh_positions(self):
        """Refresh current positions from broker"""
        if not self._context:
            return

        try:
            ret, data = self._context.position_list_query(
                trd_env=self.trade_env,
                refresh_cache=True
            )

            if ret != RET_OK:
                logger.error(f"Failed to query positions: {data}")
                return

            with self._lock:
                self._positions.clear()
                if data is not None and len(data) > 0:
                    for _, row in data.iterrows():
                        code = row['code']
                        position = Position(
                            symbol=code.split('.')[-1],
                            futu_code=code,
                            quantity=int(row.get('qty', 0)),
                            avg_cost=float(row.get('cost_price', 0)),
                            market_value=float(row.get('market_val', 0)),
                            unrealized_pnl=float(row.get('pl_val', 0)),
                            realized_pnl=float(row.get('realized_pl', 0) if 'realized_pl' in row else 0),
                        )
                        self._positions[code] = position

        except Exception as e:
            logger.error(f"Error refreshing positions: {e}")

    def get_position(self, futu_code: str) -> Position:
        """Get current position for symbol"""
        with self._lock:
            if futu_code not in self._positions:
                return Position(
                    symbol=futu_code.split('.')[-1],
                    futu_code=futu_code
                )
            return self._positions[futu_code]

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        self._refresh_positions()
        with self._lock:
            return dict(self._positions)

    def long(
        self,
        futu_code: str,
        quantity: int,
        price: float = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        """
        Open/increase long position.

        Args:
            futu_code: Symbol in Futu format (e.g., "US.TQQQ")
            quantity: Number of shares to buy
            price: Limit price (None for market order)
            order_type: Order type (MARKET, NORMAL/LIMIT, etc.)

        Returns:
            OrderResult with execution details
        """
        return self._place_order(
            futu_code=futu_code,
            side=TrdSide.BUY,
            quantity=quantity,
            price=price,
            order_type=order_type,
            action=TradingAction.LONG
        )

    def short(
        self,
        futu_code: str,
        quantity: int,
        price: float = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        """
        Open/increase short position.

        Args:
            futu_code: Symbol in Futu format
            quantity: Number of shares to short sell
            price: Limit price (None for market order)
            order_type: Order type

        Returns:
            OrderResult with execution details
        """
        return self._place_order(
            futu_code=futu_code,
            side=TrdSide.SELL_SHORT,
            quantity=quantity,
            price=price,
            order_type=order_type,
            action=TradingAction.SHORT
        )

    def flat(self, futu_code: str) -> List[OrderResult]:
        """
        Close all positions for a symbol.

        Args:
            futu_code: Symbol in Futu format

        Returns:
            List of OrderResult for closing orders
        """
        results = []
        position = self.get_position(futu_code)

        if position.is_flat:
            logger.info(f"No position to close for {futu_code}")
            return results

        if position.is_long:
            # Sell to close long
            result = self._place_order(
                futu_code=futu_code,
                side=TrdSide.SELL,
                quantity=position.abs_quantity,
                price=None,
                order_type=OrderType.MARKET,
                action=TradingAction.FLAT
            )
            results.append(result)

        elif position.is_short:
            # Buy to cover short
            result = self._place_order(
                futu_code=futu_code,
                side=TrdSide.BUY_BACK,
                quantity=position.abs_quantity,
                price=None,
                order_type=OrderType.MARKET,
                action=TradingAction.FLAT
            )
            results.append(result)

        return results

    def flat_all(self) -> List[OrderResult]:
        """Close all positions across all symbols"""
        results = []
        for futu_code in list(self._positions.keys()):
            results.extend(self.flat(futu_code))
        return results

    def _place_order(
        self,
        futu_code: str,
        side: TrdSide,
        quantity: int,
        price: float = None,
        order_type: OrderType = OrderType.MARKET,
        action: TradingAction = TradingAction.LONG,
    ) -> OrderResult:
        """
        Internal order placement with latency tracking.
        """
        if not self._context:
            logger.error("Not connected to Futu Trade API")
            return OrderResult(
                order_id=str(uuid.uuid4()),
                symbol=futu_code.split('.')[-1],
                futu_code=futu_code,
                action=action,
                side=str(side),
                requested_qty=quantity,
                status=OrderStatus.FAILED,
                error_message="Not connected"
            )

        order_id = str(uuid.uuid4())
        start_time = time.time()

        # Create order result
        result = OrderResult(
            order_id=order_id,
            symbol=futu_code.split('.')[-1],
            futu_code=futu_code,
            action=action,
            side=str(side),
            requested_qty=quantity,
            requested_price=price or 0.0,
            submit_time=datetime.now()
        )

        try:
            # For market orders, we don't need a price
            if order_type == OrderType.MARKET:
                actual_price = 0
            else:
                if price is None:
                    raise ValueError("Price required for non-market orders")
                actual_price = price

            self._trade_logger.order_placed(
                order_id=order_id,
                action=str(action.value),
                quantity=quantity,
                price=actual_price,
                order_type=str(order_type),
                symbol=futu_code
            )

            # Place order via Futu API
            ret, data = self._context.place_order(
                price=actual_price,
                qty=quantity,
                code=futu_code,
                trd_side=side,
                order_type=order_type,
                trd_env=self.trade_env,
                adjust_limit=0.05 if order_type == OrderType.MARKET else 0,  # 5% price protection
            )

            latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = latency_ms

            if ret != RET_OK:
                result.status = OrderStatus.FAILED
                result.error_message = str(data)
                self._trade_logger.order_rejected(order_id, str(data))
                logger.error(f"Order placement failed: {data}")
            else:
                # Extract Futu order ID
                if data is not None and len(data) > 0:
                    futu_order_id = str(data.iloc[0].get('order_id', order_id))
                    result.order_id = futu_order_id
                    result.status = OrderStatus.SUBMITTED

                    with self._lock:
                        self._pending_orders[futu_order_id] = result

                    logger.info(f"Order submitted: {futu_order_id} in {latency_ms:.2f}ms")

            # Track latency
            self._latency_samples.append(latency_ms)
            self._total_orders += 1
            perf_logger.latency_record("order_submit", latency_ms, success=(ret == RET_OK))

        except Exception as e:
            result.status = OrderStatus.FAILED
            result.error_message = str(e)
            result.latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Order placement error: {e}")

        return result

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if not self._context:
            return False

        try:
            ret, data = self._context.modify_order(
                modify_order_op=1,  # Cancel
                order_id=order_id,
                qty=0,
                price=0,
                trd_env=self.trade_env
            )

            if ret == RET_OK:
                self._trade_logger.order_cancelled(order_id)
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}: {data}")
                return False

        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return False

    def cancel_all_orders(self, futu_code: str = None) -> int:
        """Cancel all pending orders, optionally filtered by symbol"""
        cancelled = 0

        with self._lock:
            for order_id, order in list(self._pending_orders.items()):
                if futu_code is None or order.futu_code == futu_code:
                    if self.cancel_order(order_id):
                        cancelled += 1

        return cancelled

    def _on_order_update(self, order_id: str, order_info: dict):
        """Handle order status update callback"""
        with self._lock:
            if order_id in self._pending_orders:
                order = self._pending_orders[order_id]

                futu_status = order_info.get('order_status', 'NONE')
                order.status = self.STATUS_MAP.get(futu_status, OrderStatus.PENDING)
                order.filled_qty = order_info.get('dealt_qty', 0)

                if order_info.get('dealt_avg_price', 0) > 0:
                    order.avg_fill_price = order_info['dealt_avg_price']
                    order.calculate_slippage(order.requested_price if order.requested_price > 0 else order.avg_fill_price)

                if order.is_complete:
                    order.fill_time = datetime.now()
                    self._completed_orders[order_id] = order
                    del self._pending_orders[order_id]

                    if order.is_filled:
                        self._fill_count += 1
                        self._trade_logger.order_filled(
                            order_id=order_id,
                            fill_price=order.avg_fill_price,
                            fill_qty=order.filled_qty,
                            slippage=order.slippage,
                            latency_ms=order.latency_ms
                        )

                    # Refresh positions after fill
                    self._refresh_positions()

    def _on_deal_update(self, order_id: str, deal_info: dict):
        """Handle trade deal (fill) callback"""
        logger.debug(f"Deal update for order {order_id}: {deal_info}")

    def wait_for_fill(self, order_id: str, timeout: float = 5.0) -> OrderResult:
        """Wait for order to be filled with timeout"""
        start = time.time()

        while time.time() - start < timeout:
            with self._lock:
                if order_id in self._completed_orders:
                    return self._completed_orders[order_id]
                if order_id in self._pending_orders:
                    order = self._pending_orders[order_id]
                    if order.is_complete:
                        return order

            time.sleep(0.01)  # 10ms polling

        # Timeout - return current state
        with self._lock:
            if order_id in self._pending_orders:
                return self._pending_orders[order_id]
            if order_id in self._completed_orders:
                return self._completed_orders[order_id]

        return OrderResult(
            order_id=order_id,
            symbol="",
            futu_code="",
            action=TradingAction.HOLD,
            side="",
            requested_qty=0,
            status=OrderStatus.FAILED,
            error_message="Order not found"
        )

    @property
    def avg_latency_ms(self) -> float:
        """Get average order latency in milliseconds"""
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)

    @property
    def fill_rate(self) -> float:
        """Get overall fill rate"""
        if self._total_orders == 0:
            return 0.0
        return self._fill_count / self._total_orders

    @property
    def is_connected(self) -> bool:
        return self._context is not None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class AsyncFutuExecutor:
    """
    Async wrapper for FutuExecutor.
    Provides async/await interface for use in async trading loops.
    """

    def __init__(self, *args, **kwargs):
        self._executor = FutuExecutor(*args, **kwargs)

    async def connect(self) -> bool:
        """Async connect"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._executor.connect
        )

    async def disconnect(self):
        """Async disconnect"""
        await asyncio.get_event_loop().run_in_executor(
            None, self._executor.disconnect
        )

    async def long(self, *args, **kwargs) -> OrderResult:
        """Async long order"""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._executor.long(*args, **kwargs)
        )

    async def short(self, *args, **kwargs) -> OrderResult:
        """Async short order"""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._executor.short(*args, **kwargs)
        )

    async def flat(self, futu_code: str) -> List[OrderResult]:
        """Async flat position"""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._executor.flat(futu_code)
        )

    async def flat_all(self) -> List[OrderResult]:
        """Async flat all positions"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._executor.flat_all
        )

    async def wait_for_fill(self, order_id: str, timeout: float = 5.0) -> OrderResult:
        """Async wait for fill"""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._executor.wait_for_fill(order_id, timeout)
        )

    def get_position(self, futu_code: str) -> Position:
        """Get position (no async needed for cached data)"""
        return self._executor.get_position(futu_code)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self._executor.get_all_positions()

    @property
    def avg_latency_ms(self) -> float:
        return self._executor.avg_latency_ms

    @property
    def fill_rate(self) -> float:
        return self._executor.fill_rate

    @property
    def is_connected(self) -> bool:
        return self._executor.is_connected

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
