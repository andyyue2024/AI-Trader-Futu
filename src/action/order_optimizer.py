"""
Order Optimizer - Ultra-low latency order execution optimization
Target: 0.0014s (1.4ms) order latency
"""
import time
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

from src.core.logger import get_logger, PerformanceLogger

logger = get_logger(__name__)
perf_logger = PerformanceLogger()


@dataclass
class LatencyMetrics:
    """Latency tracking metrics"""
    # Order latency
    avg_order_latency_ms: float = 0.0
    p50_order_latency_ms: float = 0.0
    p95_order_latency_ms: float = 0.0
    p99_order_latency_ms: float = 0.0
    min_order_latency_ms: float = float('inf')
    max_order_latency_ms: float = 0.0

    # Quote latency
    avg_quote_latency_ms: float = 0.0

    # Pipeline latency
    avg_pipeline_latency_ms: float = 0.0

    # Counts
    total_orders: int = 0
    successful_orders: int = 0


class LatencyTracker:
    """
    High-precision latency tracker for order execution.
    Uses performance counter for sub-millisecond accuracy.
    """

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self._order_latencies: deque = deque(maxlen=max_samples)
        self._quote_latencies: deque = deque(maxlen=max_samples)
        self._pipeline_latencies: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def record_order_latency(self, latency_ms: float, success: bool = True):
        """Record order execution latency"""
        with self._lock:
            self._order_latencies.append((latency_ms, success))

    def record_quote_latency(self, latency_ms: float):
        """Record quote processing latency"""
        with self._lock:
            self._quote_latencies.append(latency_ms)

    def record_pipeline_latency(self, latency_ms: float):
        """Record full pipeline latency"""
        with self._lock:
            self._pipeline_latencies.append(latency_ms)

    def get_metrics(self) -> LatencyMetrics:
        """Calculate current latency metrics"""
        with self._lock:
            metrics = LatencyMetrics()

            if self._order_latencies:
                latencies = [l[0] for l in self._order_latencies]
                successful = [l for l in self._order_latencies if l[1]]

                latencies.sort()
                n = len(latencies)

                metrics.avg_order_latency_ms = sum(latencies) / n
                metrics.min_order_latency_ms = latencies[0]
                metrics.max_order_latency_ms = latencies[-1]
                metrics.p50_order_latency_ms = latencies[n // 2]
                metrics.p95_order_latency_ms = latencies[int(n * 0.95)]
                metrics.p99_order_latency_ms = latencies[int(n * 0.99)]
                metrics.total_orders = n
                metrics.successful_orders = len(successful)

            if self._quote_latencies:
                metrics.avg_quote_latency_ms = sum(self._quote_latencies) / len(self._quote_latencies)

            if self._pipeline_latencies:
                metrics.avg_pipeline_latency_ms = sum(self._pipeline_latencies) / len(self._pipeline_latencies)

            return metrics

    def is_meeting_target(self, target_ms: float = 1.4) -> bool:
        """Check if meeting latency target"""
        metrics = self.get_metrics()
        return metrics.p95_order_latency_ms <= target_ms


class OrderQueue:
    """
    High-performance order queue with priority support.
    Enables batch processing and parallel execution.
    """

    def __init__(self, max_workers: int = 4):
        self._queue = queue.PriorityQueue()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._processing = False
        self._lock = threading.Lock()

    def submit(
        self,
        order_func: Callable,
        priority: int = 5,
        *args,
        **kwargs
    ) -> asyncio.Future:
        """
        Submit order for execution.

        Args:
            order_func: Order function to execute
            priority: Priority (1=highest, 10=lowest)
            *args, **kwargs: Arguments for order function

        Returns:
            Future for order result
        """
        future = asyncio.Future()
        self._queue.put((priority, time.time(), order_func, args, kwargs, future))
        return future

    def start_processing(self):
        """Start processing orders"""
        self._processing = True
        threading.Thread(target=self._process_loop, daemon=True).start()

    def stop_processing(self):
        """Stop processing orders"""
        self._processing = False

    def _process_loop(self):
        """Main processing loop"""
        while self._processing:
            try:
                item = self._queue.get(timeout=0.001)  # 1ms timeout
                priority, submit_time, func, args, kwargs, future = item

                # Execute order
                try:
                    result = func(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)

            except queue.Empty:
                continue


class ConnectionPool:
    """
    Connection pool for Futu OpenD to reduce connection overhead.
    Maintains persistent connections for faster order execution.
    """

    def __init__(self, host: str, port: int, pool_size: int = 3):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self._connections: List = []
        self._available: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

    def initialize(self):
        """Initialize connection pool"""
        from futu import OpenSecTradeContext, TrdMarket, SecurityFirm

        for i in range(self.pool_size):
            try:
                ctx = OpenSecTradeContext(
                    host=self.host,
                    port=self.port,
                    filter_trdmarket=TrdMarket.US,
                    security_firm=SecurityFirm.FUTUSECURITIES,
                )
                self._connections.append(ctx)
                self._available.put(ctx)
                logger.info(f"Connection {i+1}/{self.pool_size} established")
            except Exception as e:
                logger.error(f"Failed to create connection {i+1}: {e}")

    def acquire(self, timeout: float = 1.0):
        """Acquire a connection from pool"""
        try:
            return self._available.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("No connections available")

    def release(self, conn):
        """Release connection back to pool"""
        self._available.put(conn)

    def close_all(self):
        """Close all connections"""
        for conn in self._connections:
            try:
                conn.close()
            except:
                pass
        self._connections.clear()


class OrderOptimizer:
    """
    Order execution optimizer for ultra-low latency.
    Implements various optimization strategies to achieve 0.0014s target.
    """

    def __init__(
        self,
        target_latency_ms: float = 1.4,
        use_connection_pool: bool = True,
        use_order_queue: bool = True,
        pool_size: int = 3,
    ):
        self.target_latency_ms = target_latency_ms
        self.use_connection_pool = use_connection_pool
        self.use_order_queue = use_order_queue

        self.latency_tracker = LatencyTracker()
        self._connection_pool: Optional[ConnectionPool] = None
        self._order_queue: Optional[OrderQueue] = None

        # Pre-computed order templates for faster serialization
        self._order_templates: Dict[str, dict] = {}

        # Warm-up state
        self._warmed_up = False

    def initialize(self, host: str, port: int):
        """Initialize optimizer components"""
        if self.use_connection_pool:
            self._connection_pool = ConnectionPool(host, port)
            self._connection_pool.initialize()

        if self.use_order_queue:
            self._order_queue = OrderQueue()
            self._order_queue.start_processing()

    def warm_up(self):
        """
        Warm up the execution path.
        Pre-computes common operations to reduce first-order latency.
        """
        if self._warmed_up:
            return

        # Pre-compile common order parameters
        symbols = ["US.TQQQ", "US.QQQ", "US.SOXL", "US.SPXL"]
        for symbol in symbols:
            self._order_templates[symbol] = {
                "code": symbol,
                "trd_side": None,  # Will be set per-order
                "order_type": None,
                "qty": 0,
                "price": 0,
            }

        self._warmed_up = True
        logger.info("Order optimizer warmed up")

    def optimize_order_params(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float = 0,
    ) -> dict:
        """
        Get optimized order parameters.
        Uses pre-computed templates for faster serialization.
        """
        if symbol in self._order_templates:
            params = self._order_templates[symbol].copy()
        else:
            params = {"code": symbol}

        params.update({
            "qty": quantity,
            "price": price,
        })

        return params

    def get_latency_metrics(self) -> LatencyMetrics:
        """Get current latency metrics"""
        return self.latency_tracker.get_metrics()

    def is_meeting_target(self) -> bool:
        """Check if meeting latency target"""
        return self.latency_tracker.is_meeting_target(self.target_latency_ms)

    def shutdown(self):
        """Shutdown optimizer"""
        if self._order_queue:
            self._order_queue.stop_processing()
        if self._connection_pool:
            self._connection_pool.close_all()


class ExecutionTimer:
    """
    High-precision execution timer using performance counter.
    Provides sub-millisecond accuracy.
    """

    def __init__(self):
        self._start_time: float = 0
        self._checkpoints: Dict[str, float] = {}

    def start(self):
        """Start timer"""
        self._start_time = time.perf_counter()
        self._checkpoints.clear()

    def checkpoint(self, name: str):
        """Record checkpoint"""
        self._checkpoints[name] = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return (time.perf_counter() - self._start_time) * 1000

    def checkpoint_elapsed_ms(self, name: str) -> float:
        """Get elapsed time from start to checkpoint"""
        if name in self._checkpoints:
            return (self._checkpoints[name] - self._start_time) * 1000
        return 0.0

    def get_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown in milliseconds"""
        breakdown = {}
        prev_time = self._start_time

        for name, timestamp in sorted(self._checkpoints.items(), key=lambda x: x[1]):
            breakdown[name] = (timestamp - prev_time) * 1000
            prev_time = timestamp

        return breakdown


# Singleton instance
_optimizer: Optional[OrderOptimizer] = None


def get_order_optimizer() -> OrderOptimizer:
    """Get global order optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = OrderOptimizer()
    return _optimizer
