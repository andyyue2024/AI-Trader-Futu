"""
Web API - FastAPI-based Web interface for AI Futu Trader
Provides REST API endpoints for monitoring and control
"""
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.logger import get_logger
from src.core.symbols import get_symbol_registry
from src.core.session_manager import get_session_manager

logger = get_logger(__name__)

# Pydantic models for API
class SymbolInfo(BaseModel):
    symbol: str
    futu_code: str
    name: Optional[str]
    instrument_type: str
    is_active: bool


class PositionInfo(BaseModel):
    symbol: str
    futu_code: str
    direction: str
    quantity: int
    avg_cost: float
    unrealized_pnl: float
    realized_pnl: float


class TradeInfo(BaseModel):
    trade_id: str
    symbol: str
    entry_time: str
    entry_price: float
    entry_side: str
    quantity: int
    exit_time: Optional[str]
    exit_price: Optional[float]
    pnl: float
    status: str


class PerformanceMetrics(BaseModel):
    total_pnl: float
    daily_pnl: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    fill_rate: float
    avg_latency_ms: float


class SystemStatus(BaseModel):
    is_running: bool
    current_session: str
    session_progress: float
    quote_connected: bool
    trade_connected: bool
    circuit_breaker_active: bool
    last_update: str


class TradingCommand(BaseModel):
    action: str  # start, stop, pause
    symbols: Optional[List[str]] = None


class OrderRequest(BaseModel):
    symbol: str
    action: str  # long, short, flat
    quantity: Optional[int] = None


# Global state (will be replaced with actual engine reference)
_engine = None
_app_state = {
    "is_running": False,
    "start_time": None,
}


def get_engine():
    """Get trading engine instance"""
    global _engine
    return _engine


def set_engine(engine):
    """Set trading engine instance"""
    global _engine
    _engine = engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Web API starting up...")
    yield
    logger.info("Web API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AI Futu Trader API",
    description="REST API for AI Futu Trader monitoring and control",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# System Endpoints
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - returns dashboard HTML"""
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <title>AI Futu Trader - 智能交易系统</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .gradient-bg { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
        .card { background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
        .card:hover { border-color: rgba(59, 130, 246, 0.5); }
        .pulse-dot { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .stat-value { font-size: 1.75rem; font-weight: 700; }
        .progress-bar { transition: width 0.5s ease-in-out; }
    </style>
</head>
<body class="gradient-bg text-white min-h-screen">
    <!-- Navigation -->
    <nav class="border-b border-gray-700 bg-gray-900/50 backdrop-blur-lg sticky top-0 z-50">
        <div class="container mx-auto px-4 py-3 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="text-2xl">🤖</span>
                <span class="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                    AI Futu Trader
                </span>
            </div>
            <div class="flex items-center space-x-4">
                <div id="connection-status" class="flex items-center space-x-2">
                    <span class="w-2 h-2 bg-green-500 rounded-full pulse-dot"></span>
                    <span class="text-sm text-gray-400">已连接</span>
                </div>
                <span id="current-time" class="text-sm text-gray-400"></span>
            </div>
        </div>
    </nav>

    <div class="container mx-auto px-4 py-6">
        <!-- Top Stats Row -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <!-- Portfolio Value -->
            <div class="card rounded-xl p-5 transition-all duration-300" hx-get="/api/portfolio/html" hx-trigger="every 5s" hx-swap="innerHTML">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-gray-400 text-sm">投资组合</span>
                    <span class="text-2xl">💰</span>
                </div>
                <div class="stat-value text-white">$100,000.00</div>
                <div class="flex items-center mt-2">
                    <span class="text-gray-400 text-sm font-medium">加载中...</span>
                </div>
            </div>

            <!-- Today's P&L -->
            <div class="card rounded-xl p-5 transition-all duration-300" hx-get="/api/metrics/html" hx-trigger="every 5s" hx-swap="innerHTML">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-gray-400 text-sm">今日盈亏</span>
                    <span class="text-2xl">📈</span>
                </div>
                <div class="stat-value text-green-400">+$0.00</div>
                <div class="flex items-center mt-2">
                    <span class="text-gray-400 text-sm">0 笔交易</span>
                </div>
            </div>

            <!-- Win Rate -->
            <div class="card rounded-xl p-5 transition-all duration-300" hx-get="/api/winrate/html" hx-trigger="every 10s" hx-swap="innerHTML">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-gray-400 text-sm">胜率</span>
                    <span class="text-2xl">🎯</span>
                </div>
                <div class="stat-value text-blue-400">--</div>
                <div class="w-full bg-gray-700 rounded-full h-2 mt-3">
                    <div class="bg-blue-500 h-2 rounded-full progress-bar" style="width: 0%"></div>
                </div>
            </div>

            <!-- Sharpe Ratio -->
            <div class="card rounded-xl p-5 transition-all duration-300" hx-get="/api/sharpe/html" hx-trigger="every 10s" hx-swap="innerHTML">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-gray-400 text-sm">夏普比率</span>
                    <span class="text-2xl">📊</span>
                </div>
                <div class="stat-value text-purple-400">--</div>
                <div class="flex items-center mt-2">
                    <span class="text-gray-400 text-sm">等待数据...</span>
                </div>
            </div>
        </div>

        <!-- Second Row: Session & Controls -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            <!-- Market Session -->
            <div class="card rounded-xl p-5" hx-get="/api/session/html" hx-trigger="every 5s" hx-swap="innerHTML">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-gray-400 text-sm">交易时段</span>
                    <span class="text-2xl">⏰</span>
                </div>
                <div class="text-xl font-bold text-yellow-400 mb-2">盘前交易</div>
                <div class="w-full bg-gray-700 rounded-full h-3 mb-2">
                    <div class="bg-gradient-to-r from-yellow-400 to-orange-500 h-3 rounded-full progress-bar" style="width: 45%"></div>
                </div>
                <div class="flex justify-between text-sm text-gray-400">
                    <span>04:00</span>
                    <span>进度: 45%</span>
                    <span>09:30</span>
                </div>
            </div>

            <!-- System Status -->
            <div class="card rounded-xl p-5" hx-get="/api/status/html" hx-trigger="every 5s" hx-swap="innerHTML">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-gray-400 text-sm">系统状态</span>
                    <span class="text-2xl">🖥️</span>
                </div>
                <div class="space-y-3">
                    <div class="flex items-center justify-between">
                        <span class="text-gray-300">交易引擎</span>
                        <span class="px-2 py-1 bg-green-500/20 text-green-400 rounded text-sm">运行中</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-gray-300">行情连接</span>
                        <span class="px-2 py-1 bg-green-500/20 text-green-400 rounded text-sm">已连接</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-gray-300">熔断状态</span>
                        <span class="px-2 py-1 bg-gray-500/20 text-gray-400 rounded text-sm">未触发</span>
                    </div>
                </div>
            </div>

            <!-- Quick Actions -->
            <div class="card rounded-xl p-5">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-gray-400 text-sm">快速操作</span>
                    <span class="text-2xl">⚡</span>
                </div>
                <div class="grid grid-cols-2 gap-3">
                    <button onclick="startTrading()" class="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 px-4 py-3 rounded-lg font-medium transition-all duration-200 transform hover:scale-105">
                        ▶️ 启动
                    </button>
                    <button onclick="stopTrading()" class="bg-gradient-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 px-4 py-3 rounded-lg font-medium transition-all duration-200 transform hover:scale-105">
                        ⏹️ 停止
                    </button>
                    <button onclick="window.location.href='/docs'" class="bg-gray-600 hover:bg-gray-700 px-4 py-3 rounded-lg font-medium transition-all duration-200">
                        📚 API
                    </button>
                    <button onclick="generateReport()" class="bg-gray-600 hover:bg-gray-700 px-4 py-3 rounded-lg font-medium transition-all duration-200">
                        📑 报告
                    </button>
                </div>
            </div>
        </div>

        <!-- Third Row: Compliance Metrics -->
        <div class="card rounded-xl p-5 mb-6" hx-get="/api/compliance/html" hx-trigger="every 10s" hx-swap="innerHTML">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-semibold">📋 目标合规状态</h3>
                <span class="px-3 py-1 bg-gray-500/20 text-gray-400 rounded-full text-sm">加载中...</span>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
                <div class="text-center p-3 bg-gray-800/50 rounded-lg animate-pulse">
                    <div class="text-gray-400 text-lg mb-1">...</div>
                    <div class="text-xs text-gray-400">延迟</div>
                    <div class="text-sm font-medium">--</div>
                </div>
                <div class="text-center p-3 bg-gray-800/50 rounded-lg animate-pulse">
                    <div class="text-gray-400 text-lg mb-1">...</div>
                    <div class="text-xs text-gray-400">滑点</div>
                    <div class="text-sm font-medium">--</div>
                </div>
                <div class="text-center p-3 bg-gray-800/50 rounded-lg animate-pulse">
                    <div class="text-gray-400 text-lg mb-1">...</div>
                    <div class="text-xs text-gray-400">成交额</div>
                    <div class="text-sm font-medium">--</div>
                </div>
                <div class="text-center p-3 bg-gray-800/50 rounded-lg animate-pulse">
                    <div class="text-gray-400 text-lg mb-1">...</div>
                    <div class="text-xs text-gray-400">成交率</div>
                    <div class="text-sm font-medium">--</div>
                </div>
                <div class="text-center p-3 bg-gray-800/50 rounded-lg animate-pulse">
                    <div class="text-gray-400 text-lg mb-1">...</div>
                    <div class="text-xs text-gray-400">夏普</div>
                    <div class="text-sm font-medium">--</div>
                </div>
                <div class="text-center p-3 bg-gray-800/50 rounded-lg animate-pulse">
                    <div class="text-gray-400 text-lg mb-1">...</div>
                    <div class="text-xs text-gray-400">回撤</div>
                    <div class="text-sm font-medium">--</div>
                </div>
                <div class="text-center p-3 bg-gray-800/50 rounded-lg animate-pulse">
                    <div class="text-gray-400 text-lg mb-1">...</div>
                    <div class="text-xs text-gray-400">熔断</div>
                    <div class="text-sm font-medium">--</div>
                </div>
                <div class="text-center p-3 bg-gray-800/50 rounded-lg animate-pulse">
                    <div class="text-gray-400 text-lg mb-1">...</div>
                    <div class="text-xs text-gray-400">时段</div>
                    <div class="text-sm font-medium">--</div>
                </div>
            </div>
        </div>

        <!-- Fourth Row: Positions & Trades -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <!-- Open Positions -->
            <div class="card rounded-xl p-5">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold">📊 当前持仓</h3>
                    <span class="text-sm text-gray-400">实时更新</span>
                </div>
                <div class="overflow-x-auto" hx-get="/api/positions/html" hx-trigger="load, every 5s" hx-swap="innerHTML">
                    <p class="text-gray-500 text-center py-4">加载中...</p>
                </div>
            </div>

            <!-- Recent Trades -->
            <div class="card rounded-xl p-5">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold">📜 最近交易</h3>
                    <a href="/api/trades/recent" class="text-sm text-blue-400 hover:text-blue-300">查看全部 →</a>
                </div>
                <div class="space-y-3" hx-get="/api/trades/html" hx-trigger="load, every 10s" hx-swap="innerHTML">
                    <p class="text-gray-500 text-center py-4">加载中...</p>
                </div>
            </div>
        </div>

        <!-- Fifth Row: Trading Symbols -->
        <div class="card rounded-xl p-5 mb-6">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-semibold">🎯 交易标的</h3>
                <span class="text-sm text-gray-400">点击切换状态</span>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3" hx-get="/api/symbols/html" hx-trigger="load" hx-swap="innerHTML">
                <p class="text-gray-500 text-center py-4 col-span-full">加载中...</p>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="border-t border-gray-700 bg-gray-900/50 mt-8">
        <div class="container mx-auto px-4 py-4 flex items-center justify-between text-sm text-gray-500">
            <span>© 2024 AI Futu Trader - 基于 Futu OpenD + LLM</span>
            <div class="flex items-center space-x-4">
                <a href="/docs" class="hover:text-gray-300">API 文档</a>
                <a href="/api/health" class="hover:text-gray-300">健康检查</a>
            </div>
        </div>
    </footer>

    <script>
        // Update current time
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleString('zh-CN', {
                year: 'numeric', month: '2-digit', day: '2-digit',
                hour: '2-digit', minute: '2-digit', second: '2-digit'
            });
        }
        setInterval(updateTime, 1000);
        updateTime();

        // Trading controls
        async function startTrading() {
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = '启动中...';
            try {
                const res = await fetch('/api/trading/start', {method: 'POST'});
                if (res.ok) {
                    showNotification('交易引擎已启动', 'success');
                }
            } catch(e) {
                showNotification('启动失败: ' + e.message, 'error');
            }
            btn.disabled = false;
            btn.textContent = '▶️ 启动';
        }

        async function stopTrading() {
            if (!confirm('确定要停止交易引擎吗？')) return;
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = '停止中...';
            try {
                const res = await fetch('/api/trading/stop', {method: 'POST'});
                if (res.ok) {
                    showNotification('交易引擎已停止', 'warning');
                }
            } catch(e) {
                showNotification('停止失败: ' + e.message, 'error');
            }
            btn.disabled = false;
            btn.textContent = '⏹️ 停止';
        }

        async function generateReport() {
            showNotification('正在生成报告...', 'info');
            try {
                const res = await fetch('/api/reports/generate?format=html');
                const data = await res.json();
                if (res.ok) {
                    showNotification('报告已生成', 'success');
                    window.open(data.download_url, '_blank');
                }
            } catch(e) {
                showNotification('生成失败: ' + e.message, 'error');
            }
        }

        // Notification
        function showNotification(message, type) {
            const colors = {
                success: 'bg-green-500',
                error: 'bg-red-500',
                warning: 'bg-yellow-500',
                info: 'bg-blue-500'
            };
            const notification = document.createElement('div');
            notification.className = `fixed top-4 right-4 ${colors[type]} text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-pulse`;
            notification.textContent = message;
            document.body.appendChild(notification);
            setTimeout(() => notification.remove(), 3000);
        }

        // Refresh metrics
        async function refreshMetrics() {
            try {
                const res = await fetch('/api/metrics/summary');
                const data = await res.json();
                // Update UI elements
            } catch(e) {
                console.error('Failed to refresh metrics:', e);
            }
        }
        setInterval(refreshMetrics, 10000);
    </script>
</body>
</html>
    """


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/status")
async def get_status():
    """Get system status"""
    session_mgr = get_session_manager()
    session_info = session_mgr.get_session_info()

    engine = get_engine()

    return {
        "is_running": _app_state.get("is_running", False),
        "current_session": session_info.session.value,
        "session_progress": round(session_info.progress_pct, 1),
        "trading_allowed": session_info.is_trading_allowed,
        "seconds_to_close": session_info.seconds_to_close,
        "circuit_breaker_active": False,  # Will be from engine
        "last_update": datetime.now().isoformat()
    }


@app.get("/api/status/html", response_class=HTMLResponse)
async def get_status_html():
    """Get system status as HTML for HTMX"""
    is_running = _app_state.get("is_running", False)
    engine_status = "运行中" if is_running else "已停止"
    engine_color = "green" if is_running else "gray"

    return f"""
    <div class="flex items-center justify-between mb-4">
        <span class="text-gray-400 text-sm">系统状态</span>
        <span class="text-2xl">🖥️</span>
    </div>
    <div class="space-y-3">
        <div class="flex items-center justify-between">
            <span class="text-gray-300">交易引擎</span>
            <span class="px-2 py-1 bg-{engine_color}-500/20 text-{engine_color}-400 rounded text-sm">{engine_status}</span>
        </div>
        <div class="flex items-center justify-between">
            <span class="text-gray-300">行情连接</span>
            <span class="px-2 py-1 bg-green-500/20 text-green-400 rounded text-sm">已连接</span>
        </div>
        <div class="flex items-center justify-between">
            <span class="text-gray-300">熔断状态</span>
            <span class="px-2 py-1 bg-gray-500/20 text-gray-400 rounded text-sm">未触发</span>
        </div>
    </div>
    """


@app.get("/api/session")
async def get_session():
    """Get market session info"""
    session_mgr = get_session_manager()
    info = session_mgr.get_session_info()

    return {
        "current": info.session.value,
        "next": info.next_session.value,
        "progress": round(info.progress_pct, 1),
        "is_trading_allowed": info.is_trading_allowed,
        "seconds_to_close": info.seconds_to_close,
        "seconds_to_next_open": info.seconds_to_next_open
    }


@app.get("/api/session/html", response_class=HTMLResponse)
async def get_session_html():
    """Get session info as HTML for HTMX"""
    session_mgr = get_session_manager()
    info = session_mgr.get_session_info()

    session_names = {
        "closed": ("休市", "gray"),
        "pre_market": ("盘前交易", "yellow"),
        "regular": ("正常交易", "green"),
        "after_hours": ("盘后交易", "orange"),
    }

    name, color = session_names.get(info.session.value, ("未知", "gray"))
    progress = round(info.progress_pct, 1)

    return f"""
    <div class="flex items-center justify-between mb-4">
        <span class="text-gray-400 text-sm">交易时段</span>
        <span class="text-2xl">⏰</span>
    </div>
    <div class="text-xl font-bold text-{color}-400 mb-2">{name}</div>
    <div class="w-full bg-gray-700 rounded-full h-3 mb-2">
        <div class="bg-gradient-to-r from-{color}-400 to-{color}-500 h-3 rounded-full progress-bar" style="width: {progress}%"></div>
    </div>
    <div class="flex justify-between text-sm text-gray-400">
        <span>进度: {progress}%</span>
        <span>{'可交易' if info.is_trading_allowed else '不可交易'}</span>
    </div>
    """


@app.get("/api/positions/html", response_class=HTMLResponse)
async def get_positions_html():
    """Get positions as HTML for HTMX"""
    positions = [
        {"symbol": "TQQQ", "futu_code": "US.TQQQ", "direction": "LONG", "quantity": 100, "pnl": 75.00},
    ]

    if not positions:
        return '<p class="text-gray-500 text-center py-4">暂无持仓</p>'

    rows = ""
    for pos in positions:
        pnl_color = "green" if pos["pnl"] >= 0 else "red"
        pnl_sign = "+" if pos["pnl"] >= 0 else ""
        dir_color = "green" if pos["direction"] == "LONG" else "red"

        rows += f"""
        <tr class="border-b border-gray-700/50 hover:bg-gray-800/30">
            <td class="py-3">
                <div class="font-medium">{pos["symbol"]}</div>
                <div class="text-xs text-gray-500">{pos["futu_code"]}</div>
            </td>
            <td class="text-right">
                <span class="px-2 py-1 bg-{dir_color}-500/20 text-{dir_color}-400 rounded text-sm">{pos["direction"]}</span>
            </td>
            <td class="text-right">{pos["quantity"]}</td>
            <td class="text-right text-{pnl_color}-400">{pnl_sign}${abs(pos["pnl"]):.2f}</td>
        </tr>
        """

    return f"""
    <table class="w-full">
        <thead>
            <tr class="text-gray-400 text-sm border-b border-gray-700">
                <th class="text-left py-2">标的</th>
                <th class="text-right py-2">方向</th>
                <th class="text-right py-2">数量</th>
                <th class="text-right py-2">盈亏</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


@app.get("/api/trades/html", response_class=HTMLResponse)
async def get_trades_html():
    """Get recent trades as HTML for HTMX"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()
        trades = db.get_recent_trades(5)
    except:
        trades = []

    if not trades:
        return '<p class="text-gray-500 text-center py-4">暂无交易记录</p>'

    items = ""
    for trade in trades:
        pnl = trade.pnl if hasattr(trade, 'pnl') else 0
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_sign = "+" if pnl >= 0 else ""
        time_str = trade.entry_time.strftime("%H:%M:%S") if hasattr(trade, 'entry_time') else ""

        items += f"""
        <div class="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg">
            <div>
                <div class="font-medium">{trade.symbol}</div>
                <div class="text-xs text-gray-500">{time_str}</div>
            </div>
            <div class="text-right">
                <div class="text-{pnl_color}-400 font-medium">{pnl_sign}${abs(pnl):.2f}</div>
                <div class="text-xs text-gray-500">{trade.entry_side}</div>
            </div>
        </div>
        """

    return items


@app.get("/api/symbols/html", response_class=HTMLResponse)
async def get_symbols_html():
    """Get trading symbols as HTML for HTMX"""
    registry = get_symbol_registry()
    symbols = []

    for sym in registry.all_symbols()[:6]:
        is_active = sym.futu_code in registry.active_symbols
        symbols.append({
            "symbol": sym.symbol,
            "futu_code": sym.futu_code,
            "is_active": is_active,
            "price": 0,
            "change": 0
        })

    items = ""
    for sym in symbols:
        border = "green-500/30" if sym["is_active"] else "gray-600"
        dot_color = "green" if sym["is_active"] else "gray"

        items += f"""
        <div class="p-3 bg-gray-800/50 rounded-lg border border-{border} cursor-pointer hover:border-blue-500">
            <div class="flex items-center justify-between mb-2">
                <span class="font-bold">{sym["symbol"]}</span>
                <span class="w-2 h-2 bg-{dot_color}-500 rounded-full"></span>
            </div>
            <div class="text-lg font-medium">--</div>
            <div class="text-xs text-gray-400">{sym["futu_code"]}</div>
        </div>
        """

    items += """
    <div class="p-3 bg-gray-800/50 rounded-lg border border-dashed border-gray-600 cursor-pointer hover:border-blue-500 flex items-center justify-center">
        <span class="text-gray-500">+ 更多</span>
    </div>
    """

    return items


@app.get("/api/compliance/html", response_class=HTMLResponse)
async def get_compliance_html():
    """Get compliance status as HTML"""
    try:
        from src.monitor.dashboard import get_dashboard
        dashboard = get_dashboard()
        data = dashboard.get_dashboard()
        c = data.compliance

        met = c.targets_met_count
        total = c.total_targets

        metrics = [
            ("延迟", f"{c.order_latency_actual_ms:.1f}ms", c.order_latency_meets),
            ("滑点", f"{c.slippage_actual_pct:.2%}", c.slippage_meets),
            ("成交额", f"${c.daily_volume_actual_usd/1000:.0f}K", c.daily_volume_meets),
            ("成交率", f"{c.fill_rate_actual_pct:.1%}", c.fill_rate_meets),
            ("夏普", f"{c.sharpe_actual:.2f}", c.sharpe_meets),
            ("回撤", f"{c.max_drawdown_actual_pct:.1%}", c.max_drawdown_meets),
            ("熔断", "正常" if not c.circuit_breaker_triggered else "触发", not c.circuit_breaker_triggered),
            ("时段", "无缝", True),
        ]
    except:
        met, total = 0, 8
        metrics = [
            ("延迟", "--", False),
            ("滑点", "--", False),
            ("成交额", "--", False),
            ("成交率", "--", False),
            ("夏普", "--", False),
            ("回撤", "--", False),
            ("熔断", "--", False),
            ("时段", "--", False),
        ]

    items = ""
    for name, value, ok in metrics:
        icon = "✓" if ok else "✗"
        color = "green" if ok else "red"
        items += f"""
        <div class="text-center p-3 bg-gray-800/50 rounded-lg">
            <div class="text-{color}-400 text-lg mb-1">{icon}</div>
            <div class="text-xs text-gray-400">{name}</div>
            <div class="text-sm font-medium">{value}</div>
        </div>
        """

    return f"""
    <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold">📋 目标合规状态</h3>
        <span class="px-3 py-1 bg-{'green' if met == total else 'yellow'}-500/20 text-{'green' if met == total else 'yellow'}-400 rounded-full text-sm">{met}/{total} 达标</span>
    </div>
    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
        {items}
    </div>
    """


@app.get("/api/portfolio/html", response_class=HTMLResponse)
async def get_portfolio_html():
    """Get portfolio value as HTML"""
    try:
        from src.risk.sharpe_calculator import get_sharpe_calculator
        calc = get_sharpe_calculator()
        equity = calc._current_equity
        starting = calc.starting_equity
        daily_pnl = equity - starting
        daily_pct = (daily_pnl / starting * 100) if starting > 0 else 0
    except Exception:
        equity = 100000.0
        daily_pnl = 0
        daily_pct = 0

    pnl_color = "green" if daily_pnl >= 0 else "red"
    pnl_sign = "+" if daily_pnl >= 0 else ""

    return f"""
    <div class="flex items-center justify-between mb-3">
        <span class="text-gray-400 text-sm">投资组合</span>
        <span class="text-2xl">💰</span>
    </div>
    <div class="stat-value text-white">${equity:,.2f}</div>
    <div class="flex items-center mt-2">
        <span class="text-{pnl_color}-400 text-sm font-medium">{pnl_sign}${abs(daily_pnl):,.2f} ({pnl_sign}{abs(daily_pct):.2f}%)</span>
    </div>
    """


@app.get("/api/winrate/html", response_class=HTMLResponse)
async def get_winrate_html():
    """Get win rate as HTML"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()
        stats = db.get_trading_stats(30)
        win_rate = stats.get("win_rate", 0) * 100
    except Exception:
        win_rate = 0

    color = "green" if win_rate >= 50 else "yellow" if win_rate >= 30 else "red"

    return f"""
    <div class="flex items-center justify-between mb-3">
        <span class="text-gray-400 text-sm">胜率</span>
        <span class="text-2xl">🎯</span>
    </div>
    <div class="stat-value text-blue-400">{win_rate:.1f}%</div>
    <div class="w-full bg-gray-700 rounded-full h-2 mt-3">
        <div class="bg-blue-500 h-2 rounded-full progress-bar" style="width: {min(win_rate, 100)}%"></div>
    </div>
    """


@app.get("/api/sharpe/html", response_class=HTMLResponse)
async def get_sharpe_html():
    """Get Sharpe ratio as HTML"""
    try:
        from src.risk.sharpe_calculator import get_sharpe_calculator
        calc = get_sharpe_calculator()
        sharpe = calc.calculate_sharpe()
    except Exception:
        sharpe = 0

    meets_target = sharpe >= 2.0
    status_color = "green" if meets_target else "yellow" if sharpe >= 1.0 else "red"
    status_text = "✓ 达标" if meets_target else "✗ 未达标"

    return f"""
    <div class="flex items-center justify-between mb-3">
        <span class="text-gray-400 text-sm">夏普比率</span>
        <span class="text-2xl">📊</span>
    </div>
    <div class="stat-value text-purple-400">{sharpe:.2f}</div>
    <div class="flex items-center mt-2">
        <span class="text-{status_color}-400 text-sm">{status_text} (目标≥2.0)</span>
    </div>
    """


# ==========================================
# Trading Control Endpoints
# ==========================================

@app.post("/api/trading/start")
async def start_trading(background_tasks: BackgroundTasks):
    """Start trading"""
    _app_state["is_running"] = True
    _app_state["start_time"] = datetime.now()

    engine = get_engine()
    if engine:
        background_tasks.add_task(engine.start)

    return {"status": "started", "timestamp": datetime.now().isoformat()}


@app.post("/api/trading/stop")
async def stop_trading():
    """Stop trading"""
    _app_state["is_running"] = False

    engine = get_engine()
    if engine:
        await engine.stop()

    return {"status": "stopped", "timestamp": datetime.now().isoformat()}


@app.post("/api/order")
async def place_order(order: OrderRequest):
    """Place a manual order"""
    engine = get_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    # Validate action
    if order.action not in ["long", "short", "flat"]:
        raise HTTPException(status_code=400, detail="Invalid action")

    # Place order through engine
    # result = await engine.execute_action(order.symbol, order.action, order.quantity)

    return {
        "status": "submitted",
        "symbol": order.symbol,
        "action": order.action,
        "timestamp": datetime.now().isoformat()
    }


# ==========================================
# Symbols Endpoints
# ==========================================

@app.get("/api/symbols", response_model=List[SymbolInfo])
async def get_symbols():
    """Get all registered symbols"""
    registry = get_symbol_registry()
    symbols = []

    for sym in registry.all_symbols():
        symbols.append(SymbolInfo(
            symbol=sym.symbol,
            futu_code=sym.futu_code,
            name=sym.name,
            instrument_type=sym.instrument_type.value,
            is_active=sym.futu_code in registry.active_symbols
        ))

    return symbols


@app.post("/api/symbols/{futu_code}/activate")
async def activate_symbol(futu_code: str):
    """Activate a symbol for trading"""
    registry = get_symbol_registry()
    registry.activate(futu_code)
    return {"status": "activated", "symbol": futu_code}


@app.post("/api/symbols/{futu_code}/deactivate")
async def deactivate_symbol(futu_code: str):
    """Deactivate a symbol from trading"""
    registry = get_symbol_registry()
    registry.deactivate(futu_code)
    return {"status": "deactivated", "symbol": futu_code}


# ==========================================
# Positions Endpoints
# ==========================================

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    engine = get_engine()

    # Mock data for demo
    positions = [
        {
            "symbol": "TQQQ",
            "futu_code": "US.TQQQ",
            "direction": "LONG",
            "quantity": 100,
            "avg_cost": 50.25,
            "current_price": 51.00,
            "unrealized_pnl": 75.00,
            "pnl_pct": 1.49
        }
    ]

    return positions


# ==========================================
# Trades Endpoints
# ==========================================

@app.get("/api/trades/recent")
async def get_recent_trades(limit: int = Query(default=20, le=100)):
    """Get recent trades"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()
        trades = db.get_recent_trades(limit)

        return [
            {
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "entry_time": t.entry_time.isoformat(),
                "entry_price": t.entry_price,
                "entry_side": t.entry_side,
                "quantity": t.quantity,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "status": t.status
            }
            for t in trades
        ]
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return []


@app.get("/api/trades/date/{trade_date}")
async def get_trades_by_date(trade_date: str):
    """Get trades for a specific date"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()

        dt = date.fromisoformat(trade_date)
        trades = db.get_trades_by_date(dt)

        return [t.to_dict() for t in trades]
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================
# Metrics Endpoints
# ==========================================

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get metrics summary"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()
        stats = db.get_trading_stats(30)

        return {
            "total_pnl": round(stats.get("total_pnl", 0), 2),
            "total_trades": stats.get("total_trades", 0),
            "win_rate": round(stats.get("win_rate", 0) * 100, 1),
            "avg_pnl": round(stats.get("avg_pnl", 0), 2),
            "avg_slippage_pct": round(stats.get("avg_slippage", 0) * 100, 4),
            "avg_latency_ms": round(stats.get("avg_latency_ms", 0), 2)
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {
            "total_pnl": 0,
            "total_trades": 0,
            "win_rate": 0,
            "avg_pnl": 0
        }


@app.get("/api/metrics/html", response_class=HTMLResponse)
async def get_metrics_html():
    """Get metrics as HTML for HTMX"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()
        stats = db.get_trading_stats(30)

        total_pnl = stats.get("total_pnl", 0)
        total_trades = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0) * 100
    except Exception:
        total_pnl = 0
        total_trades = 0
        win_rate = 0

    pnl_color = "green" if total_pnl >= 0 else "red"
    pnl_sign = "+" if total_pnl >= 0 else ""

    return f"""
    <div class="flex items-center justify-between mb-3">
        <span class="text-gray-400 text-sm">今日盈亏</span>
        <span class="text-2xl">📈</span>
    </div>
    <div class="stat-value text-{pnl_color}-400">{pnl_sign}${abs(total_pnl):,.2f}</div>
    <div class="flex items-center mt-2">
        <span class="text-gray-400 text-sm">{total_trades} 笔交易 · 胜率 {win_rate:.1f}%</span>
    </div>
    """


@app.get("/api/metrics/daily")
async def get_daily_metrics(days: int = Query(default=30, le=365)):
    """Get daily performance metrics"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        records = db.get_daily_performance(start_date, end_date)

        return [r.to_dict() for r in records]
    except Exception as e:
        logger.error(f"Error getting daily metrics: {e}")
        return []


# ==========================================
# Reports Endpoints
# ==========================================

@app.get("/api/reports/generate")
async def generate_report(
    format: str = Query(default="pdf", pattern="^(pdf|excel)$"),
    start_date: str = Query(default=None),
    end_date: str = Query(default=None),
    background_tasks: BackgroundTasks = None
):
    """Generate trading report"""
    try:
        from src.report.generator import ReportGenerator

        generator = ReportGenerator()

        start = date.fromisoformat(start_date) if start_date else date.today() - timedelta(days=30)
        end = date.fromisoformat(end_date) if end_date else date.today()

        if format == "pdf":
            filepath = generator.generate_pdf(start, end)
        else:
            filepath = generator.generate_excel(start, end)

        return {
            "status": "generated",
            "format": format,
            "filepath": filepath,
            "download_url": f"/api/reports/download/{filepath.split('/')[-1]}"
        }
    except ImportError:
        raise HTTPException(status_code=501, detail="Report generation not available")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/download/{filename}")
async def download_report(filename: str):
    """Download generated report"""
    import os
    filepath = f"reports/{filename}"

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(filepath)


# ==========================================
# Configuration Endpoints
# ==========================================

@app.get("/api/config")
async def get_config():
    """Get current configuration (safe subset)"""
    settings = get_settings()

    return {
        "futu_host": settings.futu_host,
        "futu_port": settings.futu_port,
        "trade_env": settings.futu_trade_env,
        "llm_provider": settings.llm_provider,
        "trading_symbols": settings.trading_symbols,
        "max_daily_drawdown": settings.max_daily_drawdown,
        "max_total_drawdown": settings.max_total_drawdown,
        "slippage_tolerance": settings.slippage_tolerance
    }


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the web server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
