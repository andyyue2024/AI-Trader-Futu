# 🤖 AI Futu Trader

基于富途 OpenD + LLM 的智能量化交易系统，专注于美股 ETF 和期权交易。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![Coverage](https://img.shields.io/badge/Coverage-80%25+-brightgreen.svg)

## ✨ 主要特性

### 🚀 超低延迟执行
- **下单延迟**: ≤1.4ms (目标 0.0014s)
- **全流程延迟**: ≤1s (行情 → 模型 → 下单)
- 连接池复用、订单预编译优化

### 🧠 LLM 驱动决策
- 支持 GPT-4 Turbo 和 Claude 3.5 Sonnet
- Chain-of-Thought 推理
- 技术指标 + AI 综合决策

### 📊 完善的风控
- 3% 日内自动熔断
- 15% 最大回撤限制
- 夏普比率 ≥2 目标
- 成交率 ≥95%，滑点 ≤0.2%

### ⏰ 全时段交易
- 盘前交易 (04:00-09:30 ET)
- 常规交易 (09:30-16:00 ET)
- 盘后交易 (16:00-20:00 ET)
- 自动时段切换

### 🎯 零改动扩展
- 预配置: TQQQ, QQQ, SOXL, SPXL, AAPL 等
- 一键添加新标的
- 完整期权链支持

### 🌐 Web 界面
- FastAPI REST API
- 实时仪表盘
- 交易控制和监控

### 📑 报告生成
- PDF 格式报告
- Excel 格式报告
- 定时发送功能

## 📁 项目结构

```
AIFutuTrader/
├── src/
│   ├── core/                    # 核心模块
│   │   ├── config.py            # 配置管理
│   │   ├── logger.py            # 日志系统
│   │   ├── symbols.py           # 符号注册表
│   │   ├── session_manager.py   # 交易时段管理
│   │   ├── statistics.py        # 交易统计
│   │   └── strategy_config.py   # 策略配置
│   ├── data/                    # 数据模块
│   │   ├── futu_quote.py        # 富途行情
│   │   ├── data_processor.py    # 数据处理
│   │   ├── options_data.py      # 期权数据
│   │   └── persistence.py       # 数据持久化
│   ├── action/                  # 执行模块
│   │   ├── futu_executor.py     # 订单执行
│   │   ├── position_manager.py  # 持仓管理
│   │   └── order_optimizer.py   # 订单优化器
│   ├── model/                   # 模型模块
│   │   ├── llm_agent.py         # LLM 决策
│   │   └── prompts.py           # 提示词模板
│   ├── risk/                    # 风控模块
│   │   └── risk_manager.py      # 风险管理
│   ├── monitor/                 # 监控模块
│   │   ├── metrics.py           # Prometheus 指标
│   │   ├── alerts.py            # 告警系统
│   │   ├── feishu_enhanced.py   # 增强飞书告警
│   │   └── performance.py       # 性能监控
│   ├── report/                  # 报告模块
│   │   └── generator.py         # 报告生成器
│   ├── web/                     # Web 模块
│   │   └── api.py               # FastAPI 接口
│   ├── backtest/                # 回测模块
│   │   └── engine.py            # 回测引擎
│   ├── engine.py                # 交易引擎
│   └── run.py                   # 入口文件
├── tests/                       # 测试 (19+ 单元测试文件)
│   ├── unit/
│   └── integration/
├── config/                      # 配置文件
├── docker/                      # Docker 配置
├── reports/                     # 生成的报告
├── data/                        # 数据文件
├── logs/                        # 日志文件
├── demo.py                      # 功能演示
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/AIFutuTrader.git
cd AIFutuTrader

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件:
# - FUTU_HOST, FUTU_PORT (OpenD 连接)
# - OPENAI_API_KEY 或 ANTHROPIC_API_KEY
# - FUTU_TRADE_PASSWORD (交易密码)
# - FEISHU_WEBHOOK (飞书告警，可选)
```

### 3. 启动 Futu OpenD

确保 Futu OpenD 已安装并运行:
- 下载: https://www.futunn.com/download/OpenAPI
- 启动 OpenD 并登录

### 4. 运行交易系统

```bash
# 模拟交易模式
python -m src.run --simulate --symbols US.TQQQ US.QQQ

# 真实交易模式
python -m src.run --real --symbols US.TQQQ US.QQQ

# 使用 Claude 作为 LLM
python -m src.run --simulate --llm anthropic

# 使用快速启动脚本
python start.py trade --simulate
python start.py trade --real --symbols US.TQQQ US.QQQ
```

### 5. 启动 Web 界面

```bash
# 启动 Web API
python -m src.web.api

# 或使用快速启动
python start.py web

# 访问 http://localhost:8080
```

### 6. CLI 工具

```bash
# 系统健康检查
python -m src.cli health

# 查看系统状态
python -m src.cli status

# 列出交易标的
python -m src.cli symbols

# 运行回测
python -m src.cli backtest --capital 100000 --rsi-low 30 --rsi-high 70

# 生成报告
python -m src.cli report --format pdf --days 30

# 导出交易数据
python -m src.cli export --days 30 --output trades.json
```

### 7. 功能演示

```bash
# 运行所有功能演示
python demo.py --mode all

# 仅检查系统状态
python demo.py --mode status

# 运行回测演示
python demo.py --mode backtest
```

## 🐳 Docker 部署

```bash
# 使用 Docker Compose 启动完整环境
docker-compose up -d

# 包含服务:
# - 交易系统
# - Futu OpenD
# - Prometheus
# - Grafana
```

## 📈 Web API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 仪表盘 |
| `/api/health` | GET | 健康检查 |
| `/api/status` | GET | 系统状态 |
| `/api/session` | GET | 交易时段 |
| `/api/symbols` | GET | 交易标的列表 |
| `/api/positions` | GET | 当前持仓 |
| `/api/trades/recent` | GET | 最近交易 |
| `/api/metrics/summary` | GET | 性能指标 |
| `/api/trading/start` | POST | 启动交易 |
| `/api/trading/stop` | POST | 停止交易 |
| `/api/reports/generate` | GET | 生成报告 |

## 📑 报告生成

```python
from src.report import ReportGenerator
from datetime import date, timedelta

generator = ReportGenerator()

# 生成 PDF 报告
pdf = generator.generate_pdf(
    start_date=date.today() - timedelta(days=30),
    end_date=date.today()
)

# 生成 Excel 报告
excel = generator.generate_excel(
    start_date=date.today() - timedelta(days=30),
    end_date=date.today()
)
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 带覆盖率
pytest tests/ -v --cov=src --cov-report=html

# 运行特定测试
pytest tests/unit/test_core.py -v
```

## 📊 性能指标

| 指标 | 目标 | 实现模块 |
|------|------|----------|
| 下单延迟 | ≤1.4ms | `OrderOptimizer` |
| 全流程延迟 | ≤1s | `TradingEngine` |
| 日成交额 | ≥$50,000 | `RiskManager` |
| 成交率 | ≥95% | `RiskManager` |
| 滑点 | ≤0.2% | `OrderResult` |
| 夏普比率 | ≥2 | `TradingStatistics` |
| 最大回撤 | ≤15% | `CircuitBreaker` |
| 日内熔断 | 3% | `CircuitBreaker` |
| 测试覆盖 | ≥80% | GitHub Actions |

## 🔧 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `FUTU_HOST` | OpenD 主机 | 127.0.0.1 |
| `FUTU_PORT` | OpenD 端口 | 11111 |
| `FUTU_TRADE_ENV` | 交易环境 (SIMULATE/REAL) | SIMULATE |
| `FUTU_TRADE_PASSWORD` | 交易密码 | - |
| `LLM_PROVIDER` | LLM 提供商 (openai/anthropic) | openai |
| `OPENAI_API_KEY` | OpenAI 密钥 | - |
| `ANTHROPIC_API_KEY` | Anthropic 密钥 | - |
| `TRADING_SYMBOLS` | 交易标的 (逗号分隔) | US.TQQQ,US.QQQ |
| `FEISHU_WEBHOOK` | 飞书 Webhook URL | - |
| `MAX_DAILY_DRAWDOWN` | 日最大回撤 | 0.03 |
| `MAX_TOTAL_DRAWDOWN` | 总最大回撤 | 0.15 |

## 📱 飞书告警

系统支持丰富的飞书卡片告警:

- 🚨 **熔断告警** - 触发风控时立即通知
- ⚠️ **异常告警** - 5 分钟异常检测
- 📋 **每日报告** - 每日交易汇总
- 📊 **周报** - 每周业绩报告

## ✅ 功能完成状态

- [x] 基础交易引擎
- [x] LLM 决策集成 (GPT-4/Claude)
- [x] 风险管理与熔断
- [x] 飞书告警 (增强版)
- [x] 回测引擎
- [x] Web API 界面
- [x] PDF/Excel 报告生成
- [x] 期权交易支持
- [x] 性能监控
- [x] 交易时段管理
- [x] 数据持久化
- [ ] PC 客户端 (计划中)
- [ ] 移动端 App (计划中)

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

---

**⚠️ 免责声明**: 本软件仅供学习和研究使用，不构成投资建议。使用本软件进行实盘交易的风险由用户自行承担。
