# AI Futu Trader

基于港大 Al-Trader 架构的超低延迟交易系统，使用 Futu OpenD API 实现。

## 🎯 核心特性

- **超低延迟**: 目标下单延迟 0.0014s (1.4ms)，全流程 (行情→模型→下单) ≤ 1s
- **AI 决策**: 基于 GPT-4/Claude 的 Chain-of-Thought 交易决策
- **风控管理**: 3% 日内熔断，15% 最大回撤限制，Sharpe ≥ 2
- **零改动扩展**: 支持 TQQQ/QQQ 后，1天内部署到 SPXL、SOXL、AAPL 等任意美股标的
- **全时段交易**: 盘前盘后无缝接力

## 📊 性能指标

| 指标 | 目标值 |
|------|--------|
| 下单延迟 | ≤ 1.4ms |
| 全流程延迟 | ≤ 1s |
| 日成交额 | ≥ $50,000 |
| 成交率 | ≥ 95% |
| 滑点 | ≤ 0.2% |
| Sharpe 比率 | ≥ 2 |
| 最大回撤 | ≤ 15% |

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone <repository-url>
cd AIFutuTrader

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的配置
# - Futu OpenD 连接信息
# - OpenAI/Anthropic API Key
# - 交易参数
```

### 3. 启动 Futu OpenD

确保 Futu OpenD 客户端已启动并登录。

### 4. 运行

```bash
# 模拟交易模式
python -m src.run --simulate

# 指定交易标的
python -m src.run --simulate --symbols US.TQQQ US.QQQ US.SOXL

# 使用 Anthropic Claude
python -m src.run --simulate --llm anthropic

# 真实交易模式 (需要确认)
python -m src.run --real
```

## 🐳 Docker 部署

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f ai-trader

# 停止服务
docker-compose down
```

服务端口:
- AI Trader Metrics: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## 📁 项目结构

```
AIFutuTrader/
├── src/
│   ├── core/           # 核心模块 (配置、日志、符号注册)
│   ├── data/           # 数据模块 (Futu 行情、数据处理)
│   ├── action/         # 执行模块 (Futu 下单)
│   ├── model/          # 模型模块 (LLM 决策)
│   ├── risk/           # 风控模块 (熔断、仓位)
│   ├── monitor/        # 监控模块 (Prometheus、飞书告警)
│   ├── engine.py       # 交易引擎
│   └── run.py          # 入口文件
├── tests/
│   ├── unit/           # 单元测试
│   └── integration/    # 集成测试
├── docker/             # Docker 配置
├── .github/workflows/  # CI/CD 配置
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行测试并生成覆盖率报告
pytest --cov=src --cov-report=html

# 查看覆盖率 (目标 ≥ 80%)
open htmlcov/index.html
```

## 📈 监控

### Grafana 仪表盘

预配置的仪表盘包含:
- 投资组合价值
- 每日盈亏
- 回撤监控
- Sharpe 比率
- 订单延迟 (P50/P95/P99)
- 交易频率
- 连接状态
- 熔断状态
- 成交率

### 飞书告警

支持的告警类型:
- 熔断触发 (立即通知)
- 回撤预警
- 高滑点事件
- 连接断开
- 每日交易总结

配置飞书机器人:
1. 创建飞书机器人
2. 获取 Webhook URL
3. 设置 `FEISHU_WEBHOOK_URL` 环境变量

## 🔧 扩展新标的

无需修改代码，只需更新配置:

```bash
# 方式 1: 环境变量
export TRADING_SYMBOLS=US.TQQQ,US.QQQ,US.SPXL,US.SOXL,US.AAPL

# 方式 2: 命令行参数
python -m src.run --symbols US.TQQQ US.QQQ US.SPXL US.SOXL US.AAPL

# 方式 3: 代码中动态添加
from src.core.symbols import get_symbol_registry
registry = get_symbol_registry()
registry.activate("US.SPXL", "US.SOXL", "US.AAPL")
```

### 添加期权

```python
from src.core.symbols import get_symbol_registry

registry = get_symbol_registry()

# 添加期权合约
registry.register_option(
    underlying="AAPL",
    strike=150.0,
    expiry="20240315",
    option_type="call"
)
```

## ⚠️ 风险提示

1. **本系统仅供学习研究使用**
2. 真实交易存在亏损风险
3. 请在模拟环境充分测试后再进行实盘
4. 建议从小资金开始
5. 请遵守当地法律法规

## 📄 许可证

MIT License

## 🙏 致谢

- [HKUDS/Al-Trader](https://github.com/HKUDS/Al-Trader) - 原始架构参考
- [Futu OpenD](https://openapi.futunn.com/) - 交易 API
