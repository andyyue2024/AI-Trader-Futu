"""
Configuration management for AI Futu Trader
Uses pydantic-settings for type-safe configuration with environment variable support
"""
from functools import lru_cache
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file
    """

    # ===========================================
    # Futu OpenD Configuration
    # ===========================================
    futu_host: str = Field(default="127.0.0.1", description="Futu OpenD host address")
    futu_port: int = Field(default=11111, description="Futu OpenD port")
    futu_trade_env: str = Field(
        default="SIMULATE",
        description="Trading environment: REAL or SIMULATE"
    )
    futu_trade_password: Optional[str] = Field(
        default=None,
        description="Futu trading password (required for real trading)"
    )
    futu_rsa_path: Optional[str] = Field(
        default=None,
        description="Path to RSA private key for encrypted connection"
    )

    # ===========================================
    # LLM Configuration
    # ===========================================
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: openai, anthropic, or local"
    )
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI API base URL")

    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", description="Anthropic model")

    llm_temperature: float = Field(default=0.1, description="LLM temperature for generation")
    llm_max_tokens: int = Field(default=1024, description="Max tokens for LLM response")
    llm_timeout: float = Field(default=30.0, description="LLM API timeout in seconds")

    # ===========================================
    # Trading Configuration
    # ===========================================
    trading_symbols: List[str] = Field(
        default=["US.TQQQ", "US.QQQ"],
        description="List of trading symbols in Futu format (MARKET.SYMBOL)"
    )
    default_position_size: float = Field(
        default=10000.0,
        description="Default position size in USD"
    )
    max_position_per_symbol: float = Field(
        default=50000.0,
        description="Maximum position size per symbol in USD"
    )
    slippage_tolerance: float = Field(
        default=0.002,
        description="Maximum acceptable slippage (0.2%)"
    )

    # ===========================================
    # Risk Management
    # ===========================================
    max_daily_drawdown: float = Field(
        default=0.03,
        description="Maximum daily drawdown before circuit breaker (3%)"
    )
    max_total_drawdown: float = Field(
        default=0.15,
        description="Maximum total drawdown (15%)"
    )
    min_sharpe_ratio: float = Field(
        default=2.0,
        description="Minimum target Sharpe ratio"
    )
    daily_volume_target: float = Field(
        default=50000.0,
        description="Target daily trading volume in USD"
    )
    min_fill_rate: float = Field(
        default=0.95,
        description="Minimum acceptable fill rate (95%)"
    )

    # ===========================================
    # Performance Targets
    # ===========================================
    max_order_latency_ms: float = Field(
        default=1.4,
        description="Maximum order execution latency in milliseconds (0.0014s)"
    )
    max_pipeline_latency_s: float = Field(
        default=1.0,
        description="Maximum full pipeline latency in seconds"
    )

    # ===========================================
    # Monitoring & Alerting
    # ===========================================
    prometheus_port: int = Field(default=8000, description="Prometheus metrics port")
    grafana_url: Optional[str] = Field(default=None, description="Grafana URL for dashboards")

    feishu_webhook_url: Optional[str] = Field(
        default=None,
        description="Feishu (Lark) webhook URL for alerts"
    )
    alert_cooldown_minutes: int = Field(
        default=5,
        description="Minimum interval between alerts in minutes"
    )

    # ===========================================
    # Logging Configuration
    # ===========================================
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(
        default="logs/trading.log",
        description="Log file path"
    )
    log_rotation: str = Field(default="100 MB", description="Log rotation size")
    log_retention: str = Field(default="7 days", description="Log retention period")

    # ===========================================
    # Database / State Persistence
    # ===========================================
    state_file: str = Field(
        default="data/trading_state.json",
        description="Path to trading state persistence file"
    )
    history_db: str = Field(
        default="data/trading_history.db",
        description="Path to trading history SQLite database"
    )

    @field_validator("futu_trade_env")
    @classmethod
    def validate_trade_env(cls, v: str) -> str:
        v = v.upper()
        if v not in ("REAL", "SIMULATE"):
            raise ValueError("futu_trade_env must be 'REAL' or 'SIMULATE'")
        return v

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        v = v.lower()
        if v not in ("openai", "anthropic", "local"):
            raise ValueError("llm_provider must be 'openai', 'anthropic', or 'local'")
        return v

    @field_validator("trading_symbols", mode="before")
    @classmethod
    def parse_trading_symbols(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @property
    def is_real_trading(self) -> bool:
        """Check if running in real trading mode"""
        return self.futu_trade_env == "REAL"

    @property
    def is_paper_trading(self) -> bool:
        """Check if running in paper/simulated trading mode"""
        return self.futu_trade_env == "SIMULATE"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Use this function to access settings throughout the application.
    """
    return Settings()


# Example .env file template
ENV_TEMPLATE = """
# ===========================================
# Futu OpenD Configuration
# ===========================================
FUTU_HOST=127.0.0.1
FUTU_PORT=11111
FUTU_TRADE_ENV=SIMULATE
FUTU_TRADE_PASSWORD=your_password_here
FUTU_RSA_PATH=

# ===========================================
# LLM Configuration
# ===========================================
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_BASE_URL=
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1024

# ===========================================
# Trading Configuration
# ===========================================
TRADING_SYMBOLS=US.TQQQ,US.QQQ
DEFAULT_POSITION_SIZE=10000
MAX_POSITION_PER_SYMBOL=50000
SLIPPAGE_TOLERANCE=0.002

# ===========================================
# Risk Management
# ===========================================
MAX_DAILY_DRAWDOWN=0.03
MAX_TOTAL_DRAWDOWN=0.15
MIN_SHARPE_RATIO=2.0
DAILY_VOLUME_TARGET=50000
MIN_FILL_RATE=0.95

# ===========================================
# Monitoring & Alerting
# ===========================================
PROMETHEUS_PORT=8000
GRAFANA_URL=http://localhost:3000
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-id
ALERT_COOLDOWN_MINUTES=5

# ===========================================
# Logging
# ===========================================
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log
"""
