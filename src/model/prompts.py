"""
Prompt Templates for LLM Trading Agent
Implements Chain-of-Thought reasoning for trading decisions
"""
from typing import Optional
from src.data.data_processor import MarketSnapshot
from src.action.futu_executor import Position


class PromptTemplates:
    """
    Prompt templates for the LLM trading agent.
    Based on Al-Trader's Chain-of-Thought approach.
    """

    def system_prompt(self) -> str:
        """System prompt defining the trading agent's role and behavior"""
        return """You are an expert quantitative trading agent specializing in US equities and ETFs, 
particularly leveraged ETFs like TQQQ, QQQ, SOXL, and SPXL. Your role is to analyze market data 
and make precise trading decisions.

## Your Trading Philosophy:
1. **Risk-First**: Never risk more than 2% of portfolio on a single trade
2. **Momentum-Based**: Trade with the trend, not against it
3. **Technical-Driven**: Use technical indicators as primary decision factors
4. **Quick Decisions**: Markets move fast - be decisive but calculated

## Decision Framework:
- LONG: Open or increase long position when bullish signals align
- SHORT: Open or increase short position when bearish signals align
- FLAT: Close all positions when risk is high or signals are unclear
- HOLD: No action when conditions don't warrant a trade

## Key Indicators to Consider:
1. RSI: Overbought (>70) or Oversold (<30) conditions
2. MACD: Crossovers and histogram direction
3. Moving Averages: Price relative to SMA/EMA levels
4. Volume: Confirmation of price moves
5. Bollinger Bands: Volatility and mean reversion signals
6. ADX: Trend strength

## Response Format:
Always respond with a JSON object containing your decision:
```json
{
    "action": "long|short|flat|hold",
    "confidence": 0.0-1.0,
    "position_size_pct": 0-100,
    "entry_price": <current price or limit>,
    "stop_loss": <stop loss price>,
    "take_profit": <take profit price>,
    "reasoning": "<brief explanation>",
    "key_factors": ["factor1", "factor2", ...],
    "risk_level": "low|medium|high",
    "expected_return": <percentage>,
    "risk_reward_ratio": <ratio>
}
```

Be concise but thorough in your analysis. Time is critical in trading."""

    def trading_analysis_prompt(
        self,
        snapshot: MarketSnapshot,
        current_position: Optional[Position] = None,
        portfolio_value: float = 100000.0,
        risk_budget: float = 0.02,
    ) -> str:
        """
        Generate trading analysis prompt from market snapshot.

        Args:
            snapshot: Current market data and indicators
            current_position: Current position if any
            portfolio_value: Total portfolio value
            risk_budget: Max risk per trade (fraction)
        """
        # Get context from snapshot
        market_context = snapshot.to_prompt_context()

        # Position context
        if current_position and not current_position.is_flat:
            pnl_pct = (current_position.unrealized_pnl / current_position.market_value * 100) if current_position.market_value > 0 else 0
            position_context = f"""
CURRENT POSITION:
- Direction: {"LONG" if current_position.is_long else "SHORT"}
- Quantity: {current_position.abs_quantity} shares
- Average Cost: ${current_position.avg_cost:.4f}
- Market Value: ${current_position.market_value:.2f}
- Unrealized P&L: ${current_position.unrealized_pnl:.2f} ({pnl_pct:.2f}%)
"""
        else:
            position_context = "\nCURRENT POSITION: Flat (no position)\n"

        # Portfolio context
        portfolio_context = f"""
PORTFOLIO CONTEXT:
- Total Value: ${portfolio_value:,.2f}
- Risk Budget per Trade: {risk_budget*100:.1f}% (${portfolio_value * risk_budget:,.2f})
- Maximum Position Size: ${portfolio_value * 0.25:,.2f} (25% of portfolio)
"""

        # Trading rules reminder
        rules_context = """
TRADING RULES:
1. Maximum slippage tolerance: 0.2%
2. If RSI > 75, consider closing longs or shorting
3. If RSI < 25, consider closing shorts or going long
4. Respect the trend - ADX > 25 indicates strong trend
5. Volume should confirm price action
6. For leveraged ETFs (TQQQ, SOXL), use smaller position sizes
"""

        prompt = f"""Analyze the following market data and provide a trading decision.

{market_context}
{position_context}
{portfolio_context}
{rules_context}

Based on the above data, what is your trading decision? 
Provide your analysis in the specified JSON format.
Consider both the immediate opportunity and risk management."""

        return prompt

    def risk_assessment_prompt(
        self,
        snapshot: MarketSnapshot,
        position: Position,
        daily_pnl: float,
        drawdown_pct: float,
    ) -> str:
        """Generate prompt for risk assessment"""
        return f"""Assess the current risk situation:

SYMBOL: {snapshot.symbol}
CURRENT PRICE: ${snapshot.last_price:.4f}
POSITION: {position.quantity} shares @ ${position.avg_cost:.4f}
UNREALIZED P&L: ${position.unrealized_pnl:.2f}
DAILY P&L: ${daily_pnl:.2f}
CURRENT DRAWDOWN: {drawdown_pct:.2%}

VOLATILITY:
- ATR(14): {snapshot.indicators.atr_14:.4f}
- Current Range: {(snapshot.high - snapshot.low):.4f}

Should we:
1. HOLD: Continue with current position
2. REDUCE: Reduce position size by 50%
3. EXIT: Close entire position immediately

Respond with JSON:
{{
    "action": "hold|reduce|exit",
    "urgency": "low|medium|high",
    "reasoning": "<brief explanation>",
    "suggested_stop": <price level>
}}"""

    def market_regime_prompt(self, snapshots: dict) -> str:
        """Generate prompt for overall market regime analysis"""
        symbols_data = []
        for code, snap in snapshots.items():
            symbols_data.append(
                f"- {snap.symbol}: ${snap.last_price:.2f} ({snap.change_day:+.2%}), "
                f"RSI={snap.indicators.rsi_14:.1f}, Vol={snap.indicators.volume_ratio:.1f}x"
            )

        return f"""Analyze the overall market regime based on these key symbols:

{chr(10).join(symbols_data)}

Determine:
1. Is the market BULLISH, BEARISH, or NEUTRAL?
2. Is volatility HIGH, MEDIUM, or LOW?
3. What sectors are showing strength/weakness?
4. Should we be aggressive or defensive?

Respond with JSON:
{{
    "regime": "bullish|bearish|neutral",
    "volatility": "high|medium|low",
    "recommendation": "aggressive|moderate|defensive",
    "sector_strength": ["sector1", "sector2"],
    "sector_weakness": ["sector1", "sector2"],
    "notes": "<brief analysis>"
}}"""

    def reflection_prompt(
        self,
        recent_trades: list,
        win_rate: float,
        avg_return: float,
    ) -> str:
        """Generate prompt for trade reflection and learning"""
        trades_summary = []
        for trade in recent_trades[-5:]:
            trades_summary.append(
                f"- {trade.get('symbol')}: {trade.get('action')} at ${trade.get('entry_price'):.2f}, "
                f"Exit: ${trade.get('exit_price', 0):.2f}, P&L: {trade.get('pnl', 0):+.2f}%"
            )

        return f"""Review recent trading performance and suggest improvements:

RECENT TRADES:
{chr(10).join(trades_summary) if trades_summary else "No recent trades"}

PERFORMANCE METRICS:
- Win Rate: {win_rate:.1%}
- Average Return: {avg_return:+.2%}

Questions to answer:
1. What patterns led to winning trades?
2. What patterns led to losing trades?
3. Should we adjust our entry/exit criteria?
4. Are there any systematic errors?

Respond with JSON:
{{
    "winning_patterns": ["pattern1", "pattern2"],
    "losing_patterns": ["pattern1", "pattern2"],
    "suggested_adjustments": ["adjustment1", "adjustment2"],
    "confidence_calibration": "increase|maintain|decrease",
    "position_sizing": "increase|maintain|decrease",
    "key_learnings": "<summary>"
}}"""
