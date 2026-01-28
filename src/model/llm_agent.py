"""
LLM Agent - AI-powered trading decision engine
Based on Al-Trader architecture with Chain-of-Thought reasoning
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import re

from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logger import get_logger, TradeLogger, PerformanceLogger
from src.data.data_processor import MarketSnapshot
from src.action.futu_executor import TradingAction, Position
from .prompts import PromptTemplates

logger = get_logger(__name__)
perf_logger = PerformanceLogger()


class DecisionConfidence(Enum):
    """Confidence level for trading decisions"""
    HIGH = "high"        # >80% confidence
    MEDIUM = "medium"    # 50-80% confidence
    LOW = "low"          # <50% confidence


@dataclass
class TradingDecision:
    """Trading decision from LLM analysis"""
    action: TradingAction
    symbol: str
    futu_code: str
    confidence: DecisionConfidence
    confidence_score: float  # 0.0 to 1.0

    # Position sizing
    position_size_pct: float = 0.0  # Percentage of available capital
    suggested_quantity: int = 0

    # Price targets
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Reasoning
    reasoning: str = ""
    key_factors: List[str] = field(default_factory=list)

    # Risk assessment
    risk_level: str = "medium"
    expected_return: float = 0.0
    risk_reward_ratio: float = 0.0

    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    model_latency_ms: float = 0.0

    # Raw model output
    raw_response: str = ""

    @property
    def should_execute(self) -> bool:
        """Check if decision should be executed based on confidence"""
        return (
            self.action != TradingAction.HOLD and
            self.confidence_score >= 0.6 and
            self.confidence != DecisionConfidence.LOW
        )

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "symbol": self.symbol,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "position_size_pct": self.position_size_pct,
            "suggested_quantity": self.suggested_quantity,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "risk_level": self.risk_level,
            "model_latency_ms": self.model_latency_ms,
            "timestamp": self.timestamp.isoformat()
        }


class LLMAgent:
    """
    LLM-based trading decision agent.
    Implements Chain-of-Thought reasoning for long/short/flat decisions.
    """

    def __init__(
        self,
        provider: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        settings = get_settings()
        self.provider = (provider or settings.llm_provider).lower()
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = settings.llm_timeout

        # Initialize client based on provider
        self._client = None
        self._model = None

        if self.provider == "openai":
            self._init_openai(
                api_key or settings.openai_api_key,
                model or settings.openai_model,
                settings.openai_base_url
            )
        elif self.provider == "anthropic":
            self._init_anthropic(
                api_key or settings.anthropic_api_key,
                model or settings.anthropic_model
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        self._prompts = PromptTemplates()
        self._trade_logger = TradeLogger()

        # Decision history for learning
        self._decision_history: List[TradingDecision] = []
        self._max_history = 100

    def _init_openai(self, api_key: str, model: str, base_url: str = None):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI, AsyncOpenAI

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url

            self._client = OpenAI(**kwargs)
            self._async_client = AsyncOpenAI(**kwargs)
            self._model = model
            logger.info(f"Initialized OpenAI client with model: {model}")

        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def _init_anthropic(self, api_key: str, model: str):
        """Initialize Anthropic client"""
        try:
            from anthropic import Anthropic, AsyncAnthropic

            self._client = Anthropic(api_key=api_key)
            self._async_client = AsyncAnthropic(api_key=api_key)
            self._model = model
            logger.info(f"Initialized Anthropic client with model: {model}")

        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def analyze(
        self,
        snapshot: MarketSnapshot,
        current_position: Position = None,
        portfolio_value: float = 100000.0,
        risk_budget: float = 0.02,
    ) -> TradingDecision:
        """
        Analyze market snapshot and generate trading decision.

        Args:
            snapshot: Current market snapshot with indicators
            current_position: Current position for the symbol
            portfolio_value: Total portfolio value
            risk_budget: Maximum risk per trade as fraction of portfolio

        Returns:
            TradingDecision with action and reasoning
        """
        start_time = time.time()

        try:
            # Build prompt
            system_prompt = self._prompts.system_prompt()
            user_prompt = self._prompts.trading_analysis_prompt(
                snapshot=snapshot,
                current_position=current_position,
                portfolio_value=portfolio_value,
                risk_budget=risk_budget
            )

            # Call LLM
            if self.provider == "openai":
                response = self._call_openai(system_prompt, user_prompt)
            else:
                response = self._call_anthropic(system_prompt, user_prompt)

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            decision = self._parse_decision(
                response=response,
                snapshot=snapshot,
                latency_ms=latency_ms
            )

            # Log decision
            self._trade_logger.signal_generated(
                signal=decision.action.value,
                confidence=decision.confidence_score,
                reasoning=decision.reasoning[:200],
                symbol=snapshot.futu_code
            )

            # Store in history
            self._decision_history.append(decision)
            if len(self._decision_history) > self._max_history:
                self._decision_history.pop(0)

            perf_logger.latency_record("llm_analysis", latency_ms)

            return decision

        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return TradingDecision(
                action=TradingAction.HOLD,
                symbol=snapshot.symbol,
                futu_code=snapshot.futu_code,
                confidence=DecisionConfidence.LOW,
                confidence_score=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                model_latency_ms=latency_ms
            )

    async def analyze_async(
        self,
        snapshot: MarketSnapshot,
        current_position: Position = None,
        portfolio_value: float = 100000.0,
        risk_budget: float = 0.02,
    ) -> TradingDecision:
        """Async version of analyze"""
        start_time = time.time()

        try:
            system_prompt = self._prompts.system_prompt()
            user_prompt = self._prompts.trading_analysis_prompt(
                snapshot=snapshot,
                current_position=current_position,
                portfolio_value=portfolio_value,
                risk_budget=risk_budget
            )

            if self.provider == "openai":
                response = await self._call_openai_async(system_prompt, user_prompt)
            else:
                response = await self._call_anthropic_async(system_prompt, user_prompt)

            latency_ms = (time.time() - start_time) * 1000

            decision = self._parse_decision(
                response=response,
                snapshot=snapshot,
                latency_ms=latency_ms
            )

            self._trade_logger.signal_generated(
                signal=decision.action.value,
                confidence=decision.confidence_score,
                reasoning=decision.reasoning[:200],
                symbol=snapshot.futu_code
            )

            self._decision_history.append(decision)
            if len(self._decision_history) > self._max_history:
                self._decision_history.pop(0)

            perf_logger.latency_record("llm_analysis_async", latency_ms)

            return decision

        except Exception as e:
            logger.error(f"Async LLM analysis error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return TradingDecision(
                action=TradingAction.HOLD,
                symbol=snapshot.symbol,
                futu_code=snapshot.futu_code,
                confidence=DecisionConfidence.LOW,
                confidence_score=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                model_latency_ms=latency_ms
            )

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API synchronously"""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )
        return response.choices[0].message.content

    async def _call_openai_async(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API asynchronously"""
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )
        return response.choices[0].message.content

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic API synchronously"""
        response = self._client.messages.create(
            model=self._model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.content[0].text

    async def _call_anthropic_async(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic API asynchronously"""
        response = await self._async_client.messages.create(
            model=self._model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.content[0].text

    def _parse_decision(
        self,
        response: str,
        snapshot: MarketSnapshot,
        latency_ms: float
    ) -> TradingDecision:
        """Parse LLM response into TradingDecision"""

        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._parse_json_decision(data, snapshot, response, latency_ms)
            except json.JSONDecodeError:
                pass

        # Fallback to text parsing
        return self._parse_text_decision(response, snapshot, latency_ms)

    def _parse_json_decision(
        self,
        data: dict,
        snapshot: MarketSnapshot,
        raw_response: str,
        latency_ms: float
    ) -> TradingDecision:
        """Parse JSON-formatted decision"""

        # Parse action
        action_str = data.get("action", "hold").lower()
        action_map = {
            "long": TradingAction.LONG,
            "buy": TradingAction.LONG,
            "short": TradingAction.SHORT,
            "sell": TradingAction.SHORT,
            "flat": TradingAction.FLAT,
            "close": TradingAction.FLAT,
            "hold": TradingAction.HOLD,
        }
        action = action_map.get(action_str, TradingAction.HOLD)

        # Parse confidence
        confidence_score = float(data.get("confidence", data.get("confidence_score", 0.5)))
        if confidence_score > 0.8:
            confidence = DecisionConfidence.HIGH
        elif confidence_score > 0.5:
            confidence = DecisionConfidence.MEDIUM
        else:
            confidence = DecisionConfidence.LOW

        return TradingDecision(
            action=action,
            symbol=snapshot.symbol,
            futu_code=snapshot.futu_code,
            confidence=confidence,
            confidence_score=confidence_score,
            position_size_pct=float(data.get("position_size_pct", data.get("position_size", 0))),
            suggested_quantity=int(data.get("quantity", data.get("suggested_quantity", 0))),
            entry_price=float(data.get("entry_price", snapshot.last_price)),
            stop_loss=float(data.get("stop_loss", 0)),
            take_profit=float(data.get("take_profit", 0)),
            reasoning=data.get("reasoning", data.get("analysis", "")),
            key_factors=data.get("key_factors", data.get("factors", [])),
            risk_level=data.get("risk_level", "medium"),
            expected_return=float(data.get("expected_return", 0)),
            risk_reward_ratio=float(data.get("risk_reward_ratio", data.get("rr_ratio", 0))),
            model_latency_ms=latency_ms,
            raw_response=raw_response
        )

    def _parse_text_decision(
        self,
        response: str,
        snapshot: MarketSnapshot,
        latency_ms: float
    ) -> TradingDecision:
        """Fallback text parsing for non-JSON responses"""

        response_lower = response.lower()

        # Detect action
        if any(word in response_lower for word in ["long", "buy", "bullish"]):
            action = TradingAction.LONG
        elif any(word in response_lower for word in ["short", "sell", "bearish"]):
            action = TradingAction.SHORT
        elif any(word in response_lower for word in ["flat", "close", "exit"]):
            action = TradingAction.FLAT
        else:
            action = TradingAction.HOLD

        # Estimate confidence from language
        if any(word in response_lower for word in ["strong", "high confidence", "definitely", "clearly"]):
            confidence = DecisionConfidence.HIGH
            confidence_score = 0.85
        elif any(word in response_lower for word in ["moderate", "medium", "likely"]):
            confidence = DecisionConfidence.MEDIUM
            confidence_score = 0.65
        else:
            confidence = DecisionConfidence.LOW
            confidence_score = 0.4

        return TradingDecision(
            action=action,
            symbol=snapshot.symbol,
            futu_code=snapshot.futu_code,
            confidence=confidence,
            confidence_score=confidence_score,
            reasoning=response[:500],
            model_latency_ms=latency_ms,
            raw_response=response
        )

    def get_recent_decisions(self, count: int = 10) -> List[TradingDecision]:
        """Get recent decision history"""
        return self._decision_history[-count:]

    def get_decision_stats(self) -> dict:
        """Get statistics on recent decisions"""
        if not self._decision_history:
            return {}

        actions = [d.action.value for d in self._decision_history]
        confidences = [d.confidence_score for d in self._decision_history]
        latencies = [d.model_latency_ms for d in self._decision_history]

        return {
            "total_decisions": len(self._decision_history),
            "action_distribution": {
                "long": actions.count("long"),
                "short": actions.count("short"),
                "flat": actions.count("flat"),
                "hold": actions.count("hold")
            },
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies)
        }
