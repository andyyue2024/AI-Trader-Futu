"""
Unit tests for model/LLM agent module
"""
import pytest
import json
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock


class TestTradingDecision:
    """Test TradingDecision class"""

    def test_decision_creation(self):
        """Test TradingDecision creation"""
        from src.model.llm_agent import TradingDecision, DecisionConfidence
        from src.action.futu_executor import TradingAction

        decision = TradingDecision(
            action=TradingAction.LONG,
            symbol="TQQQ",
            futu_code="US.TQQQ",
            confidence=DecisionConfidence.HIGH,
            confidence_score=0.85,
            reasoning="Strong bullish momentum"
        )

        assert decision.action == TradingAction.LONG
        assert decision.confidence == DecisionConfidence.HIGH
        assert decision.should_execute is True

    def test_decision_should_execute(self):
        """Test should_execute property"""
        from src.model.llm_agent import TradingDecision, DecisionConfidence
        from src.action.futu_executor import TradingAction

        # Should execute: high confidence, non-hold action
        exec_decision = TradingDecision(
            action=TradingAction.LONG,
            symbol="TQQQ",
            futu_code="US.TQQQ",
            confidence=DecisionConfidence.HIGH,
            confidence_score=0.85
        )
        assert exec_decision.should_execute is True

        # Should not execute: HOLD action
        hold_decision = TradingDecision(
            action=TradingAction.HOLD,
            symbol="TQQQ",
            futu_code="US.TQQQ",
            confidence=DecisionConfidence.HIGH,
            confidence_score=0.85
        )
        assert hold_decision.should_execute is False

        # Should not execute: low confidence
        low_conf_decision = TradingDecision(
            action=TradingAction.LONG,
            symbol="TQQQ",
            futu_code="US.TQQQ",
            confidence=DecisionConfidence.LOW,
            confidence_score=0.4
        )
        assert low_conf_decision.should_execute is False

    def test_decision_to_dict(self):
        """Test decision dictionary conversion"""
        from src.model.llm_agent import TradingDecision, DecisionConfidence
        from src.action.futu_executor import TradingAction

        decision = TradingDecision(
            action=TradingAction.LONG,
            symbol="TQQQ",
            futu_code="US.TQQQ",
            confidence=DecisionConfidence.HIGH,
            confidence_score=0.85,
            position_size_pct=5.0,
            stop_loss=49.00,
            take_profit=52.00,
            reasoning="Test reasoning"
        )

        data = decision.to_dict()

        assert data["action"] == "long"
        assert data["symbol"] == "TQQQ"
        assert data["confidence"] == "high"
        assert data["confidence_score"] == 0.85
        assert data["stop_loss"] == 49.00


class TestDecisionConfidence:
    """Test DecisionConfidence enum"""

    def test_confidence_levels(self):
        """Test confidence level values"""
        from src.model.llm_agent import DecisionConfidence

        assert DecisionConfidence.HIGH.value == "high"
        assert DecisionConfidence.MEDIUM.value == "medium"
        assert DecisionConfidence.LOW.value == "low"


class TestLLMAgent:
    """Test LLMAgent class"""

    def test_agent_initialization_openai(self, mock_settings):
        """Test agent initialization with OpenAI"""
        with patch('src.model.llm_agent.get_settings', return_value=mock_settings):
            with patch('src.model.llm_agent.OpenAI') as mock_openai:
                with patch('src.model.llm_agent.AsyncOpenAI') as mock_async:
                    from src.model.llm_agent import LLMAgent

                    agent = LLMAgent(provider="openai", api_key="test-key")

                    assert agent.provider == "openai"

    def test_parse_json_decision(self, mock_settings, sample_market_snapshot):
        """Test parsing JSON decision from LLM response"""
        with patch('src.model.llm_agent.get_settings', return_value=mock_settings):
            with patch('src.model.llm_agent.OpenAI'):
                with patch('src.model.llm_agent.AsyncOpenAI'):
                    from src.model.llm_agent import LLMAgent
                    from src.action.futu_executor import TradingAction

                    agent = LLMAgent(provider="openai", api_key="test-key")

                    response = '''
                    Based on my analysis:
                    {
                        "action": "long",
                        "confidence": 0.75,
                        "position_size_pct": 5,
                        "entry_price": 50.00,
                        "stop_loss": 49.00,
                        "take_profit": 52.00,
                        "reasoning": "Strong bullish momentum",
                        "key_factors": ["MACD bullish", "RSI neutral"],
                        "risk_level": "medium"
                    }
                    '''

                    decision = agent._parse_decision(
                        response=response,
                        snapshot=sample_market_snapshot,
                        latency_ms=500.0
                    )

                    assert decision.action == TradingAction.LONG
                    assert decision.confidence_score == 0.75
                    assert decision.stop_loss == 49.00

    def test_parse_text_decision(self, mock_settings, sample_market_snapshot):
        """Test parsing text decision when JSON is not available"""
        with patch('src.model.llm_agent.get_settings', return_value=mock_settings):
            with patch('src.model.llm_agent.OpenAI'):
                with patch('src.model.llm_agent.AsyncOpenAI'):
                    from src.model.llm_agent import LLMAgent
                    from src.action.futu_executor import TradingAction

                    agent = LLMAgent(provider="openai", api_key="test-key")

                    # No JSON in response
                    response = "Based on the strong bullish momentum, I recommend going LONG with high confidence."

                    decision = agent._parse_decision(
                        response=response,
                        snapshot=sample_market_snapshot,
                        latency_ms=500.0
                    )

                    assert decision.action == TradingAction.LONG

    def test_decision_stats(self, mock_settings, sample_market_snapshot):
        """Test decision statistics tracking"""
        with patch('src.model.llm_agent.get_settings', return_value=mock_settings):
            with patch('src.model.llm_agent.OpenAI'):
                with patch('src.model.llm_agent.AsyncOpenAI'):
                    from src.model.llm_agent import LLMAgent, TradingDecision, DecisionConfidence
                    from src.action.futu_executor import TradingAction

                    agent = LLMAgent(provider="openai", api_key="test-key")

                    # Add some decisions to history
                    for _ in range(5):
                        agent._decision_history.append(TradingDecision(
                            action=TradingAction.LONG,
                            symbol="TQQQ",
                            futu_code="US.TQQQ",
                            confidence=DecisionConfidence.HIGH,
                            confidence_score=0.8,
                            model_latency_ms=500.0
                        ))

                    stats = agent.get_decision_stats()

                    assert stats["total_decisions"] == 5
                    assert stats["avg_confidence"] == 0.8
                    assert stats["action_distribution"]["long"] == 5


class TestPromptTemplates:
    """Test PromptTemplates class"""

    def test_system_prompt(self):
        """Test system prompt generation"""
        from src.model.prompts import PromptTemplates

        templates = PromptTemplates()
        prompt = templates.system_prompt()

        assert "trading agent" in prompt.lower()
        assert "LONG" in prompt
        assert "SHORT" in prompt
        assert "FLAT" in prompt
        assert "JSON" in prompt

    def test_trading_analysis_prompt(self, sample_market_snapshot, sample_position):
        """Test trading analysis prompt generation"""
        from src.model.prompts import PromptTemplates

        templates = PromptTemplates()
        prompt = templates.trading_analysis_prompt(
            snapshot=sample_market_snapshot,
            current_position=sample_position,
            portfolio_value=100000.0,
            risk_budget=0.02
        )

        assert "TQQQ" in prompt
        assert "CURRENT POSITION" in prompt
        assert "PORTFOLIO CONTEXT" in prompt
        assert "TRADING RULES" in prompt

    def test_risk_assessment_prompt(self, sample_market_snapshot, sample_position):
        """Test risk assessment prompt generation"""
        from src.model.prompts import PromptTemplates

        templates = PromptTemplates()
        prompt = templates.risk_assessment_prompt(
            snapshot=sample_market_snapshot,
            position=sample_position,
            daily_pnl=-500.0,
            drawdown_pct=0.02
        )

        assert "TQQQ" in prompt
        assert "DRAWDOWN" in prompt
        assert "HOLD" in prompt
        assert "EXIT" in prompt

    def test_market_regime_prompt(self, sample_market_snapshot):
        """Test market regime analysis prompt"""
        from src.model.prompts import PromptTemplates

        templates = PromptTemplates()
        snapshots = {"US.TQQQ": sample_market_snapshot}

        prompt = templates.market_regime_prompt(snapshots)

        assert "TQQQ" in prompt
        assert "BULLISH" in prompt
        assert "BEARISH" in prompt
