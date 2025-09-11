"""
Tests for the CompensationAnalystAgent.

This module contains comprehensive tests for the compensation analyst agent,
including workflow orchestration, state management, error handling, and 
LangGraph integration. Tests ensure zero-warning compliance with project standards.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, Optional

from agents.compensation_analyst_agent import (
    CompensationAnalystAgent,
    CompensationAnalysisState,
    MIN_CONFIDENCE_THRESHOLD,
    RETRY_CONFIDENCE_THRESHOLD,
)
from src.schemas.core import CompBand


class TestCompensationAnalysisState:
    """Test suite for CompensationAnalysisState TypedDict."""

    def test_state_creation(self):
        """Test creating a valid state dictionary."""
        state = CompensationAnalysisState(
            job_title="Software Engineer",
            location="California",
            status="initialized",
            error_message=None,
            retry_count=0,
            soc_mapping=None,
            validation_report=None,
            geographic_info=None,
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        assert state["job_title"] == "Software Engineer"
        assert state["location"] == "California"
        assert state["status"] == "initialized"
        assert state["retry_count"] == 0

    def test_state_optional_fields(self):
        """Test state with optional fields."""
        state = CompensationAnalysisState(
            job_title="Data Scientist",
            location=None,  # Optional field
            status="processing",
            error_message="Test error",
            retry_count=1,
            soc_mapping={"soc_code": "15-2051", "confidence": 0.9},
            validation_report={"valid": True},
            geographic_info={"code": "national"},
            comp_band={"p50": 120000},
            analysis_metadata={"version": "1.0"},
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:05:00Z",
        )
        
        assert state["location"] is None
        assert state["error_message"] == "Test error"
        assert state["soc_mapping"]["confidence"] == 0.9


class TestCompensationAnalystAgent:
    """Test suite for CompensationAnalystAgent."""

    @pytest.fixture
    def agent(self) -> CompensationAnalystAgent:
        """Create a CompensationAnalystAgent instance for testing."""
        return CompensationAnalystAgent(
            min_confidence=0.6,
            max_retries=2,
            enable_checkpoints=False,  # Disable for testing
        )

    @pytest.fixture
    def mock_compensation_tool(self, agent):
        """Mock compensation analysis tool."""
        mock_tool = Mock()
        # Replace the actual tool instance
        agent.compensation_tool = mock_tool
        yield mock_tool

    @pytest.fixture
    def sample_comp_band(self) -> CompBand:
        """Create a sample CompBand for testing."""
        return CompBand(
            occupation_code="15-1252",
            geography="California",
            p25=125000.0,
            p50=155000.0,
            p75=195000.0,
            sources=["https://www.bls.gov/oes/current/oes151252.htm"],
            as_of=datetime(2024, 5, 1, tzinfo=timezone.utc),
            currency="USD",
        )

    # Agent Initialization Tests

    def test_agent_initialization_default(self):
        """Test agent initialization with default parameters."""
        agent = CompensationAnalystAgent()
        
        assert agent.min_confidence == MIN_CONFIDENCE_THRESHOLD
        assert agent.max_retries == 3
        assert hasattr(agent, 'compensation_tool')
        assert hasattr(agent, 'graph')
        assert hasattr(agent, 'compiled_graph')

    def test_agent_initialization_custom_params(self):
        """Test agent initialization with custom parameters."""
        agent = CompensationAnalystAgent(
            min_confidence=0.7,
            max_retries=5,
            enable_checkpoints=True,
        )
        
        assert agent.min_confidence == 0.7
        assert agent.max_retries == 5
        assert agent.checkpointer is not None

    def test_agent_graph_structure(self, agent: CompensationAnalystAgent):
        """Test that the workflow graph has correct structure."""
        # Test that all required nodes are present
        graph = agent.graph
        node_names = set(graph.nodes.keys())
        
        expected_nodes = {
            "initialize", "validate_input", "analyze_compensation",
            "validate_results", "handle_error", "finalize"
        }
        
        assert expected_nodes.issubset(node_names)

    # Individual Node Tests

    def test_initialize_analysis_node(self, agent: CompensationAnalystAgent):
        """Test the initialize analysis workflow node."""
        input_state = CompensationAnalysisState(
            job_title="Software Engineer",
            location="California",
            status="initialized",
            error_message=None,
            retry_count=0,
            soc_mapping=None,
            validation_report=None,
            geographic_info=None,
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._initialize_analysis(input_state)
        
        assert result_state["status"] == "initializing"
        assert result_state["retry_count"] == 0
        assert result_state["error_message"] is None
        assert result_state["started_at"] is not None
        assert "agent_config" in result_state["analysis_metadata"]

    def test_validate_input_node_success(self, agent: CompensationAnalystAgent, mock_compensation_tool):
        """Test successful input validation."""
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": True,
            "soc_mapping_confidence": 0.9,
            "valid_location": True,
            "geographic_code": "CA",
            "warnings": [],
            "soc_code": "15-1252",
        }
        
        input_state = CompensationAnalysisState(
            job_title="Software Engineer",
            location="California",
            status="initializing",
            error_message=None,
            retry_count=0,
            soc_mapping=None,
            validation_report=None,
            geographic_info=None,
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._validate_input(input_state)
        
        assert result_state["status"] == "input_validated"
        assert result_state["soc_mapping"]["confidence"] == 0.9
        assert result_state["geographic_info"]["geographic_code"] == "CA"
        assert result_state["error_message"] is None

    def test_validate_input_node_invalid_title(self, agent: CompensationAnalystAgent, mock_compensation_tool):
        """Test input validation with invalid job title."""
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": False,
            "soc_mapping_confidence": 0.0,
            "valid_location": True,
            "geographic_code": "national",
            "warnings": ["Job title is required"],
        }
        
        input_state = CompensationAnalysisState(
            job_title="",
            location=None,
            status="initializing",
            error_message=None,
            retry_count=0,
            soc_mapping=None,
            validation_report=None,
            geographic_info=None,
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._validate_input(input_state)
        
        assert result_state["status"] == "validation_failed"
        assert result_state["error_message"] == "Invalid or missing job title"

    def test_validate_input_node_low_confidence(self, agent: CompensationAnalystAgent, mock_compensation_tool):
        """Test input validation with low SOC mapping confidence."""
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": True,
            "soc_mapping_confidence": 0.3,  # Below threshold
            "valid_location": True,
            "geographic_code": "national",
            "warnings": [],
        }
        
        input_state = CompensationAnalysisState(
            job_title="Unicorn Wrangler",
            location=None,
            status="initializing",
            error_message=None,
            retry_count=0,
            soc_mapping=None,
            validation_report=None,
            geographic_info=None,
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._validate_input(input_state)
        
        assert result_state["status"] == "validation_failed"
        assert "No SOC mapping found" in result_state["error_message"]

    def test_analyze_compensation_node_success(self, agent: CompensationAnalystAgent, mock_compensation_tool, sample_comp_band):
        """Test successful compensation analysis."""
        mock_compensation_tool.analyze_compensation.return_value = sample_comp_band
        
        input_state = CompensationAnalysisState(
            job_title="Software Engineer",
            location="California",
            status="input_validated",
            error_message=None,
            retry_count=0,
            soc_mapping={"soc_code": "15-1252", "confidence": 0.9},
            validation_report={},
            geographic_info={"geographic_code": "CA"},
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._analyze_compensation(input_state)
        
        assert result_state["status"] == "analysis_completed"
        assert result_state["comp_band"] is not None
        assert result_state["comp_band"]["occupation_code"] == "15-1252"
        assert result_state["comp_band"]["p50"] == 155000.0
        assert result_state["error_message"] is None

    def test_analyze_compensation_node_failure(self, agent: CompensationAnalystAgent, mock_compensation_tool):
        """Test compensation analysis failure."""
        # Ensure the mock raises an exception
        mock_compensation_tool.analyze_compensation = Mock(side_effect=RuntimeError("Analysis failed"))
        
        input_state = CompensationAnalysisState(
            job_title="Software Engineer",
            location="California",
            status="input_validated",
            error_message=None,
            retry_count=0,
            soc_mapping={"soc_code": "15-1252", "confidence": 0.9},
            validation_report={},
            geographic_info={"geographic_code": "CA"},
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._analyze_compensation(input_state)
        
        assert result_state["status"] == "analysis_failed"
        assert "Analysis failed" in result_state["error_message"]
        assert result_state["comp_band"] is None

    def test_validate_results_node_success(self, agent: CompensationAnalystAgent):
        """Test successful results validation."""
        comp_band_data = {
            "occupation_code": "15-1252",
            "geography": "California",
            "p25": 125000.0,
            "p50": 155000.0,
            "p75": 195000.0,
            "sources": ["https://www.bls.gov/oes/"],
            "as_of": "2024-05-01T00:00:00+00:00",
            "currency": "USD",
        }
        
        input_state = CompensationAnalysisState(
            job_title="Software Engineer",
            location="California",
            status="analysis_completed",
            error_message=None,
            retry_count=0,
            soc_mapping={"confidence": 0.9},
            validation_report={},
            geographic_info={},
            comp_band=comp_band_data,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._validate_results(input_state)
        
        assert result_state["status"] == "results_validated"
        assert "results_validation" in result_state["validation_report"]
        assert result_state["error_message"] is None

    def test_validate_results_node_no_data(self, agent: CompensationAnalystAgent):
        """Test results validation with no compensation data."""
        input_state = CompensationAnalysisState(
            job_title="Software Engineer",
            location="California",
            status="analysis_completed",
            error_message=None,
            retry_count=0,
            soc_mapping={},
            validation_report={},
            geographic_info={},
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._validate_results(input_state)
        
        assert result_state["status"] == "results_validation_failed"
        assert "No compensation data available" in result_state["error_message"]

    def test_handle_error_node(self, agent: CompensationAnalystAgent):
        """Test error handling node."""
        input_state = CompensationAnalysisState(
            job_title="Software Engineer",
            location="California",
            status="analysis_failed",
            error_message="Test error",
            retry_count=0,
            soc_mapping=None,
            validation_report=None,
            geographic_info=None,
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._handle_error(input_state)
        
        assert result_state["retry_count"] == 1
        assert result_state["status"] == "retrying"

    def test_handle_error_max_retries(self, agent: CompensationAnalystAgent):
        """Test error handling when max retries exceeded."""
        input_state = CompensationAnalysisState(
            job_title="Software Engineer",
            location="California",
            status="analysis_failed",
            error_message="Test error",
            retry_count=2,  # Already at max for this agent (max_retries=2)
            soc_mapping=None,
            validation_report=None,
            geographic_info=None,
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._handle_error(input_state)
        
        assert result_state["retry_count"] == 3
        assert result_state["status"] == "max_retries_exceeded"

    def test_finalize_results_node(self, agent: CompensationAnalystAgent):
        """Test results finalization node."""
        comp_band_data = {
            "occupation_code": "15-1252",
            "p50": 155000.0,
            "currency": "USD",
        }
        
        input_state = CompensationAnalysisState(
            job_title="Software Engineer",
            location="California",
            status="results_validated",
            error_message=None,
            retry_count=1,
            soc_mapping={},
            validation_report={},
            geographic_info={},
            comp_band=comp_band_data,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )
        
        result_state = agent._finalize_results(input_state)
        
        assert result_state["status"] == "completed"
        assert result_state["completed_at"] is not None
        assert "completion_summary" in result_state["analysis_metadata"]
        assert result_state["analysis_metadata"]["completion_summary"]["total_retries"] == 1

    # Routing Tests

    def test_input_validation_routing(self, agent: CompensationAnalystAgent):
        """Test input validation routing logic."""
        # Test successful validation
        state = {"status": "input_validated", "retry_count": 0}
        assert agent._input_validation_routing(state) == "proceed"
        
        # Test validation failure with retries available
        state = {"status": "validation_failed", "retry_count": 0}
        assert agent._input_validation_routing(state) == "retry"
        
        # Test validation failure with max retries exceeded
        state = {"status": "validation_failed", "retry_count": 5}
        assert agent._input_validation_routing(state) == "failed"

    def test_analysis_routing(self, agent: CompensationAnalystAgent):
        """Test analysis routing logic."""
        # Test successful analysis
        state = {"status": "analysis_completed", "retry_count": 0}
        assert agent._analysis_routing(state) == "validate"
        
        # Test analysis failure with retries available
        state = {"status": "analysis_failed", "retry_count": 0}
        assert agent._analysis_routing(state) == "retry"
        
        # Test analysis failure with max retries exceeded
        state = {"status": "analysis_failed", "retry_count": 5}
        assert agent._analysis_routing(state) == "failed"

    def test_validation_routing(self, agent: CompensationAnalystAgent):
        """Test validation routing logic."""
        # Test successful validation
        state = {"status": "results_validated", "retry_count": 0}
        assert agent._validation_routing(state) == "success"
        
        # Test validation issues but accept results anyway
        state = {"status": "results_validation_failed", "retry_count": 5}
        assert agent._validation_routing(state) == "success"

    def test_error_handling_routing(self, agent: CompensationAnalystAgent):
        """Test error handling routing logic."""
        # Test retry decision
        state = {"status": "retrying"}
        assert agent._error_handling_routing(state) == "retry"
        
        # Test failure decision
        state = {"status": "max_retries_exceeded"}
        assert agent._error_handling_routing(state) == "failed"

    # Utility Method Tests

    def test_calculate_completeness_score(self, agent: CompensationAnalystAgent):
        """Test data completeness score calculation."""
        # Test complete data
        comp_band = {
            "p25": 100000,
            "p50": 130000,
            "p75": 170000,
            "occupation_code": "15-1252",
            "sources": ["http://example.com"],
        }
        score = agent._calculate_completeness_score(comp_band)
        assert score == 1.0
        
        # Test partial data
        comp_band = {
            "p50": 130000,
            "occupation_code": "15-1252",
        }
        score = agent._calculate_completeness_score(comp_band)
        assert score == 0.4  # 2 out of 5 checks

    # Integration Tests (Full Workflow)

    @pytest.mark.integration
    def test_full_workflow_success(self, agent: CompensationAnalystAgent, mock_compensation_tool, sample_comp_band):
        """Test complete successful workflow execution."""
        # Setup mock responses
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": True,
            "soc_mapping_confidence": 0.9,
            "valid_location": True,
            "geographic_code": "CA",
            "warnings": [],
            "soc_code": "15-1252",
        }
        mock_compensation_tool.analyze_compensation.return_value = sample_comp_band
        
        # Execute full workflow
        result = agent.analyze_compensation("Software Engineer", "California")
        
        assert result["status"] == "completed"
        assert "comp_band_object" in result
        assert isinstance(result["comp_band_object"], CompBand)
        assert result["comp_band_object"].p50 == 155000.0

    @pytest.mark.integration 
    def test_full_workflow_with_retries(self, agent: CompensationAnalystAgent, mock_compensation_tool, sample_comp_band):
        """Test workflow execution with retries."""
        # Setup mock to fail first, then succeed
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": True,
            "soc_mapping_confidence": 0.9,
            "valid_location": True,
            "geographic_code": "CA",
            "warnings": [],
            "soc_code": "15-1252",
        }
        
        # First call fails, second succeeds
        mock_compensation_tool.analyze_compensation = Mock(side_effect=[
            RuntimeError("Temporary failure"),
            sample_comp_band
        ])
        
        result = agent.analyze_compensation("Software Engineer", "California")
        
        assert result["status"] == "completed"
        assert result["analysis_metadata"]["completion_summary"]["total_retries"] >= 1

    @pytest.mark.integration
    def test_full_workflow_failure(self, agent: CompensationAnalystAgent, mock_compensation_tool):
        """Test complete workflow failure after max retries."""
        # Setup validation to fail
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": False,
            "soc_mapping_confidence": 0.0,
            "warnings": ["Job title is required"],
        }
        
        result = agent.analyze_compensation("", "California")
        
        assert result["status"] == "max_retries_exceeded"
        assert "error_message" in result

    # Synchronous Interface Tests

    def test_synchronous_analyze_compensation_success(self, agent: CompensationAnalystAgent, mock_compensation_tool, sample_comp_band):
        """Test synchronous analysis interface."""
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": True,
            "soc_mapping_confidence": 0.9,
            "valid_location": True,
            "geographic_code": "CA",
            "warnings": [],
            "soc_code": "15-1252",
        }
        mock_compensation_tool.analyze_compensation.return_value = sample_comp_band
        
        result = agent.analyze_compensation("Data Scientist", "New York")
        
        assert isinstance(result, dict)
        assert result["job_title"] == "Data Scientist"
        assert result["location"] == "New York"
        assert "comp_band_object" in result

    def test_synchronous_analyze_with_thread_id(self, agent: CompensationAnalystAgent, mock_compensation_tool, sample_comp_band):
        """Test synchronous analysis with thread ID for checkpointing."""
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": True,
            "soc_mapping_confidence": 0.9,
            "valid_location": True,
            "geographic_code": "national",
            "warnings": [],
            "soc_code": "15-2051",
        }
        mock_compensation_tool.analyze_compensation.return_value = sample_comp_band
        
        result = agent.analyze_compensation("Data Scientist", thread_id="test-thread-123")
        
        assert result["job_title"] == "Data Scientist"
        # Should complete successfully with thread ID

    # Asynchronous Interface Tests

    @pytest.mark.asyncio
    async def test_asynchronous_analyze_compensation_success(self, agent: CompensationAnalystAgent, mock_compensation_tool, sample_comp_band):
        """Test asynchronous analysis interface."""
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": True,
            "soc_mapping_confidence": 0.9,
            "valid_location": True,
            "geographic_code": "CA",
            "warnings": [],
            "soc_code": "15-1252",
        }
        mock_compensation_tool.analyze_compensation.return_value = sample_comp_band
        
        result = await agent.analyze_compensation_async("Software Engineer", "California")
        
        assert isinstance(result, dict)
        assert result["job_title"] == "Software Engineer"
        assert result["location"] == "California"
        assert "comp_band_object" in result

    @pytest.mark.asyncio
    async def test_asynchronous_analyze_with_error(self, agent: CompensationAnalystAgent, mock_compensation_tool):
        """Test asynchronous analysis with error handling."""
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": True,
            "soc_mapping_confidence": 0.2,  # Low confidence
            "warnings": [],
        }
        
        result = await agent.analyze_compensation_async("Unknown Job Title")
        
        assert result["status"] in ["max_retries_exceeded", "workflow_failed"]
        assert "error_message" in result

    # Error Handling and Edge Cases

    def test_workflow_exception_handling(self, agent: CompensationAnalystAgent):
        """Test workflow exception handling."""
        # Create a mock that raises an exception during graph compilation/execution
        with patch.object(agent.compiled_graph, 'invoke', side_effect=Exception("Workflow error")):
            result = agent.analyze_compensation("Software Engineer", "California")
            
            assert result["status"] == "workflow_failed"
            assert "Workflow error" in result["error_message"]

    @pytest.mark.asyncio
    async def test_async_workflow_exception_handling(self, agent: CompensationAnalystAgent):
        """Test async workflow exception handling."""
        with patch.object(agent.compiled_graph, 'ainvoke', side_effect=Exception("Async workflow error")):
            result = await agent.analyze_compensation_async("Software Engineer", "California")
            
            assert result["status"] == "workflow_failed"
            assert "Async workflow error" in result["error_message"]

    # Configuration and State Tests

    def test_agent_configuration_validation(self):
        """Test agent configuration validation."""
        # Test invalid configuration
        agent = CompensationAnalystAgent(
            min_confidence=1.5,  # Invalid value > 1.0
            max_retries=-1,      # Invalid negative value
        )
        
        # Agent should still initialize but may have unexpected behavior
        assert hasattr(agent, 'min_confidence')
        assert hasattr(agent, 'max_retries')

    def test_state_serialization(self, agent: CompensationAnalystAgent, sample_comp_band):
        """Test proper state serialization for CompBand objects."""
        # Test that CompBand gets properly serialized and deserialized
        comp_band_dict = sample_comp_band.model_dump()
        comp_band_dict["as_of"] = sample_comp_band.as_of.isoformat()
        comp_band_dict["sources"] = [str(url) for url in sample_comp_band.sources]
        
        # Should be able to recreate CompBand from serialized data
        recreated_comp_band = CompBand(**{
            **comp_band_dict,
            "as_of": datetime.fromisoformat(comp_band_dict["as_of"])
        })
        
        assert recreated_comp_band.occupation_code == sample_comp_band.occupation_code
        assert recreated_comp_band.p50 == sample_comp_band.p50

    # Performance and Load Tests

    @pytest.mark.performance
    def test_multiple_concurrent_analyses(self, mock_compensation_tool, sample_comp_band):
        """Test multiple concurrent compensation analyses."""
        mock_compensation_tool.validate_inputs.return_value = {
            "valid_job_title": True,
            "soc_mapping_confidence": 0.9,
            "valid_location": True,
            "geographic_code": "CA",
            "warnings": [],
            "soc_code": "15-1252",
        }
        mock_compensation_tool.analyze_compensation.return_value = sample_comp_band
        
        agents = [CompensationAnalystAgent(enable_checkpoints=False) for _ in range(3)]
        job_titles = ["Software Engineer", "Data Scientist", "Product Manager"]
        
        results = []
        for agent, title in zip(agents, job_titles):
            result = agent.analyze_compensation(title, "California")
            results.append(result)
        
        assert len(results) == 3
        assert all(r["status"] == "completed" for r in results)
        assert results[0]["job_title"] == "Software Engineer"
        assert results[1]["job_title"] == "Data Scientist"
        assert results[2]["job_title"] == "Product Manager"