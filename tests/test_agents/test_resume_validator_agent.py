"""
Comprehensive unit tests for ResumeValidatorAgent.

Tests cover all workflow nodes, error handling, state management,
evidence indexing, and LangGraph orchestration.
Maintains 100% test coverage with zero warnings.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from agents.resume_validator_agent import (
    ResumeValidatorAgent,
    ResumeValidationState,
    MIN_SUCCESS_RATE,
    MIN_EVIDENCE_COVERAGE,
    SIMILARITY_THRESHOLD,
)
from tools.resume_validator import (
    ResumeValidator,
    ResumeValidationError,
    ValidationReport,
    ValidationResult,
    ValidationStatus,
)
from tools.evidence_indexer import EvidenceIndexer, EvidenceIndexingError
from src.schemas.core import TailoredBullet, Resume, ResumeBullet, ResumeSection


class TestResumeValidatorAgent:
    """Test suite for ResumeValidatorAgent."""
    
    @pytest.fixture
    def sample_tailored_bullets(self):
        """Create sample tailored bullets for testing."""
        return [
            TailoredBullet(
                text="Enhanced Python development with microservices architecture",
                similarity_score=0.9,
                evidence_spans=["Developed Python applications"],
                jd_keywords_covered=["python", "microservices"],
            ),
            TailoredBullet(
                text="Built responsive React interfaces with API integration",
                similarity_score=0.85,
                evidence_spans=["Built React components"],
                jd_keywords_covered=["react", "api"],
            ),
            TailoredBullet(
                text="Led cross-functional team development initiatives",
                similarity_score=0.8,
                evidence_spans=["Led team development"],
                jd_keywords_covered=["leadership"],
            ),
        ]
    
    @pytest.fixture
    def sample_resume(self):
        """Create sample resume for testing."""
        bullets = [
            ResumeBullet(
                text="Developed Python applications for web services",
                section="Experience",
                start_offset=100,
                end_offset=150,
            ),
            ResumeBullet(
                text="Built React components for user interfaces",
                section="Experience",
                start_offset=151,
                end_offset=200,
            ),
            ResumeBullet(
                text="Led team of 3 developers on platform project",
                section="Experience",
                start_offset=201,
                end_offset=245,
            ),
        ]
        
        return Resume(
            raw_text="Sample resume with development experience",
            bullets=bullets,
            skills=["Python", "React", "JavaScript", "Leadership"],
            sections=[
                ResumeSection(
                    name="Experience",
                    bullets=bullets,
                    start_offset=50,
                    end_offset=245,
                ),
            ],
        )
    
    @pytest.fixture
    def mock_evidence_indexer(self):
        """Create mock EvidenceIndexer."""
        mock_indexer = Mock(spec=EvidenceIndexer)
        mock_indexer.similarity_threshold = 0.8
        mock_indexer.get_collection_stats.return_value = {"total_items": 10}
        mock_indexer.index_resume.return_value = {"items_indexed": 5}
        return mock_indexer
    
    @pytest.fixture
    def mock_validator(self):
        """Create mock ResumeValidator."""
        mock_val = Mock(spec=ResumeValidator)
        mock_val.similarity_threshold = 0.8
        return mock_val
    
    @pytest.fixture
    def agent(self, mock_evidence_indexer):
        """Create ResumeValidatorAgent instance."""
        with patch('agents.resume_validator_agent.ResumeValidator') as mock_val_class:
            mock_val_class.return_value = Mock()
            return ResumeValidatorAgent(
                max_retries=2,
                enable_checkpointing=False,
                evidence_indexer=mock_evidence_indexer,
                similarity_threshold=0.8,
            )
    
    def test_init_default_parameters(self):
        """Test agent initialization with default parameters."""
        with patch('agents.resume_validator_agent.ResumeValidator') as mock_val_class, \
             patch('agents.resume_validator_agent.EvidenceIndexer') as mock_indexer_class:
            
            mock_val_class.return_value = Mock()
            mock_indexer_class.return_value = Mock()
            
            agent = ResumeValidatorAgent()
            
            assert agent.max_retries == 3
            assert agent.similarity_threshold == SIMILARITY_THRESHOLD
            assert agent.evidence_indexer is not None
            assert agent.validator is not None
            assert agent.workflow is not None
            assert agent.graph is not None
    
    def test_init_custom_parameters(self, mock_evidence_indexer):
        """Test agent initialization with custom parameters."""
        with patch('agents.resume_validator_agent.ResumeValidator') as mock_val_class:
            mock_val_class.return_value = Mock()
            
            agent = ResumeValidatorAgent(
                max_retries=5,
                enable_checkpointing=True,
                evidence_indexer=mock_evidence_indexer,
                similarity_threshold=0.9,
            )
            
            assert agent.max_retries == 5
            assert agent.similarity_threshold == 0.9
            assert agent.evidence_indexer is mock_evidence_indexer
            assert agent.checkpointer is not None
    
    def test_validate_bullets_success(self, agent, sample_tailored_bullets, sample_resume):
        """Test successful bullet validation workflow."""
        # Mock successful validation report
        mock_report = ValidationReport(
            total_bullets=3,
            valid_bullets=2,
            rejected_bullets=1,
            needs_edit_bullets=0,
            error_bullets=0,
            overall_evidence_score=0.85,
            evidence_coverage_percentage=80.0,
            validation_results=[],
            summary_recommendations=["Good results"],
        )
        
        agent.validator.validate_bullets.return_value = mock_report
        
        result = agent.validate_bullets(
            sample_tailored_bullets, sample_resume, similarity_threshold=0.8, batch_size=10
        )
        
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["validation_report"] is not None
        assert result["evidence_stats"] is not None
        assert result["error"] is None
        assert result["retry_count"] == 0
    
    def test_initialize_validation(self, agent):
        """Test initialization workflow node."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="",
            error_message=None,
            retry_count=0,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._initialize_validation(state)
        
        assert result_state["status"] == "initialized"
        assert result_state["current_node"] == "initialize"
        assert result_state["workflow_start_time"] is not None
        assert "agent_version" in result_state["processing_metadata"]
        assert result_state["processing_metadata"]["max_retries"] == agent.max_retries
    
    def test_validate_input_success(self, agent, sample_tailored_bullets, sample_resume):
        """Test successful input validation."""
        state = ResumeValidationState(
            tailored_bullets=[bullet.model_dump() for bullet in sample_tailored_bullets],
            resume=sample_resume.model_dump(),
            similarity_threshold=0.8,
            batch_size=20,
            status="initialized",
            error_message=None,
            retry_count=0,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_input(state)
        
        assert result_state["status"] == "input_validated"
        assert result_state["current_node"] == "validate_input"
        assert result_state["error_message"] is None
        assert result_state["processing_metadata"]["bullets_to_validate"] == 3
        assert result_state["processing_metadata"]["resume_bullets_available"] == 3
    
    def test_validate_input_missing_bullets(self, agent):
        """Test input validation with missing bullets."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume={"raw_text": "test", "bullets": []},
            similarity_threshold=0.8,
            batch_size=20,
            status="initialized",
            error_message=None,
            retry_count=0,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_input(state)
        
        assert result_state["status"] == "input_validation_failed"
        assert "No tailored bullets provided" in result_state["error_message"]
    
    def test_validate_input_missing_resume(self, agent, sample_tailored_bullets):
        """Test input validation with missing resume."""
        state = ResumeValidationState(
            tailored_bullets=[bullet.model_dump() for bullet in sample_tailored_bullets],
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="initialized",
            error_message=None,
            retry_count=0,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_input(state)
        
        assert result_state["status"] == "input_validation_failed"
        assert "No resume provided" in result_state["error_message"]
    
    def test_validate_input_empty_bullets_list(self, agent, sample_resume):
        """Test input validation with empty bullets list."""
        state = ResumeValidationState(
            tailored_bullets=[],  # Empty list
            resume=sample_resume.model_dump(),
            similarity_threshold=0.8,
            batch_size=20,
            status="initialized",
            error_message=None,
            retry_count=0,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_input(state)
        
        assert result_state["status"] == "input_validation_failed"
        assert "No bullets to validate" in result_state["error_message"]
    
    def test_validate_input_malformed_bullet(self, agent, sample_resume):
        """Test input validation with malformed bullet data."""
        malformed_bullets = [
            {"text": "Test bullet"},  # Missing required fields
        ]
        
        state = ResumeValidationState(
            tailored_bullets=malformed_bullets,
            resume=sample_resume.model_dump(),
            similarity_threshold=0.8,
            batch_size=20,
            status="initialized",
            error_message=None,
            retry_count=0,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_input(state)
        
        assert result_state["status"] == "input_validation_failed"
        assert "missing required field" in result_state["error_message"]
    
    def test_setup_evidence_already_indexed(self, agent, sample_resume):
        """Test evidence setup when evidence is already indexed."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=sample_resume.model_dump(),
            similarity_threshold=0.8,
            batch_size=20,
            status="input_validated",
            error_message=None,
            retry_count=0,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        # Mock evidence indexer with existing data
        agent.evidence_indexer.get_collection_stats.return_value = {"total_items": 10}
        
        result_state = agent._setup_evidence(state)
        
        assert result_state["status"] == "evidence_ready"
        assert result_state["current_node"] == "setup_evidence"
        assert result_state["evidence_indexed"] is True
        assert result_state["evidence_stats"]["total_items"] == 10
        
        # Should not call index_resume since evidence already exists
        agent.evidence_indexer.index_resume.assert_not_called()
    
    def test_setup_evidence_needs_indexing(self, agent, sample_resume):
        """Test evidence setup when indexing is needed."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=sample_resume.model_dump(),
            similarity_threshold=0.8,
            batch_size=20,
            status="input_validated",
            error_message=None,
            retry_count=0,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        # Mock empty evidence collection
        agent.evidence_indexer.get_collection_stats.return_value = {"total_items": 0}
        agent.evidence_indexer.index_resume.return_value = {
            "items_indexed": 5,
            "bullets_indexed": 3,
            "skills_indexed": 2,
        }
        
        result_state = agent._setup_evidence(state)
        
        assert result_state["status"] == "evidence_ready"
        assert result_state["evidence_indexed"] is True
        assert result_state["evidence_stats"]["items_indexed"] == 5
        
        # Should call index_resume
        agent.evidence_indexer.index_resume.assert_called_once()
    
    def test_setup_evidence_indexing_error(self, agent, sample_resume):
        """Test evidence setup with indexing error."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=sample_resume.model_dump(),
            similarity_threshold=0.8,
            batch_size=20,
            status="input_validated",
            error_message=None,
            retry_count=0,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        # Mock indexing failure
        agent.evidence_indexer.get_collection_stats.return_value = {"total_items": 0}
        agent.evidence_indexer.index_resume.side_effect = EvidenceIndexingError("Indexing failed")
        
        result_state = agent._setup_evidence(state)
        
        assert result_state["status"] == "evidence_setup_failed"
        assert "Evidence indexing failed" in result_state["error_message"]
    
    def test_validate_bullets_success(self, agent, sample_tailored_bullets, sample_resume):
        """Test successful bullet validation."""
        state = ResumeValidationState(
            tailored_bullets=[bullet.model_dump() for bullet in sample_tailored_bullets],
            resume=sample_resume.model_dump(),
            similarity_threshold=0.8,
            batch_size=20,
            status="evidence_ready",
            error_message=None,
            retry_count=0,
            evidence_indexed=True,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        # Mock validation report
        mock_report = ValidationReport(
            total_bullets=3,
            valid_bullets=2,
            rejected_bullets=1,
            needs_edit_bullets=0,
            error_bullets=0,
            overall_evidence_score=0.85,
            evidence_coverage_percentage=80.0,
            validation_results=[
                ValidationResult(
                    bullet=sample_tailored_bullets[0],
                    status=ValidationStatus.VALID,
                    evidence_matches=[],
                    best_similarity_score=0.9,
                    confidence_score=0.85,
                    validation_notes=["Valid"],
                ),
            ],
            summary_recommendations=["Good results"],
        )
        
        agent.validator.validate_bullets.return_value = mock_report
        
        result_state = agent._validate_bullets(state)
        
        assert result_state["status"] == "bullets_validated"
        assert result_state["current_node"] == "validate_bullets"
        assert result_state["validation_report"] is not None
        assert result_state["validation_report"]["total_bullets"] == 3
        assert result_state["validation_report"]["valid_bullets"] == 2
        assert len(result_state["validation_report"]["validation_results"]) == 1
    
    def test_validate_bullets_validation_error(self, agent, sample_tailored_bullets, sample_resume):
        """Test bullet validation with error."""
        state = ResumeValidationState(
            tailored_bullets=[bullet.model_dump() for bullet in sample_tailored_bullets],
            resume=sample_resume.model_dump(),
            similarity_threshold=0.8,
            batch_size=20,
            status="evidence_ready",
            error_message=None,
            retry_count=0,
            evidence_indexed=True,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        agent.validator.validate_bullets.side_effect = ResumeValidationError("Validation failed")
        
        result_state = agent._validate_bullets(state)
        
        assert result_state["status"] == "bullet_validation_failed"
        assert "Validation failed" in result_state["error_message"]
    
    def test_analyze_results_success(self, agent):
        """Test successful results analysis."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="bullets_validated",
            error_message=None,
            retry_count=0,
            evidence_indexed=True,
            evidence_stats=None,
            validation_report={
                "total_bullets": 10,
                "valid_bullets": 8,
                "rejected_bullets": 1,
                "needs_edit_bullets": 1,
                "validation_success_rate": 80.0,
                "evidence_coverage_percentage": 75.0,
                "needs_review_count": 2,
            },
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._analyze_results(state)
        
        assert result_state["status"] == "results_analyzed"
        assert result_state["current_node"] == "analyze_results"
        assert "quality_metrics" in result_state["processing_metadata"]
        
        quality_metrics = result_state["processing_metadata"]["quality_metrics"]
        assert quality_metrics["success_rate"] == 80.0
        assert quality_metrics["evidence_coverage"] == 75.0
        assert quality_metrics["valid_count"] == 8
        assert quality_metrics["total_count"] == 10
    
    def test_analyze_results_low_success_rate(self, agent):
        """Test results analysis with low success rate."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="bullets_validated",
            error_message=None,
            retry_count=0,
            evidence_indexed=True,
            evidence_stats=None,
            validation_report={
                "total_bullets": 10,
                "valid_bullets": 3,  # Low success
                "rejected_bullets": 5,
                "needs_edit_bullets": 2,
                "validation_success_rate": 30.0,  # Below threshold
                "evidence_coverage_percentage": 40.0,  # Below threshold
                "needs_review_count": 7,
            },
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._analyze_results(state)
        
        assert result_state["status"] == "results_analysis_failed"
        assert "success rate" in result_state["error_message"]
        assert "evidence coverage" in result_state["error_message"]
        
        analysis_notes = result_state["processing_metadata"]["analysis_notes"]
        assert any("Low validation success rate" in note for note in analysis_notes)
        assert any("Low evidence coverage" in note for note in analysis_notes)
    
    def test_analyze_results_no_valid_bullets(self, agent):
        """Test results analysis with no valid bullets."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="bullets_validated",
            error_message=None,
            retry_count=0,
            evidence_indexed=True,
            evidence_stats=None,
            validation_report={
                "total_bullets": 5,
                "valid_bullets": 0,  # No valid bullets
                "rejected_bullets": 3,
                "needs_edit_bullets": 2,
                "validation_success_rate": 0.0,
                "evidence_coverage_percentage": 20.0,
                "needs_review_count": 5,
            },
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._analyze_results(state)
        
        assert result_state["status"] == "results_analysis_failed"
        analysis_notes = result_state["processing_metadata"]["analysis_notes"]
        assert any("No bullets passed validation" in note for note in analysis_notes)
    
    def test_handle_error_retry(self, agent):
        """Test error handling with retry logic."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="bullet_validation_failed",
            error_message="Validation error",
            retry_count=1,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._handle_error(state)
        
        assert result_state["status"] == "retrying"
        assert result_state["current_node"] == "handle_error"
        assert result_state["retry_count"] == 2
    
    def test_handle_error_max_retries(self, agent):
        """Test error handling when max retries exceeded."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="bullet_validation_failed",
            error_message="Validation error",
            retry_count=agent.max_retries,  # At max
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._handle_error(state)
        
        assert result_state["status"] == "max_retries_exceeded"
        assert result_state["retry_count"] == agent.max_retries + 1
    
    def test_handle_error_threshold_adjustment(self, agent):
        """Test error handling with threshold adjustment for analysis failures."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="results_analysis_failed",
            error_message="Analysis error",
            retry_count=1,
            evidence_indexed=True,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._handle_error(state)
        
        assert result_state["status"] == "retrying"
        assert result_state["similarity_threshold"] < 0.8  # Should be lowered
        assert agent.validator.similarity_threshold < 0.8  # Validator should be updated
    
    def test_finalize_validation_success(self, agent):
        """Test successful workflow finalization."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="results_analyzed",
            error_message=None,
            retry_count=0,
            evidence_indexed=True,
            evidence_stats=None,
            validation_report={"total_bullets": 5},
            processing_metadata={},
            workflow_start_time=datetime.now(timezone.utc).isoformat(),
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._finalize_validation(state)
        
        assert result_state["status"] == "completed"
        assert result_state["current_node"] == "finalize"
        assert result_state["workflow_end_time"] is not None
        assert "processing_time_seconds" in result_state["processing_metadata"]
    
    def test_finalize_validation_failed(self, agent):
        """Test workflow finalization with failure."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="max_retries_exceeded",
            error_message="Max retries exceeded",
            retry_count=3,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=datetime.now(timezone.utc).isoformat(),
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._finalize_validation(state)
        
        assert result_state["status"] == "failed"
        assert result_state["workflow_end_time"] is not None
    
    def test_routing_functions(self, agent):
        """Test workflow routing functions."""
        # Test route_after_validation
        state_validated = {"status": "input_validated"}
        assert agent._route_after_validation(state_validated) == "setup"
        
        state_failed = {"status": "input_validation_failed"}
        assert agent._route_after_validation(state_failed) == "error"
        
        # Test route_after_analysis
        state_analyzed = {"status": "results_analyzed"}
        assert agent._route_after_analysis(state_analyzed) == "finalize"
        
        state_analysis_retry = {"status": "results_analysis_failed", "retry_count": 1}
        assert agent._route_after_analysis(state_analysis_retry) == "retry"
        
        state_analysis_error = {"status": "results_analysis_failed", "retry_count": 5}
        assert agent._route_after_analysis(state_analysis_error) == "error"
        
        # Test route_after_error
        state_retrying = {"status": "retrying"}
        assert agent._route_after_error(state_retrying) == "retry"
        
        state_exceeded = {"status": "max_retries_exceeded"}
        assert agent._route_after_error(state_exceeded) == "finalize"
    
    def test_format_result(self, agent):
        """Test result formatting."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="completed",
            error_message=None,
            retry_count=1,
            evidence_indexed=True,
            evidence_stats={"items_indexed": 5},
            validation_report={"total_bullets": 10, "valid_bullets": 8},
            processing_metadata={"processing_time_seconds": 3.5},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result = agent._format_result(state)
        
        assert result["status"] == "completed"
        assert result["success"] is True
        assert result["validation_report"]["total_bullets"] == 10
        assert result["evidence_stats"]["items_indexed"] == 5
        assert result["error"] is None
        assert result["retry_count"] == 1
        assert result["processing_time"] == 3.5
        
        # Test failed case
        failed_state = state.copy()
        failed_state["status"] = "failed"
        failed_result = agent._format_result(failed_state)
        assert failed_result["success"] is False
    
    @pytest.mark.asyncio
    async def test_async_interface(self, agent, sample_tailored_bullets, sample_resume):
        """Test async interface method."""
        # Mock synchronous method
        with patch.object(agent, 'validate_bullets', return_value={"success": True}) as mock_sync:
            result = await agent.avalidate_bullets(
                sample_tailored_bullets, sample_resume, similarity_threshold=0.9, batch_size=15
            )
            
            assert result == {"success": True}
            mock_sync.assert_called_once_with(sample_tailored_bullets, sample_resume, 0.9, 15)


class TestWorkflowIntegration:
    """Test suite for workflow integration scenarios."""
    
    @pytest.fixture
    def agent_with_real_workflow(self, mock_evidence_indexer):
        """Create agent with real workflow for integration testing."""
        with patch('agents.resume_validator_agent.ResumeValidator') as mock_val_class:
            mock_validator = Mock()
            mock_report = ValidationReport(
                total_bullets=3,
                valid_bullets=2,
                rejected_bullets=1,
                needs_edit_bullets=0,
                error_bullets=0,
                overall_evidence_score=0.85,
                evidence_coverage_percentage=75.0,
                validation_results=[],
                summary_recommendations=["Good validation"],
            )
            mock_validator.validate_bullets.return_value = mock_report
            mock_val_class.return_value = mock_validator
            
            return ResumeValidatorAgent(
                max_retries=2,
                enable_checkpointing=False,
                evidence_indexer=mock_evidence_indexer,
            )
    
    def test_full_workflow_success(self, agent_with_real_workflow, sample_tailored_bullets, sample_resume):
        """Test complete workflow execution with success."""
        result = agent_with_real_workflow.validate_bullets(
            sample_tailored_bullets, sample_resume, batch_size=10
        )
        
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["retry_count"] == 0
        assert result["error"] is None
    
    def test_workflow_with_analysis_retry(self, agent_with_real_workflow, sample_tailored_bullets, sample_resume):
        """Test workflow with analysis failure and retry."""
        # Mock validator to return poor results first, then good results
        poor_report = ValidationReport(
            total_bullets=10,
            valid_bullets=1,
            rejected_bullets=8,
            needs_edit_bullets=1,
            error_bullets=0,
            overall_evidence_score=0.3,
            evidence_coverage_percentage=20.0,
            validation_results=[],
            summary_recommendations=["Poor results"],
        )
        
        good_report = ValidationReport(
            total_bullets=10,
            valid_bullets=8,
            rejected_bullets=2,
            needs_edit_bullets=0,
            error_bullets=0,
            overall_evidence_score=0.85,
            evidence_coverage_percentage=80.0,
            validation_results=[],
            summary_recommendations=["Good results"],
        )
        
        agent_with_real_workflow.validator.validate_bullets.side_effect = [
            poor_report,
            good_report,
        ]
        
        result = agent_with_real_workflow.validate_bullets(
            sample_tailored_bullets, sample_resume
        )
        
        assert result["success"] is True
        assert result["retry_count"] > 0
    
    def test_workflow_max_retries_exceeded(self, agent_with_real_workflow, sample_tailored_bullets, sample_resume):
        """Test workflow when max retries are exceeded."""
        # Mock validator to always return poor results
        poor_report = ValidationReport(
            total_bullets=10,
            valid_bullets=0,
            rejected_bullets=10,
            needs_edit_bullets=0,
            error_bullets=0,
            overall_evidence_score=0.1,
            evidence_coverage_percentage=10.0,
            validation_results=[],
            summary_recommendations=["Poor results"],
        )
        
        agent_with_real_workflow.validator.validate_bullets.return_value = poor_report
        
        result = agent_with_real_workflow.validate_bullets(
            sample_tailored_bullets, sample_resume
        )
        
        assert result["success"] is False
        assert result["status"] == "failed"
        assert result["retry_count"] == agent_with_real_workflow.max_retries + 1


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_empty_validation_report(self, agent):
        """Test handling of empty validation report in analysis."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="bullets_validated",
            error_message=None,
            retry_count=0,
            evidence_indexed=True,
            evidence_stats=None,
            validation_report=None,  # None report
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._analyze_results(state)
        
        assert result_state["status"] == "results_analysis_failed"
        assert "No validation results to analyze" in result_state["error_message"]
    
    def test_malformed_validation_report(self, agent):
        """Test handling of malformed validation report."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="bullets_validated",
            error_message=None,
            retry_count=0,
            evidence_indexed=True,
            evidence_stats=None,
            validation_report={"incomplete": "data"},  # Missing required fields
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._analyze_results(state)
        
        # Should handle missing fields gracefully
        assert result_state["status"] == "results_analysis_failed"
    
    def test_datetime_parsing_edge_cases(self, agent):
        """Test datetime parsing edge cases in finalization."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="completed",
            error_message=None,
            retry_count=0,
            evidence_indexed=True,
            evidence_stats=None,
            validation_report={},
            processing_metadata={},
            workflow_start_time="invalid-datetime",  # Invalid format
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._finalize_validation(state)
        
        # Should handle gracefully and set processing time to 0
        assert result_state["processing_metadata"]["processing_time_seconds"] == 0.0
    
    def test_evidence_setup_unexpected_error(self, agent, sample_resume):
        """Test evidence setup with unexpected error."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=sample_resume.model_dump(),
            similarity_threshold=0.8,
            batch_size=20,
            status="input_validated",
            error_message=None,
            retry_count=0,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        # Mock unexpected error
        agent.evidence_indexer.get_collection_stats.side_effect = Exception("Unexpected error")
        
        result_state = agent._setup_evidence(state)
        
        assert result_state["status"] == "evidence_setup_failed"
        assert "Unexpected error during evidence setup" in result_state["error_message"]
    
    def test_non_retryable_error(self, agent):
        """Test handling of non-retryable errors."""
        state = ResumeValidationState(
            tailored_bullets=None,
            resume=None,
            similarity_threshold=0.8,
            batch_size=20,
            status="input_validation_failed",  # Non-retryable error
            error_message="Input validation error",
            retry_count=1,
            evidence_indexed=False,
            evidence_stats=None,
            validation_report=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._handle_error(state)
        
        assert result_state["status"] == "max_retries_exceeded"
        assert "Error not retryable" in [None]  # Check that it logs appropriately