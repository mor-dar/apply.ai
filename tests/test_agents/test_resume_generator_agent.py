"""
Comprehensive unit tests for ResumeGeneratorAgent.

Tests cover all workflow nodes, error handling, state management,
retry logic, and LangGraph orchestration.
Maintains 100% test coverage with zero warnings.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from agents.resume_generator_agent import (
    ResumeGeneratorAgent,
    ResumeGenerationState,
    MIN_COVERAGE_PCT,
    MIN_BULLETS,
    SIMILARITY_THRESHOLD,
)
from tools.resume_generator import ResumeGenerator, ResumeGenerationError, GenerationMetrics
from tools.evidence_indexer import EvidenceIndexer
from src.schemas.core import JobPosting, Resume, ResumeBullet, TailoredBullet, Requirement


class TestResumeGeneratorAgent:
    """Test suite for ResumeGeneratorAgent."""
    
    @pytest.fixture
    def sample_job_posting(self):
        """Create a sample JobPosting for testing."""
        return JobPosting(
            title="Senior Software Engineer",
            company="TechCorp",
            location="San Francisco, CA",
            text="We are looking for a senior software engineer with Python and React experience.",
            keywords=["python", "react", "microservices", "api"],
            requirements=[
                Requirement(text="5+ years of experience", must_have=True),
                Requirement(text="Python expertise", must_have=True),
            ],
        )
    
    @pytest.fixture
    def sample_resume(self):
        """Create a sample Resume for testing."""
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
                text="Designed RESTful APIs for microservices",
                section="Experience",
                start_offset=201,
                end_offset=245,
            ),
        ]
        
        return Resume(
            raw_text="Sample resume with Python and React experience",
            bullets=bullets,
            skills=["Python", "React", "JavaScript", "API"],
            sections=[],
        )
    
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
        ]
    
    @pytest.fixture
    def mock_evidence_indexer(self):
        """Create mock EvidenceIndexer."""
        mock_indexer = Mock(spec=EvidenceIndexer)
        mock_indexer.similarity_threshold = 0.8
        return mock_indexer
    
    @pytest.fixture
    def mock_generator(self):
        """Create mock ResumeGenerator."""
        mock_gen = Mock(spec=ResumeGenerator)
        mock_gen.extract_keywords_from_job.return_value = ["python", "react"]
        return mock_gen
    
    @pytest.fixture
    def agent(self, mock_evidence_indexer):
        """Create ResumeGeneratorAgent instance."""
        with patch('agents.resume_generator_agent.ResumeGenerator') as mock_gen_class:
            mock_gen_class.return_value = Mock()
            return ResumeGeneratorAgent(
                max_retries=2,
                enable_checkpointing=False,
                evidence_indexer=mock_evidence_indexer,
            )
    
    def test_init_default_parameters(self):
        """Test agent initialization with default parameters."""
        with patch('agents.resume_generator_agent.ResumeGenerator') as mock_gen_class:
            mock_generator = Mock()
            mock_gen_class.return_value = mock_generator
            
            agent = ResumeGeneratorAgent()
            
            assert agent.max_retries == 3
            assert agent.generator is not None
            assert agent.workflow is not None
            assert agent.graph is not None
    
    def test_init_custom_parameters(self, mock_evidence_indexer):
        """Test agent initialization with custom parameters."""
        with patch('agents.resume_generator_agent.ResumeGenerator') as mock_gen_class:
            mock_gen_class.return_value = Mock()
            
            agent = ResumeGeneratorAgent(
                max_retries=5,
                enable_checkpointing=True,
                evidence_indexer=mock_evidence_indexer,
            )
            
            assert agent.max_retries == 5
            assert agent.evidence_indexer is mock_evidence_indexer
            assert agent.checkpointer is not None
    
    def test_generate_tailored_bullets_success(self, agent, sample_job_posting, sample_resume, sample_tailored_bullets):
        """Test successful bullet generation workflow."""
        # Mock the generator to return successful results
        mock_metrics = GenerationMetrics(
            total_keywords=4,
            covered_keywords=2,
            total_bullets_generated=2,
            bullets_above_threshold=2,
            average_similarity_score=0.875,
            keyword_coverage_percentage=50.0,
        )
        
        agent.generator.generate_resume_bullets.return_value = (
            sample_tailored_bullets,
            mock_metrics,
            [{"original_text": "test", "tailored_text": "test"}],
        )
        
        result = agent.generate_tailored_bullets(
            sample_job_posting, sample_resume, max_bullets=10, similarity_threshold=0.8
        )
        
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["tailored_bullets"] is not None
        assert result["metrics"] is not None
        assert result["diff_summaries"] is not None
        assert result["error"] is None
        assert result["retry_count"] == 0
    
    def test_initialize_generation(self, agent):
        """Test initialization workflow node."""
        state = ResumeGenerationState(
            job_posting=None,
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="",
            error_message=None,
            retry_count=0,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._initialize_generation(state)
        
        assert result_state["status"] == "initialized"
        assert result_state["current_node"] == "initialize"
        assert result_state["workflow_start_time"] is not None
        assert "agent_version" in result_state["processing_metadata"]
        assert result_state["processing_metadata"]["max_retries"] == agent.max_retries
    
    def test_validate_input_success(self, agent, sample_job_posting, sample_resume):
        """Test successful input validation."""
        state = ResumeGenerationState(
            job_posting=sample_job_posting.model_dump(),
            resume=sample_resume.model_dump(),
            max_bullets=20,
            similarity_threshold=0.8,
            status="initialized",
            error_message=None,
            retry_count=0,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_input(state)
        
        assert result_state["status"] == "input_validated"
        assert result_state["current_node"] == "validate_input"
        assert result_state["error_message"] is None
        assert "bullets_to_validate" not in result_state["processing_metadata"]  # This is for validator
    
    def test_validate_input_missing_job_posting(self, agent):
        """Test input validation with missing job posting."""
        state = ResumeGenerationState(
            job_posting=None,
            resume={"raw_text": "test", "bullets": [], "skills": []},
            max_bullets=20,
            similarity_threshold=0.8,
            status="initialized",
            error_message=None,
            retry_count=0,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_input(state)
        
        assert result_state["status"] == "input_validation_failed"
        assert "No job posting provided" in result_state["error_message"]
    
    def test_validate_input_missing_resume(self, agent, sample_job_posting):
        """Test input validation with missing resume."""
        state = ResumeGenerationState(
            job_posting=sample_job_posting.model_dump(),
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="initialized",
            error_message=None,
            retry_count=0,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_input(state)
        
        assert result_state["status"] == "input_validation_failed"
        assert "No resume provided" in result_state["error_message"]
    
    def test_validate_input_empty_bullets(self, agent, sample_job_posting):
        """Test input validation with empty resume bullets."""
        resume_data = {
            "raw_text": "test",
            "bullets": [],  # Empty bullets
            "skills": [],
        }
        
        state = ResumeGenerationState(
            job_posting=sample_job_posting.model_dump(),
            resume=resume_data,
            max_bullets=20,
            similarity_threshold=0.8,
            status="initialized",
            error_message=None,
            retry_count=0,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_input(state)
        
        assert result_state["status"] == "input_validation_failed"
        assert "no bullets to tailor" in result_state["error_message"]
    
    def test_extract_keywords_success(self, agent, sample_job_posting):
        """Test successful keyword extraction."""
        state = ResumeGenerationState(
            job_posting=sample_job_posting.model_dump(),
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="input_validated",
            error_message=None,
            retry_count=0,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        agent.generator.extract_keywords_from_job.return_value = ["python", "react", "api"]
        
        result_state = agent._extract_keywords(state)
        
        assert result_state["status"] == "keywords_extracted"
        assert result_state["current_node"] == "extract_keywords"
        assert result_state["processing_metadata"]["total_keywords_extracted"] == 3
        assert len(result_state["processing_metadata"]["extracted_keywords"]) <= 20
    
    def test_extract_keywords_error(self, agent, sample_job_posting):
        """Test keyword extraction with error."""
        state = ResumeGenerationState(
            job_posting=sample_job_posting.model_dump(),
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="input_validated",
            error_message=None,
            retry_count=0,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        agent.generator.extract_keywords_from_job.side_effect = ResumeGenerationError("Keyword extraction failed")
        
        result_state = agent._extract_keywords(state)
        
        assert result_state["status"] == "keyword_extraction_failed"
        assert "Keyword extraction failed" in result_state["error_message"]
    
    def test_generate_bullets_success(self, agent, sample_job_posting, sample_resume, sample_tailored_bullets):
        """Test successful bullet generation."""
        state = ResumeGenerationState(
            job_posting=sample_job_posting.model_dump(),
            resume=sample_resume.model_dump(),
            max_bullets=20,
            similarity_threshold=0.8,
            status="keywords_extracted",
            error_message=None,
            retry_count=0,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        mock_metrics = GenerationMetrics(
            total_keywords=4,
            covered_keywords=2,
            total_bullets_generated=2,
            bullets_above_threshold=2,
            average_similarity_score=0.875,
            keyword_coverage_percentage=50.0,
        )
        
        agent.generator.generate_resume_bullets.return_value = (
            sample_tailored_bullets,
            mock_metrics,
            [{"original_text": "test", "tailored_text": "enhanced test"}],
        )
        
        result_state = agent._generate_bullets(state)
        
        assert result_state["status"] == "bullets_generated"
        assert result_state["current_node"] == "generate_bullets"
        assert result_state["tailored_bullets"] is not None
        assert result_state["generation_metrics"] is not None
        assert result_state["diff_summaries"] is not None
        assert len(result_state["tailored_bullets"]) == 2
    
    def test_generate_bullets_error(self, agent, sample_job_posting, sample_resume):
        """Test bullet generation with error."""
        state = ResumeGenerationState(
            job_posting=sample_job_posting.model_dump(),
            resume=sample_resume.model_dump(),
            max_bullets=20,
            similarity_threshold=0.8,
            status="keywords_extracted",
            error_message=None,
            retry_count=0,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        agent.generator.generate_resume_bullets.side_effect = ResumeGenerationError("Generation failed")
        
        result_state = agent._generate_bullets(state)
        
        assert result_state["status"] == "bullet_generation_failed"
        assert "Generation failed" in result_state["error_message"]
    
    def test_validate_results_success(self, agent):
        """Test successful results validation."""
        state = ResumeGenerationState(
            job_posting=None,
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="bullets_generated",
            error_message=None,
            retry_count=0,
            tailored_bullets=[{"text": "test"}] * 10,  # 10 bullets
            generation_metrics={
                "keyword_coverage_percentage": 60.0,  # Above threshold
                "bullets_above_threshold": 8,
                "average_similarity_score": 0.85,
            },
            diff_summaries=[],
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_results(state)
        
        assert result_state["status"] == "results_validated"
        assert result_state["current_node"] == "validate_results"
        assert "quality_metrics" in result_state["processing_metadata"]
    
    def test_validate_results_low_coverage(self, agent):
        """Test results validation with low coverage."""
        state = ResumeGenerationState(
            job_posting=None,
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="bullets_generated",
            error_message=None,
            retry_count=0,
            tailored_bullets=[{"text": "test"}] * 3,  # Few bullets
            generation_metrics={
                "keyword_coverage_percentage": 10.0,  # Low coverage
                "bullets_above_threshold": 1,
                "average_similarity_score": 0.75,  # Low similarity
            },
            diff_summaries=[],
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._validate_results(state)
        
        assert result_state["status"] == "results_validation_failed"
        assert "coverage" in result_state["error_message"]
    
    def test_handle_error_retry(self, agent):
        """Test error handling with retry logic."""
        state = ResumeGenerationState(
            job_posting=None,
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="bullet_generation_failed",
            error_message="Generation error",
            retry_count=1,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
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
        state = ResumeGenerationState(
            job_posting=None,
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="bullet_generation_failed",
            error_message="Generation error",
            retry_count=agent.max_retries,  # At max
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._handle_error(state)
        
        assert result_state["status"] == "max_retries_exceeded"
        assert result_state["retry_count"] == agent.max_retries + 1
    
    def test_handle_error_threshold_adjustment(self, agent):
        """Test error handling with threshold adjustment for validation failures."""
        state = ResumeGenerationState(
            job_posting=None,
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="results_validation_failed",
            error_message="Validation error",
            retry_count=1,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._handle_error(state)
        
        assert result_state["status"] == "retrying"
        assert result_state["similarity_threshold"] < 0.8  # Should be lowered
    
    def test_finalize_generation_success(self, agent):
        """Test successful workflow finalization."""
        state = ResumeGenerationState(
            job_posting=None,
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="results_validated",
            error_message=None,
            retry_count=0,
            tailored_bullets=[{"text": "test"}],
            generation_metrics={},
            diff_summaries=[],
            processing_metadata={},
            workflow_start_time=datetime.now(timezone.utc).isoformat(),
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._finalize_generation(state)
        
        assert result_state["status"] == "completed"
        assert result_state["current_node"] == "finalize"
        assert result_state["workflow_end_time"] is not None
        assert "processing_time_seconds" in result_state["processing_metadata"]
    
    def test_finalize_generation_failed(self, agent):
        """Test workflow finalization with failure."""
        state = ResumeGenerationState(
            job_posting=None,
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="max_retries_exceeded",
            error_message="Max retries exceeded",
            retry_count=3,
            tailored_bullets=None,
            generation_metrics=None,
            diff_summaries=None,
            processing_metadata={},
            workflow_start_time=datetime.now(timezone.utc).isoformat(),
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._finalize_generation(state)
        
        assert result_state["status"] == "failed"
        assert result_state["workflow_end_time"] is not None
    
    def test_routing_functions(self, agent):
        """Test workflow routing functions."""
        # Test route_after_validation
        state_validated = {"status": "input_validated"}
        assert agent._route_after_validation(state_validated) == "extract"
        
        state_failed = {"status": "input_validation_failed"}
        assert agent._route_after_validation(state_failed) == "error"
        
        # Test route_after_results
        state_results_ok = {"status": "results_validated"}
        assert agent._route_after_results(state_results_ok) == "finalize"
        
        state_results_retry = {"status": "results_validation_failed", "retry_count": 1}
        assert agent._route_after_results(state_results_retry) == "retry"
        
        state_results_error = {"status": "results_validation_failed", "retry_count": 5}
        assert agent._route_after_results(state_results_error) == "error"
        
        # Test route_after_error
        state_retrying = {"status": "retrying"}
        assert agent._route_after_error(state_retrying) == "retry"
        
        state_exceeded = {"status": "max_retries_exceeded"}
        assert agent._route_after_error(state_exceeded) == "finalize"
    
    def test_format_result(self, agent):
        """Test result formatting."""
        state = ResumeGenerationState(
            job_posting=None,
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="completed",
            error_message=None,
            retry_count=1,
            tailored_bullets=[{"text": "test"}],
            generation_metrics={"coverage": 80.0},
            diff_summaries=[{"diff": "test"}],
            processing_metadata={"processing_time_seconds": 2.5},
            workflow_start_time=None,
            workflow_end_time=None,
            current_node=None,
        )
        
        result = agent._format_result(state)
        
        assert result["status"] == "completed"
        assert result["success"] is True
        assert result["tailored_bullets"] == [{"text": "test"}]
        assert result["metrics"] == {"coverage": 80.0}
        assert result["diff_summaries"] == [{"diff": "test"}]
        assert result["error"] is None
        assert result["retry_count"] == 1
        assert result["processing_time"] == 2.5
        
        # Test failed case
        failed_state = state.copy()
        failed_state["status"] = "failed"
        failed_result = agent._format_result(failed_state)
        assert failed_result["success"] is False
    
    @pytest.mark.asyncio
    async def test_async_interface(self, agent, sample_job_posting, sample_resume):
        """Test async interface method."""
        # Mock synchronous method
        with patch.object(agent, 'generate_tailored_bullets', return_value={"success": True}) as mock_sync:
            result = await agent.agenerate_tailored_bullets(
                sample_job_posting, sample_resume, max_bullets=15, similarity_threshold=0.9
            )
            
            assert result == {"success": True}
            mock_sync.assert_called_once_with(sample_job_posting, sample_resume, 15, 0.9)


class TestWorkflowIntegration:
    """Test suite for workflow integration scenarios."""
    
    @pytest.fixture
    def agent_with_real_workflow(self, mock_evidence_indexer):
        """Create agent with real workflow for integration testing."""
        with patch('agents.resume_generator_agent.ResumeGenerator') as mock_gen_class:
            mock_generator = Mock()
            mock_generator.extract_keywords_from_job.return_value = ["python", "react"]
            mock_generator.generate_resume_bullets.return_value = (
                [Mock(model_dump=Mock(return_value={"text": "test"}))],
                Mock(
                    total_keywords=2,
                    covered_keywords=1,
                    total_bullets_generated=1,
                    bullets_above_threshold=1,
                    average_similarity_score=0.9,
                    keyword_coverage_percentage=50.0,
                ),
                [{"original": "test", "tailored": "enhanced test"}],
            )
            mock_gen_class.return_value = mock_generator
            
            return ResumeGeneratorAgent(
                max_retries=2,
                enable_checkpointing=False,
                evidence_indexer=mock_evidence_indexer,
            )
    
    def test_full_workflow_success(self, agent_with_real_workflow, sample_job_posting, sample_resume):
        """Test complete workflow execution with success."""
        result = agent_with_real_workflow.generate_tailored_bullets(
            sample_job_posting, sample_resume, max_bullets=10
        )
        
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["retry_count"] == 0
        assert result["error"] is None
    
    def test_workflow_with_validation_retry(self, agent_with_real_workflow, sample_job_posting, sample_resume):
        """Test workflow with validation failure and retry."""
        # Mock generator to return poor results first, then good results
        poor_metrics = Mock(
            total_keywords=10,
            covered_keywords=1,
            total_bullets_generated=2,
            bullets_above_threshold=0,
            average_similarity_score=0.5,
            keyword_coverage_percentage=10.0,
        )
        
        good_metrics = Mock(
            total_keywords=10,
            covered_keywords=7,
            total_bullets_generated=8,
            bullets_above_threshold=8,
            average_similarity_score=0.9,
            keyword_coverage_percentage=70.0,
        )
        
        agent_with_real_workflow.generator.generate_resume_bullets.side_effect = [
            ([Mock(model_dump=Mock(return_value={"text": "poor"}))], poor_metrics, []),
            ([Mock(model_dump=Mock(return_value={"text": "good"}))] * 8, good_metrics, []),
        ]
        
        result = agent_with_real_workflow.generate_tailored_bullets(
            sample_job_posting, sample_resume
        )
        
        assert result["success"] is True
        assert result["retry_count"] > 0
    
    def test_workflow_max_retries_exceeded(self, agent_with_real_workflow, sample_job_posting, sample_resume):
        """Test workflow when max retries are exceeded."""
        # Mock generator to always return poor results
        poor_metrics = Mock(
            total_keywords=10,
            covered_keywords=0,
            total_bullets_generated=0,
            bullets_above_threshold=0,
            average_similarity_score=0.0,
            keyword_coverage_percentage=0.0,
        )
        
        agent_with_real_workflow.generator.generate_resume_bullets.return_value = ([], poor_metrics, [])
        
        result = agent_with_real_workflow.generate_tailored_bullets(
            sample_job_posting, sample_resume
        )
        
        assert result["success"] is False
        assert result["status"] == "failed"
        assert result["retry_count"] == agent_with_real_workflow.max_retries + 1


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_empty_job_posting_fields(self, agent):
        """Test handling of job posting with empty fields."""
        empty_job = JobPosting(
            title="",
            company="",
            text="",
            keywords=[],
            requirements=[],
        )
        
        resume = Resume(
            raw_text="test",
            bullets=[ResumeBullet(text="test", section="test", start_offset=0, end_offset=4)],
            skills=[],
            sections=[],
        )
        
        result = agent.generate_tailored_bullets(empty_job, resume)
        
        # Should handle gracefully, likely with warnings
        assert "status" in result
    
    def test_malformed_state_data(self, agent):
        """Test handling of malformed state data."""
        # Test with missing required fields
        malformed_state = {
            "job_posting": {"title": "test"},  # Missing required fields
            "resume": {"raw_text": "test"},  # Missing bullets
            "max_bullets": 20,
            "similarity_threshold": 0.8,
            "status": "initialized",
            "error_message": None,
            "retry_count": 0,
            "tailored_bullets": None,
            "generation_metrics": None,
            "diff_summaries": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }
        
        result_state = agent._validate_input(malformed_state)
        assert result_state["status"] == "input_validation_failed"
    
    def test_datetime_parsing_edge_cases(self, agent):
        """Test datetime parsing edge cases in finalization."""
        state = ResumeGenerationState(
            job_posting=None,
            resume=None,
            max_bullets=20,
            similarity_threshold=0.8,
            status="completed",
            error_message=None,
            retry_count=0,
            tailored_bullets=[],
            generation_metrics={},
            diff_summaries=[],
            processing_metadata={},
            workflow_start_time="invalid-datetime",  # Invalid format
            workflow_end_time=None,
            current_node=None,
        )
        
        result_state = agent._finalize_generation(state)
        
        # Should handle gracefully and set processing time to 0
        assert result_state["processing_metadata"]["processing_time_seconds"] == 0.0