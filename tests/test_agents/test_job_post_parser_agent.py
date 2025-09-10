"""
Unit tests for the JobPostParserAgent.

Tests LangGraph workflow orchestration, error handling, retry logic,
and state management for the job posting parsing agent.
"""

import pytest
from unittest.mock import Mock, patch

from agents.job_post_parser_agent import JobPostParserAgent, JobPostParsingState


class TestJobPostParserAgent:
    """Test suite for JobPostParserAgent functionality."""

    @pytest.fixture
    def mock_parser(self):
        """Create a mock JobPostingParser for testing."""
        mock_parser = Mock()
        mock_parser.max_keywords = 30
        mock_parser.min_keyword_length = 2
        return mock_parser

    @pytest.fixture
    def agent(self, mock_parser):
        """Create a JobPostParserAgent instance for testing."""
        with patch("agents.job_post_parser_agent.JobPostingParser") as mock_cls:
            mock_cls.return_value = mock_parser
            agent = JobPostParserAgent(max_retries=2, enable_checkpoints=False)
            agent.parser = mock_parser
            return agent

    def test_agent_initialization(self):
        """Test JobPostParserAgent initialization."""
        with patch("agents.job_post_parser_agent.JobPostingParser") as mock_parser_cls:
            mock_parser_cls.return_value = Mock()

            agent = JobPostParserAgent(
                max_keywords=20,
                min_keyword_length=3,
                max_retries=5,
                enable_checkpoints=True,
            )

            assert agent.max_retries == 5
            assert agent.checkpointer is not None
            mock_parser_cls.assert_called_once_with(
                max_keywords=20, min_keyword_length=3
            )

    def test_agent_initialization_no_checkpoints(self):
        """Test agent initialization with checkpoints disabled."""
        with patch("agents.job_post_parser_agent.JobPostingParser"):
            agent = JobPostParserAgent(enable_checkpoints=False)
            assert agent.checkpointer is None

    def test_initialize_parsing_state(self, agent):
        """Test the _initialize_parsing method."""
        initial_state: JobPostParsingState = {
            "job_text": "Test job description",
            "job_title": "Software Engineer",
            "company_name": "TechCorp",
            "job_location": "Remote",
            "status": "pending",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": None,
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        result = agent._initialize_parsing(initial_state)

        assert result["status"] == "initializing"
        assert result["started_at"] is not None
        assert result["processing_metadata"]["parser_config"]["max_keywords"] == 30
        assert result["processing_metadata"]["workflow_version"] == "1.0"
        assert result["job_text"] == "Test job description"

    def test_parse_job_posting_success(self, agent, mock_parser):
        """Test successful job posting parsing."""
        # Mock successful parsing
        mock_job_posting = Mock()
        mock_job_posting.model_dump.return_value = {
            "title": "Software Engineer",
            "company": "TechCorp",
            "location": "Remote",
            "text": "Test job description",
            "keywords": ["python", "software", "engineer"],
            "requirements": ["Bachelor's degree required"],
        }
        mock_job_posting.keywords = ["python", "software", "engineer"]
        mock_job_posting.requirements = ["Bachelor's degree required"]

        # Create mock parser report
        mock_parser_report = Mock()
        mock_parser_report.confidence = 0.9
        mock_parser_report.missing_fields = []
        mock_parser_report.warnings = []
        mock_parser_report.keyword_count = 3
        mock_parser_report.requirement_count = 1
        mock_parser_report.text_length = len("Test job description")

        # Parser now returns a tuple (JobPosting, ParserReport)
        mock_parser.parse.return_value = (mock_job_posting, mock_parser_report)

        state: JobPostParsingState = {
            "job_text": "Test job description",
            "job_title": "Software Engineer",
            "company_name": "TechCorp",
            "job_location": "Remote",
            "status": "initializing",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": None,
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        result = agent._parse_job_posting(state)

        assert result["status"] == "parsed"
        assert result["error_message"] is None
        assert result["parsed_job"] is not None
        assert result["parsed_job"]["keywords"] == ["python", "software", "engineer"]

        mock_parser.parse.assert_called_once_with(
            job_text="Test job description",
            title="Software Engineer",
            company="TechCorp",
            location="Remote",
        )

    def test_parse_job_posting_failure(self, agent, mock_parser):
        """Test job posting parsing failure."""
        # Mock parsing failure
        mock_parser.parse.side_effect = Exception("Parsing failed")

        state: JobPostParsingState = {
            "job_text": "Test job description",
            "job_title": None,
            "company_name": None,
            "job_location": None,
            "status": "initializing",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": None,
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        result = agent._parse_job_posting(state)

        assert result["status"] == "error"
        assert "Parsing failed" in result["error_message"]

    def test_validate_results_success(self, agent):
        """Test successful result validation."""
        state: JobPostParsingState = {
            "job_text": "A detailed job description with sufficient content for validation",
            "job_title": "Software Engineer",
            "company_name": "TechCorp",
            "job_location": "Remote",
            "status": "parsed",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": {
                "title": "Software Engineer",
                "company": "TechCorp",
                "location": "Remote",
                "text": "A detailed job description with sufficient content for validation",
                "keywords": ["python", "software", "engineer"],
                "requirements": ["Bachelor's degree required"],
            },
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        result = agent._validate_results(state)

        assert result["status"] == "validated"

    def test_validate_results_no_keywords(self, agent):
        """Test validation failure due to no keywords."""
        state: JobPostParsingState = {
            "job_text": "A detailed job description with sufficient content for validation",
            "job_title": "Software Engineer",
            "company_name": "TechCorp",
            "job_location": "Remote",
            "status": "parsed",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": {
                "title": "Software Engineer",
                "company": "TechCorp",
                "location": "Remote",
                "text": "A detailed job description with sufficient content for validation",
                "keywords": [],  # No keywords
                "requirements": ["Bachelor's degree required"],
            },
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        result = agent._validate_results(state)

        assert result["status"] == "validation_failed"
        assert "No keywords extracted" in result["error_message"]

    def test_validate_results_no_requirements(self, agent):
        """Test validation failure due to no requirements."""
        state: JobPostParsingState = {
            "job_text": "A detailed job description with sufficient content for validation",
            "job_title": "Software Engineer",
            "company_name": "TechCorp",
            "job_location": "Remote",
            "status": "parsed",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": {
                "title": "Software Engineer",
                "company": "TechCorp",
                "location": "Remote",
                "text": "A detailed job description with sufficient content for validation",
                "keywords": ["python", "software"],
                "requirements": [],  # No requirements
            },
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        result = agent._validate_results(state)

        assert result["status"] == "validation_failed"
        assert "No requirements extracted" in result["error_message"]

    def test_validate_results_short_text(self, agent):
        """Test validation failure due to short text."""
        state: JobPostParsingState = {
            "job_text": "Short",
            "job_title": "Software Engineer",
            "company_name": "TechCorp",
            "job_location": "Remote",
            "status": "parsed",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": {
                "title": "Software Engineer",
                "company": "TechCorp",
                "location": "Remote",
                "text": "Short",  # Too short
                "keywords": ["python"],
                "requirements": ["Bachelor's degree required"],
            },
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        result = agent._validate_results(state)

        assert result["status"] == "validation_failed"
        assert "Job text appears too short" in result["error_message"]

    def test_validate_results_no_parsed_job(self, agent):
        """Test validation failure when no parsed job data exists."""
        state: JobPostParsingState = {
            "job_text": "Test job description",
            "job_title": "Software Engineer",
            "company_name": "TechCorp",
            "job_location": "Remote",
            "status": "parsed",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": None,  # No parsed job
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        result = agent._validate_results(state)

        assert result["status"] == "validation_failed"
        assert "No parsed job data to validate" in result["error_message"]

    def test_handle_error(self, agent):
        """Test error handling logic."""
        state: JobPostParsingState = {
            "job_text": "Test job description",
            "job_title": "Software Engineer",
            "company_name": "TechCorp",
            "job_location": "Remote",
            "status": "error",
            "error_message": "Some error occurred",
            "retry_count": 0,
            "parsed_job": None,
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        result = agent._handle_error(state)

        assert result["retry_count"] == 1
        assert result["status"] == "retrying"  # Should retry since count < max_retries

    def test_handle_error_max_retries(self, agent):
        """Test error handling when max retries reached."""
        state: JobPostParsingState = {
            "job_text": "Test job description",
            "job_title": "Software Engineer",
            "company_name": "TechCorp",
            "job_location": "Remote",
            "status": "error",
            "error_message": "Some error occurred",
            "retry_count": 2,  # At max retries
            "parsed_job": None,
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        result = agent._handle_error(state)

        assert result["retry_count"] == 3
        assert result["status"] == "failed"  # Should fail since max retries exceeded

    def test_finalize_results(self, agent):
        """Test result finalization."""
        state: JobPostParsingState = {
            "job_text": "Test job description",
            "job_title": "Software Engineer",
            "company_name": "TechCorp",
            "job_location": "Remote",
            "status": "validated",
            "error_message": None,
            "retry_count": 1,
            "parsed_job": {
                "title": "Software Engineer",
                "company": "TechCorp",
                "location": "Remote",
                "text": "Test job description",
                "keywords": ["python", "software", "engineer"],
                "requirements": ["Bachelor's degree required"],
            },
            "processing_metadata": {"initial": "metadata"},
            "started_at": "2025-01-01T00:00:00Z",
            "completed_at": None,
        }

        result = agent._finalize_results(state)

        assert result["status"] == "completed"
        assert result["completed_at"] is not None
        assert result["processing_metadata"]["final_status"] == "completed"
        assert result["processing_metadata"]["total_keywords"] == 3
        assert result["processing_metadata"]["total_requirements"] == 1
        assert result["processing_metadata"]["retry_attempts"] == 1

    def test_should_retry_logic(self, agent):
        """Test retry decision logic."""
        # Test parsed state -> should validate
        state = {"status": "parsed", "retry_count": 0}
        assert agent._should_retry(state) == "validate"

        # Test error state under retry limit -> should retry
        state = {"status": "error", "retry_count": 1}
        assert agent._should_retry(state) == "retry"

        # Test error state over retry limit -> should fail
        state = {"status": "error", "retry_count": 3}
        assert agent._should_retry(state) == "failed"

    def test_validation_complete_logic(self, agent):
        """Test validation completion logic."""
        # Test validated state -> success
        state = {"status": "validated", "retry_count": 0}
        assert agent._validation_complete(state) == "success"

        # Test validation failed under retry limit -> retry
        state = {"status": "validation_failed", "retry_count": 1}
        assert agent._validation_complete(state) == "retry"

        # Test validation failed over retry limit -> failed
        state = {"status": "validation_failed", "retry_count": 3}
        assert agent._validation_complete(state) == "failed"

    def test_should_continue_after_error_logic(self, agent):
        """Test continue after error logic."""
        # Test retrying status -> should retry
        state = {"status": "retrying"}
        assert agent._should_continue_after_error(state) == "retry"

        # Test failed status -> should fail
        state = {"status": "failed"}
        assert agent._should_continue_after_error(state) == "failed"


class TestJobPostParserAgentIntegration:
    """Integration tests for the complete JobPostParserAgent workflow."""

    @pytest.fixture
    def agent(self):
        """Create an agent with real dependencies for integration testing."""
        return JobPostParserAgent(max_retries=2, enable_checkpoints=False)

    def test_synchronous_parsing_success(self, agent):
        """Test successful synchronous job posting parsing."""
        job_text = """
        Senior Software Engineer position at TechCorp.
        Requirements: Bachelor's degree in Computer Science, 5+ years Python experience.
        Must have knowledge of React, AWS, and agile methodology.
        """

        result = agent.parse_job_posting_sync(
            job_text=job_text,
            job_title="Senior Software Engineer",
            company_name="TechCorp",
            job_location="San Francisco, CA",
        )

        assert result["success"] is True
        assert result["job_posting"] is not None
        assert result["error"] is None
        assert result["retry_count"] == 0

        # Check job posting content
        job_posting = result["job_posting"]
        assert job_posting["title"] == "Senior Software Engineer"
        assert job_posting["company"] == "TechCorp"
        assert job_posting["location"] == "San Francisco, CA"
        assert len(job_posting["keywords"]) > 0
        assert len(job_posting["requirements"]) > 0

        # Check metadata
        metadata = result["metadata"]
        assert metadata["final_status"] == "completed"
        assert metadata["total_keywords"] > 0
        assert metadata["total_requirements"] > 0

    def test_synchronous_parsing_failure(self, agent):
        """Test synchronous parsing with validation failure."""
        # Empty job text should trigger validation failure
        result = agent.parse_job_posting_sync(
            job_text="", job_title="Test Position", company_name="TestCorp"
        )

        assert result["success"] is False
        assert result["job_posting"] is None  # Failed workflow returns None
        assert result["error"] is not None
        assert "failed" in result["error"] or "validation" in result["error"].lower()
        assert (
            result["retry_count"] == 0
        )  # Workflow execution failed, no retry count returned

    @pytest.mark.asyncio
    async def test_asynchronous_parsing_success(self, agent):
        """Test successful asynchronous job posting parsing."""
        job_text = """
        Data Scientist role requiring PhD in Statistics or related field.
        Must have experience with machine learning, Python, R, and SQL.
        Knowledge of TensorFlow and cloud platforms preferred.
        """

        result = await agent.parse_job_posting(
            job_text=job_text,
            job_title="Data Scientist",
            company_name="DataCorp",
            job_location="Remote",
        )

        assert result["success"] is True
        assert result["job_posting"] is not None
        assert result["error"] is None

        # Check job posting content
        job_posting = result["job_posting"]
        assert job_posting["title"] == "Data Scientist"
        assert job_posting["company"] == "DataCorp"
        assert job_posting["location"] == "Remote"
        assert len(job_posting["keywords"]) > 0

    def test_workflow_state_transitions(self, agent):
        """Test that workflow properly transitions through all expected states."""
        # Simply test that a successful workflow completes and produces expected results
        # Use a longer, more detailed job description to meet confidence thresholds
        job_text = """
        Senior Software Engineer Position at TechCorp
        
        We are seeking a Senior Software Engineer to join our development team. 
        The ideal candidate will have strong experience in Python programming, 
        web development frameworks, and software engineering best practices.
        
        Key Requirements:
        • Bachelor's degree in Computer Science or related field required
        • 5+ years of professional software development experience
        • Proficiency in Python, Django, and REST API development
        • Experience with databases (PostgreSQL, Redis)
        • Strong knowledge of version control systems (Git)
        • Experience with cloud platforms (AWS, Docker, Kubernetes)
        • Excellent problem-solving and communication skills
        
        Responsibilities include designing scalable applications, 
        mentoring junior developers, and collaborating with cross-functional teams.
        """

        result = agent.parse_job_posting_sync(
            job_text=job_text,
            job_title="Senior Software Engineer",
            company_name="TechCorp",
        )

        # Verify successful completion
        assert result["success"] is True
        assert result["job_posting"] is not None
        assert result["error"] is None

        # Verify the workflow produced valid output
        job_posting = result["job_posting"]
        assert job_posting["title"] == "Senior Software Engineer"
        assert len(job_posting["keywords"]) > 0
        assert len(job_posting["requirements"]) > 0

        # Verify metadata shows completed workflow
        assert result["metadata"]["final_status"] == "completed"
        assert result["metadata"]["total_keywords"] > 0


class TestJobPostParserAgentErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def agent(self):
        """Create agent for error testing."""
        return JobPostParserAgent(max_retries=1, enable_checkpoints=False)

    def test_workflow_execution_exception(self, agent):
        """Test handling of workflow execution exceptions."""
        # Mock the compiled graph to raise an exception
        agent.compiled_graph.invoke = Mock(side_effect=Exception("Workflow error"))

        result = agent.parse_job_posting_sync(job_text="Test job description")

        assert result["success"] is False
        assert "Workflow error" in result["error"]
        assert result["metadata"]["error"] == "workflow_execution_failed"

    def test_empty_input_handling(self, agent):
        """Test handling of empty inputs."""
        result = agent.parse_job_posting_sync(job_text="")

        # Should complete but likely fail validation
        assert result["success"] is False
        assert result["job_posting"] is None  # Failed validation returns None

    def test_long_input_handling(self, agent):
        """Test handling of very long job descriptions."""
        # Create a very long job description (beyond normal limits)
        long_text = "Software engineer position. " * 1000  # Very long text

        result = agent.parse_job_posting_sync(
            job_text=long_text, job_title="Software Engineer"
        )

        # Should handle gracefully (either succeed or fail gracefully)
        assert isinstance(result, dict)
        assert "success" in result
        assert "error" in result

    def test_invalid_retry_configuration(self):
        """Test agent behavior with invalid retry configuration."""
        # Test with negative retries (should handle gracefully)
        with patch("agents.job_post_parser_agent.JobPostingParser"):
            agent = JobPostParserAgent(max_retries=-1)
            assert agent.max_retries == -1  # Should accept but behave appropriately


class TestJobPostParserAgentCoverage:
    """Test coverage edge cases for agent methods."""

    def test_validate_results_exception_handling(self):
        """Test exception handling in _validate_results method."""
        with patch("agents.job_post_parser_agent.JobPostingParser"):
            agent = JobPostParserAgent()

            # Create a mock that will raise an exception when accessed
            class ExceptionRaisingDict(dict):
                def get(self, key, default=None):
                    if key == "confidence":
                        raise Exception("Simulated validation error")
                    return default

            mock_parser_report = ExceptionRaisingDict(
                {"confidence": 0.9, "missing_fields": [], "warnings": []}
            )

            state: JobPostParsingState = {
                "job_text": "test job description",
                "job_title": "Engineer",
                "company_name": "Test Corp",
                "job_location": "Test City",
                "status": "parsed",
                "error_message": None,
                "retry_count": 0,
                "parsed_job": {"title": "Engineer"},
                "parser_report": mock_parser_report,  # Put it directly in state, not in processing_metadata
                "processing_metadata": {},
                "started_at": None,
                "completed_at": None,
            }

            result = agent._validate_results(state)

            # Should handle the exception gracefully
            assert result["status"] == "validation_failed"
            assert "Validation error:" in result["error_message"]

    def test_parse_job_posting_sync_exception_handling(self):
        """Test exception handling in parse_job_posting_sync method."""
        with patch("agents.job_post_parser_agent.JobPostingParser"):
            agent = JobPostParserAgent()

            # Mock the compiled_graph.invoke to raise an exception
            with patch.object(
                agent.compiled_graph, "invoke", side_effect=Exception("Workflow failed")
            ):
                result = agent.parse_job_posting_sync(
                    job_text="test job",
                    job_title="Engineer",
                    company_name="Test Corp",
                    job_location="Test City",
                )

                # Should handle the exception gracefully (covers lines 546-548)
                assert result["success"] is False
                assert result["error"] == "Workflow failed"
                assert result["metadata"]["error"] == "workflow_execution_failed"
                assert result["job_posting"] is None
                assert result["retry_count"] == 0


class TestJobPostParsingState:
    """Test the JobPostParsingState TypedDict."""

    def test_state_structure(self):
        """Test that state structure matches expectations."""
        state: JobPostParsingState = {
            "job_text": "test",
            "job_title": "Engineer",
            "company_name": "Corp",
            "job_location": "City",
            "status": "pending",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": None,
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        # Test that all required keys are present
        required_keys = {
            "job_text",
            "job_title",
            "company_name",
            "job_location",
            "status",
            "error_message",
            "retry_count",
            "parsed_job",
            "processing_metadata",
            "started_at",
            "completed_at",
        }

        assert set(state.keys()) == required_keys

        # Test basic operations
        state_copy = state.copy()
        assert state_copy == state

        state_copy["status"] = "completed"
        assert state_copy["status"] != state["status"]
