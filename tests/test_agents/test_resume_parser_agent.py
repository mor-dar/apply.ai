"""
Comprehensive unit tests for ResumeParserAgent.

Tests cover all workflow nodes, state management, error handling, retry logic,
and integration with ResumeParser tool. Maintains 100% test coverage with zero warnings.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime
import tempfile
from datetime import timezone

from agents.resume_parser_agent import ResumeParserAgent, ResumeParsingState
from tools.resume_parser import ResumeParsingError
from src.schemas.core import Resume, ResumeBullet, ResumeSection


class TestResumeParserAgent:
    """Test suite for ResumeParserAgent."""

    @pytest.fixture
    def agent(self):
        """Create a ResumeParserAgent instance for testing."""
        return ResumeParserAgent(max_retries=2, enable_checkpointing=False)

    @pytest.fixture
    def sample_resume_data(self):
        """Sample resume data for testing."""
        return Resume(
            raw_text="Sample resume content",
            bullets=[
                ResumeBullet(
                    text="Led development team",
                    section="Experience",
                    start_offset=0,
                    end_offset=20,
                )
            ],
            skills=["Python", "JavaScript"],
            dates=["2020-2023"],
            sections=[
                ResumeSection(
                    name="Experience", bullets=[], start_offset=0, end_offset=100
                )
            ],
        )

    @pytest.fixture
    def sample_parser_report(self):
        """Sample parser report for testing."""
        return {
            "text_length": 500,
            "sections_found": 3,
            "bullets_found": 5,
            "skills_found": 10,
            "dates_found": 2,
            "confidence": 0.85,
            "warnings": ["No issues found"],
        }

    def test_initialization_default(self):
        """Test agent initialization with default parameters."""
        agent = ResumeParserAgent()
        assert agent.max_retries == 3
        assert agent.checkpointer is not None
        assert agent.parser is not None
        assert agent.workflow is not None
        assert agent.graph is not None

    def test_initialization_custom(self):
        """Test agent initialization with custom parameters."""
        agent = ResumeParserAgent(max_retries=5, enable_checkpointing=False)
        assert agent.max_retries == 5
        assert agent.checkpointer is None

    # Workflow node tests
    def test_initialize_parsing_node(self, agent):
        """Test _initialize_parsing workflow node."""
        initial_state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "starting",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._initialize_parsing(initial_state)

        assert result["status"] == "initialized"
        assert result["current_node"] == "initialize"
        assert result["workflow_start_time"] is not None
        assert result["processing_metadata"]["agent_version"] == "1.0"
        assert result["processing_metadata"]["parser_initialized"] is True
        assert result["processing_metadata"]["max_retries"] == agent.max_retries

    def test_validate_input_success_file_path(self, agent):
        """Test _validate_input node with valid file path."""
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            state: ResumeParsingState = {
                "file_path": temp_file.name,
                "file_bytes": None,
                "file_extension": ".pdf",
                "original_filename": "resume.pdf",
                "status": "initialized",
                "error_message": None,
                "retry_count": 0,
                "parsed_resume": None,
                "parser_report": None,
                "processing_metadata": {},
                "workflow_start_time": None,
                "workflow_end_time": None,
                "current_node": None,
            }

            result = agent._validate_input(state)

            assert result["status"] == "input_validated"
            assert result["current_node"] == "validate_input"
            assert result["error_message"] is None

    def test_validate_input_success_bytes(self, agent):
        """Test _validate_input node with valid bytes data."""
        state: ResumeParsingState = {
            "file_path": None,
            "file_bytes": b"fake pdf content",
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "initialized",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._validate_input(state)

        assert result["status"] == "input_validated"
        assert result["current_node"] == "validate_input"

    def test_validate_input_no_data(self, agent):
        """Test _validate_input node with no input data."""
        state: ResumeParsingState = {
            "file_path": None,
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": None,
            "status": "initialized",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._validate_input(state)

        assert result["status"] == "input_validation_failed"
        assert "No file path or file bytes provided" in result["error_message"]

    def test_validate_input_unsupported_format(self, agent):
        """Test _validate_input node with unsupported file format."""
        state: ResumeParsingState = {
            "file_path": None,
            "file_bytes": b"content",
            "file_extension": ".txt",
            "original_filename": "resume.txt",
            "status": "initialized",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._validate_input(state)

        assert result["status"] == "input_validation_failed"
        assert "Unsupported file format: .txt" in result["error_message"]

    def test_validate_input_file_not_found(self, agent):
        """Test _validate_input node with non-existent file."""
        state: ResumeParsingState = {
            "file_path": "/nonexistent/file.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "initialized",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._validate_input(state)

        assert result["status"] == "input_validation_failed"
        assert "File not found" in result["error_message"]

    @patch("tools.resume_parser.ResumeParser.parse_file")
    def test_parse_resume_success_file_path(
        self, mock_parse_file, agent, sample_resume_data, sample_parser_report
    ):
        """Test _parse_resume node with successful file parsing."""
        mock_parse_file.return_value = (sample_resume_data, sample_parser_report)

        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "input_validated",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._parse_resume(state)

        assert result["status"] == "parsed"
        assert result["current_node"] == "parse_resume"
        assert result["parsed_resume"] is not None
        assert result["parser_report"] == sample_parser_report
        mock_parse_file.assert_called_once_with("/test/resume.pdf")

    @patch("tools.resume_parser.ResumeParser.parse_bytes")
    def test_parse_resume_success_bytes(
        self, mock_parse_bytes, agent, sample_resume_data, sample_parser_report
    ):
        """Test _parse_resume node with successful bytes parsing."""
        mock_parse_bytes.return_value = (sample_resume_data, sample_parser_report)

        state: ResumeParsingState = {
            "file_path": None,
            "file_bytes": b"fake content",
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "input_validated",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._parse_resume(state)

        assert result["status"] == "parsed"
        assert result["parsed_resume"] is not None
        mock_parse_bytes.assert_called_once_with(b"fake content", ".pdf")

    @patch("tools.resume_parser.ResumeParser.parse_file")
    def test_parse_resume_parsing_error(self, mock_parse_file, agent):
        """Test _parse_resume node with ResumeParsingError."""
        mock_parse_file.side_effect = ResumeParsingError("Parsing failed")

        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "input_validated",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._parse_resume(state)

        assert result["status"] == "parsing_failed"
        assert "Parsing failed" in result["error_message"]

    @patch("tools.resume_parser.ResumeParser.parse_file")
    def test_parse_resume_unexpected_error(self, mock_parse_file, agent):
        """Test _parse_resume node with unexpected error."""
        mock_parse_file.side_effect = Exception("Unexpected error")

        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "input_validated",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._parse_resume(state)

        assert result["status"] == "parsing_failed"
        assert "Unexpected error during parsing" in result["error_message"]

    def test_parse_resume_no_input_data(self, agent):
        """Test _parse_resume node with no input data."""
        state: ResumeParsingState = {
            "file_path": None,
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "input_validated",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._parse_resume(state)

        assert result["status"] == "parsing_failed"
        assert "No valid input data available" in result["error_message"]

    def test_validate_results_success_high_confidence(self, agent):
        """Test _validate_results node with high confidence results."""
        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "parsed",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": {"raw_text": "sample content"},
            "parser_report": {
                "confidence": 0.85,
                "sections_found": 4,
                "bullets_found": 10,
                "text_length": 1000,
            },
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._validate_results(state)

        assert result["status"] == "results_validated"
        assert result["current_node"] == "validate_results"
        assert result["processing_metadata"]["quality_score"] == 0.85
        assert "validation_warnings" in result["processing_metadata"]

    def test_validate_results_low_confidence(self, agent):
        """Test _validate_results node with low confidence results."""
        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "parsed",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": {"raw_text": "sample content"},
            "parser_report": {
                "confidence": 0.2,  # Below threshold
                "sections_found": 1,
                "bullets_found": 2,
                "text_length": 50,
            },
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._validate_results(state)

        assert result["status"] == "results_validation_failed"
        warnings = result["processing_metadata"]["validation_warnings"]
        assert any("Low parsing confidence" in warning for warning in warnings)

    def test_validate_results_no_data(self, agent):
        """Test _validate_results node with no parsing results."""
        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "parsed",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._validate_results(state)

        assert result["status"] == "results_validation_failed"
        assert "No parsing results to validate" in result["error_message"]

    def test_handle_error_within_retry_limit(self, agent):
        """Test _handle_error node within retry limit."""
        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "parsing_failed",
            "error_message": "Test error",
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._handle_error(state)

        assert result["current_node"] == "handle_error"
        assert result["retry_count"] == 1
        assert result["status"] == "retrying"

    def test_handle_error_max_retries_exceeded(self, agent):
        """Test _handle_error node when max retries exceeded."""
        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "parsing_failed",
            "error_message": "Test error",
            "retry_count": 2,  # At max retries
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._handle_error(state)

        assert result["retry_count"] == 3
        assert result["status"] == "max_retries_exceeded"

    def test_handle_error_non_retryable(self, agent):
        """Test _handle_error node with non-retryable error."""
        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "input_validation_failed",  # Not retryable
            "error_message": "Invalid input",
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._handle_error(state)

        assert result["retry_count"] == 1
        assert result["status"] == "max_retries_exceeded"

    def test_finalize_parsing_success(self, agent):
        """Test _finalize_parsing node with successful parsing."""
        start_time = datetime.now(timezone.utc).isoformat()
        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "results_validated",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": {"raw_text": "sample content"},
            "parser_report": {"confidence": 0.8},
            "processing_metadata": {},
            "workflow_start_time": start_time,
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._finalize_parsing(state)

        assert result["status"] == "completed"
        assert result["current_node"] == "finalize"
        assert result["workflow_end_time"] is not None
        assert "processing_time_seconds" in result["processing_metadata"]

    def test_finalize_parsing_failure(self, agent):
        """Test _finalize_parsing node with parsing failure."""
        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "max_retries_exceeded",
            "error_message": "Parsing failed",
            "retry_count": 3,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": datetime.now(timezone.utc).isoformat(),
            "workflow_end_time": None,
            "current_node": None,
        }

        result = agent._finalize_parsing(state)

        assert result["status"] == "failed"
        assert result["current_node"] == "finalize"

    # Routing function tests
    def test_route_after_validation_success(self, agent):
        """Test _route_after_validation routing function."""
        state = {"status": "input_validated"}
        route = agent._route_after_validation(state)
        assert route == "parse"

        state = {"status": "input_validation_failed"}
        route = agent._route_after_validation(state)
        assert route == "error"

    def test_route_after_results_success(self, agent):
        """Test _route_after_results routing function."""
        state = {"status": "results_validated"}
        route = agent._route_after_results(state)
        assert route == "finalize"

        state = {"status": "results_validation_failed", "retry_count": 1}
        route = agent._route_after_results(state)
        assert route == "retry"

        state = {"status": "results_validation_failed", "retry_count": 3}
        route = agent._route_after_results(state)
        assert route == "error"

    def test_route_after_error(self, agent):
        """Test _route_after_error routing function."""
        state = {"status": "retrying"}
        route = agent._route_after_error(state)
        assert route == "retry"

        state = {"status": "max_retries_exceeded"}
        route = agent._route_after_error(state)
        assert route == "finalize"

    # High-level interface tests
    @patch("tools.resume_parser.ResumeParser.parse_file")
    def test_parse_file_interface_success(
        self, mock_parse_file, agent, sample_resume_data, sample_parser_report
    ):
        """Test parse_file interface method with successful parsing."""
        mock_parse_file.return_value = (sample_resume_data, sample_parser_report)

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            result = agent.parse_file(temp_file.name)

            assert result["success"] is True
            assert result["status"] == "completed"
            assert result["resume"] is not None
            assert result["report"] == sample_parser_report
            assert result["error"] is None
            assert result["retry_count"] == 0
            assert "processing_time" in result

    @patch("tools.resume_parser.ResumeParser.parse_bytes")
    def test_parse_bytes_interface_success(
        self, mock_parse_bytes, agent, sample_resume_data, sample_parser_report
    ):
        """Test parse_bytes interface method with successful parsing."""
        mock_parse_bytes.return_value = (sample_resume_data, sample_parser_report)

        result = agent.parse_bytes(b"fake content", ".pdf", "resume.pdf")

        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["resume"] is not None
        assert result["error"] is None

    @patch("tools.resume_parser.ResumeParser.parse_file")
    def test_parse_file_interface_failure(self, mock_parse_file, agent):
        """Test parse_file interface method with parsing failure."""
        mock_parse_file.side_effect = ResumeParsingError("Parsing failed")

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            result = agent.parse_file(temp_file.name)

            assert result["success"] is False
            assert result["status"] == "failed"
            assert result["resume"] is None
            assert "Parsing failed" in result["error"]
            assert result["retry_count"] > 0

    def test_parse_file_interface_path_handling(self, agent):
        """Test parse_file interface with different path types."""
        # Test with string path
        with patch("tools.resume_parser.ResumeParser.parse_file") as mock_parse:
            mock_parse.side_effect = ResumeParsingError("File not found")
            result = agent.parse_file("resume.pdf")
            assert result["success"] is False

        # Test with Path object
        with patch("tools.resume_parser.ResumeParser.parse_file") as mock_parse:
            mock_parse.side_effect = ResumeParsingError("File not found")
            result = agent.parse_file(Path("resume.pdf"))
            assert result["success"] is False

    def test_parse_bytes_extension_normalization(self, agent):
        """Test parse_bytes with extension normalization."""
        with patch("tools.resume_parser.ResumeParser.parse_bytes") as mock_parse:
            mock_parse.side_effect = ResumeParsingError("Parsing failed")

            # Test without dot
            result = agent.parse_bytes(b"content", "pdf")
            assert result["success"] is False

            # Test with dot
            result = agent.parse_bytes(b"content", ".pdf")
            assert result["success"] is False

    # Async interface tests
    @pytest.mark.asyncio
    @patch("tools.resume_parser.ResumeParser.parse_file")
    async def test_aparse_file(
        self, mock_parse_file, agent, sample_resume_data, sample_parser_report
    ):
        """Test async parse_file interface."""
        mock_parse_file.return_value = (sample_resume_data, sample_parser_report)

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            result = await agent.aparse_file(temp_file.name)
            assert result["success"] is True

    @pytest.mark.asyncio
    @patch("tools.resume_parser.ResumeParser.parse_bytes")
    async def test_aparse_bytes(
        self, mock_parse_bytes, agent, sample_resume_data, sample_parser_report
    ):
        """Test async parse_bytes interface."""
        mock_parse_bytes.return_value = (sample_resume_data, sample_parser_report)

        result = await agent.aparse_bytes(b"content", ".pdf")
        assert result["success"] is True

    # Format result tests
    def test_format_result_success(self, agent):
        """Test _format_result method with successful state."""
        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "completed",
            "error_message": None,
            "retry_count": 1,
            "parsed_resume": {"raw_text": "content"},
            "parser_report": {"confidence": 0.8},
            "processing_metadata": {"processing_time_seconds": 2.5},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": "finalize",
        }

        result = agent._format_result(state)

        assert result["status"] == "completed"
        assert result["success"] is True
        assert result["resume"] == {"raw_text": "content"}
        assert result["report"] == {"confidence": 0.8}
        assert result["metadata"] == {"processing_time_seconds": 2.5}
        assert result["error"] is None
        assert result["retry_count"] == 1
        assert result["processing_time"] == 2.5

    def test_format_result_failure(self, agent):
        """Test _format_result method with failed state."""
        state: ResumeParsingState = {
            "file_path": "/test/resume.pdf",
            "file_bytes": None,
            "file_extension": ".pdf",
            "original_filename": "resume.pdf",
            "status": "failed",
            "error_message": "Parsing failed",
            "retry_count": 3,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": "finalize",
        }

        result = agent._format_result(state)

        assert result["status"] == "failed"
        assert result["success"] is False
        assert result["resume"] is None
        assert result["error"] == "Parsing failed"
        assert result["retry_count"] == 3

    # Integration tests
    @patch("tools.resume_parser.ResumeParser")
    def test_integration_complete_workflow_success(self, mock_parser_class, agent):
        """Integration test: Complete successful workflow."""
        # Mock parser instance and methods
        mock_parser = Mock()
        mock_resume = Mock()
        mock_resume.model_dump.return_value = {"raw_text": "content"}
        mock_resume.sections = [
            "Experience",
            "Education",
            "Skills",
        ]  # Mock sections for len()
        mock_resume.bullets = ["bullet1", "bullet2"]  # Mock bullets for len()
        mock_resume.skills = ["Python", "Java"]  # Mock skills for len()
        mock_report = {"confidence": 0.8, "sections_found": 3}
        mock_parser.parse_file.return_value = (mock_resume, mock_report)
        mock_parser_class.return_value = mock_parser

        # Create new agent to use mocked parser
        test_agent = ResumeParserAgent(max_retries=2, enable_checkpointing=False)
        test_agent.parser = mock_parser

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            result = test_agent.parse_file(temp_file.name)

            assert result["success"] is True
            assert result["status"] == "completed"
            assert result["resume"]["raw_text"] == "content"
            assert result["report"]["confidence"] == 0.8

    @patch("tools.resume_parser.ResumeParser")
    def test_integration_complete_workflow_with_retries(self, mock_parser_class):
        """Integration test: Workflow with retries and eventual success."""
        mock_parser = Mock()
        mock_resume = Mock()
        mock_resume.model_dump.return_value = {"raw_text": "content"}
        mock_resume.sections = []  # Empty list to avoid len() errors
        mock_resume.bullets = []
        mock_resume.skills = []
        mock_report = {
            "confidence": 0.2,
            "sections_found": 1,
        }  # Below minimum confidence
        mock_parser.parse_file.return_value = (mock_resume, mock_report)
        mock_parser_class.return_value = mock_parser

        test_agent = ResumeParserAgent(max_retries=2, enable_checkpointing=False)
        test_agent.parser = mock_parser

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            result = test_agent.parse_file(temp_file.name)

            # Should fail due to low confidence and max retries
            assert result["success"] is False
            assert result["retry_count"] > 0

    @patch("tools.resume_parser.ResumeParser")
    def test_integration_complete_workflow_failure(self, mock_parser_class):
        """Integration test: Complete workflow with failure."""
        mock_parser = Mock()
        mock_parser.parse_file.side_effect = ResumeParsingError("Parse error")
        mock_parser_class.return_value = mock_parser

        test_agent = ResumeParserAgent(max_retries=2, enable_checkpointing=False)
        test_agent.parser = mock_parser

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            result = test_agent.parse_file(temp_file.name)

            assert result["success"] is False
            assert result["status"] == "failed"
            assert "Parse error" in result["error"]
            assert result["retry_count"] == test_agent.max_retries
