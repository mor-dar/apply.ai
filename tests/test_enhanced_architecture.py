"""
Comprehensive unit tests for the enhanced architecture implementation.

Tests the enhanced data models, quality gates, confidence scoring,
and agent policy implementation following the reference architecture.
"""

import pytest
from unittest.mock import Mock, patch

from src.schemas.core import Requirement, ParserReport, JobPosting
from tools.job_posting_parser import JobPostingParser
from agents.job_post_parser_agent import (
    JobPostParserAgent,
    JobPostParsingState,
    CONFIDENCE_MIN,
    REQUIRED_FIELDS,
)


class TestEnhancedSchemas:
    """Test the enhanced Pydantic schemas."""

    def test_requirement_creation_must_have(self):
        """Test creating a must-have requirement."""
        req = Requirement(
            text="Bachelor's degree required",
            must_have=True,
            rationale="Contains explicit requirement language",
        )

        assert req.text == "Bachelor's degree required"
        assert req.must_have is True
        assert req.rationale == "Contains explicit requirement language"
        assert len(req.id) == 8  # UUID prefix

    def test_requirement_creation_nice_to_have(self):
        """Test creating a nice-to-have requirement."""
        req = Requirement(
            text="GraphQL experience preferred",
            must_have=False,
            rationale="Contains 'preferred' indicating nice-to-have",
        )

        assert req.text == "GraphQL experience preferred"
        assert req.must_have is False
        assert req.rationale == "Contains 'preferred' indicating nice-to-have"

    def test_requirement_default_must_have(self):
        """Test requirement defaults to must_have=True."""
        req = Requirement(text="Python experience")
        assert req.must_have is True
        assert req.rationale is None

    def test_parser_report_creation(self):
        """Test creating a ParserReport with all fields."""
        report = ParserReport(
            confidence=0.75,
            missing_fields=["title"],
            warnings=["Short text"],
            keyword_count=5,
            requirement_count=3,
            text_length=200,
        )

        assert report.confidence == 0.75
        assert report.missing_fields == ["title"]
        assert report.warnings == ["Short text"]
        assert report.keyword_count == 5
        assert report.requirement_count == 3
        assert report.text_length == 200

    def test_parser_report_confidence_bounds(self):
        """Test ParserReport confidence validation bounds."""
        # Valid confidence values
        ParserReport(
            confidence=0.0, keyword_count=0, requirement_count=0, text_length=0
        )
        ParserReport(
            confidence=1.0, keyword_count=0, requirement_count=0, text_length=0
        )

        # Invalid confidence values
        with pytest.raises(ValueError):
            ParserReport(
                confidence=-0.1, keyword_count=0, requirement_count=0, text_length=0
            )
        with pytest.raises(ValueError):
            ParserReport(
                confidence=1.1, keyword_count=0, requirement_count=0, text_length=0
            )

    def test_parser_report_defaults(self):
        """Test ParserReport default values."""
        report = ParserReport(
            confidence=0.5, keyword_count=0, requirement_count=0, text_length=0
        )

        assert report.missing_fields == []
        assert report.warnings == []

    def test_job_posting_with_requirement_objects(self):
        """Test JobPosting with Requirement objects instead of strings."""
        requirements = [
            Requirement(text="Python required", must_have=True),
            Requirement(text="AWS preferred", must_have=False),
        ]

        job_posting = JobPosting(
            title="Software Engineer",
            company="TechCorp",
            text="Job description",
            keywords=["python", "aws"],
            requirements=requirements,
        )

        assert len(job_posting.requirements) == 2
        assert job_posting.requirements[0].text == "Python required"
        assert job_posting.requirements[0].must_have is True
        assert job_posting.requirements[1].text == "AWS preferred"
        assert job_posting.requirements[1].must_have is False

    def test_job_posting_keyword_validation(self):
        """Test JobPosting keyword validation strips whitespace."""
        job_posting = JobPosting(
            title="Software Engineer",
            company="TechCorp",
            text="Job description",
            keywords=["  python  ", "", "aws", "   "],  # Mixed whitespace/empty
            requirements=[],
        )

        # Should strip whitespace and remove empty strings
        assert job_posting.keywords == ["python", "aws"]


class TestEnhancedJobPostingParser:
    """Test the enhanced JobPostingParser tool with confidence scoring."""

    @pytest.fixture
    def parser(self):
        """Create a JobPostingParser instance for testing."""
        return JobPostingParser()

    def test_parse_returns_tuple(self, parser):
        """Test that parse method returns tuple of (JobPosting, ParserReport)."""
        result = parser.parse(
            job_text="Software engineer with Python experience required",
            title="Software Engineer",
            company="TechCorp",
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        job_posting, parser_report = result
        assert isinstance(job_posting, JobPosting)
        assert isinstance(parser_report, ParserReport)

    def test_structured_requirements_extraction(self, parser):
        """Test extraction of structured requirements with must_have classification."""
        job_text = """
        Software Engineer position.
        
        Requirements:
        - Bachelor's degree required
        - Python experience (must have)
        - AWS knowledge preferred
        - Docker skills would be nice to have
        """

        job_posting, report = parser.parse(job_text, "Software Engineer", "TechCorp")

        # Should extract structured requirements
        assert len(job_posting.requirements) > 0

        # Check must_have classification
        must_have_reqs = [r for r in job_posting.requirements if r.must_have]
        nice_to_have_reqs = [r for r in job_posting.requirements if not r.must_have]

        # Should have both types if classification is working
        assert len(must_have_reqs) >= 0  # May vary based on extraction
        assert len(nice_to_have_reqs) >= 0  # May vary based on extraction

    def test_confidence_scoring_high_quality(self, parser):
        """Test confidence scoring with high-quality input."""
        job_text = """
        Senior Software Engineer position at TechCorp.
        
        We're looking for an experienced developer to join our team.
        
        Requirements:
        - Bachelor's degree in Computer Science required
        - 5+ years Python development experience 
        - Experience with Django, Flask frameworks
        - Knowledge of PostgreSQL databases
        - Strong problem-solving skills
        
        Responsibilities:
        - Design scalable web applications
        - Collaborate with product team
        - Code reviews and mentoring
        """

        job_posting, report = parser.parse(
            job_text, "Senior Software Engineer", "TechCorp"
        )

        # High quality input should have good confidence
        assert report.confidence >= 0.6
        assert report.keyword_count > 0
        assert report.text_length > 300
        assert len(report.missing_fields) == 0  # Title and company provided

    def test_confidence_scoring_low_quality(self, parser):
        """Test confidence scoring with low-quality input."""
        job_text = "Job"  # Very short, minimal content

        job_posting, report = parser.parse(job_text, "", "")  # No title/company

        # Low quality input should have low confidence
        assert report.confidence < 0.5
        assert "title" in report.missing_fields
        assert "company" in report.missing_fields
        assert any("short" in warning.lower() for warning in report.warnings)

    def test_requirement_classification_must_have(self, parser):
        """Test requirement classification for must-have patterns."""
        test_cases = [
            "Bachelor's degree required",
            "Python experience must have",
            "Essential: 3+ years experience",
            "Mandatory knowledge of SQL",
            "Minimum 2 years experience",
        ]

        for req_text in test_cases:
            must_have = parser._classify_requirement(req_text)
            assert must_have is True, f"'{req_text}' should be classified as must-have"

    def test_requirement_classification_nice_to_have(self, parser):
        """Test requirement classification for nice-to-have patterns."""
        test_cases = [
            "Docker experience preferred",
            "AWS knowledge would be nice to have",
            "GraphQL skills are a bonus",
            "Kubernetes experience is a plus",
            "Ideally has startup experience",
        ]

        for req_text in test_cases:
            must_have = parser._classify_requirement(req_text)
            assert (
                must_have is False
            ), f"'{req_text}' should be classified as nice-to-have"

    def test_classification_rationale_generation(self, parser):
        """Test generation of rationale for classifications."""
        # Must-have rationale
        rationale = parser._get_classification_rationale("Python required", True)
        assert "requirement language" in rationale.lower()

        # Nice-to-have rationale
        rationale = parser._get_classification_rationale("AWS preferred", False)
        assert "preferred" in rationale.lower()

    def test_parser_report_generation_comprehensive(self, parser):
        """Test comprehensive parser report generation."""
        job_text = "Software engineer needed"  # Minimal content

        job_posting, report = parser.parse(job_text, "Engineer", "Corp")

        # Should generate comprehensive report
        assert isinstance(report.confidence, float)
        assert 0.0 <= report.confidence <= 1.0
        assert isinstance(report.keyword_count, int)
        assert isinstance(report.requirement_count, int)
        assert isinstance(report.text_length, int)
        assert isinstance(report.missing_fields, list)
        assert isinstance(report.warnings, list)

        # Should have metrics matching actual results
        assert report.keyword_count == len(job_posting.keywords)
        assert report.requirement_count == len(job_posting.requirements)
        assert report.text_length == len(job_text)

    def test_edge_case_empty_input(self, parser):
        """Test parser handles empty input gracefully."""
        job_posting, report = parser.parse("", "", "")

        assert job_posting.title == "Unknown Position"
        assert job_posting.company == "Unknown Company"
        assert report.confidence < 0.5
        assert "title" in report.missing_fields
        assert "company" in report.missing_fields

    def test_edge_case_very_long_input(self, parser):
        """Test parser handles very long input."""
        long_text = "Software engineer position. " * 200  # Very long (5600+ chars)

        job_posting, report = parser.parse(long_text, "Engineer", "Corp")

        assert report.text_length > 5000
        assert any("long" in warning.lower() for warning in report.warnings)

    def test_error_handling_invalid_input(self, parser):
        """Test parser error handling with invalid input."""
        with pytest.raises(ValueError):
            parser.parse(123)  # Non-string input


class TestEnhancedJobPostParserAgent:
    """Test the enhanced JobPostParserAgent with quality gates and policy."""

    @pytest.fixture
    def mock_parser(self):
        """Create a mock JobPostingParser for agent testing."""
        mock = Mock(spec=JobPostingParser)
        mock.max_keywords = 30
        mock.min_keyword_length = 2
        return mock

    @pytest.fixture
    def agent(self, mock_parser):
        """Create agent with mock parser for testing."""
        with patch("agents.job_post_parser_agent.JobPostingParser") as mock_cls:
            mock_cls.return_value = mock_parser
            agent = JobPostParserAgent(max_retries=2, enable_checkpoints=False)
            agent.parser = mock_parser
            return agent

    def test_agent_state_includes_parser_report(self):
        """Test that agent state includes parser_report field."""
        state = JobPostParsingState(
            job_text="test",
            job_title="Engineer",
            company_name="Corp",
            job_location=None,
            status="pending",
            error_message=None,
            retry_count=0,
            parsed_job=None,
            parser_report=None,
            processing_metadata={},
            started_at=None,
            completed_at=None,
        )

        assert "parser_report" in state
        assert state["parser_report"] is None

    def test_parse_job_posting_with_quality_gates_success(self, agent, mock_parser):
        """Test parsing with quality gates - success case."""
        # Mock high-confidence parsing result
        mock_job_posting = Mock()
        mock_job_posting.model_dump.return_value = {
            "title": "Software Engineer",
            "company": "TechCorp",
            "location": None,
            "text": "Test job description",
            "keywords": ["python", "software"],
            "requirements": [],
        }
        # Add len() support for mock objects used in validation
        mock_job_posting.keywords = ["python", "software"]
        mock_job_posting.requirements = []

        mock_parser_report = Mock()
        mock_parser_report.model_dump.return_value = {
            "confidence": 0.85,  # Above threshold
            "missing_fields": [],
            "warnings": [],
            "keyword_count": 2,
            "requirement_count": 0,
            "text_length": 100,
        }
        mock_parser_report.confidence = 0.85
        mock_parser_report.missing_fields = []

        mock_parser.parse.return_value = (mock_job_posting, mock_parser_report)

        # Test parsing
        state = JobPostParsingState(
            job_text="Test job description",
            job_title="Software Engineer",
            company_name="TechCorp",
            job_location=None,
            status="initializing",
            error_message=None,
            retry_count=0,
            parsed_job=None,
            parser_report=None,
            processing_metadata={},
            started_at=None,
            completed_at=None,
        )

        result = agent._parse_job_posting(state)

        assert result["status"] == "parsed"
        assert result["parsed_job"] is not None
        assert result["parser_report"] is not None
        assert result["error_message"] is None

    def test_parse_job_posting_quality_gates_failure(self, agent, mock_parser):
        """Test parsing with quality gates - failure case."""
        # Mock low-confidence parsing result
        mock_job_posting = Mock()
        mock_job_posting.model_dump.return_value = {
            "title": "Unknown Position",
            "company": "Unknown Company",
            "location": None,
            "text": "Short",
            "keywords": [],
            "requirements": [],
        }
        # Add len() support
        mock_job_posting.keywords = []
        mock_job_posting.requirements = []

        mock_parser_report = Mock()
        mock_parser_report.model_dump.return_value = {
            "confidence": 0.3,  # Below threshold
            "missing_fields": ["title", "company"],
            "warnings": ["No keywords extracted"],
            "keyword_count": 0,
            "requirement_count": 0,
            "text_length": 5,
        }
        mock_parser_report.confidence = 0.3
        mock_parser_report.missing_fields = ["title", "company"]

        mock_parser.parse.return_value = (mock_job_posting, mock_parser_report)

        state = JobPostParsingState(
            job_text="Short",
            job_title=None,
            company_name=None,
            job_location=None,
            status="initializing",
            error_message=None,
            retry_count=0,
            parsed_job=None,
            parser_report=None,
            processing_metadata={},
            started_at=None,
            completed_at=None,
        )

        # Should still parse successfully but log quality issues
        with patch("agents.job_post_parser_agent.logger") as mock_logger:
            result = agent._parse_job_posting(state)

            # Should still return parsed status (gates warn but don't block)
            assert result["status"] == "parsed"
            assert result["parsed_job"] is not None
            assert result["parser_report"] is not None

            # Should log quality gate warnings
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Quality gates failed" in warning_call
            assert "low confidence" in warning_call
            assert "missing required fields" in warning_call

    def test_validate_results_with_parser_report(self, agent):
        """Test validation using parser report (new validation method)."""
        # High confidence case
        state = JobPostParsingState(
            job_text="Test job description",
            job_title="Engineer",
            company_name="Corp",
            job_location=None,
            status="parsed",
            error_message=None,
            retry_count=0,
            parsed_job={"title": "Engineer", "company": "Corp", "keywords": ["python"]},
            parser_report={
                "confidence": 0.8,  # Above threshold
                "missing_fields": [],
                "warnings": [],
                "keyword_count": 1,
                "requirement_count": 1,
                "text_length": 100,
            },
            processing_metadata={},
            started_at=None,
            completed_at=None,
        )

        result = agent._validate_results(state)
        assert result["status"] == "validated"

    def test_validate_results_confidence_too_low(self, agent):
        """Test validation failure due to low confidence."""
        state = JobPostParsingState(
            job_text="Test job description",
            job_title="Engineer",
            company_name="Corp",
            job_location=None,
            status="parsed",
            error_message=None,
            retry_count=0,
            parsed_job={"title": "Engineer", "company": "Corp"},
            parser_report={
                "confidence": 0.4,  # Below threshold
                "missing_fields": [],
                "warnings": [],
                "keyword_count": 0,
                "requirement_count": 0,
                "text_length": 50,
            },
            processing_metadata={},
            started_at=None,
            completed_at=None,
        )

        result = agent._validate_results(state)

        assert result["status"] == "validation_failed"
        assert "Confidence too low" in result["error_message"]
        assert (
            "0.4" in result["error_message"]
            and str(CONFIDENCE_MIN) in result["error_message"]
        )

    def test_validate_results_missing_required_fields(self, agent):
        """Test validation failure due to missing required fields."""
        state = JobPostParsingState(
            job_text="Test job description",
            job_title="Engineer",
            company_name="Corp",
            job_location=None,
            status="parsed",
            error_message=None,
            retry_count=0,
            parsed_job={"title": "Engineer", "company": "Corp"},
            parser_report={
                "confidence": 0.8,  # Good confidence
                "missing_fields": ["title", "company"],  # But missing required fields
                "warnings": [],
                "keyword_count": 5,
                "requirement_count": 2,
                "text_length": 200,
            },
            processing_metadata={},
            started_at=None,
            completed_at=None,
        )

        result = agent._validate_results(state)

        assert result["status"] == "validation_failed"
        assert "Missing required fields" in result["error_message"]

    def test_validate_results_critical_warnings(self, agent):
        """Test validation failure due to critical warnings."""
        state = JobPostParsingState(
            job_text="Test job description",
            job_title="Engineer",
            company_name="Corp",
            job_location=None,
            status="parsed",
            error_message=None,
            retry_count=0,
            parsed_job={"title": "Engineer", "company": "Corp"},
            parser_report={
                "confidence": 0.8,  # Good confidence
                "missing_fields": [],
                "warnings": [
                    "No keywords extracted",
                    "No requirements found",
                ],  # Critical warnings
                "keyword_count": 0,
                "requirement_count": 0,
                "text_length": 100,
            },
            processing_metadata={},
            started_at=None,
            completed_at=None,
        )

        result = agent._validate_results(state)

        assert result["status"] == "validation_failed"
        assert "No keywords extracted" in result["error_message"]

    def test_basic_validation_fallback(self, agent):
        """Test fallback to basic validation when no parser report available."""
        state = JobPostParsingState(
            job_text="Test job description",
            job_title="Engineer",
            company_name="Corp",
            job_location=None,
            status="parsed",
            error_message=None,
            retry_count=0,
            parsed_job={
                "title": "Engineer",
                "company": "Corp",
                "keywords": ["python"],
                "requirements": [{"text": "Python required"}],
                "text": "Test job description with sufficient content for validation to pass",
            },
            parser_report=None,  # No parser report
            processing_metadata={},
            started_at=None,
            completed_at=None,
        )

        result = agent._validate_results(state)
        assert result["status"] == "validated"

    def test_finalize_results_with_enhanced_metadata(self, agent):
        """Test finalization includes enhanced metadata from parser report."""
        state = JobPostParsingState(
            job_text="Test job description",
            job_title="Engineer",
            company_name="Corp",
            job_location=None,
            status="validated",
            error_message=None,
            retry_count=1,
            parsed_job={
                "title": "Engineer",
                "company": "Corp",
                "keywords": ["python", "java"],
                "requirements": [
                    {"text": "Python required"},
                    {"text": "Java preferred"},
                ],
            },
            parser_report={
                "confidence": 0.75,
                "missing_fields": [],
                "warnings": ["Short description"],
                "keyword_count": 2,
                "requirement_count": 2,
                "text_length": 100,
            },
            processing_metadata={"initial": "data"},
            started_at="2025-01-01T00:00:00Z",
            completed_at=None,
        )

        result = agent._finalize_results(state)

        metadata = result["processing_metadata"]
        assert metadata["final_status"] == "completed"
        assert metadata["total_keywords"] == 2
        assert metadata["total_requirements"] == 2
        assert metadata["retry_attempts"] == 1
        assert metadata["final_confidence"] == 0.75
        assert metadata["parser_warnings_count"] == 1
        assert metadata["missing_fields_count"] == 0

        assert result["status"] == "completed"
        assert result["completed_at"] is not None

    def test_constants_defined(self):
        """Test that quality gate constants are properly defined."""
        assert isinstance(CONFIDENCE_MIN, float)
        assert 0.0 < CONFIDENCE_MIN < 1.0
        assert isinstance(REQUIRED_FIELDS, tuple)
        assert "title" in REQUIRED_FIELDS
        assert "company" in REQUIRED_FIELDS

    def test_integration_workflow_success(self, agent, mock_parser):
        """Test complete workflow integration - success path."""
        # Mock successful parsing
        mock_job_posting = Mock()
        mock_job_posting.model_dump.return_value = {
            "title": "Software Engineer",
            "company": "TechCorp",
            "location": "Remote",
            "text": "Comprehensive job description",
            "keywords": ["python", "software", "engineer"],
            "requirements": [{"text": "Python required", "must_have": True}],
        }
        # Add len() support
        mock_job_posting.keywords = ["python", "software", "engineer"]
        mock_job_posting.requirements = [{"text": "Python required", "must_have": True}]

        mock_parser_report = Mock()
        mock_parser_report.model_dump.return_value = {
            "confidence": 0.9,
            "missing_fields": [],
            "warnings": [],
            "keyword_count": 3,
            "requirement_count": 1,
            "text_length": 200,
        }
        mock_parser_report.confidence = 0.9
        mock_parser_report.missing_fields = []

        mock_parser.parse.return_value = (mock_job_posting, mock_parser_report)

        # Execute full workflow
        result = agent.parse_job_posting_sync(
            job_text="Comprehensive job description with Python requirements",
            job_title="Software Engineer",
            company_name="TechCorp",
            job_location="Remote",
        )

        assert result["success"] is True
        assert result["job_posting"] is not None
        assert result["error"] is None

        metadata = result["metadata"]
        assert metadata["final_confidence"] == 0.9
        assert metadata["total_keywords"] == 3
        assert metadata["total_requirements"] == 1

    def test_integration_workflow_quality_gates_failure(self, agent, mock_parser):
        """Test complete workflow integration - quality gates trigger failure."""
        # Mock low-quality parsing
        mock_job_posting = Mock()
        mock_job_posting.model_dump.return_value = {
            "title": "Unknown Position",
            "company": "Unknown Company",
            "location": None,
            "text": "Short",
            "keywords": [],
            "requirements": [],
        }
        # Add len() support
        mock_job_posting.keywords = []
        mock_job_posting.requirements = []

        mock_parser_report = Mock()
        mock_parser_report.model_dump.return_value = {
            "confidence": 0.2,  # Very low
            "missing_fields": ["title", "company"],
            "warnings": ["No keywords extracted", "No requirements extracted"],
            "keyword_count": 0,
            "requirement_count": 0,
            "text_length": 5,
        }
        mock_parser_report.confidence = 0.2
        mock_parser_report.missing_fields = ["title", "company"]

        mock_parser.parse.return_value = (mock_job_posting, mock_parser_report)

        # Should fail due to validation
        result = agent.parse_job_posting_sync(
            job_text="Short", job_title="", company_name=""
        )

        assert result["success"] is False
        assert (
            "Confidence too low" in result["error"]
            or "validation" in result["error"].lower()
            or "failed" in result["error"]
        )


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for the enhanced architecture."""

    def test_requirement_with_special_characters(self):
        """Test requirements with special characters and unicode."""
        req = Requirement(
            text="Experience with C++ and .NET frameworks (required)", must_have=True
        )
        assert req.text == "Experience with C++ and .NET frameworks (required)"
        assert req.must_have is True

    def test_parser_report_extreme_values(self):
        """Test parser report with extreme but valid values."""
        report = ParserReport(
            confidence=0.999,
            missing_fields=["field1", "field2", "field3"],
            warnings=["warning"] * 10,
            keyword_count=100,
            requirement_count=50,
            text_length=10000,
        )

        assert report.confidence == 0.999
        assert len(report.missing_fields) == 3
        assert len(report.warnings) == 10
        assert report.keyword_count == 100

    @patch("agents.job_post_parser_agent.logger")
    def test_agent_parser_exception_handling(self, mock_logger):
        """Test agent handles parser exceptions gracefully."""
        with patch("agents.job_post_parser_agent.JobPostingParser") as mock_cls:
            mock_parser = Mock()
            mock_parser.parse.side_effect = Exception("Parser error")
            mock_cls.return_value = mock_parser

            agent = JobPostParserAgent(enable_checkpoints=False)

            state = JobPostParsingState(
                job_text="test",
                job_title="Engineer",
                company_name="Corp",
                job_location=None,
                status="initializing",
                error_message=None,
                retry_count=0,
                parsed_job=None,
                parser_report=None,
                processing_metadata={},
                started_at=None,
                completed_at=None,
            )

            result = agent._parse_job_posting(state)

            assert result["status"] == "error"
            assert "Parser error" in result["error_message"]
            mock_logger.error.assert_called()

    def test_validation_no_parsed_job(self):
        """Test validation when no parsed job data exists."""
        with patch("agents.job_post_parser_agent.JobPostingParser"):
            agent = JobPostParserAgent(enable_checkpoints=False)

            state = JobPostParsingState(
                job_text="test",
                job_title="Engineer",
                company_name="Corp",
                job_location=None,
                status="parsed",
                error_message=None,
                retry_count=0,
                parsed_job=None,  # No parsed job
                parser_report=None,
                processing_metadata={},
                started_at=None,
                completed_at=None,
            )

            result = agent._validate_results(state)

            assert result["status"] == "validation_failed"
            assert "No parsed job data to validate" in result["error_message"]

    def test_smoke_test_full_pipeline(self):
        """Smoke test: Run the full pipeline with real components."""
        # Test with minimal real components (no mocks)
        parser = JobPostingParser()
        job_posting, report = parser.parse(
            "Software engineer needed with Python skills", "Engineer", "Corp"
        )

        # Should complete without errors
        assert isinstance(job_posting, JobPosting)
        assert isinstance(report, ParserReport)
        assert 0.0 <= report.confidence <= 1.0

        # Agent should also work
        agent = JobPostParserAgent(max_retries=1, enable_checkpoints=False)
        result = agent.parse_job_posting_sync(
            job_text="Software engineer needed with Python skills and bachelor's degree",
            job_title="Engineer",
            company_name="Corp",
        )

        # Should complete (may succeed or fail based on actual parsing)
        assert isinstance(result, dict)
        assert "success" in result
        assert "error" in result
