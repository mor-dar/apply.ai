"""
Comprehensive unit tests for ResumeParser tool.

Tests cover all functionality including PDF/DOCX parsing, structure extraction,
error handling, and edge cases. Maintains 100% test coverage with zero warnings.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile

from tools.resume_parser import ResumeParser, ResumeParsingError
from src.schemas.core import Resume, ResumeSection


class TestResumeParser:
    """Test suite for ResumeParser tool."""

    @pytest.fixture
    def parser(self):
        """Create a ResumeParser instance for testing."""
        return ResumeParser()

    @pytest.fixture
    def sample_resume_text(self):
        """Sample resume text for testing."""
        return """
John Doe
Software Engineer

EXPERIENCE
• Led development of microservices architecture serving 1M+ users
• Implemented CI/CD pipeline reducing deployment time by 50%
• Mentored 5 junior developers in Python and React technologies

EDUCATION
Bachelor of Science in Computer Science
University of Technology, 2020

SKILLS
Python, Java, JavaScript, React, Node.js, AWS, Docker, Kubernetes

PROJECTS
• E-commerce Platform: Built scalable online store using MERN stack
• Data Analysis Tool: Created Python application for financial data processing
"""

    @pytest.fixture
    def minimal_resume_text(self):
        """Minimal resume text for edge case testing."""
        return "John Doe\nSoftware Engineer\n\nExperience at ABC Corp"

    def test_initialization(self, parser):
        """Test ResumeParser initialization."""
        assert isinstance(parser, ResumeParser)
        assert len(parser.section_patterns) > 0
        assert len(parser.bullet_patterns) > 0
        assert len(parser.date_patterns) > 0
        assert len(parser.skills_indicators) > 0

    # File parsing tests
    def test_parse_file_nonexistent(self, parser):
        """Test parsing non-existent file raises error."""
        with pytest.raises(ResumeParsingError, match="File not found"):
            parser.parse_file("/nonexistent/file.pdf")

    def test_parse_file_unsupported_format(self, parser):
        """Test parsing unsupported file format raises error."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            with pytest.raises(ResumeParsingError, match="Unsupported file format"):
                parser.parse_file(temp_file.name)

    @patch("pdfplumber.open")
    def test_parse_pdf_file_success(self, mock_pdf_open, parser, sample_resume_text):
        """Test successful PDF file parsing."""
        # Mock PDF structure
        mock_page = Mock()
        mock_page.extract_text.return_value = sample_resume_text
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            resume, report = parser.parse_file(temp_file.name)

            assert isinstance(resume, Resume)
            assert resume.raw_text == sample_resume_text
            assert len(resume.sections) > 0
            assert len(resume.bullets) > 0
            assert isinstance(report, dict)
            assert report["confidence"] > 0

    @patch("pdfplumber.open")
    def test_parse_pdf_file_empty(self, mock_pdf_open, parser):
        """Test PDF file with no extractable text."""
        mock_page = Mock()
        mock_page.extract_text.return_value = None
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            with pytest.raises(ResumeParsingError, match="No text could be extracted"):
                parser.parse_file(temp_file.name)

    @patch("docx.Document")
    def test_parse_docx_file_success(self, mock_document, parser, sample_resume_text):
        """Test successful DOCX file parsing."""
        # Mock DOCX structure
        mock_paragraph = Mock()
        mock_paragraph.text = sample_resume_text
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph]
        mock_document.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".docx") as temp_file:
            resume, report = parser.parse_file(temp_file.name)

            assert isinstance(resume, Resume)
            assert resume.raw_text == sample_resume_text
            assert isinstance(report, dict)

    @patch("docx.Document")
    def test_parse_docx_file_empty(self, mock_document, parser):
        """Test DOCX file with no extractable text."""
        mock_doc = Mock()
        mock_doc.paragraphs = []
        mock_document.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".docx") as temp_file:
            with pytest.raises(ResumeParsingError, match="No text could be extracted"):
                parser.parse_file(temp_file.name)

    # Bytes parsing tests
    @patch("pdfplumber.open")
    def test_parse_pdf_bytes_success(self, mock_pdf_open, parser, sample_resume_text):
        """Test successful PDF bytes parsing."""
        mock_page = Mock()
        mock_page.extract_text.return_value = sample_resume_text
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        fake_bytes = b"fake pdf content"
        resume, report = parser.parse_bytes(fake_bytes, ".pdf")

        assert isinstance(resume, Resume)
        assert resume.raw_text == sample_resume_text
        mock_pdf_open.assert_called_once()

    @patch("docx.Document")
    def test_parse_docx_bytes_success(self, mock_document, parser, sample_resume_text):
        """Test successful DOCX bytes parsing."""
        mock_paragraph = Mock()
        mock_paragraph.text = sample_resume_text
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph]
        mock_document.return_value = mock_doc

        fake_bytes = b"fake docx content"
        resume, report = parser.parse_bytes(fake_bytes, ".docx")

        assert isinstance(resume, Resume)
        assert resume.raw_text == sample_resume_text

    def test_parse_bytes_unsupported_format(self, parser):
        """Test bytes parsing with unsupported format."""
        with pytest.raises(ResumeParsingError, match="Unsupported file format"):
            parser.parse_bytes(b"content", ".txt")

    # Structure parsing tests
    def test_parse_structure_empty_text(self, parser):
        """Test structure parsing with empty text."""
        with pytest.raises(ResumeParsingError, match="Empty document"):
            parser._parse_structure("")

    def test_parse_structure_whitespace_only(self, parser):
        """Test structure parsing with whitespace only."""
        with pytest.raises(ResumeParsingError, match="Empty document"):
            parser._parse_structure("   \n\t  \n  ")

    def test_parse_structure_success(self, parser, sample_resume_text):
        """Test successful structure parsing."""
        resume, report = parser._parse_structure(sample_resume_text)

        assert isinstance(resume, Resume)
        assert resume.raw_text == sample_resume_text
        assert len(resume.sections) > 0
        assert len(resume.bullets) > 0
        assert len(resume.skills) > 0
        assert len(resume.dates) > 0

        # Verify report structure
        assert isinstance(report, dict)
        required_keys = [
            "text_length",
            "sections_found",
            "bullets_found",
            "skills_found",
            "dates_found",
            "confidence",
            "warnings",
        ]
        for key in required_keys:
            assert key in report

    # Section extraction tests
    def test_extract_sections_with_headers(self, parser):
        """Test section extraction with clear headers."""
        text = """
EXPERIENCE
Led development of microservices
EDUCATION
Bachelor of Science
SKILLS
Python, Java, JavaScript
"""
        sections = parser._extract_sections(text)

        assert len(sections) >= 3
        section_names = [s.name for s in sections]
        assert any("EXPERIENCE" in name for name in section_names)
        assert any("EDUCATION" in name for name in section_names)
        assert any("SKILLS" in name for name in section_names)

    def test_extract_sections_no_headers(self, parser):
        """Test section extraction with no clear headers."""
        text = "Just some random resume content without clear sections."
        sections = parser._extract_sections(text)

        assert len(sections) == 1
        assert sections[0].name == "Content"

    def test_extract_sections_case_variations(self, parser):
        """Test section extraction with different case variations."""
        text = """
Experience
Work experience at companies
education
University degree
Technical Skills
Programming languages
"""
        sections = parser._extract_sections(text)
        assert len(sections) >= 3

    # Bullet extraction tests
    def test_extract_bullets_unicode_bullets(self, parser):
        """Test bullet extraction with Unicode bullet characters."""
        text = """
EXPERIENCE
• First bullet point
• Second bullet point
▪ Third bullet point
◦ Fourth bullet point
"""
        sections = parser._extract_sections(text)
        bullets = parser._extract_bullets(text, sections)

        assert len(bullets) == 4
        assert bullets[0].text == "First bullet point"
        assert bullets[1].text == "Second bullet point"
        assert bullets[2].text == "Third bullet point"
        assert bullets[3].text == "Fourth bullet point"

    def test_extract_bullets_ascii_bullets(self, parser):
        """Test bullet extraction with ASCII bullet characters."""
        text = """
EXPERIENCE
- First bullet point
* Second bullet point
+ Third bullet point
"""
        sections = parser._extract_sections(text)
        bullets = parser._extract_bullets(text, sections)

        assert len(bullets) == 3
        assert bullets[0].text == "First bullet point"
        assert bullets[1].text == "Second bullet point"
        assert bullets[2].text == "Third bullet point"

    def test_extract_bullets_numbered_lists(self, parser):
        """Test bullet extraction with numbered lists."""
        text = """
EXPERIENCE
1. First numbered item
2. Second numbered item
3) Third numbered item
"""
        sections = parser._extract_sections(text)
        bullets = parser._extract_bullets(text, sections)

        assert len(bullets) == 3
        assert bullets[0].text == "First numbered item"
        assert bullets[1].text == "Second numbered item"
        assert bullets[2].text == "Third numbered item"

    def test_extract_bullets_empty_bullets_filtered(self, parser):
        """Test that empty bullet points are filtered out."""
        text = """
EXPERIENCE
• Valid bullet point
•   
• Another valid bullet
•
"""
        sections = parser._extract_sections(text)
        bullets = parser._extract_bullets(text, sections)

        assert len(bullets) == 2
        assert bullets[0].text == "Valid bullet point"
        assert bullets[1].text == "Another valid bullet"

    def test_extract_bullets_section_assignment(self, parser):
        """Test that bullets are correctly assigned to sections."""
        text = """
EXPERIENCE
• Experience bullet 1
• Experience bullet 2
EDUCATION
• Education bullet 1
SKILLS
• Skills bullet 1
"""
        sections = parser._extract_sections(text)
        bullets = parser._extract_bullets(text, sections)

        # Check bullets are assigned to correct sections
        experience_bullets = [b for b in bullets if "EXPERIENCE" in b.section]
        education_bullets = [b for b in bullets if "EDUCATION" in b.section]
        skills_bullets = [b for b in bullets if "SKILLS" in b.section]

        assert len(experience_bullets) == 2
        assert len(education_bullets) == 1
        assert len(skills_bullets) == 1

    # Skills extraction tests
    def test_extract_skills_from_skills_section(self, parser):
        """Test skills extraction from dedicated skills section."""
        text = """
EXPERIENCE
Led development teams
SKILLS:
Python, Java, JavaScript, React, Node.js, AWS, Docker
Machine Learning, Data Analysis
EDUCATION
Bachelor's degree
"""
        skills = parser._extract_skills(text)

        assert len(skills) > 0
        # Should find some technical skills
        skills_lower = [s.lower() for s in skills]
        assert any("python" in skill for skill in skills_lower)

    def test_extract_skills_technical_indicators(self, parser):
        """Test skills extraction using technical indicators."""
        text = """
Developed applications using Python and JavaScript.
Worked extensively with AWS cloud services.
Experience with machine learning algorithms.
"""
        skills = parser._extract_skills(text)

        expected_skills = {"Python", "Javascript", "Aws", "Machine Learning"}
        skills_set = set(skills)
        # Should find at least some of the expected skills
        assert len(skills_set.intersection(expected_skills)) > 0

    def test_extract_skills_no_skills_section(self, parser, minimal_resume_text):
        """Test skills extraction when no skills section exists."""
        skills = parser._extract_skills(minimal_resume_text)
        # Might still find some skills from indicators, but could be empty
        assert isinstance(skills, list)

    # Date extraction tests
    def test_extract_dates_various_formats(self, parser):
        """Test date extraction with various formats."""
        text = """
January 2020 - Present
06/2019 - 12/2020
2018-2019
Mar 15, 2021
May 2022 - current
"""
        dates = parser._extract_dates(text)

        assert len(dates) > 0
        # Should find several date patterns
        assert any("2020" in date for date in dates)
        assert any("2019" in date for date in dates)

    def test_extract_dates_no_dates(self, parser):
        """Test date extraction when no dates present."""
        text = "No dates in this text at all"
        dates = parser._extract_dates(text)
        assert dates == []

    # Confidence calculation tests
    def test_calculate_confidence_full_resume(self, parser, sample_resume_text):
        """Test confidence calculation for full resume."""
        resume, _ = parser._parse_structure(sample_resume_text)
        confidence = parser._calculate_confidence(resume)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high for good resume

    def test_calculate_confidence_minimal_resume(self, parser, minimal_resume_text):
        """Test confidence calculation for minimal resume."""
        resume, _ = parser._parse_structure(minimal_resume_text)
        confidence = parser._calculate_confidence(resume)

        assert 0.0 <= confidence <= 1.0
        # Should be lower for minimal resume
        assert confidence < 0.8

    # Warning generation tests
    def test_generate_warnings_full_resume(self, parser, sample_resume_text):
        """Test warning generation for full resume."""
        resume, _ = parser._parse_structure(sample_resume_text)
        warnings = parser._generate_warnings(resume)

        assert isinstance(warnings, list)
        # Good resume should have few warnings
        assert len(warnings) <= 2

    def test_generate_warnings_minimal_resume(self, parser, minimal_resume_text):
        """Test warning generation for minimal resume."""
        resume, _ = parser._parse_structure(minimal_resume_text)
        warnings = parser._generate_warnings(resume)

        assert isinstance(warnings, list)
        # Minimal resume should generate more warnings
        assert len(warnings) > 0

    def test_generate_warnings_specific_conditions(self, parser):
        """Test specific warning conditions."""
        # Test empty sections warning
        empty_sections_text = "Just text without structure"
        resume, _ = parser._parse_structure(empty_sections_text)
        warnings = parser._generate_warnings(resume)
        assert any("section" in warning.lower() for warning in warnings)

    # Error handling tests
    @patch("pdfplumber.open")
    def test_parse_pdf_file_exception(self, mock_pdf_open, parser):
        """Test PDF parsing with exception."""
        mock_pdf_open.side_effect = Exception("PDF parsing error")

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            with pytest.raises(ResumeParsingError, match="Error parsing"):
                parser.parse_file(temp_file.name)

    @patch("docx.Document")
    def test_parse_docx_file_exception(self, mock_document, parser):
        """Test DOCX parsing with exception."""
        mock_document.side_effect = Exception("DOCX parsing error")

        with tempfile.NamedTemporaryFile(suffix=".docx") as temp_file:
            with pytest.raises(ResumeParsingError, match="Error parsing"):
                parser.parse_file(temp_file.name)

    def test_parse_bytes_exception(self, parser):
        """Test bytes parsing with exception."""
        with patch("pdfplumber.open", side_effect=Exception("Bytes parsing error")):
            with pytest.raises(ResumeParsingError, match="Error parsing bytes"):
                parser.parse_bytes(b"content", ".pdf")

    # Edge cases and boundary tests
    def test_find_bullet_section_fallback(self, parser):
        """Test bullet section assignment fallback."""
        sections = [
            ResumeSection(
                name="Test Section", bullets=[], start_offset=0, end_offset=100
            )
        ]

        # Test with offset outside any section
        section_name = parser._find_bullet_section(200, sections)
        assert section_name == "Test Section"  # Should fallback to first section

    def test_find_bullet_section_no_sections(self, parser):
        """Test bullet section assignment with no sections."""
        section_name = parser._find_bullet_section(0, [])
        assert section_name == "Content"  # Should fallback to default

    def test_extract_docx_text_empty_paragraphs(self, parser):
        """Test DOCX text extraction with empty paragraphs."""
        mock_doc = Mock()
        mock_paragraph1 = Mock()
        mock_paragraph1.text = ""
        mock_paragraph2 = Mock()
        mock_paragraph2.text = "   "  # Whitespace only
        mock_paragraph3 = Mock()
        mock_paragraph3.text = "Valid content"
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2, mock_paragraph3]

        result = parser._extract_docx_text(mock_doc)
        assert result == "Valid content"

    def test_extract_docx_text_all_empty(self, parser):
        """Test DOCX text extraction with all empty paragraphs."""
        mock_doc = Mock()
        mock_paragraph = Mock()
        mock_paragraph.text = "   "
        mock_doc.paragraphs = [mock_paragraph]

        with pytest.raises(ResumeParsingError, match="No text could be extracted"):
            parser._extract_docx_text(mock_doc)

    # Integration tests
    @patch("pdfplumber.open")
    def test_integration_pdf_to_resume_complete(self, mock_pdf_open, parser):
        """Integration test: PDF parsing to complete Resume object."""
        resume_text = """
John Smith
Senior Software Engineer

EXPERIENCE
• Led development of microservices architecture
• Implemented CI/CD pipeline
• Managed team of 5 developers

EDUCATION
MS Computer Science, MIT, 2018
BS Computer Science, Stanford, 2016

SKILLS
Python, Java, React, AWS, Docker, Kubernetes

PROJECTS
• E-commerce platform serving 1M+ users
• Real-time data processing system
"""

        mock_page = Mock()
        mock_page.extract_text.return_value = resume_text
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            resume, report = parser.parse_file(temp_file.name)

            # Verify Resume object completeness
            assert resume.raw_text == resume_text
            assert len(resume.sections) >= 4
            assert len(resume.bullets) >= 5
            assert len(resume.skills) >= 5
            assert len(resume.dates) >= 2

            # Verify report completeness
            assert report["confidence"] > 0.5
            assert report["sections_found"] >= 4
            assert report["bullets_found"] >= 5
            assert isinstance(report["warnings"], list)
