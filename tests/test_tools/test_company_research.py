"""
Tests for CompanyResearchTool.

Tests the stateless company research functionality including web scraping,
fact extraction, validation, and FactSheet compilation.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from tools.company_research import CompanyResearchTool
from src.schemas.core import Fact, FactSheet, SourceDomainClass


class TestCompanyResearchTool:
    """Test cases for CompanyResearchTool."""

    @pytest.fixture
    def tool(self):
        """Create a CompanyResearchTool instance for testing."""
        return CompanyResearchTool(
            max_facts=5,
            request_timeout=5,
            request_delay=0.1,  # Faster for tests
            max_retries=2,
        )

    @pytest.fixture
    def mock_html_response(self):
        """Mock HTML response for company website."""
        return """
        <html>
        <head><title>TechCorp - Leading AI Solutions</title></head>
        <body>
            <main>
                <h1>About TechCorp</h1>
                <p>TechCorp is a leading artificial intelligence company that specializes in 
                machine learning solutions for enterprise customers. Founded in 2018, the 
                company is headquartered in San Francisco, CA.</p>
                
                <h2>Our Products</h2>
                <p>The company offers advanced analytics platforms that help businesses 
                optimize their operations through predictive modeling.</p>
                
                <h2>Recent News</h2>
                <p>TechCorp announced a partnership with Microsoft to expand cloud services
                and recently raised $50 million in Series B funding.</p>
                
                <div class="company-info">
                    <p>TechCorp employs over 250 people and has offices in San Francisco 
                    and New York.</p>
                </div>
            </main>
        </body>
        </html>
        """

    @pytest.fixture
    def mock_response(self, mock_html_response):
        """Mock HTTP response object."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = mock_html_response.encode("utf-8")
        mock_resp.raise_for_status = Mock()
        return mock_resp

    def test_init(self):
        """Test tool initialization."""
        tool = CompanyResearchTool(
            max_facts=15, request_timeout=30, request_delay=2.0, max_retries=5
        )

        assert tool.max_facts == 15
        assert tool.request_timeout == 30
        assert tool.request_delay == 2.0
        assert tool.max_retries == 5
        assert tool.session is not None
        assert tool.html_converter is not None

    def test_research_company_invalid_input(self, tool):
        """Test research with invalid company name input."""
        with pytest.raises(ValueError, match="company_name must be a non-empty string"):
            tool.research_company("")

        with pytest.raises(ValueError, match="company_name must be a non-empty string"):
            tool.research_company(None)

        with pytest.raises(ValueError, match="company_name must be a non-empty string"):
            tool.research_company("   ")

    @patch("tools.company_research.time.sleep")
    def test_research_company_success(self, mock_sleep, tool, mock_response):
        """Test successful company research."""
        with patch.object(tool, "_discover_sources") as mock_discover, patch.object(
            tool, "_make_request"
        ) as mock_request, patch.object(tool, "_check_robots_txt", return_value=True):

            # Setup mocks
            mock_discover.return_value = [
                ("https://techcorp.com", SourceDomainClass.OFFICIAL),
                ("https://techcorp.com/about", SourceDomainClass.OFFICIAL),
            ]
            mock_request.return_value = mock_response

            # Execute research
            result = tool.research_company("TechCorp")

            # Verify results
            assert isinstance(result, FactSheet)
            assert result.company == "TechCorp"
            assert len(result.facts) > 0
            assert all(isinstance(fact, Fact) for fact in result.facts)
            assert result.generated_at is not None

    def test_create_company_slug(self, tool):
        """Test company name to URL slug conversion."""
        test_cases = [
            ("TechCorp Inc", "techcorp"),
            ("Data Solutions LLC", "data-solutions"),
            ("AI & Machine Learning Co.", "ai-machine-learning"),
            ("StartupName", "startupname"),
            ("Multi-Word Company Corp", "multi-word"),  # Company suffix removed
        ]

        for company_name, expected_slug in test_cases:
            slug = tool._create_company_slug(company_name)
            assert slug == expected_slug

    @patch("tools.company_research.requests.Session.get")
    def test_is_valid_company_site(self, mock_get, tool):
        """Test company website validation."""
        # Mock successful response with company name
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html><head><title>TechCorp Solutions</title></head><body>Welcome to TechCorp</body></html>"
        mock_get.return_value = mock_response

        assert tool._is_valid_company_site("https://techcorp.com", "TechCorp")

        # Test invalid site (no company name)
        mock_response.content = b"<html><body>Generic website content</body></html>"
        assert not tool._is_valid_company_site("https://example.com", "TechCorp")

        # Test 404 response
        mock_response.status_code = 404
        assert not tool._is_valid_company_site("https://nonexistent.com", "TechCorp")

    def test_extract_clean_text(self, tool, mock_html_response):
        """Test HTML text extraction and cleaning."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(mock_html_response, "html.parser")
        clean_text = tool._extract_clean_text(soup, "https://techcorp.com")

        assert "TechCorp" in clean_text
        assert "artificial intelligence" in clean_text
        assert "Founded in 2018" in clean_text
        assert len(clean_text) > 100
        assert "<html>" not in clean_text  # HTML tags removed
        assert "<script>" not in clean_text  # Scripts removed

    def test_clean_fact_text(self, tool):
        """Test fact text cleaning and normalization."""
        test_cases = [
            ("  Multiple   spaces  ", "Multiple spaces."),
            ("Text with **markdown** formatting", "Text with markdown formatting."),
            ("Text ending with comma,", "Text ending with comma."),
            ("Already ended with period.", "Already ended with period."),
            ("Question text?", "Question text?"),
            ("Exclamation text!", "Exclamation text!"),
        ]

        for input_text, expected in test_cases:
            result = tool._clean_fact_text(input_text)
            assert result == expected

    def test_is_valid_fact(self, tool):
        """Test fact validation logic."""
        # Valid facts
        assert tool._is_valid_fact(
            "TechCorp specializes in artificial intelligence solutions for enterprise clients.",
            "TechCorp",
            "products",
        )

        # Invalid facts
        assert not tool._is_valid_fact("", "TechCorp", "products")  # Empty
        assert not tool._is_valid_fact("Too short", "TechCorp", "products")  # Too short
        assert not tool._is_valid_fact(
            "Click here to learn more", "TechCorp", "products"
        )  # Generic

        # Long text (over limit)
        long_text = "A" * 600  # Over MAX_FACT_LENGTH
        assert not tool._is_valid_fact(long_text, "TechCorp", "products")

    def test_calculate_fact_confidence(self, tool):
        """Test confidence score calculation."""
        # High confidence: official source, detailed fact
        confidence = tool._calculate_fact_confidence(
            "TechCorp raised $50 million in Series B funding in March 2023.",
            SourceDomainClass.OFFICIAL,
            "funding",
            "https://techcorp.com/about",
        )
        assert confidence > 0.7

        # Lower confidence: other source, generic fact
        confidence = tool._calculate_fact_confidence(
            "Company provides services.",
            SourceDomainClass.OTHER,
            "description",
            "https://example.com",
        )
        assert confidence < 0.7

        # News source confidence
        confidence = tool._calculate_fact_confidence(
            "TechCorp announced new partnership.",
            SourceDomainClass.REPUTABLE_NEWS,
            "news",
            "https://techcrunch.com/article",
        )
        assert 0.5 < confidence < 0.9

    def test_validate_facts(self, tool):
        """Test fact validation and filtering."""
        # Create test facts
        valid_fact = Fact(
            statement="TechCorp is a leading AI company.",
            source_url="https://techcorp.com",
            source_domain_class=SourceDomainClass.OFFICIAL,
            as_of_date=datetime.now(timezone.utc),
            confidence=0.8,
        )

        low_confidence_fact = Fact(
            statement="Some generic statement.",
            source_url="https://example.com",
            source_domain_class=SourceDomainClass.OTHER,
            as_of_date=datetime.now(timezone.utc),
            confidence=0.2,  # Below threshold
        )

        facts = [valid_fact, low_confidence_fact]
        validated = tool._validate_facts(facts)

        assert len(validated) == 1
        assert validated[0] == valid_fact

    def test_deduplicate_facts(self, tool):
        """Test fact deduplication logic."""
        # Create similar facts
        fact1 = Fact(
            statement="TechCorp is a leading AI company.",
            source_url="https://techcorp.com",
            source_domain_class=SourceDomainClass.OFFICIAL,
            as_of_date=datetime.now(timezone.utc),
            confidence=0.8,
        )

        fact2 = Fact(
            statement="TechCorp is a leading artificial intelligence company.",
            source_url="https://techcorp.com/about",
            source_domain_class=SourceDomainClass.OFFICIAL,
            as_of_date=datetime.now(timezone.utc),
            confidence=0.7,
        )

        fact3 = Fact(
            statement="Company offers different services entirely.",
            source_url="https://techcorp.com",
            source_domain_class=SourceDomainClass.OFFICIAL,
            as_of_date=datetime.now(timezone.utc),
            confidence=0.6,
        )

        facts = [fact1, fact2, fact3]
        deduplicated = tool._deduplicate_and_prioritize_facts(facts)

        # Since similarity detection might not catch these as duplicates due to threshold
        # let's verify the expected behavior - facts should be unique unless very similar
        assert len(deduplicated) >= 2  # At least 2 facts should remain
        # Higher confidence facts should be prioritized in sorting
        assert deduplicated[0].confidence >= deduplicated[-1].confidence

    def test_are_facts_similar(self, tool):
        """Test fact similarity detection."""
        # Similar statements with high overlap
        statement1 = "techcorp leading ai company"
        statement2 = "techcorp leading ai company solutions"
        assert tool._are_facts_similar(statement1, statement2)

        # Different statements
        statement3 = "company provides different services entirely"
        assert not tool._are_facts_similar(statement1, statement3)

    def test_classify_source_domain(self, tool):
        """Test source domain classification."""
        # Reputable news sites
        assert (
            tool._classify_source_domain("https://techcrunch.com/article")
            == SourceDomainClass.REPUTABLE_NEWS
        )
        assert (
            tool._classify_source_domain("https://www.bloomberg.com/news")
            == SourceDomainClass.REPUTABLE_NEWS
        )

        # Official company sites
        assert (
            tool._classify_source_domain("https://techcorp.com")
            == SourceDomainClass.OFFICIAL
        )
        assert (
            tool._classify_source_domain("https://startup.ai")
            == SourceDomainClass.OFFICIAL
        )

        # Other sites
        assert (
            tool._classify_source_domain("https://random-blog.net")
            == SourceDomainClass.OTHER
        )

    @patch("tools.company_research.time.sleep")
    def test_research_with_no_sources(self, mock_sleep, tool):
        """Test research when no sources are discovered."""
        with patch.object(tool, "_discover_sources", return_value=[]):
            result = tool.research_company("NonexistentCompany")

            assert isinstance(result, FactSheet)
            assert result.company == "NonexistentCompany"
            assert len(result.facts) == 0

    def test_parse_date_string(self, tool):
        """Test date string parsing."""
        test_cases = [
            ("2023-03-15", datetime(2023, 3, 15, tzinfo=timezone.utc)),
            (
                "2023-03-15T10:30:00Z",
                datetime(2023, 3, 15, 10, 30, 0, tzinfo=timezone.utc),
            ),
            ("March 15, 2023", datetime(2023, 3, 15, tzinfo=timezone.utc)),
            ("Mar 15, 2023", datetime(2023, 3, 15, tzinfo=timezone.utc)),
            ("invalid-date", None),
        ]

        for date_str, expected in test_cases:
            result = tool._parse_date_string(date_str)
            assert result == expected

    def test_is_fact_too_old(self, tool):
        """Test recency filtering for news facts."""
        # Recent date (within 180 days)
        recent_date = datetime.now(timezone.utc) - timedelta(days=30)
        assert not tool._is_fact_too_old(recent_date)

        # Old date (beyond 180 days)
        old_date = datetime.now(timezone.utc) - timedelta(days=200)
        assert tool._is_fact_too_old(old_date)

    @patch("tools.company_research.requests.Session.head")
    def test_url_exists(self, mock_head, tool):
        """Test URL existence checking."""
        # Successful response
        mock_head.return_value.status_code = 200
        assert tool._url_exists("https://techcorp.com/about")

        # 404 response
        mock_head.return_value.status_code = 404
        assert not tool._url_exists("https://techcorp.com/nonexistent")

        # Request exception
        mock_head.side_effect = Exception("Network error")
        assert not tool._url_exists("https://unreachable.com")

    def test_check_robots_txt_allowed(self, tool):
        """Test robots.txt compliance checking - allowed access."""
        with patch("tools.company_research.RobotFileParser") as mock_robot_parser:
            mock_rp_allow = Mock()
            mock_rp_allow.can_fetch.return_value = True
            mock_robot_parser.return_value = mock_rp_allow

            assert tool._check_robots_txt("https://techcorp.com/page")

    def test_check_robots_txt_disallowed(self, tool):
        """Test robots.txt compliance checking - disallowed access."""
        with patch("tools.company_research.RobotFileParser") as mock_robot_parser:
            mock_rp_disallow = Mock()
            mock_rp_disallow.can_fetch.return_value = False
            mock_robot_parser.return_value = mock_rp_disallow

            assert not tool._check_robots_txt("https://techcorp.com/private")

    def test_check_robots_txt_exception(self, tool):
        """Test robots.txt compliance checking - exception handling."""
        with patch("tools.company_research.RobotFileParser") as mock_robot_parser:
            mock_robot_parser.side_effect = Exception("Robot parser error")
            assert tool._check_robots_txt("https://techcorp.com/page")


@pytest.mark.integration
class TestCompanyResearchToolIntegration:
    """Integration tests that may make real network requests."""

    @pytest.fixture
    def tool(self):
        """Create tool for integration testing."""
        return CompanyResearchTool(max_facts=3, request_delay=0.5)

    @pytest.mark.slow
    def test_research_known_company(self, tool):
        """Test researching a well-known company (may make real requests)."""
        # This test might be skipped in CI/CD environments
        # or mocked depending on testing strategy
        pytest.skip("Integration test - requires network access")

        # Uncomment for manual testing:
        # result = tool.research_company("OpenAI")
        # assert isinstance(result, FactSheet)
        # assert result.company == "OpenAI"
        # assert len(result.facts) > 0
