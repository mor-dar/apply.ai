"""
Comprehensive tests for core Pydantic schemas.

Tests cover validation, serialization, and business logic for all schema classes
used in the apply.ai job application copilot.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.schemas.core import (
    JobPosting,
    Resume,
    ResumeBullet,
    ResumeSection,
    Fact,
    FactSheet,
    TailoredBullet,
    CoverLetter,
    CompBand,
    Metrics,
    SourceDomainClass,
)


class TestJobPosting:
    """Test JobPosting schema validation and functionality."""

    def test_valid_job_posting(self):
        """Test creation of valid JobPosting instance."""
        job = JobPosting(
            title="Senior Software Engineer",
            company="TechCorp",
            location="San Francisco, CA",
            text="We are looking for a senior software engineer...",
            keywords=["Python", "Django", "PostgreSQL"],
            requirements=["5+ years experience", "Bachelor's degree"],
        )

        assert job.title == "Senior Software Engineer"
        assert job.company == "TechCorp"
        assert job.location == "San Francisco, CA"
        assert len(job.keywords) == 3
        assert len(job.requirements) == 2

    def test_minimal_job_posting(self):
        """Test JobPosting with only required fields."""
        job = JobPosting(title="Engineer", company="Company", text="Job description")

        assert job.title == "Engineer"
        assert job.company == "Company"
        assert job.text == "Job description"
        assert job.location is None
        assert job.keywords == []
        assert job.requirements == []

    def test_job_text_too_long(self):
        """Test validation failure for overly long job text."""
        with pytest.raises(ValidationError) as exc_info:
            JobPosting(
                title="Engineer",
                company="Company",
                text="x" * 10001,  # Exceeds 10000 char limit
            )

        assert "Job description text too long" in str(exc_info.value)

    def test_keywords_requirements_whitespace_cleaning(self):
        """Test that empty and whitespace-only entries are cleaned."""
        job = JobPosting(
            title="Engineer",
            company="Company",
            text="Description",
            keywords=["Python", "  ", "", "Django", "   React   "],
            requirements=["", "5 years", "  ", "Degree", "   "],
        )

        assert job.keywords == ["Python", "Django", "React"]
        assert job.requirements == ["5 years", "Degree"]


class TestResumeBullet:
    """Test ResumeBullet schema validation."""

    def test_valid_resume_bullet(self):
        """Test creation of valid ResumeBullet."""
        bullet = ResumeBullet(
            text="Developed Python applications",
            section="Experience",
            start_offset=100,
            end_offset=130,
        )

        assert bullet.text == "Developed Python applications"
        assert bullet.section == "Experience"
        assert bullet.start_offset == 100
        assert bullet.end_offset == 130
        assert bullet.embedding is None

    def test_bullet_with_embedding(self):
        """Test ResumeBullet with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        bullet = ResumeBullet(
            text="Managed team projects",
            section="Experience",
            start_offset=50,
            end_offset=75,
            embedding=embedding,
        )

        assert bullet.embedding == embedding

    def test_empty_bullet_text_validation(self):
        """Test validation failure for empty bullet text."""
        with pytest.raises(ValidationError) as exc_info:
            ResumeBullet(
                text="   ",  # Whitespace only
                section="Experience",
                start_offset=0,
                end_offset=10,
            )

        assert "Bullet text cannot be empty" in str(exc_info.value)


class TestResumeSection:
    """Test ResumeSection schema validation."""

    def test_valid_resume_section(self):
        """Test creation of valid ResumeSection."""
        bullets = [
            ResumeBullet(
                text="First bullet",
                section="Experience",
                start_offset=10,
                end_offset=20,
            ),
            ResumeBullet(
                text="Second bullet",
                section="Experience",
                start_offset=25,
                end_offset=35,
            ),
        ]

        section = ResumeSection(
            name="Experience", bullets=bullets, start_offset=0, end_offset=50
        )

        assert section.name == "Experience"
        assert len(section.bullets) == 2
        assert section.start_offset == 0
        assert section.end_offset == 50

    def test_empty_section(self):
        """Test ResumeSection with no bullets."""
        section = ResumeSection(name="Education", start_offset=100, end_offset=120)

        assert section.name == "Education"
        assert section.bullets == []


class TestResume:
    """Test Resume schema validation."""

    def test_valid_resume(self):
        """Test creation of valid Resume."""
        bullets = [
            ResumeBullet(
                text="Developed apps",
                section="Experience",
                start_offset=10,
                end_offset=25,
            )
        ]
        sections = [
            ResumeSection(
                name="Experience", bullets=bullets, start_offset=0, end_offset=50
            )
        ]

        resume = Resume(
            raw_text="Resume content here...",
            bullets=bullets,
            skills=["Python", "JavaScript"],
            dates=["2020-2023", "2018-2020"],
            sections=sections,
        )

        assert resume.raw_text == "Resume content here..."
        assert len(resume.bullets) == 1
        assert len(resume.skills) == 2
        assert len(resume.dates) == 2
        assert len(resume.sections) == 1

    def test_empty_resume_text_validation(self):
        """Test validation failure for empty resume text."""
        with pytest.raises(ValidationError) as exc_info:
            Resume(raw_text="   ")  # Whitespace only

        assert "Resume text cannot be empty" in str(exc_info.value)

    def test_skills_whitespace_cleaning(self):
        """Test that skills list is cleaned of empty entries."""
        resume = Resume(
            raw_text="Resume text",
            skills=["Python", "  ", "", "JavaScript", "   Django   "],
        )

        assert resume.skills == ["Python", "JavaScript", "Django"]


class TestFact:
    """Test Fact schema validation."""

    def test_valid_fact(self):
        """Test creation of valid Fact."""
        fact = Fact(
            statement="Company was founded in 2010",
            source_url="https://company.com/about",
            source_domain_class=SourceDomainClass.OFFICIAL,
            as_of_date=datetime(2024, 1, 1),
            confidence=0.9,
        )

        assert fact.statement == "Company was founded in 2010"
        assert str(fact.source_url) == "https://company.com/about"
        assert fact.source_domain_class == SourceDomainClass.OFFICIAL
        assert fact.confidence == 0.9

    def test_confidence_range_validation(self):
        """Test confidence score must be between 0 and 1."""
        # Test valid confidence values
        Fact(
            statement="Valid fact",
            source_url="https://example.com",
            source_domain_class=SourceDomainClass.OTHER,
            as_of_date=datetime.now(),
            confidence=0.0,
        )

        Fact(
            statement="Valid fact",
            source_url="https://example.com",
            source_domain_class=SourceDomainClass.OTHER,
            as_of_date=datetime.now(),
            confidence=1.0,
        )

        # Test invalid confidence values
        with pytest.raises(ValidationError):
            Fact(
                statement="Invalid fact",
                source_url="https://example.com",
                source_domain_class=SourceDomainClass.OTHER,
                as_of_date=datetime.now(),
                confidence=-0.1,
            )

        with pytest.raises(ValidationError):
            Fact(
                statement="Invalid fact",
                source_url="https://example.com",
                source_domain_class=SourceDomainClass.OTHER,
                as_of_date=datetime.now(),
                confidence=1.1,
            )

    def test_empty_statement_validation(self):
        """Test validation failure for empty fact statement."""
        with pytest.raises(ValidationError) as exc_info:
            Fact(
                statement="   ",  # Whitespace only
                source_url="https://example.com",
                source_domain_class=SourceDomainClass.OTHER,
                as_of_date=datetime.now(),
                confidence=0.8,
            )

        assert "Fact statement cannot be empty" in str(exc_info.value)

    def test_statement_too_long(self):
        """Test validation failure for overly long statement."""
        with pytest.raises(ValidationError) as exc_info:
            Fact(
                statement="x" * 501,  # Exceeds 500 char limit
                source_url="https://example.com",
                source_domain_class=SourceDomainClass.OTHER,
                as_of_date=datetime.now(),
                confidence=0.8,
            )

        assert "Fact statement too long" in str(exc_info.value)

    def test_source_domain_class_enum(self):
        """Test SourceDomainClass enum values."""
        assert SourceDomainClass.OFFICIAL == "official"
        assert SourceDomainClass.REPUTABLE_NEWS == "reputable_news"
        assert SourceDomainClass.OTHER == "other"


class TestFactSheet:
    """Test FactSheet schema validation."""

    def test_valid_factsheet(self):
        """Test creation of valid FactSheet."""
        facts = [
            Fact(
                statement="Company has 1000+ employees",
                source_url="https://company.com/about",
                source_domain_class=SourceDomainClass.OFFICIAL,
                as_of_date=datetime(2024, 1, 1),
                confidence=0.95,
            )
        ]

        factsheet = FactSheet(company="TechCorp", facts=facts)

        assert factsheet.company == "TechCorp"
        assert len(factsheet.facts) == 1
        assert isinstance(factsheet.generated_at, datetime)

    def test_empty_company_validation(self):
        """Test validation failure for empty company name."""
        with pytest.raises(ValidationError) as exc_info:
            FactSheet(company="   ")  # Whitespace only

        assert "Company name cannot be empty" in str(exc_info.value)

    def test_too_many_facts_validation(self):
        """Test validation failure for too many facts."""
        facts = []
        for i in range(21):  # Exceeds 20 fact limit
            facts.append(
                Fact(
                    statement=f"Fact {i}",
                    source_url="https://example.com",
                    source_domain_class=SourceDomainClass.OTHER,
                    as_of_date=datetime.now(),
                    confidence=0.8,
                )
            )

        with pytest.raises(ValidationError) as exc_info:
            FactSheet(company="Company", facts=facts)

        assert "Too many facts in factsheet" in str(exc_info.value)


class TestTailoredBullet:
    """Test TailoredBullet schema validation."""

    def test_valid_tailored_bullet(self):
        """Test creation of valid TailoredBullet."""
        bullet = TailoredBullet(
            text="Implemented scalable Python microservices using Django",
            original_bullet_id=1,
            evidence_spans=["Developed Python applications", "Used Django framework"],
            similarity_score=0.85,
            jd_keywords_covered=["Python", "microservices", "Django"],
        )

        assert bullet.text == "Implemented scalable Python microservices using Django"
        assert bullet.original_bullet_id == 1
        assert len(bullet.evidence_spans) == 2
        assert bullet.similarity_score == 0.85
        assert len(bullet.jd_keywords_covered) == 3

    def test_similarity_score_threshold_validation(self):
        """Test validation failure for low similarity score."""
        with pytest.raises(ValidationError) as exc_info:
            TailoredBullet(
                text="New bullet text",
                similarity_score=0.7,  # Below 0.8 threshold
                evidence_spans=["Original text"],
                jd_keywords_covered=["keyword"],
            )

        assert "Similarity score must be ≥ 0.8 for evidence validation" in str(
            exc_info.value
        )

    def test_empty_text_validation(self):
        """Test validation failure for empty bullet text."""
        with pytest.raises(ValidationError) as exc_info:
            TailoredBullet(
                text="   ",  # Whitespace only
                similarity_score=0.9,
                evidence_spans=["Evidence"],
                jd_keywords_covered=["keyword"],
            )

        assert "Tailored bullet text cannot be empty" in str(exc_info.value)


class TestCoverLetter:
    """Test CoverLetter schema validation."""

    def test_valid_cover_letter(self):
        """Test creation of valid CoverLetter."""
        letter = CoverLetter(
            intro="Dear Hiring Manager, I am writing to apply...",
            body_points=[
                "My experience with Python aligns with your requirements.",
                "I am excited about the company's mission.",
            ],
            closing="Thank you for your consideration.",
            sources=["https://company.com/about", "Annual report 2023"],
            company="TechCorp",
            position="Software Engineer",
        )

        assert letter.intro.startswith("Dear Hiring Manager")
        assert len(letter.body_points) == 2
        assert letter.closing.startswith("Thank you")
        assert len(letter.sources) == 2
        assert letter.company == "TechCorp"
        assert letter.position == "Software Engineer"

    def test_body_points_validation(self):
        """Test body points count validation."""
        # Test minimum body points (1)
        CoverLetter(
            intro="Intro",
            body_points=["One point"],
            closing="Closing",
            company="Company",
            position="Position",
        )

        # Test maximum body points (5)
        CoverLetter(
            intro="Intro",
            body_points=["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
            closing="Closing",
            company="Company",
            position="Position",
        )

        # Test too few body points
        with pytest.raises(ValidationError) as exc_info:
            CoverLetter(
                intro="Intro",
                body_points=[],  # Empty
                closing="Closing",
                company="Company",
                position="Position",
            )

        assert "Cover letter must have 1-5 body points" in str(exc_info.value)

        # Test too many body points
        with pytest.raises(ValidationError) as exc_info:
            CoverLetter(
                intro="Intro",
                body_points=["P1", "P2", "P3", "P4", "P5", "P6"],  # 6 points
                closing="Closing",
                company="Company",
                position="Position",
            )

        assert "Cover letter must have 1-5 body points" in str(exc_info.value)

    def test_empty_paragraphs_validation(self):
        """Test validation failure for empty paragraphs."""
        with pytest.raises(ValidationError):
            CoverLetter(
                intro="   ",  # Whitespace only
                body_points=["Valid point"],
                closing="Valid closing",
                company="Company",
                position="Position",
            )

        with pytest.raises(ValidationError):
            CoverLetter(
                intro="Valid intro",
                body_points=["Valid point"],
                closing="   ",  # Whitespace only
                company="Company",
                position="Position",
            )


class TestCompBand:
    """Test CompBand schema validation."""

    def test_valid_comp_band(self):
        """Test creation of valid CompBand."""
        comp = CompBand(
            occupation_code="15-1252",
            geography="California",
            p25=85000.0,
            p50=105000.0,
            p75=130000.0,
            sources=["https://bls.gov/oes/current/oes151252.htm"],
            as_of=datetime(2024, 5, 1),
            currency="USD",
        )

        assert comp.occupation_code == "15-1252"
        assert comp.geography == "California"
        assert comp.p25 == 85000.0
        assert comp.p50 == 105000.0
        assert comp.p75 == 130000.0
        assert comp.currency == "USD"

    def test_optional_salary_fields(self):
        """Test CompBand with optional salary fields."""
        comp = CompBand(
            occupation_code="15-1252",
            geography="National",
            p50=100000.0,  # Only median available
            as_of=datetime(2024, 1, 1),
        )

        assert comp.p25 is None
        assert comp.p50 == 100000.0
        assert comp.p75 is None

    def test_negative_salary_validation(self):
        """Test validation failure for negative salary values."""
        with pytest.raises(ValidationError):
            CompBand(
                occupation_code="15-1252",
                geography="State",
                p50=-1000.0,  # Negative salary
                as_of=datetime.now(),
            )

    def test_zero_salary_validation(self):
        """Test validation failure for zero salary values."""
        with pytest.raises(ValidationError):
            CompBand(
                occupation_code="15-1252",
                geography="State",
                p50=0.0,  # Zero salary
                as_of=datetime.now(),
            )

    def test_empty_occupation_code_validation(self):
        """Test validation failure for empty occupation code."""
        with pytest.raises(ValidationError) as exc_info:
            CompBand(
                occupation_code="   ",  # Whitespace only
                geography="State",
                as_of=datetime.now(),
            )

        assert "Occupation code cannot be empty" in str(exc_info.value)


class TestMetrics:
    """Test Metrics schema validation."""

    def test_valid_metrics(self):
        """Test creation of valid Metrics."""
        metrics = Metrics(
            jd_coverage_pct=75.5,
            readability_grade=10.2,
            evidence_mapped_ratio=0.96,
            total_tailored_bullets=10,
            validated_bullets=9,
        )

        assert metrics.jd_coverage_pct == 75.5
        assert metrics.readability_grade == 10.2
        assert metrics.evidence_mapped_ratio == 0.96
        assert metrics.total_tailored_bullets == 10
        assert metrics.validated_bullets == 9

    def test_coverage_percentage_range(self):
        """Test JD coverage percentage must be 0-100."""
        # Valid range
        Metrics(
            jd_coverage_pct=0.0,
            evidence_mapped_ratio=1.0,
            total_tailored_bullets=5,
            validated_bullets=5,
        )

        Metrics(
            jd_coverage_pct=100.0,
            evidence_mapped_ratio=1.0,
            total_tailored_bullets=5,
            validated_bullets=5,
        )

        # Invalid range
        with pytest.raises(ValidationError):
            Metrics(
                jd_coverage_pct=-1.0,  # Below 0
                evidence_mapped_ratio=1.0,
                total_tailored_bullets=5,
                validated_bullets=5,
            )

        with pytest.raises(ValidationError):
            Metrics(
                jd_coverage_pct=101.0,  # Above 100
                evidence_mapped_ratio=1.0,
                total_tailored_bullets=5,
                validated_bullets=5,
            )

    def test_evidence_ratio_range(self):
        """Test evidence mapped ratio must be 0-1."""
        # Valid range
        Metrics(
            jd_coverage_pct=50.0,
            evidence_mapped_ratio=0.0,
            total_tailored_bullets=5,
            validated_bullets=0,
        )

        Metrics(
            jd_coverage_pct=50.0,
            evidence_mapped_ratio=1.0,
            total_tailored_bullets=5,
            validated_bullets=5,
        )

        # Invalid range
        with pytest.raises(ValidationError):
            Metrics(
                jd_coverage_pct=50.0,
                evidence_mapped_ratio=-0.1,  # Below 0
                total_tailored_bullets=5,
                validated_bullets=5,
            )

        with pytest.raises(ValidationError):
            Metrics(
                jd_coverage_pct=50.0,
                evidence_mapped_ratio=1.1,  # Above 1
                total_tailored_bullets=5,
                validated_bullets=5,
            )

    def test_bullet_count_validation(self):
        """Test validated bullets cannot exceed total bullets."""
        with pytest.raises(ValidationError) as exc_info:
            Metrics(
                jd_coverage_pct=50.0,
                evidence_mapped_ratio=1.0,
                total_tailored_bullets=5,
                validated_bullets=6,  # Exceeds total
            )

        assert "Validated bullets cannot exceed total bullets" in str(exc_info.value)

    def test_optional_readability_grade(self):
        """Test metrics with optional readability grade."""
        metrics = Metrics(
            jd_coverage_pct=80.0,
            evidence_mapped_ratio=0.95,
            total_tailored_bullets=8,
            validated_bullets=8,
        )

        assert metrics.readability_grade is None


class TestSchemaIntegration:
    """Integration tests across multiple schema types."""

    def test_complete_application_workflow(self):
        """Test creating instances of all schemas in a realistic workflow."""
        # Job posting
        job = JobPosting(
            title="Senior Python Developer",
            company="TechCorp",
            location="San Francisco, CA",
            text="We seek a senior Python developer with Django experience...",
            keywords=["Python", "Django", "PostgreSQL", "REST API"],
            requirements=["5+ years Python", "Django framework", "Database design"],
        )

        # Resume
        bullets = [
            ResumeBullet(
                text="Developed Django web applications",
                section="Experience",
                start_offset=100,
                end_offset=135,
            ),
            ResumeBullet(
                text="Designed PostgreSQL database schemas",
                section="Experience",
                start_offset=140,
                end_offset=175,
            ),
        ]

        resume = Resume(
            raw_text="Senior Software Engineer at StartupCorp...",
            bullets=bullets,
            skills=["Python", "Django", "PostgreSQL", "Git"],
            dates=["2020-2024", "2018-2020"],
            sections=[
                ResumeSection(
                    name="Experience", bullets=bullets, start_offset=50, end_offset=200
                )
            ],
        )

        # Company research
        facts = [
            Fact(
                statement="TechCorp specializes in fintech solutions",
                source_url="https://techcorp.com/about",
                source_domain_class=SourceDomainClass.OFFICIAL,
                as_of_date=datetime(2024, 1, 15),
                confidence=0.95,
            ),
            Fact(
                statement="TechCorp raised $50M Series B in 2023",
                source_url="https://techcrunch.com/techcorp-funding",
                source_domain_class=SourceDomainClass.REPUTABLE_NEWS,
                as_of_date=datetime(2023, 12, 1),
                confidence=0.88,
            ),
        ]

        factsheet = FactSheet(company="TechCorp", facts=facts)

        # Tailored resume bullets
        tailored_bullets = [
            TailoredBullet(
                text="Architected Django REST APIs for fintech applications handling $1M+ daily transactions",
                original_bullet_id=0,
                evidence_spans=["Developed Django web applications"],
                similarity_score=0.87,
                jd_keywords_covered=["Python", "Django", "REST API"],
            ),
            TailoredBullet(
                text="Optimized PostgreSQL queries for high-volume financial data processing",
                original_bullet_id=1,
                evidence_spans=["Designed PostgreSQL database schemas"],
                similarity_score=0.83,
                jd_keywords_covered=["PostgreSQL", "Database design"],
            ),
        ]

        # Cover letter
        cover_letter = CoverLetter(
            intro="Dear TechCorp Hiring Team, I am excited to apply for the Senior Python Developer position.",
            body_points=[
                "My 4+ years of Django development experience aligns perfectly with your tech stack requirements.",
                "Having worked extensively with PostgreSQL in high-transaction environments, I understand the database optimization needs for fintech applications.",
                "TechCorp's recent $50M Series B funding and focus on innovative fintech solutions strongly appeals to my career interests.",
            ],
            closing="I look forward to contributing to TechCorp's continued growth and innovation.",
            sources=[
                "https://techcorp.com/about",
                "https://techcrunch.com/techcorp-funding",
            ],
            company="TechCorp",
            position="Senior Python Developer",
        )

        # Compensation data
        comp_band = CompBand(
            occupation_code="15-1252",
            geography="California",
            p25=120000.0,
            p50=145000.0,
            p75=170000.0,
            sources=["https://bls.gov/oes/current/oes151252.htm"],
            as_of=datetime(2024, 5, 1),
        )

        # Quality metrics
        metrics = Metrics(
            jd_coverage_pct=85.0,
            readability_grade=9.5,
            evidence_mapped_ratio=1.0,
            total_tailored_bullets=2,
            validated_bullets=2,
        )

        # Verify all instances are valid
        assert job.title == "Senior Python Developer"
        assert len(resume.bullets) == 2
        assert len(factsheet.facts) == 2
        assert len(tailored_bullets) == 2
        assert len(cover_letter.body_points) == 3
        assert comp_band.p50 == 145000.0
        assert metrics.evidence_mapped_ratio == 1.0

    def test_schema_serialization(self):
        """Test that all schemas can be serialized to/from dict."""
        job = JobPosting(
            title="Engineer",
            company="Company",
            text="Job description",
            keywords=["Python"],
        )

        # Test dict serialization
        job_dict = job.model_dump()
        assert job_dict["title"] == "Engineer"
        assert job_dict["keywords"] == ["Python"]

        # Test recreation from dict
        job_recreated = JobPosting(**job_dict)
        assert job_recreated.title == job.title
        assert job_recreated.keywords == job.keywords

    def test_schema_json_serialization(self):
        """Test JSON serialization of schemas with datetime fields."""
        fact = Fact(
            statement="Test fact",
            source_url="https://example.com",
            source_domain_class=SourceDomainClass.OTHER,
            as_of_date=datetime(2024, 1, 1, 12, 0, 0),
            confidence=0.9,
        )

        # Test JSON serialization
        fact_json = fact.model_dump_json()
        assert '"statement":"Test fact"' in fact_json
        assert '"confidence":0.9' in fact_json
        assert "2024-01-01T12:00:00" in fact_json

        # Test recreation from JSON
        fact_recreated = Fact.model_validate_json(fact_json)
        assert fact_recreated.statement == fact.statement
        assert fact_recreated.confidence == fact.confidence
        assert fact_recreated.as_of_date == fact.as_of_date
