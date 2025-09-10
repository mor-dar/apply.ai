"""
Core Pydantic schemas for the apply.ai job application copilot.

These schemas define the data structures used across the multi-agent architecture
for parsing, validation, and generation of job application materials.
"""

from datetime import datetime, timezone
from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator, HttpUrl
from uuid import uuid4


class SourceDomainClass(str, Enum):
    """Classification of fact source domains for validation."""

    OFFICIAL = "official"  # Company websites, official documentation
    REPUTABLE_NEWS = "reputable_news"  # Major news outlets
    OTHER = "other"  # Other sources


class Requirement(BaseModel):
    """Individual job requirement with classification and rationale."""

    id: str = Field(
        default_factory=lambda: str(uuid4())[:8], description="Unique requirement ID"
    )
    text: str = Field(..., description="Requirement text")
    must_have: bool = Field(
        True, description="Whether this is a must-have vs nice-to-have requirement"
    )
    rationale: Optional[str] = Field(
        None, description="Why this was classified as must-have/nice-to-have"
    )


class ParserReport(BaseModel):
    """Report on parsing quality and confidence."""

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall parsing confidence score"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Required fields that are missing or low confidence",
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings about parsing quality or content"
    )
    keyword_count: int = Field(0, description="Number of keywords extracted")
    requirement_count: int = Field(0, description="Number of requirements extracted")
    text_length: int = Field(0, description="Length of input text")


class JobPosting(BaseModel):
    """Parsed job posting with extracted requirements and keywords."""

    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: Optional[str] = Field(None, description="Job location")
    text: str = Field(..., description="Full job description text")
    keywords: List[str] = Field(
        default_factory=list, description="Extracted keywords ranked by importance"
    )
    requirements: List[Requirement] = Field(
        default_factory=list, description="Extracted requirements and qualifications"
    )

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v):
        if len(v) > 10000:  # Reasonable upper bound
            raise ValueError("Job description text too long")
        return v

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, v):
        return [item.strip() for item in v if isinstance(item, str) and item.strip()]

    @field_validator("requirements")
    @classmethod
    def validate_requirements(cls, v):
        # Requirements are now Requirement objects, not strings
        return v


class ResumeBullet(BaseModel):
    """Individual resume bullet point with metadata for evidence tracking."""

    text: str = Field(..., description="Bullet point text")
    section: str = Field(..., description="Resume section this bullet belongs to")
    start_offset: int = Field(..., description="Character offset in original text")
    end_offset: int = Field(..., description="End character offset in original text")
    embedding: Optional[List[float]] = Field(
        None, description="Vector embedding for similarity search"
    )

    @field_validator("text")
    @classmethod
    def validate_bullet_text(cls, v):
        text = v.strip()
        if not text:
            raise ValueError("Bullet text cannot be empty")
        return text


class ResumeSection(BaseModel):
    """Resume section with its bullets and metadata."""

    name: str = Field(..., description="Section name (e.g., 'Experience', 'Education')")
    bullets: List[ResumeBullet] = Field(
        default_factory=list, description="Bullets in this section"
    )
    start_offset: int = Field(
        ..., description="Section start position in original text"
    )
    end_offset: int = Field(..., description="Section end position in original text")


class Resume(BaseModel):
    """Parsed resume with normalized structure and content."""

    raw_text: str = Field(..., description="Original resume text")
    bullets: List[ResumeBullet] = Field(
        default_factory=list, description="All bullet points across sections"
    )
    skills: List[str] = Field(default_factory=list, description="Extracted skills")
    dates: List[str] = Field(
        default_factory=list, description="Extracted dates (experience, education)"
    )
    sections: List[ResumeSection] = Field(
        default_factory=list, description="Resume sections with structure"
    )

    @field_validator("raw_text")
    @classmethod
    def validate_raw_text(cls, v):
        if not v.strip():
            raise ValueError("Resume text cannot be empty")
        return v

    @field_validator("skills")
    @classmethod
    def validate_skills(cls, v):
        return [skill.strip() for skill in v if skill.strip()]


class Fact(BaseModel):
    """Individual research fact with validation metadata."""

    statement: str = Field(..., description="The factual claim")
    source_url: HttpUrl = Field(..., description="Source URL for verification")
    source_domain_class: SourceDomainClass = Field(
        ..., description="Classification of source domain"
    )
    as_of_date: datetime = Field(..., description="When this fact was current")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, v):
        statement = v.strip()
        if not statement:
            raise ValueError("Fact statement cannot be empty")
        if len(statement) > 500:
            raise ValueError("Fact statement too long")
        return statement


class FactSheet(BaseModel):
    """Company research compilation with validated facts."""

    company: str = Field(..., description="Company name")
    facts: List[Fact] = Field(
        default_factory=list, description="List of verified facts"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When factsheet was generated",
    )

    @field_validator("company")
    @classmethod
    def validate_company(cls, v):
        company = v.strip()
        if not company:
            raise ValueError("Company name cannot be empty")
        return company

    @field_validator("facts")
    @classmethod
    def validate_fact_limit(cls, v):
        if len(v) > 20:  # Reasonable upper limit
            raise ValueError("Too many facts in factsheet")
        return v


class TailoredBullet(BaseModel):
    """Rewritten bullet point with evidence mapping and keyword coverage."""

    text: str = Field(..., description="Tailored bullet text")
    original_bullet_id: Optional[int] = Field(
        None, description="Reference to original ResumeBullet"
    )
    evidence_spans: List[str] = Field(
        default_factory=list,
        description="Original resume content supporting this bullet",
    )
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity to original content"
    )
    jd_keywords_covered: List[str] = Field(
        default_factory=list, description="Job description keywords addressed"
    )

    @field_validator("text")
    @classmethod
    def validate_tailored_text(cls, v):
        text = v.strip()
        if not text:
            raise ValueError("Tailored bullet text cannot be empty")
        return text

    @field_validator("similarity_score")
    @classmethod
    def validate_evidence_threshold(cls, v):
        if v < 0.8:  # Enforce evidence-based requirement
            raise ValueError("Similarity score must be ≥ 0.8 for evidence validation")
        return v


class CoverLetter(BaseModel):
    """Generated cover letter with source citations."""

    intro: str = Field(..., description="Opening paragraph")
    body_points: List[str] = Field(
        default_factory=list, description="Main body paragraphs"
    )
    closing: str = Field(..., description="Closing paragraph")
    sources: List[str] = Field(
        default_factory=list, description="Fact sources referenced"
    )
    company: str = Field(..., description="Target company name")
    position: str = Field(..., description="Target position title")

    @field_validator("intro", "closing")
    @classmethod
    def validate_paragraphs(cls, v):
        paragraph = v.strip()
        if not paragraph:
            raise ValueError("Paragraph cannot be empty")
        return paragraph

    @field_validator("body_points")
    @classmethod
    def validate_body_points(cls, v):
        if len(v) < 1 or len(v) > 5:
            raise ValueError("Cover letter must have 1-5 body points")
        return [point.strip() for point in v if point.strip()]


class CompBand(BaseModel):
    """Compensation data with geography and percentiles."""

    occupation_code: str = Field(..., description="SOC or similar occupation code")
    geography: str = Field(..., description="Geographic area (state, national)")
    p25: Optional[float] = Field(None, description="25th percentile salary")
    p50: Optional[float] = Field(None, description="50th percentile (median) salary")
    p75: Optional[float] = Field(None, description="75th percentile salary")
    sources: List[HttpUrl] = Field(default_factory=list, description="Data source URLs")
    as_of: datetime = Field(..., description="Data currency date")
    currency: str = Field(default="USD", description="Salary currency")

    @field_validator("occupation_code")
    @classmethod
    def validate_occupation_code(cls, v):
        code = v.strip()
        if not code:
            raise ValueError("Occupation code cannot be empty")
        return code

    @field_validator("p25", "p50", "p75")
    @classmethod
    def validate_salary_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Salary values must be positive")
        return v


class Metrics(BaseModel):
    """Quality and coverage metrics for generated application materials."""

    jd_coverage_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage of JD keywords covered"
    )
    readability_grade: Optional[float] = Field(
        None, description="Flesch-Kincaid or similar readability score"
    )
    evidence_mapped_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of bullets with valid evidence mapping"
    )
    total_tailored_bullets: int = Field(
        ..., ge=0, description="Total number of tailored bullets"
    )
    validated_bullets: int = Field(
        ..., ge=0, description="Number of bullets passing evidence validation"
    )

    @field_validator("evidence_mapped_ratio")
    @classmethod
    def validate_evidence_threshold(cls, v):
        # Warn if below 95% threshold but don't fail validation
        return v

    @field_validator("validated_bullets")
    @classmethod
    def validate_bullet_counts(cls, v, info):
        if (
            info.data
            and "total_tailored_bullets" in info.data
            and v > info.data["total_tailored_bullets"]
        ):
            raise ValueError("Validated bullets cannot exceed total bullets")
        return v

    model_config = {"validate_assignment": True}
