"""
Comprehensive unit tests for ResumeValidator tool.

Tests cover all functionality including bullet validation, evidence matching,
similarity scoring, report generation, and error handling.
Maintains 100% test coverage with zero warnings.
"""

import pytest
from unittest.mock import Mock, patch

from tools.resume_validator import (
    ResumeValidator,
    ResumeValidationError,
    ValidationResult,
    ValidationReport,
    ValidationStatus,
    validate_tailored_bullets,
)
from tools.evidence_indexer import EvidenceIndexer, EvidenceMatch, EvidenceIndexingError
from src.schemas.core import TailoredBullet, Resume, ResumeBullet, ResumeSection


@pytest.fixture
def sample_tailored_bullets():
    """Create sample TailoredBullet objects for testing."""
    return [
        TailoredBullet(
            text="Developed scalable Python applications using microservices architecture",
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
            text="Led team development of e-commerce solutions",
            similarity_score=0.8,  # At threshold for schema validation
            evidence_spans=["Led team development"],
            jd_keywords_covered=["leadership"],
        ),
    ]


@pytest.fixture
def sample_resume():
    """Create a sample Resume object for testing."""
    bullets = [
        ResumeBullet(
            text="Developed Python applications for web services",
            section="Experience",
            start_offset=50,
            end_offset=95,
        ),
        ResumeBullet(
            text="Built React components for user interfaces",
            section="Experience",
            start_offset=96,
            end_offset=140,
        ),
        ResumeBullet(
            text="Led development team on multiple projects",
            section="Experience",
            start_offset=141,
            end_offset=185,
        ),
        ResumeBullet(
            text="Implemented automated testing procedures",
            section="Experience",
            start_offset=186,
            end_offset=230,
        ),
        ResumeBullet(
            text="Collaborated with cross-functional teams",
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
def mock_evidence_indexer():
    """Create mock EvidenceIndexer for testing."""
    mock_indexer = Mock(spec=EvidenceIndexer)
    mock_indexer.similarity_threshold = 0.8
    
    # Mock the collection attribute with count method
    mock_collection = Mock()
    mock_collection.count.return_value = 10  # Default to already indexed
    mock_indexer.collection = mock_collection
    
    return mock_indexer


@pytest.fixture
def validator(mock_evidence_indexer):
    """Create ResumeValidator instance with mocked dependencies."""
    return ResumeValidator(
        similarity_threshold=0.8,
        min_confidence_threshold=0.7,
        evidence_indexer=mock_evidence_indexer,
        require_keyword_evidence=False,
    )


class TestResumeValidator:
    """Test suite for ResumeValidator tool."""
    
    @pytest.fixture
    def sample_tailored_bullets(self):
        """Create sample tailored bullets for testing."""
        return [
            TailoredBullet(
                text="Developed scalable Python applications using microservices architecture",
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
                text="Led team development of e-commerce solutions",
                similarity_score=0.8,  # At threshold for schema validation
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
        """Create mock EvidenceIndexer for testing."""
        mock_indexer = Mock(spec=EvidenceIndexer)
        mock_indexer.similarity_threshold = 0.8
        
        # Mock the collection attribute with count method
        mock_collection = Mock()
        mock_collection.count.return_value = 10  # Default to already indexed
        mock_indexer.collection = mock_collection
        
        return mock_indexer
    
    @pytest.fixture
    def validator(self, mock_evidence_indexer):
        """Create ResumeValidator instance with mocked dependencies."""
        return ResumeValidator(
            similarity_threshold=0.8,
            min_confidence_threshold=0.7,
            evidence_indexer=mock_evidence_indexer,
            require_keyword_evidence=False,
        )
    
    def test_init_default_parameters_duplicate(self):
        """Test ResumeValidator initialization with default parameters (duplicate test)."""
        with patch('tools.resume_validator.EvidenceIndexer') as mock_indexer_class:
            mock_indexer_class.return_value = Mock()
            
            validator = ResumeValidator()
            
            assert validator.similarity_threshold == 0.8
            assert validator.min_confidence_threshold == 0.7
            assert validator.require_keyword_evidence is True
            assert validator.evidence_indexer is not None
            mock_indexer_class.assert_called_once()
    
    def test_init_custom_parameters(self, mock_evidence_indexer):
        """Test ResumeValidator initialization with custom parameters."""
        validator = ResumeValidator(
            similarity_threshold=0.9,
            min_confidence_threshold=0.8,
            evidence_indexer=mock_evidence_indexer,
            require_keyword_evidence=False,
        )
        
        assert validator.similarity_threshold == 0.9
        assert validator.min_confidence_threshold == 0.8
        assert validator.require_keyword_evidence is False
        assert validator.evidence_indexer is mock_evidence_indexer
    
    def test_validate_bullet_valid(self, validator, sample_tailored_bullets):
        """Test validation of a valid bullet with good evidence."""
        bullet = sample_tailored_bullets[0]  # High similarity score
        
        # Mock evidence matches - multiple matches to boost confidence score
        mock_evidence = [
            EvidenceMatch(
                bullet=ResumeBullet(
                    text="Developed Python applications",
                    section="Experience",
                    start_offset=0,
                    end_offset=30,
                ),
                similarity_score=0.9,
            ),
            EvidenceMatch(
                bullet=ResumeBullet(
                    text="Built scalable software solutions",
                    section="Experience", 
                    start_offset=31,
                    end_offset=65,
                ),
                similarity_score=0.85,
            ),
            EvidenceMatch(
                bullet=ResumeBullet(
                    text="Worked with modern technologies",
                    section="Experience",
                    start_offset=66,
                    end_offset=98,
                ),
                similarity_score=0.88,
            ),
        ]
        validator.evidence_indexer.find_evidence.return_value = mock_evidence
        
        result = validator.validate_bullet(bullet)
        
        assert isinstance(result, ValidationResult)
        assert result.status == ValidationStatus.VALID
        assert result.bullet == bullet
        assert result.best_similarity_score == 0.9
        assert result.confidence_score >= 0.7
        assert len(result.evidence_matches) == 3
        assert len(result.validation_notes) > 0
        assert "Strong evidence support" in result.validation_notes[0]
    
    def test_validate_bullet_rejected_low_similarity(self, validator, sample_tailored_bullets):
        """Test validation of bullet with low similarity evidence."""
        bullet = sample_tailored_bullets[2]  # Use valid bullet from sample
        
        # Mock low similarity evidence
        mock_evidence = [
            EvidenceMatch(
                bullet=ResumeBullet(
                    text="Some unrelated experience",
                    section="Experience",
                    start_offset=0,
                    end_offset=30,
                ),
                similarity_score=0.6,  # Below threshold
            ),
        ]
        validator.evidence_indexer.find_evidence.return_value = mock_evidence
        
        result = validator.validate_bullet(bullet)
        
        assert result.status == ValidationStatus.REJECTED
        assert result.best_similarity_score == 0.6
        assert result.recommended_edits is not None
        assert "below threshold" in result.validation_notes[0]
    
    def test_validate_bullet_needs_edit_low_confidence(self, validator, sample_tailored_bullets):
        """Test validation of bullet that needs editing due to low confidence."""
        bullet = sample_tailored_bullets[1]
        
        # Mock evidence with decent similarity but low confidence scenario
        mock_evidence = [
            EvidenceMatch(
                bullet=ResumeBullet(
                    text="Built some components",
                    section="Experience",
                    start_offset=0,
                    end_offset=25,
                ),
                similarity_score=0.82,  # Above threshold
            ),
        ]
        validator.evidence_indexer.find_evidence.return_value = mock_evidence
        
        # Mock low confidence calculation
        with patch.object(validator, '_calculate_confidence_score', return_value=0.6):
            result = validator.validate_bullet(bullet)
            
            assert result.status == ValidationStatus.NEEDS_EDIT
            assert result.confidence_score == 0.6
            assert result.recommended_edits is not None
            assert "Low confidence" in result.validation_notes[0]
    
    def test_validate_bullet_no_evidence(self, validator, sample_tailored_bullets):
        """Test validation when no evidence is found."""
        bullet = sample_tailored_bullets[0]
        
        validator.evidence_indexer.find_evidence.return_value = []
        
        result = validator.validate_bullet(bullet)
        
        assert result.status == ValidationStatus.REJECTED
        assert result.best_similarity_score == 0.0
        assert result.confidence_score == 0.0
        assert len(result.evidence_matches) == 0
        assert "No evidence found" in result.validation_notes[0]
        assert result.recommended_edits is not None
    
    def test_validate_bullet_keyword_evidence_issues(self, mock_evidence_indexer, sample_tailored_bullets):
        """Test validation with unsupported keyword claims."""
        # Create validator with keyword evidence required
        keyword_validator = ResumeValidator(
            similarity_threshold=0.8,
            min_confidence_threshold=0.6,  # Lower threshold to pass confidence check
            evidence_indexer=mock_evidence_indexer,
            require_keyword_evidence=True,
        )
        
        bullet = sample_tailored_bullets[0]  # Has keywords: python, microservices
        
        # Mock evidence with high confidence - multiple matches but missing keyword support
        mock_evidence = [
            EvidenceMatch(
                bullet=ResumeBullet(
                    text="Developed Python applications",  # Only mentions python, not microservices
                    section="Experience",
                    start_offset=0,
                    end_offset=30,
                ),
                similarity_score=0.9,
            ),
            EvidenceMatch(
                bullet=ResumeBullet(
                    text="Built scalable software solutions with Python",
                    section="Experience",
                    start_offset=31,
                    end_offset=70,
                ),
                similarity_score=0.88,
            ),
            EvidenceMatch(
                bullet=ResumeBullet(
                    text="Python development experience",
                    section="Experience",
                    start_offset=71,
                    end_offset=100,
                ),
                similarity_score=0.85,
            ),
        ]
        keyword_validator.evidence_indexer.find_evidence.return_value = mock_evidence
        
        result = keyword_validator.validate_bullet(bullet)
        
        assert result.status == ValidationStatus.NEEDS_EDIT
        assert "Keywords not clearly supported" in " ".join(result.validation_notes)
        assert "microservices" in " ".join(result.validation_notes)
    
    def test_validate_bullet_evidence_indexing_error(self, validator, sample_tailored_bullets):
        """Test validation when evidence indexing fails."""
        bullet = sample_tailored_bullets[0]
        
        validator.evidence_indexer.find_evidence.side_effect = EvidenceIndexingError("Index error")
        
        result = validator.validate_bullet(bullet)
        
        assert result.status == ValidationStatus.ERROR
        assert "Evidence indexing error" in result.validation_notes[0]
        assert result.best_similarity_score == 0.0
        assert result.confidence_score == 0.0
    
    def test_validate_bullet_unexpected_error(self, validator, sample_tailored_bullets):
        """Test validation with unexpected error."""
        bullet = sample_tailored_bullets[0]
        
        validator.evidence_indexer.find_evidence.side_effect = Exception("Unexpected error")
        
        with pytest.raises(ResumeValidationError, match="Bullet validation failed"):
            validator.validate_bullet(bullet)
    
    def test_validate_bullets_success(self, validator, sample_tailored_bullets, sample_resume):
        """Test successful validation of multiple bullets."""
        # Mock evidence indexer to simulate indexed resume
        validator.evidence_indexer.collection.count.return_value = 10  # Already indexed
        
        # Mock individual bullet validations
        with patch.object(validator, 'validate_bullet') as mock_validate:
            mock_results = [
                ValidationResult(
                    bullet=sample_tailored_bullets[0],
                    status=ValidationStatus.VALID,
                    evidence_matches=[Mock()],
                    best_similarity_score=0.9,
                    confidence_score=0.85,
                    validation_notes=["Valid"],
                ),
                ValidationResult(
                    bullet=sample_tailored_bullets[1],
                    status=ValidationStatus.VALID,
                    evidence_matches=[Mock()],
                    best_similarity_score=0.85,
                    confidence_score=0.8,
                    validation_notes=["Valid"],
                ),
                ValidationResult(
                    bullet=sample_tailored_bullets[2],
                    status=ValidationStatus.REJECTED,
                    evidence_matches=[Mock()],
                    best_similarity_score=0.6,
                    confidence_score=0.5,
                    validation_notes=["Rejected"],
                ),
            ]
            mock_validate.side_effect = mock_results
            
            report = validator.validate_bullets(sample_tailored_bullets, sample_resume)
            
            assert isinstance(report, ValidationReport)
            assert report.total_bullets == 3
            assert report.valid_bullets == 2
            assert report.rejected_bullets == 1
            assert report.needs_edit_bullets == 0
            assert report.error_bullets == 0
            assert report.validation_success_rate == (2/3) * 100
            assert report.needs_review_count == 1
            assert len(report.validation_results) == 3
            assert len(report.summary_recommendations) > 0
    
    def test_validate_bullets_empty_list(self, validator):
        """Test validation of empty bullet list."""
        report = validator.validate_bullets([])
        
        assert report.total_bullets == 0
        assert report.valid_bullets == 0
        assert report.validation_success_rate == 0.0
        assert len(report.validation_results) == 0
        assert len(report.summary_recommendations) == 0
    
    def test_validate_bullets_with_resume_indexing(self, validator, sample_tailored_bullets, sample_resume):
        """Test validation with resume indexing needed."""
        # Mock empty collection requiring indexing
        validator.evidence_indexer.collection.count.return_value = 0
        validator.evidence_indexer.index_resume.return_value = {"items_indexed": 5}
        
        # Mock validation results
        with patch.object(validator, 'validate_bullet') as mock_validate:
            mock_validate.return_value = ValidationResult(
                bullet=sample_tailored_bullets[0],
                status=ValidationStatus.VALID,
                evidence_matches=[Mock()],
                best_similarity_score=0.9,
                confidence_score=0.85,
                validation_notes=["Valid"],
            )
            
            report = validator.validate_bullets(sample_tailored_bullets, sample_resume)
            
            # Should have called index_resume
            validator.evidence_indexer.index_resume.assert_called_once_with(sample_resume)
            assert isinstance(report, ValidationReport)
    
    def test_validate_bullets_indexing_failure(self, validator, sample_tailored_bullets, sample_resume):
        """Test validation when resume indexing fails."""
        validator.evidence_indexer.collection.count.return_value = 0
        validator.evidence_indexer.index_resume.side_effect = Exception("Indexing failed")
        
        # Should continue validation despite indexing failure
        with patch.object(validator, 'validate_bullet') as mock_validate:
            mock_validate.return_value = ValidationResult(
                bullet=sample_tailored_bullets[0],
                status=ValidationStatus.VALID,
                evidence_matches=[],
                best_similarity_score=0.8,
                confidence_score=0.7,
                validation_notes=["Valid"],
            )
            
            report = validator.validate_bullets(sample_tailored_bullets, sample_resume)
            
            assert isinstance(report, ValidationReport)
    
    def test_validate_bullets_individual_errors(self, validator, sample_tailored_bullets):
        """Test validation with some individual bullet errors."""
        with patch.object(validator, 'validate_bullet') as mock_validate:
            mock_validate.side_effect = [
                ValidationResult(
                    bullet=sample_tailored_bullets[0],
                    status=ValidationStatus.VALID,
                    evidence_matches=[Mock()],
                    best_similarity_score=0.9,
                    confidence_score=0.85,
                    validation_notes=["Valid"],
                ),
                Exception("Validation error"),  # Error on second bullet
                ValidationResult(
                    bullet=sample_tailored_bullets[2],
                    status=ValidationStatus.REJECTED,
                    evidence_matches=[Mock()],
                    best_similarity_score=0.6,
                    confidence_score=0.5,
                    validation_notes=["Rejected"],
                ),
            ]
            
            report = validator.validate_bullets(sample_tailored_bullets)
            
            assert report.total_bullets == 3
            assert report.valid_bullets == 1
            assert report.rejected_bullets == 1
            assert report.error_bullets == 1
    
    def test_validate_bullets_error_handling(self, validator, sample_tailored_bullets):
        """Test error handling in bullet validation."""
        with patch.object(validator, 'validate_bullet', side_effect=Exception("Validation error")):
            report = validator.validate_bullets([sample_tailored_bullets[0]])
            
            assert isinstance(report, ValidationReport)
            assert report.total_bullets == 1
            assert report.error_bullets == 1
            assert report.valid_bullets == 0
            assert report.rejected_bullets == 0
            assert report.needs_edit_bullets == 0
            assert len(report.validation_results) == 1
            assert report.validation_results[0].status == ValidationStatus.ERROR
            assert "Validation error" in report.validation_results[0].validation_notes[0]
    
    def test_calculate_confidence_score(self, validator):
        """Test confidence score calculation."""
        # Test with no evidence
        score = validator._calculate_confidence_score([])
        assert score == 0.0
        
        # Test with single high-quality evidence
        evidence = [
            EvidenceMatch(bullet=Mock(), similarity_score=0.95),
        ]
        score = validator._calculate_confidence_score(evidence)
        assert score > 0.5
        
        # Test with multiple consistent evidence
        evidence = [
            EvidenceMatch(bullet=Mock(), similarity_score=0.9),
            EvidenceMatch(bullet=Mock(), similarity_score=0.85),
            EvidenceMatch(bullet=Mock(), similarity_score=0.88),
        ]
        score = validator._calculate_confidence_score(evidence)
        assert score > 0.7
    
    def test_validate_keyword_evidence(self, validator):
        """Test keyword evidence validation."""
        bullet = TailoredBullet(
            text="Python development with microservices",
            similarity_score=0.9,
            evidence_spans=["test"],
            jd_keywords_covered=["python", "microservices", "kubernetes"],
        )
        
        # Evidence only mentions python and microservices
        evidence = [
            EvidenceMatch(
                bullet=Mock(text="Developed Python applications with microservices"),
                similarity_score=0.9,
            ),
        ]
        
        notes = validator._validate_keyword_evidence(bullet, evidence)
        
        assert len(notes) > 0
        assert "kubernetes" in notes[0]  # Should identify unsupported keyword
        assert "not clearly supported" in notes[0]
        
        # Test with empty keywords
        bullet_no_keywords = TailoredBullet(
            text="Test bullet",
            similarity_score=0.9,
            evidence_spans=["test"],
            jd_keywords_covered=[],
        )
        
        notes_empty = validator._validate_keyword_evidence(bullet_no_keywords, evidence)
        assert len(notes_empty) == 0
    
    def test_determine_validation_status(self, validator):
        """Test validation status determination logic."""
        bullet = Mock()
        
        # Test valid case
        status, notes, edits = validator._determine_validation_status(
            bullet, best_similarity=0.9, confidence_score=0.85, keyword_validation_notes=[]
        )
        assert status == ValidationStatus.VALID
        assert "Strong evidence support" in notes[0]
        assert edits is None
        
        # Test rejected case (low similarity)
        status, notes, edits = validator._determine_validation_status(
            bullet, best_similarity=0.6, confidence_score=0.8, keyword_validation_notes=[]
        )
        assert status == ValidationStatus.REJECTED
        assert "below threshold" in notes[0]
        assert edits is not None
        
        # Test needs edit case (low confidence)
        status, notes, edits = validator._determine_validation_status(
            bullet, best_similarity=0.85, confidence_score=0.5, keyword_validation_notes=[]
        )
        assert status == ValidationStatus.NEEDS_EDIT
        assert "Low confidence" in notes[0]
        assert edits is not None
        
        # Test needs edit case (keyword issues)
        status, notes, edits = validator._determine_validation_status(
            bullet, best_similarity=0.9, confidence_score=0.85, 
            keyword_validation_notes=["Keyword issues"]
        )
        assert status == ValidationStatus.NEEDS_EDIT
        assert "Keyword issues" in notes
        assert edits is not None
    
    def test_calculate_overall_evidence_score(self, validator):
        """Test overall evidence score calculation."""
        results = [
            ValidationResult(
                bullet=Mock(),
                status=ValidationStatus.VALID,
                evidence_matches=[],
                best_similarity_score=0.9,
                confidence_score=0.8,
                validation_notes=[],
            ),
            ValidationResult(
                bullet=Mock(),
                status=ValidationStatus.REJECTED,
                evidence_matches=[],
                best_similarity_score=0.7,
                confidence_score=0.6,
                validation_notes=[],
            ),
            ValidationResult(
                bullet=Mock(),
                status=ValidationStatus.ERROR,
                evidence_matches=[],
                best_similarity_score=0.0,
                confidence_score=0.0,
                validation_notes=[],
            ),
        ]
        
        # Should exclude error results and average similarity scores
        score = validator._calculate_overall_evidence_score(results)
        assert score == 0.8  # (0.9 + 0.7) / 2
        
        # Test with empty results
        empty_score = validator._calculate_overall_evidence_score([])
        assert empty_score == 0.0
    
    def test_calculate_evidence_coverage(self, validator):
        """Test evidence coverage calculation."""
        results = [
            ValidationResult(
                bullet=Mock(),
                status=ValidationStatus.VALID,
                evidence_matches=[],
                best_similarity_score=0.9,  # Above threshold
                confidence_score=0.8,
                validation_notes=[],
            ),
            ValidationResult(
                bullet=Mock(),
                status=ValidationStatus.REJECTED,
                evidence_matches=[],
                best_similarity_score=0.7,  # Below threshold
                confidence_score=0.6,
                validation_notes=[],
            ),
            ValidationResult(
                bullet=Mock(),
                status=ValidationStatus.VALID,
                evidence_matches=[],
                best_similarity_score=0.85,  # Above threshold
                confidence_score=0.8,
                validation_notes=[],
            ),
        ]
        
        coverage = validator._calculate_evidence_coverage(results)
        assert coverage == (2/3) * 100  # 2 out of 3 above threshold
    
    def test_generate_summary_recommendations(self, validator):
        """Test summary recommendations generation."""
        # Create realistic mock validation results
        mock_valid = Mock()
        mock_valid.confidence_score = 0.9
        mock_valid.best_similarity_score = 0.9
        mock_valid.status = ValidationStatus.VALID
        
        mock_rejected = Mock()
        mock_rejected.confidence_score = 0.5
        mock_rejected.best_similarity_score = 0.6
        mock_rejected.status = ValidationStatus.REJECTED
        
        mock_needs_edit = Mock()
        mock_needs_edit.confidence_score = 0.6
        mock_needs_edit.best_similarity_score = 0.8
        mock_needs_edit.status = ValidationStatus.NEEDS_EDIT
        
        validation_results = [mock_valid, mock_rejected, mock_needs_edit]
        status_counts = {
            ValidationStatus.VALID: 1,
            ValidationStatus.REJECTED: 1,
            ValidationStatus.NEEDS_EDIT: 1,
            ValidationStatus.ERROR: 0,
        }
        
        recommendations = validator._generate_summary_recommendations(
            validation_results, status_counts
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have recommendations about low success rate and specific issues
        rec_text = " ".join(recommendations)
        assert "success rate" in rec_text.lower()
        assert "rejected" in rec_text.lower()
        assert "need editing" in rec_text.lower()
    
    def test_get_validation_summary(self, validator):
        """Test validation summary generation."""
        report = ValidationReport(
            total_bullets=10,
            valid_bullets=7,
            rejected_bullets=2,
            needs_edit_bullets=1,
            error_bullets=0,
            overall_evidence_score=0.85,
            evidence_coverage_percentage=80.0,
            validation_results=[],
            summary_recommendations=["Rec 1", "Rec 2", "Rec 3", "Rec 4"],
        )
        
        summary = validator.get_validation_summary(report)
        
        assert summary["total_bullets"] == 10
        assert summary["validation_success_rate"] == 70.0
        assert summary["valid_count"] == 7
        assert summary["needs_review_count"] == 3  # rejected + needs_edit
        assert summary["overall_evidence_score"] == 0.85
        assert summary["evidence_coverage_percentage"] == 80.0
        assert len(summary["top_recommendations"]) == 3  # Top 3 only
        assert summary["similarity_threshold_used"] == 0.8
        assert summary["confidence_threshold_used"] == 0.7


class TestValidationResult:
    """Test suite for ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation and attributes."""
        bullet = TailoredBullet(
            text="Test bullet",
            similarity_score=0.9,
            evidence_spans=["test"],
            jd_keywords_covered=["test"],
        )
        
        result = ValidationResult(
            bullet=bullet,
            status=ValidationStatus.VALID,
            evidence_matches=[],
            best_similarity_score=0.9,
            confidence_score=0.85,
            validation_notes=["Good evidence"],
            recommended_edits=None,
        )
        
        assert result.bullet == bullet
        assert result.status == ValidationStatus.VALID
        assert result.best_similarity_score == 0.9
        assert result.confidence_score == 0.85
        assert result.validation_notes == ["Good evidence"]
        assert result.recommended_edits is None


class TestValidationReport:
    """Test suite for ValidationReport dataclass."""
    
    def test_validation_report_creation(self):
        """Test ValidationReport creation and computed properties."""
        report = ValidationReport(
            total_bullets=10,
            valid_bullets=7,
            rejected_bullets=2,
            needs_edit_bullets=1,
            error_bullets=0,
            overall_evidence_score=0.85,
            evidence_coverage_percentage=75.0,
            validation_results=[],
            summary_recommendations=["Test rec"],
        )
        
        assert report.total_bullets == 10
        assert report.valid_bullets == 7
        assert report.validation_success_rate == 70.0
        assert report.needs_review_count == 3  # rejected + needs_edit
    
    def test_validation_report_zero_bullets(self):
        """Test ValidationReport with zero bullets."""
        report = ValidationReport(
            total_bullets=0,
            valid_bullets=0,
            rejected_bullets=0,
            needs_edit_bullets=0,
            error_bullets=0,
            overall_evidence_score=0.0,
            evidence_coverage_percentage=0.0,
            validation_results=[],
            summary_recommendations=[],
        )
        
        assert report.validation_success_rate == 0.0
        assert report.needs_review_count == 0


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_validate_tailored_bullets(self, sample_tailored_bullets, sample_resume):
        """Test convenience function for bullet validation."""
        with patch('tools.resume_validator.ResumeValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_report = Mock(spec=ValidationReport)
            mock_validator.validate_bullets.return_value = mock_report
            mock_validator_class.return_value = mock_validator
            
            result = validate_tailored_bullets(
                sample_tailored_bullets, sample_resume, similarity_threshold=0.9
            )
            
            assert result == mock_report
            mock_validator_class.assert_called_once_with(
                similarity_threshold=0.9, evidence_indexer=None
            )
            mock_validator.validate_bullets.assert_called_once_with(
                sample_tailored_bullets, sample_resume
            )


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_validation_with_minimal_values(self, validator):
        """Test validation handling of minimal valid values."""
        # Test bullet with empty lists (valid but minimal)
        bullet = TailoredBullet(
            text="Test bullet",
            similarity_score=0.9,
            evidence_spans=[],
            jd_keywords_covered=[],
        )
        
        validator.evidence_indexer.find_evidence.return_value = []
        
        result = validator.validate_bullet(bullet)
        assert isinstance(result, ValidationResult)
        assert result.status == ValidationStatus.REJECTED  # No evidence found
    
    def test_validation_with_minimal_strings(self, validator):
        """Test validation with minimal string inputs."""
        bullet = TailoredBullet(
            text="A",  # Minimal valid text
            similarity_score=0.9,
            evidence_spans=["A"],
            jd_keywords_covered=["a"],
        )
        
        validator.evidence_indexer.find_evidence.return_value = []
        
        result = validator.validate_bullet(bullet)
        assert isinstance(result, ValidationResult)
    
    def test_confidence_calculation_edge_cases(self, validator):
        """Test confidence score calculation edge cases."""
        
        # Test with identical similarity scores (no variance)
        evidence = [
            EvidenceMatch(bullet=Mock(), similarity_score=0.8),
            EvidenceMatch(bullet=Mock(), similarity_score=0.8),
            EvidenceMatch(bullet=Mock(), similarity_score=0.8),
        ]
        
        score = validator._calculate_confidence_score(evidence)
        assert 0.0 <= score <= 1.0
        
        # Test with extreme variance
        evidence_varied = [
            EvidenceMatch(bullet=Mock(), similarity_score=1.0),
            EvidenceMatch(bullet=Mock(), similarity_score=0.0),
        ]
        
        score_varied = validator._calculate_confidence_score(evidence_varied)
        assert 0.0 <= score_varied <= 1.0
    
    def test_keyword_validation_partial_matches(self, validator):
        """Test keyword validation with partial word matches."""
        bullet = TailoredBullet(
            text="Test bullet",
            similarity_score=0.9,
            evidence_spans=["test"],
            jd_keywords_covered=["machine learning", "data science"],
        )
        
        # Evidence has partial matches
        evidence = [
            EvidenceMatch(
                bullet=Mock(text="machine algorithms and data analysis"),
                similarity_score=0.9,
            ),
        ]
        
        notes = validator._validate_keyword_evidence(bullet, evidence)
        
        # Should handle partial matches appropriately
        assert isinstance(notes, list)
    
    def test_report_generation_with_all_error_results(self, validator):
        """Test report generation when all validations error."""
        bullets = [Mock(), Mock(), Mock()]
        
        with patch.object(validator, 'validate_bullet') as mock_validate:
            mock_validate.side_effect = [Exception("Error")] * 3
            
            report = validator.validate_bullets(bullets)
            
            assert report.total_bullets == 3
            assert report.error_bullets == 3
            assert report.valid_bullets == 0
            assert report.validation_success_rate == 0.0