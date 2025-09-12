"""
Comprehensive unit tests for ResumeGenerator tool.

Tests cover all functionality including keyword extraction, bullet mapping,
tailored bullet generation, similarity calculation, and error handling.
Maintains 100% test coverage with zero warnings.
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from tools.resume_generator import (
    ResumeGenerator,
    ResumeGenerationError,
    KeywordMapping,
    GenerationMetrics,
    generate_tailored_resume,
)
from tools.evidence_indexer import EvidenceIndexer
from src.schemas.core import (
    JobPosting,
    Resume,
    ResumeBullet,
    ResumeSection,
    TailoredBullet,
    Requirement,
)


@pytest.fixture
def sample_job_posting():
    """Create a sample JobPosting object for testing."""
    return JobPosting(
        title="Senior Software Engineer",
        company="TechCorp",
        location="San Francisco, CA",
        text="We are looking for a senior software engineer with experience in Python, React, and microservices architecture. Must have 5+ years of experience.",
        keywords=["python", "react", "microservices", "api", "javascript", "sql"],
        requirements=[
            Requirement(text="5+ years of software development experience", must_have=True),
            Requirement(text="Experience with Python and React", must_have=True),
            Requirement(text="Knowledge of microservices architecture", must_have=False),
        ],
    )


@pytest.fixture
def sample_resume():
    """Create a sample Resume object for testing."""
    bullets = [
        ResumeBullet(
            text="Developed scalable web applications using Python and Django framework",
            section="Experience",
            start_offset=100,
            end_offset=165,
        ),
        ResumeBullet(
            text="Built responsive frontend interfaces with React and JavaScript",
            section="Experience",
            start_offset=166,
            end_offset=225,
        ),
        ResumeBullet(
            text="Designed RESTful APIs for microservices communication",
            section="Experience",
            start_offset=226,
            end_offset=280,
        ),
        ResumeBullet(
            text="Led team of 3 developers on e-commerce platform project",
            section="Experience",
            start_offset=281,
            end_offset=340,
        ),
    ]
    
    sections = [
        ResumeSection(
            name="Experience",
            bullets=bullets,
            start_offset=50,
            end_offset=340,
        ),
    ]
    
    return Resume(
        raw_text="Sample resume text with experience in Python, React, and web development.",
        bullets=bullets,
        skills=["Python", "React", "JavaScript", "Django", "SQL", "Git"],
        sections=sections,
    )


@pytest.fixture
def mock_evidence_indexer():
    """Create mock EvidenceIndexer for testing."""
    mock_indexer = Mock(spec=EvidenceIndexer)
    mock_indexer.similarity_threshold = 0.8
    mock_indexer.find_evidence.return_value = []
    return mock_indexer


@pytest.fixture
def generator(mock_evidence_indexer):
    """Create ResumeGenerator instance with mocked dependencies."""
    with patch('tools.resume_generator.spacy.load') as mock_spacy, \
         patch('tools.resume_generator.TfidfVectorizer') as mock_tfidf:
        
        # Mock spaCy
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.similarity.return_value = 0.85
        mock_doc.ents = []
        mock_doc.noun_chunks = []
        mock_doc.__iter__ = Mock(return_value=iter([Mock(pos_="VERB", lemma_="developed", text="developed")]))
        mock_nlp.return_value = mock_doc
        mock_spacy.return_value = mock_nlp
        
        # Mock TF-IDF
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.return_value.toarray.return_value = [[0.5, 0.3, 0.2]]
        mock_vectorizer.get_feature_names_out.return_value = ["python", "react", "microservices"]
        mock_tfidf.return_value = mock_vectorizer
        
        return ResumeGenerator(
            similarity_threshold=0.8,
            evidence_indexer=mock_evidence_indexer,
        )


class TestResumeGenerator:
    """Test suite for ResumeGenerator tool."""
    
    def test_init_default_parameters(self):
        """Test ResumeGenerator initialization with default parameters."""
        with patch('tools.resume_generator.spacy.load') as mock_spacy, \
             patch('tools.resume_generator.TfidfVectorizer') as mock_tfidf, \
             patch('tools.resume_generator.EvidenceIndexer') as mock_indexer_class:
            
            mock_spacy.return_value = Mock()
            mock_tfidf.return_value = Mock()
            mock_indexer_class.return_value = Mock()
            
            generator = ResumeGenerator()
            
            assert generator.similarity_threshold == 0.8
            assert generator.min_keyword_frequency == 1
            assert generator.max_bullets_per_keyword == 3
            assert generator.evidence_indexer is not None
            mock_indexer_class.assert_called_once()
    
    def test_init_custom_parameters(self, mock_evidence_indexer):
        """Test ResumeGenerator initialization with custom parameters."""
        with patch('tools.resume_generator.spacy.load') as mock_spacy, \
             patch('tools.resume_generator.TfidfVectorizer'):
            
            mock_spacy.return_value = Mock()
            
            generator = ResumeGenerator(
                similarity_threshold=0.9,
                min_keyword_frequency=2,
                max_bullets_per_keyword=5,
                evidence_indexer=mock_evidence_indexer,
            )
            
            assert generator.similarity_threshold == 0.9
            assert generator.min_keyword_frequency == 2
            assert generator.max_bullets_per_keyword == 5
            assert generator.evidence_indexer is mock_evidence_indexer
    
    def test_init_spacy_error(self):
        """Test ResumeGenerator initialization when spaCy model is not found."""
        with patch('tools.resume_generator.spacy.load') as mock_spacy:
            mock_spacy.side_effect = OSError("Model not found")
            
            with pytest.raises(ResumeGenerationError, match="spaCy English model not found"):
                ResumeGenerator()
    
    def test_extract_keywords_from_job_basic(self, generator, sample_job_posting):
        """Test basic keyword extraction from job posting."""
        keywords = generator.extract_keywords_from_job(sample_job_posting)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        
        # Should include original keywords
        for keyword in sample_job_posting.keywords:
            assert keyword.lower() in [k.lower() for k in keywords]
    
    def test_extract_keywords_from_job_empty_keywords(self, generator):
        """Test keyword extraction when job has no pre-extracted keywords."""
        job_posting = JobPosting(
            title="Developer",
            company="TestCorp",
            text="Looking for developer with Python experience",
            keywords=[],  # Empty keywords
            requirements=[],
        )
        
        keywords = generator.extract_keywords_from_job(job_posting)
        assert isinstance(keywords, list)
    
    def test_extract_keywords_error_handling(self, generator):
        """Test keyword extraction error handling."""
        with patch.object(generator, 'nlp', side_effect=Exception("NLP error")):
            with pytest.raises(ResumeGenerationError, match="Failed to extract keywords"):
                generator.extract_keywords_from_job(JobPosting(
                    title="Test", company="Test", text="Test", keywords=[], requirements=[]
                ))
    
    def test_map_keywords_to_bullets_basic(self, generator, sample_resume):
        """Test basic keyword to bullet mapping."""
        keywords = ["python", "react", "microservices"]
        mappings = generator.map_keywords_to_bullets(keywords, sample_resume)
        
        assert isinstance(mappings, list)
        
        for mapping in mappings:
            assert isinstance(mapping, KeywordMapping)
            assert mapping.keyword in keywords
            assert isinstance(mapping.matched_bullets, list)
            assert isinstance(mapping.similarity_scores, list)
            assert len(mapping.matched_bullets) == len(mapping.similarity_scores)
    
    def test_map_keywords_to_bullets_empty_inputs(self, generator):
        """Test keyword mapping with empty inputs."""
        # Empty keywords
        mappings = generator.map_keywords_to_bullets([], Mock())
        assert mappings == []
        
        # Empty resume bullets
        resume = Resume(raw_text="test", bullets=[], skills=[], sections=[])
        mappings = generator.map_keywords_to_bullets(["test"], resume)
        assert mappings == []
    
    def test_generate_tailored_bullets_basic(self, generator):
        """Test basic tailored bullet generation."""
        # Create sample keyword mappings
        sample_bullets = [
            ResumeBullet(text="Developed Python applications", section="Experience", start_offset=0, end_offset=30),
            ResumeBullet(text="Built React components", section="Experience", start_offset=31, end_offset=55),
        ]
        
        mappings = [
            KeywordMapping(
                keyword="python",
                matched_bullets=sample_bullets[:1],
                similarity_scores=[0.9],
                priority_score=0.9,
            ),
            KeywordMapping(
                keyword="react",
                matched_bullets=sample_bullets[1:],
                similarity_scores=[0.85],
                priority_score=0.85,
            ),
        ]
        
        with patch.object(generator, '_rewrite_bullet_for_keyword') as mock_rewrite:
            mock_bullet = TailoredBullet(
                text="Enhanced Python development",
                similarity_score=0.85,
                evidence_spans=["Developed Python applications"],
                jd_keywords_covered=["python"],
            )
            mock_rewrite.return_value = mock_bullet
            
            tailored_bullets = generator.generate_tailored_bullets(mappings, max_bullets=10)
            
            assert isinstance(tailored_bullets, list)
            assert len(tailored_bullets) > 0
            assert all(isinstance(b, TailoredBullet) for b in tailored_bullets)
    
    def test_generate_tailored_bullets_max_bullets_limit(self, generator):
        """Test that generation respects max_bullets limit."""
        sample_bullets = [
            ResumeBullet(text=f"Bullet {i}", section="Experience", start_offset=i*10, end_offset=(i+1)*10)
            for i in range(10)
        ]
        
        mappings = [
            KeywordMapping(
                keyword=f"keyword{i}",
                matched_bullets=[sample_bullets[i]],
                similarity_scores=[0.9],
                priority_score=0.9,
            )
            for i in range(10)
        ]
        
        with patch.object(generator, '_rewrite_bullet_for_keyword') as mock_rewrite:
            mock_bullet = TailoredBullet(
                text="Test bullet",
                similarity_score=0.85,
                evidence_spans=["original"],
                jd_keywords_covered=["test"],
            )
            mock_rewrite.return_value = mock_bullet
            
            tailored_bullets = generator.generate_tailored_bullets(mappings, max_bullets=5)
            
            assert len(tailored_bullets) <= 5
    
    def test_generate_tailored_bullets_error_handling(self, generator):
        """Test error handling in bullet generation."""
        mappings = [Mock()]
        
        with patch.object(generator, '_rewrite_bullet_for_keyword', side_effect=Exception("Rewrite error")):
            with pytest.raises(ResumeGenerationError, match="Failed to generate tailored bullets"):
                generator.generate_tailored_bullets(mappings)
    
    def test_calculate_keyword_bullet_similarity(self, generator):
        """Test keyword-bullet similarity calculation."""
        # Mock nlp similarity method to return predictable results
        mock_doc = Mock()
        mock_doc.similarity.return_value = 0.85
        generator.nlp.return_value = mock_doc
        
        # Direct match
        similarity = generator._calculate_keyword_bullet_similarity("python", "Developed Python applications")
        assert similarity >= 0.8
        
        # Mock for no match scenario
        mock_doc.similarity.return_value = 0.2
        similarity = generator._calculate_keyword_bullet_similarity("java", "Developed Python applications")
        assert similarity < 0.8
    
    def test_rewrite_bullet_for_keyword_success(self, generator):
        """Test successful bullet rewriting."""
        original_bullet = ResumeBullet(
            text="Developed web applications using Python",
            section="Experience",
            start_offset=0,
            end_offset=40,
        )
        
        # Mock nlp to support subscriptable operations used in _optimize_bullet_structure
        mock_token = Mock()
        mock_token.pos_ = "VERB"
        mock_token.lemma_ = "developed"
        mock_token.text = "Developed"
        
        mock_doc = Mock()
        mock_doc.__bool__ = Mock(return_value=True)
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_token)
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        
        generator.nlp.return_value = mock_doc
        
        with patch.object(generator, '_calculate_text_similarity', return_value=0.85):
            result = generator._rewrite_bullet_for_keyword(original_bullet, "microservices", 0.9)
            
            assert result is not None
            assert isinstance(result, TailoredBullet)
            assert result.similarity_score >= generator.similarity_threshold
            assert "microservices" in result.jd_keywords_covered
    
    def test_rewrite_bullet_for_keyword_low_similarity(self, generator):
        """Test bullet rewriting when similarity is too low."""
        original_bullet = ResumeBullet(
            text="Developed web applications",
            section="Experience",
            start_offset=0,
            end_offset=25,
        )
        
        with patch.object(generator, '_calculate_text_similarity', return_value=0.7):
            result = generator._rewrite_bullet_for_keyword(original_bullet, "test", 0.8)
            
            assert result is None
    
    def test_inject_keyword(self, generator):
        """Test keyword injection strategies."""
        # Keyword already present
        text = "Developed Python applications"
        result = generator._inject_keyword(text, "python")
        assert result == text
        
        # Keyword not present - should be injected
        text = "Developed web applications"
        result = generator._inject_keyword(text, "microservices")
        assert "microservices" in result.lower()
        assert len(result) > len(text)
    
    def test_enhance_with_technical_terms(self, generator):
        """Test technical term enhancement."""
        text = "Developed Python applications"
        result = generator._enhance_with_technical_terms(text, "python")
        
        # Should return enhanced text (may be same if no enhancements apply)
        assert isinstance(result, str)
        assert len(result) >= len(text)
    
    def test_optimize_bullet_structure(self, generator):
        """Test bullet structure optimization."""
        # Mock spaCy doc processing
        mock_token = Mock()
        mock_token.pos_ = "VERB"
        mock_token.lemma_ = "developed"
        mock_token.text = "developed"
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_doc.__bool__ = Mock(return_value=True)
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_token)
        generator.nlp.return_value = mock_doc
        
        # Test capitalization
        text = "developed applications"
        result = generator._optimize_bullet_structure(text)
        assert result[0].isupper()
        
        # Test ending punctuation
        text = "Developed applications"
        result = generator._optimize_bullet_structure(text)
        assert result.endswith(".")
    
    def test_calculate_text_similarity(self, generator):
        """Test text similarity calculation."""
        # Mock nlp similarity to return predictable results
        mock_doc1 = Mock()
        mock_doc2 = Mock()
        mock_doc1.similarity.return_value = 1.0
        generator.nlp.side_effect = [mock_doc1, mock_doc2]
        
        # Identical texts
        similarity = generator._calculate_text_similarity("test text", "test text")
        assert similarity == 1.0
        
        # Different texts
        mock_doc1.similarity.return_value = 0.3
        generator.nlp.side_effect = [mock_doc1, mock_doc2]
        similarity = generator._calculate_text_similarity("test text", "different content")
        assert 0.0 <= similarity < 1.0
    
    def test_generate_diff_summary(self, generator, sample_resume):
        """Test diff summary generation."""
        tailored_bullets = [
            TailoredBullet(
                text="Enhanced Python development with microservices",
                similarity_score=0.85,
                evidence_spans=["Developed Python applications"],
                jd_keywords_covered=["python", "microservices"],
            ),
        ]
        
        diff_summaries = generator.generate_diff_summary(tailored_bullets, sample_resume.bullets)
        
        assert isinstance(diff_summaries, list)
        assert len(diff_summaries) == len(tailored_bullets)
        
        for summary in diff_summaries:
            assert "original_text" in summary
            assert "tailored_text" in summary
            assert "similarity_score" in summary
            assert "keywords_targeted" in summary
    
    def test_calculate_generation_metrics(self, generator):
        """Test generation metrics calculation."""
        job_keywords = ["python", "react", "microservices", "api"]
        tailored_bullets = [
            TailoredBullet(
                text="Python development",
                similarity_score=0.9,
                evidence_spans=["original"],
                jd_keywords_covered=["python"],
            ),
            TailoredBullet(
                text="React components",
                similarity_score=0.85,
                evidence_spans=["original"],
                jd_keywords_covered=["react"],
            ),
        ]
        
        metrics = generator.calculate_generation_metrics(job_keywords, tailored_bullets)
        
        assert isinstance(metrics, GenerationMetrics)
        assert metrics.total_keywords == len(job_keywords)
        assert metrics.covered_keywords == 2
        assert metrics.total_bullets_generated == len(tailored_bullets)
        assert metrics.keyword_coverage_percentage == 50.0  # 2/4 keywords covered
        assert metrics.average_similarity_score == 0.875  # (0.9 + 0.85) / 2
    
    def test_generate_resume_bullets_integration(self, generator, sample_job_posting, sample_resume):
        """Test complete resume bullet generation pipeline."""
        with patch.object(generator, 'extract_keywords_from_job', return_value=["python", "react"]), \
             patch.object(generator, 'map_keywords_to_bullets') as mock_map, \
             patch.object(generator, 'generate_tailored_bullets') as mock_generate:
            
            # Mock keyword mappings
            mock_map.return_value = [Mock()]
            
            # Mock tailored bullets
            mock_bullets = [
                TailoredBullet(
                    text="Python development",
                    similarity_score=0.9,
                    evidence_spans=["original"],
                    jd_keywords_covered=["python"],
                ),
            ]
            mock_generate.return_value = mock_bullets
            
            tailored_bullets, metrics, diff_summaries = generator.generate_resume_bullets(
                sample_job_posting, sample_resume, max_bullets=10
            )
            
            assert isinstance(tailored_bullets, list)
            assert isinstance(metrics, GenerationMetrics)
            assert isinstance(diff_summaries, list)
            
            mock_map.assert_called_once()
            mock_generate.assert_called_once()
    
    def test_generate_resume_bullets_error_handling(self, generator, sample_job_posting, sample_resume):
        """Test error handling in complete generation pipeline."""
        with patch.object(generator, 'extract_keywords_from_job', side_effect=Exception("Extraction error")):
            with pytest.raises(ResumeGenerationError, match="Resume generation failed"):
                generator.generate_resume_bullets(sample_job_posting, sample_resume)


class TestKeywordMapping:
    """Test suite for KeywordMapping dataclass."""
    
    def test_keyword_mapping_creation(self):
        """Test KeywordMapping creation and attributes."""
        bullets = [
            ResumeBullet(text="Test bullet", section="Experience", start_offset=0, end_offset=10),
        ]
        
        mapping = KeywordMapping(
            keyword="test",
            matched_bullets=bullets,
            similarity_scores=[0.9],
            priority_score=0.9,
        )
        
        assert mapping.keyword == "test"
        assert mapping.matched_bullets == bullets
        assert mapping.similarity_scores == [0.9]
        assert mapping.priority_score == 0.9


class TestGenerationMetrics:
    """Test suite for GenerationMetrics dataclass."""
    
    def test_generation_metrics_creation(self):
        """Test GenerationMetrics creation and attributes."""
        metrics = GenerationMetrics(
            total_keywords=10,
            covered_keywords=7,
            total_bullets_generated=15,
            bullets_above_threshold=12,
            average_similarity_score=0.85,
            keyword_coverage_percentage=70.0,
        )
        
        assert metrics.total_keywords == 10
        assert metrics.covered_keywords == 7
        assert metrics.total_bullets_generated == 15
        assert metrics.bullets_above_threshold == 12
        assert metrics.average_similarity_score == 0.85
        assert metrics.keyword_coverage_percentage == 70.0


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_generate_tailored_resume(self, sample_job_posting, sample_resume):
        """Test convenience function for resume generation."""
        with patch('tools.resume_generator.ResumeGenerator') as mock_generator_class:
            mock_generator = Mock()
            mock_generator.generate_resume_bullets.return_value = ([], Mock(), [])
            mock_generator_class.return_value = mock_generator
            
            result = generate_tailored_resume(
                sample_job_posting, sample_resume, similarity_threshold=0.9, max_bullets=15
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 3
            
            mock_generator_class.assert_called_once_with(similarity_threshold=0.9)
            mock_generator.generate_resume_bullets.assert_called_once_with(
                sample_job_posting, sample_resume, 15
            )


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_empty_job_posting(self, generator):
        """Test handling of empty job posting."""
        job_posting = JobPosting(
            title="", company="", text="", keywords=[], requirements=[]
        )
        
        keywords = generator.extract_keywords_from_job(job_posting)
        assert isinstance(keywords, list)  # Should handle gracefully
    
    def test_empty_resume(self, generator):
        """Test handling of empty resume."""
        resume = Resume(raw_text="Empty resume", bullets=[], skills=[], sections=[])
        
        mappings = generator.map_keywords_to_bullets(["test"], resume)
        assert mappings == []
    
    def test_very_short_text(self, generator):
        """Test handling of very short text inputs."""
        similarity = generator._calculate_text_similarity("a", "b")
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_special_characters_in_text(self, generator):
        """Test handling of special characters in text."""
        text = "Developed APIs with special chars: @#$%^&*()"
        result = generator._inject_keyword(text, "microservices")
        assert isinstance(result, str)
        
        # Should handle special characters gracefully
        similarity = generator._calculate_text_similarity(text, "Different text with @#$%")
        assert isinstance(similarity, float)