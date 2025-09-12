"""
Resume validation tool for evidence-based validation of tailored resume bullets.

This tool implements comprehensive validation of generated resume bullets using
semantic similarity search and evidence indexing to ensure all claims are backed
by actual resume content. It prevents fabrication by enforcing strict similarity
thresholds and provides detailed validation reports.

The tool is designed to be used by the ResumeValidatorAgent for orchestration
within the LangGraph workflow.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.schemas.core import (
    TailoredBullet,
    Resume,
    ResumeBullet,
)
from tools.evidence_indexer import EvidenceIndexer, EvidenceMatch, EvidenceIndexingError


logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Status enum for bullet validation results."""
    
    VALID = "valid"
    REJECTED = "rejected"
    NEEDS_EDIT = "needs_edit"
    ERROR = "error"


class ResumeValidationError(Exception):
    """Raised when resume validation encounters unrecoverable errors."""
    pass


@dataclass
class ValidationResult:
    """Individual bullet validation result with evidence and recommendations."""
    
    bullet: TailoredBullet
    status: ValidationStatus
    evidence_matches: List[EvidenceMatch]
    best_similarity_score: float
    confidence_score: float
    validation_notes: List[str]
    recommended_edits: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for a set of bullets."""
    
    total_bullets: int
    valid_bullets: int
    rejected_bullets: int
    needs_edit_bullets: int
    error_bullets: int
    
    overall_evidence_score: float
    evidence_coverage_percentage: float
    
    validation_results: List[ValidationResult]
    summary_recommendations: List[str]
    
    @property
    def validation_success_rate(self) -> float:
        """Calculate percentage of successfully validated bullets."""
        if self.total_bullets == 0:
            return 0.0
        return (self.valid_bullets / self.total_bullets) * 100
    
    @property
    def needs_review_count(self) -> int:
        """Count bullets that need review (rejected + needs_edit)."""
        return self.rejected_bullets + self.needs_edit_bullets


class ResumeValidator:
    """
    Pure tool for validating tailored resume bullets against original evidence.
    
    This tool provides stateless, deterministic validation of generated resume bullets
    by comparing them against original resume content using semantic similarity and
    evidence indexing. It ensures all claims are backed by verifiable evidence.
    
    Features:
    - Evidence-based validation using EvidenceIndexer
    - Configurable similarity thresholds for different validation levels
    - Comprehensive validation reports with detailed feedback
    - Keyword coverage analysis and validation
    - Automatic recommendation generation for failed validations
    - Support for both individual bullet and batch validation
    - Detailed logging and error handling
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_confidence_threshold: float = 0.7,
        evidence_indexer: Optional[EvidenceIndexer] = None,
        require_keyword_evidence: bool = True,
    ):
        """
        Initialize the resume validator with configuration and evidence indexer.
        
        Args:
            similarity_threshold: Minimum similarity score for validation (τ threshold)
            min_confidence_threshold: Minimum confidence for evidence matches
            evidence_indexer: EvidenceIndexer instance for similarity search
            require_keyword_evidence: Whether to require evidence for claimed keywords
        """
        self.similarity_threshold = similarity_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.require_keyword_evidence = require_keyword_evidence
        
        # Initialize or use provided evidence indexer
        self.evidence_indexer = evidence_indexer or EvidenceIndexer(
            similarity_threshold=similarity_threshold
        )
        
        logger.info(
            f"Initialized ResumeValidator with similarity_threshold={similarity_threshold}, "
            f"confidence_threshold={min_confidence_threshold}"
        )
    
    def validate_bullet(
        self,
        bullet: TailoredBullet,
        top_k_evidence: int = 5,
    ) -> ValidationResult:
        """
        Validate a single tailored bullet against evidence.
        
        Args:
            bullet: TailoredBullet to validate
            top_k_evidence: Number of top evidence matches to consider
            
        Returns:
            ValidationResult with evidence and recommendations
            
        Raises:
            ResumeValidationError: If validation fails due to system error
        """
        try:
            logger.debug(f"Validating bullet: '{bullet.text[:50]}...'")
            
            # Find evidence for the bullet text
            evidence_matches = self.evidence_indexer.find_evidence(
                claim_text=bullet.text,
                top_k=top_k_evidence,
                min_similarity=0.5,  # Lower threshold to get more candidates
            )
            
            if not evidence_matches:
                return ValidationResult(
                    bullet=bullet,
                    status=ValidationStatus.REJECTED,
                    evidence_matches=[],
                    best_similarity_score=0.0,
                    confidence_score=0.0,
                    validation_notes=["No evidence found for this bullet"],
                    recommended_edits="Revise bullet to align more closely with resume content",
                )
            
            # Find best evidence match
            best_match = evidence_matches[0]  # Already sorted by similarity
            best_similarity = best_match.similarity_score
            
            # Calculate confidence score based on evidence strength
            confidence_score = self._calculate_confidence_score(evidence_matches)
            
            # Validate keywords coverage if required
            keyword_validation_notes = []
            if self.require_keyword_evidence and bullet.jd_keywords_covered:
                keyword_validation_notes = self._validate_keyword_evidence(
                    bullet, evidence_matches
                )
            
            # Determine validation status
            status, validation_notes, recommended_edits = self._determine_validation_status(
                bullet, best_similarity, confidence_score, keyword_validation_notes
            )
            
            result = ValidationResult(
                bullet=bullet,
                status=status,
                evidence_matches=evidence_matches,
                best_similarity_score=best_similarity,
                confidence_score=confidence_score,
                validation_notes=validation_notes,
                recommended_edits=recommended_edits,
            )
            
            logger.debug(
                f"Bullet validation result: {status.value}, "
                f"similarity={best_similarity:.3f}, confidence={confidence_score:.3f}"
            )
            
            return result
            
        except EvidenceIndexingError as e:
            logger.error(f"Evidence indexing error during validation: {e}")
            return ValidationResult(
                bullet=bullet,
                status=ValidationStatus.ERROR,
                evidence_matches=[],
                best_similarity_score=0.0,
                confidence_score=0.0,
                validation_notes=[f"Evidence indexing error: {str(e)}"],
                recommended_edits=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error during bullet validation: {e}")
            raise ResumeValidationError(f"Bullet validation failed: {str(e)}") from e
    
    def validate_bullets(
        self,
        bullets: List[TailoredBullet],
        resume: Optional[Resume] = None,
    ) -> ValidationReport:
        """
        Validate a list of tailored bullets and generate comprehensive report.
        
        Args:
            bullets: List of TailoredBullet objects to validate
            resume: Optional Resume object for additional context
            
        Returns:
            ValidationReport with comprehensive analysis
            
        Raises:
            ResumeValidationError: If validation fails due to system error
        """
        try:
            logger.info(f"Starting validation of {len(bullets)} bullets")
            
            if not bullets:
                return ValidationReport(
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
            
            # Index resume if provided and not already indexed
            if resume and self.evidence_indexer.collection.count() == 0:
                try:
                    self.evidence_indexer.index_resume(resume)
                    logger.info("Indexed resume for validation")
                except Exception as e:
                    logger.warning(f"Failed to index resume: {e}")
            
            # Validate each bullet
            validation_results = []
            status_counts = {status: 0 for status in ValidationStatus}
            
            for i, bullet in enumerate(bullets):
                try:
                    result = self.validate_bullet(bullet)
                    validation_results.append(result)
                    status_counts[result.status] += 1
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Validated {i + 1}/{len(bullets)} bullets")
                        
                except Exception as e:
                    logger.error(f"Failed to validate bullet {i}: {e}")
                    error_result = ValidationResult(
                        bullet=bullet,
                        status=ValidationStatus.ERROR,
                        evidence_matches=[],
                        best_similarity_score=0.0,
                        confidence_score=0.0,
                        validation_notes=[f"Validation error: {str(e)}"],
                    )
                    validation_results.append(error_result)
                    status_counts[ValidationStatus.ERROR] += 1
            
            # Calculate overall metrics
            overall_evidence_score = self._calculate_overall_evidence_score(validation_results)
            evidence_coverage_pct = self._calculate_evidence_coverage(validation_results)
            
            # Generate summary recommendations
            summary_recommendations = self._generate_summary_recommendations(
                validation_results, status_counts
            )
            
            report = ValidationReport(
                total_bullets=len(bullets),
                valid_bullets=status_counts[ValidationStatus.VALID],
                rejected_bullets=status_counts[ValidationStatus.REJECTED],
                needs_edit_bullets=status_counts[ValidationStatus.NEEDS_EDIT],
                error_bullets=status_counts[ValidationStatus.ERROR],
                overall_evidence_score=overall_evidence_score,
                evidence_coverage_percentage=evidence_coverage_pct,
                validation_results=validation_results,
                summary_recommendations=summary_recommendations,
            )
            
            logger.info(
                f"Validation completed - {report.valid_bullets}/{report.total_bullets} valid "
                f"({report.validation_success_rate:.1f}% success rate)"
            )
            
            return report
            
        except Exception as e:
            raise ResumeValidationError(f"Bullet validation failed: {str(e)}") from e
    
    def _calculate_confidence_score(self, evidence_matches: List[EvidenceMatch]) -> float:
        """
        Calculate confidence score based on evidence match quality.
        
        Args:
            evidence_matches: List of evidence matches
            
        Returns:
            Confidence score between 0 and 1
        """
        if not evidence_matches:
            return 0.0
        
        # Weight by similarity scores and evidence diversity
        similarities = [match.similarity_score for match in evidence_matches]
        
        # Best match weight: 50%
        best_similarity = similarities[0]
        best_score = best_similarity * 0.5
        
        # Consistency weight: 30% (how consistent are multiple matches)
        if len(similarities) > 1:
            consistency = 1 - np.std(similarities)  # Lower std = higher consistency
            consistency_score = max(0, consistency) * 0.3
        else:
            consistency_score = 0.15  # Penalty for single evidence
        
        # Coverage weight: 20% (how many evidence pieces we have)
        coverage_score = min(len(evidence_matches) / 3, 1.0) * 0.2
        
        total_confidence = best_score + consistency_score + coverage_score
        return min(1.0, total_confidence)
    
    def _validate_keyword_evidence(
        self,
        bullet: TailoredBullet,
        evidence_matches: List[EvidenceMatch],
    ) -> List[str]:
        """
        Validate that claimed keywords are supported by evidence.
        
        Args:
            bullet: TailoredBullet with keyword claims
            evidence_matches: Evidence supporting the bullet
            
        Returns:
            List of validation notes about keyword coverage
        """
        notes = []
        
        if not bullet.jd_keywords_covered:
            return notes
        
        # Check if keywords appear in evidence
        evidence_text = " ".join(match.bullet.text.lower() for match in evidence_matches)
        
        unsupported_keywords = []
        for keyword in bullet.jd_keywords_covered:
            keyword_lower = keyword.lower()
            if keyword_lower not in evidence_text:
                # Check for partial matches or related terms
                keyword_words = keyword_lower.split()
                found_words = sum(1 for word in keyword_words if word in evidence_text)
                
                if found_words < len(keyword_words) * 0.5:  # Less than 50% word overlap
                    unsupported_keywords.append(keyword)
        
        if unsupported_keywords:
            notes.append(
                f"Keywords not clearly supported by evidence: {', '.join(unsupported_keywords)}"
            )
        
        return notes
    
    def _determine_validation_status(
        self,
        bullet: TailoredBullet,
        best_similarity: float,
        confidence_score: float,
        keyword_validation_notes: List[str],
    ) -> Tuple[ValidationStatus, List[str], Optional[str]]:
        """
        Determine validation status based on evidence analysis.
        
        Args:
            bullet: TailoredBullet being validated
            best_similarity: Best evidence similarity score
            confidence_score: Overall confidence score
            keyword_validation_notes: Keyword validation issues
            
        Returns:
            Tuple of (status, validation_notes, recommended_edits)
        """
        validation_notes = []
        recommended_edits = None
        
        # Check primary similarity threshold
        if best_similarity < self.similarity_threshold:
            status = ValidationStatus.REJECTED
            validation_notes.append(
                f"Similarity score {best_similarity:.3f} below threshold {self.similarity_threshold}"
            )
            recommended_edits = (
                "Rewrite bullet to align more closely with original resume content. "
                "Consider using more specific details from your actual experience."
            )
        
        # Check confidence threshold
        elif confidence_score < self.min_confidence_threshold:
            status = ValidationStatus.NEEDS_EDIT
            validation_notes.append(
                f"Low confidence score {confidence_score:.3f} (threshold: {self.min_confidence_threshold})"
            )
            recommended_edits = (
                "Consider strengthening the bullet with more specific details "
                "or focusing on well-documented achievements."
            )
        
        # Check keyword evidence issues
        elif keyword_validation_notes:
            status = ValidationStatus.NEEDS_EDIT
            validation_notes.extend(keyword_validation_notes)
            recommended_edits = (
                "Review keyword claims and ensure they are supported by your "
                "actual experience or remove unsupported keywords."
            )
        
        # All checks passed
        else:
            status = ValidationStatus.VALID
            validation_notes.append(
                f"Strong evidence support (similarity: {best_similarity:.3f}, "
                f"confidence: {confidence_score:.3f})"
            )
        
        return status, validation_notes, recommended_edits
    
    def _calculate_overall_evidence_score(self, validation_results: List[ValidationResult]) -> float:
        """
        Calculate overall evidence score across all validation results.
        
        Args:
            validation_results: List of ValidationResult objects
            
        Returns:
            Overall evidence score between 0 and 1
        """
        if not validation_results:
            return 0.0
        
        valid_results = [
            result for result in validation_results
            if result.status != ValidationStatus.ERROR
        ]
        
        if not valid_results:
            return 0.0
        
        # Average of best similarity scores
        similarity_scores = [result.best_similarity_score for result in valid_results]
        return np.mean(similarity_scores)
    
    def _calculate_evidence_coverage(self, validation_results: List[ValidationResult]) -> float:
        """
        Calculate percentage of bullets with adequate evidence coverage.
        
        Args:
            validation_results: List of ValidationResult objects
            
        Returns:
            Evidence coverage percentage
        """
        if not validation_results:
            return 0.0
        
        adequately_covered = sum(
            1 for result in validation_results
            if result.best_similarity_score >= self.similarity_threshold
        )
        
        return (adequately_covered / len(validation_results)) * 100
    
    def _generate_summary_recommendations(
        self,
        validation_results: List[ValidationResult],
        status_counts: Dict[ValidationStatus, int],
    ) -> List[str]:
        """
        Generate summary recommendations based on validation results.
        
        Args:
            validation_results: List of ValidationResult objects
            status_counts: Count of each validation status
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        total_bullets = len(validation_results)
        if total_bullets == 0:
            return recommendations
        
        # Overall quality assessment
        valid_ratio = status_counts[ValidationStatus.VALID] / total_bullets
        if valid_ratio < 0.5:
            recommendations.append(
                f"Low validation success rate ({valid_ratio:.1%}). "
                "Consider using more conservative bullet generation settings."
            )
        elif valid_ratio < 0.8:
            recommendations.append(
                f"Moderate validation success rate ({valid_ratio:.1%}). "
                "Review rejected bullets for potential improvements."
            )
        
        # Specific issues
        if status_counts[ValidationStatus.REJECTED] > 0:
            recommendations.append(
                f"{status_counts[ValidationStatus.REJECTED]} bullets were rejected due to "
                "insufficient evidence. Consider regenerating with higher similarity requirements."
            )
        
        if status_counts[ValidationStatus.NEEDS_EDIT] > 0:
            recommendations.append(
                f"{status_counts[ValidationStatus.NEEDS_EDIT]} bullets need editing. "
                "Review keyword claims and evidence support."
            )
        
        if status_counts[ValidationStatus.ERROR] > 0:
            recommendations.append(
                f"{status_counts[ValidationStatus.ERROR]} bullets had validation errors. "
                "Check system configuration and resume indexing."
            )
        
        # Evidence quality insights
        low_confidence_count = sum(
            1 for result in validation_results
            if result.confidence_score < self.min_confidence_threshold
        )
        
        if low_confidence_count > total_bullets * 0.3:
            recommendations.append(
                "Many bullets have low confidence scores. "
                "Consider focusing on well-documented achievements with clear evidence."
            )
        
        return recommendations
    
    def get_validation_summary(self, report: ValidationReport) -> Dict[str, Any]:
        """
        Get a concise validation summary for reporting.
        
        Args:
            report: ValidationReport to summarize
            
        Returns:
            Dictionary with key validation metrics
        """
        return {
            "total_bullets": report.total_bullets,
            "validation_success_rate": report.validation_success_rate,
            "valid_count": report.valid_bullets,
            "needs_review_count": report.needs_review_count,
            "overall_evidence_score": report.overall_evidence_score,
            "evidence_coverage_percentage": report.evidence_coverage_percentage,
            "top_recommendations": report.summary_recommendations[:3],
            "similarity_threshold_used": self.similarity_threshold,
            "confidence_threshold_used": self.min_confidence_threshold,
        }


# Convenience functions for simple validation
def validate_tailored_bullets(
    bullets: List[TailoredBullet],
    resume: Optional[Resume] = None,
    similarity_threshold: float = 0.8,
    evidence_indexer: Optional[EvidenceIndexer] = None,
) -> ValidationReport:
    """
    Convenience function for validating bullets with default validator.
    
    Args:
        bullets: List of TailoredBullet objects to validate
        resume: Optional Resume object for evidence indexing
        similarity_threshold: Minimum similarity threshold for validation
        evidence_indexer: Optional EvidenceIndexer instance
        
    Returns:
        ValidationReport with validation results
    """
    validator = ResumeValidator(
        similarity_threshold=similarity_threshold,
        evidence_indexer=evidence_indexer,
    )
    return validator.validate_bullets(bullets, resume)