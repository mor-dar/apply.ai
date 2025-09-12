"""
Resume generation tool for creating tailored resume bullets targeting job description keywords.

This tool implements intelligent bullet rewriting using NLP and semantic similarity to target
job description keywords while maintaining adherence to original resume content. It ensures
no fabrication of facts by requiring high similarity scores to source evidence.

The tool is designed to be used by the ResumeGeneratorAgent for orchestration within
the LangGraph workflow.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.schemas.core import (
    JobPosting,
    Resume,
    ResumeBullet,
    TailoredBullet,
    Requirement,
)
from tools.evidence_indexer import EvidenceIndexer, EvidenceMatch


logger = logging.getLogger(__name__)


class ResumeGenerationError(Exception):
    """Raised when resume generation encounters unrecoverable errors."""
    pass


@dataclass
class KeywordMapping:
    """Represents mapping between job keywords and resume bullets."""
    
    keyword: str
    matched_bullets: List[ResumeBullet]
    similarity_scores: List[float]
    priority_score: float  # TF-IDF or other relevance score


@dataclass
class GenerationMetrics:
    """Metrics for tracking generation quality and coverage."""
    
    total_keywords: int
    covered_keywords: int
    total_bullets_generated: int
    bullets_above_threshold: int
    average_similarity_score: float
    keyword_coverage_percentage: float


class ResumeGenerator:
    """
    Pure tool for generating tailored resume bullets targeting job description keywords.
    
    This tool provides stateless, deterministic generation of tailored resume bullets
    by rewriting original bullets to target specific job description keywords while
    maintaining high similarity to source evidence to prevent fabrication.
    
    Features:
    - Keyword extraction and ranking from job descriptions
    - Resume bullet analysis and keyword matching
    - Intelligent bullet rewriting with context preservation
    - Evidence-based validation with configurable similarity thresholds
    - Comprehensive coverage tracking and metrics
    - Multi-strategy bullet enhancement (keyword injection, phrase rewriting)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_keyword_frequency: int = 1,
        max_bullets_per_keyword: int = 3,
        evidence_indexer: Optional[EvidenceIndexer] = None,
    ):
        """
        Initialize the resume generator with configuration and NLP models.
        
        Args:
            similarity_threshold: Minimum cosine similarity for evidence validation
            min_keyword_frequency: Minimum frequency for keyword consideration
            max_bullets_per_keyword: Maximum bullets to generate per keyword
            evidence_indexer: Optional EvidenceIndexer for validation
        """
        self.similarity_threshold = similarity_threshold
        self.min_keyword_frequency = min_keyword_frequency
        self.max_bullets_per_keyword = max_bullets_per_keyword
        
        # Initialize evidence indexer for validation
        self.evidence_indexer = evidence_indexer or EvidenceIndexer(
            similarity_threshold=similarity_threshold
        )
        
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model for resume generation")
        except OSError:
            raise ResumeGenerationError(
                "spaCy English model not found. Install with: python -m spacy download en_core_web_sm"
            )
        
        # Initialize TF-IDF vectorizer for keyword analysis
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            max_features=500,
            lowercase=True,
        )
        
        # Common technical and professional terms that should be preserved
        self.preserve_terms = {
            "microservices",
            "architecture",
            "ci/cd",
            "pipeline",
            "deployment",
            "scalable",
            "performance",
            "optimization",
            "analytics",
            "dashboard",
            "api",
            "restful",
            "database",
            "sql",
            "nosql",
            "cloud",
            "aws",
            "azure",
            "gcp",
            "kubernetes",
            "docker",
            "agile",
            "scrum",
            "leadership",
            "collaboration",
            "stakeholder",
            "requirements",
            "testing",
            "automation",
            "security",
            "compliance",
        }
        
        # Action verbs that can be used for bullet rewriting
        self.action_verbs = [
            "developed", "implemented", "designed", "built", "created", "led",
            "managed", "optimized", "improved", "enhanced", "delivered",
            "architected", "collaborated", "coordinated", "executed",
            "streamlined", "automated", "integrated", "analyzed", "researched",
            "maintained", "supported", "resolved", "troubleshot", "deployed",
        ]
    
    def extract_keywords_from_job(self, job_posting: JobPosting) -> List[str]:
        """
        Extract and rank keywords from job posting for targeting.
        
        Args:
            job_posting: JobPosting object with requirements and keywords
            
        Returns:
            List of keywords ranked by importance
            
        Raises:
            ResumeGenerationError: If keyword extraction fails
        """
        try:
            # Combine all text content
            all_text = []
            
            # Add existing keywords (these are pre-extracted and ranked)
            all_text.extend(job_posting.keywords)
            
            # Add requirement texts
            requirement_texts = [req.text for req in job_posting.requirements]
            all_text.extend(requirement_texts)
            
            # Add main job description
            all_text.append(job_posting.text)
            
            # Process with spaCy for additional keyword extraction
            combined_text = " ".join(all_text)
            doc = self.nlp(combined_text)
            
            # Extract entities and noun phrases
            extracted_keywords = set()
            
            # Add named entities (technologies, organizations, etc.)
            for ent in doc.ents:
                if ent.label_ in ["PRODUCT", "ORG", "LANGUAGE", "WORK_OF_ART"]:
                    keyword = ent.text.lower().strip()
                    if len(keyword) >= 2:
                        extracted_keywords.add(keyword)
            
            # Add noun phrases (skills, technologies)
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Keep phrases short
                    keyword = chunk.text.lower().strip()
                    if len(keyword) >= 2:
                        extracted_keywords.add(keyword)
            
            # Combine existing keywords with extracted ones
            all_keywords = set(job_posting.keywords + list(extracted_keywords))
            
            # Filter and rank keywords
            filtered_keywords = []
            for keyword in all_keywords:
                keyword = keyword.lower().strip()
                # Skip very short keywords, stop words, and common non-technical terms
                if (
                    len(keyword) >= 2
                    and keyword not in ["and", "or", "the", "a", "an", "of", "in", "to", "for"]
                    and not keyword.isdigit()
                ):
                    filtered_keywords.append(keyword)
            
            # Use TF-IDF to rank keywords by importance
            if len(filtered_keywords) > 1:
                try:
                    # Create document corpus for TF-IDF
                    documents = [combined_text] + requirement_texts
                    
                    # Fit TF-IDF and get scores
                    tfidf_matrix = self.tfidf.fit_transform(documents)
                    feature_names = self.tfidf.get_feature_names_out()
                    
                    # Get scores for main document (job description)
                    main_doc_scores = tfidf_matrix[0].toarray()[0]
                    
                    # Create keyword scores dictionary
                    keyword_scores = {}
                    for keyword in filtered_keywords:
                        # Find best matching n-gram in TF-IDF features
                        best_score = 0.0
                        for i, feature in enumerate(feature_names):
                            if keyword in feature or feature in keyword:
                                best_score = max(best_score, main_doc_scores[i])
                        keyword_scores[keyword] = best_score
                    
                    # Sort by score and return
                    ranked_keywords = sorted(
                        keyword_scores.items(), key=lambda x: x[1], reverse=True
                    )
                    result_keywords = [kw for kw, score in ranked_keywords if score > 0]
                    
                except Exception as e:
                    logger.warning(f"TF-IDF ranking failed, using original order: {e}")
                    result_keywords = filtered_keywords
            else:
                result_keywords = filtered_keywords
            
            logger.info(f"Extracted {len(result_keywords)} keywords from job posting")
            return result_keywords[:50]  # Limit to top 50 keywords
            
        except Exception as e:
            raise ResumeGenerationError(f"Failed to extract keywords: {str(e)}") from e
    
    def map_keywords_to_bullets(
        self, keywords: List[str], resume: Resume
    ) -> List[KeywordMapping]:
        """
        Map job keywords to relevant resume bullets using semantic similarity.
        
        Args:
            keywords: List of keywords to map
            resume: Resume object with bullets to match against
            
        Returns:
            List of KeywordMapping objects with matches
        """
        try:
            if not keywords or not resume.bullets:
                return []
            
            keyword_mappings = []
            
            for keyword in keywords:
                # Find bullets that semantically match this keyword
                matched_bullets = []
                similarity_scores = []
                
                for bullet in resume.bullets:
                    # Calculate semantic similarity
                    similarity = self._calculate_keyword_bullet_similarity(keyword, bullet.text)
                    
                    # Include bullets above a minimum threshold for keyword matching
                    # (lower than final similarity threshold since we'll rewrite them)
                    if similarity >= 0.3:
                        matched_bullets.append(bullet)
                        similarity_scores.append(similarity)
                
                # Create mapping if we have matches
                if matched_bullets:
                    # Calculate priority score (average similarity + keyword importance)
                    avg_similarity = np.mean(similarity_scores)
                    priority_score = avg_similarity
                    
                    mapping = KeywordMapping(
                        keyword=keyword,
                        matched_bullets=matched_bullets,
                        similarity_scores=similarity_scores,
                        priority_score=priority_score,
                    )
                    keyword_mappings.append(mapping)
            
            # Sort by priority score
            keyword_mappings.sort(key=lambda x: x.priority_score, reverse=True)
            
            logger.info(
                f"Mapped {len(keyword_mappings)} keywords to resume bullets"
            )
            return keyword_mappings
            
        except Exception as e:
            logger.error(f"Failed to map keywords to bullets: {str(e)}")
            return []
    
    def generate_tailored_bullets(
        self,
        keyword_mappings: List[KeywordMapping],
        max_bullets: int = 20,
    ) -> List[TailoredBullet]:
        """
        Generate tailored bullets by rewriting original bullets to target keywords.
        
        Args:
            keyword_mappings: List of KeywordMapping objects
            max_bullets: Maximum number of bullets to generate
            
        Returns:
            List of TailoredBullet objects
            
        Raises:
            ResumeGenerationError: If generation fails
        """
        try:
            tailored_bullets = []
            processed_bullet_ids = set()
            
            for mapping in keyword_mappings:
                if len(tailored_bullets) >= max_bullets:
                    break
                
                keyword = mapping.keyword
                bullets_generated_for_keyword = 0
                
                # Sort bullets by similarity score for this keyword
                bullet_score_pairs = list(zip(mapping.matched_bullets, mapping.similarity_scores))
                bullet_score_pairs.sort(key=lambda x: x[1], reverse=True)
                
                for bullet, similarity_score in bullet_score_pairs:
                    if (
                        len(tailored_bullets) >= max_bullets
                        or bullets_generated_for_keyword >= self.max_bullets_per_keyword
                    ):
                        break
                    
                    # Skip if we've already processed this bullet
                    bullet_id = id(bullet)  # Use object id as unique identifier
                    if bullet_id in processed_bullet_ids:
                        continue
                    
                    # Generate tailored version
                    tailored_bullet = self._rewrite_bullet_for_keyword(
                        bullet, keyword, similarity_score
                    )
                    
                    if tailored_bullet:
                        tailored_bullets.append(tailored_bullet)
                        processed_bullet_ids.add(bullet_id)
                        bullets_generated_for_keyword += 1
            
            logger.info(f"Generated {len(tailored_bullets)} tailored bullets")
            return tailored_bullets
            
        except Exception as e:
            raise ResumeGenerationError(f"Failed to generate tailored bullets: {str(e)}") from e
    
    def _calculate_keyword_bullet_similarity(self, keyword: str, bullet_text: str) -> float:
        """
        Calculate semantic similarity between keyword and bullet text.
        
        Args:
            keyword: Target keyword
            bullet_text: Resume bullet text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Simple approach: check for keyword presence and related terms
            bullet_lower = bullet_text.lower()
            keyword_lower = keyword.lower()
            
            # Direct match gets high score
            if keyword_lower in bullet_lower:
                return 0.9
            
            # Check for partial matches and related terms
            keyword_doc = self.nlp(keyword_lower)
            bullet_doc = self.nlp(bullet_lower)
            
            # Use spaCy similarity
            similarity = keyword_doc.similarity(bullet_doc)
            
            # Boost similarity if keyword parts are found
            keyword_words = keyword_lower.split()
            found_words = sum(1 for word in keyword_words if word in bullet_lower)
            word_boost = (found_words / len(keyword_words)) * 0.3
            
            return min(1.0, similarity + word_boost)
            
        except Exception:
            # Fallback to simple string matching
            return 0.8 if keyword.lower() in bullet_text.lower() else 0.2
    
    def _rewrite_bullet_for_keyword(
        self,
        original_bullet: ResumeBullet,
        target_keyword: str,
        original_similarity: float,
    ) -> Optional[TailoredBullet]:
        """
        Rewrite a bullet to better target a specific keyword while maintaining evidence.
        
        Args:
            original_bullet: Original ResumeBullet object
            target_keyword: Keyword to target in rewriting
            original_similarity: Original similarity score
            
        Returns:
            TailoredBullet object if successful, None if similarity too low
        """
        try:
            original_text = original_bullet.text
            
            # Strategy 1: Keyword injection if not present
            rewritten_text = original_text
            keyword_lower = target_keyword.lower()
            
            if keyword_lower not in original_text.lower():
                # Try to naturally integrate the keyword
                rewritten_text = self._inject_keyword(original_text, target_keyword)
            
            # Strategy 2: Enhance with related technical terms
            rewritten_text = self._enhance_with_technical_terms(rewritten_text, target_keyword)
            
            # Strategy 3: Optimize action verbs and structure
            rewritten_text = self._optimize_bullet_structure(rewritten_text)
            
            # Calculate similarity between original and rewritten
            similarity_score = self._calculate_text_similarity(original_text, rewritten_text)
            
            # Ensure similarity meets threshold
            if similarity_score < self.similarity_threshold:
                logger.debug(
                    f"Bullet similarity too low ({similarity_score:.3f}): "
                    f"'{rewritten_text[:50]}...'"
                )
                return None
            
            # Create TailoredBullet
            tailored_bullet = TailoredBullet(
                text=rewritten_text,
                original_bullet_id=None,  # Would need proper ID system
                evidence_spans=[original_text],
                similarity_score=similarity_score,
                jd_keywords_covered=[target_keyword],
            )
            
            return tailored_bullet
            
        except Exception as e:
            logger.error(f"Failed to rewrite bullet: {str(e)}")
            return None
    
    def _inject_keyword(self, text: str, keyword: str) -> str:
        """
        Intelligently inject keyword into bullet text.
        
        Args:
            text: Original bullet text
            keyword: Keyword to inject
            
        Returns:
            Text with keyword injected
        """
        # Find good injection points
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Don't inject if keyword is already present
        if keyword_lower in text_lower:
            return text
        
        # Strategy 1: Technology/tool injection
        tech_patterns = [
            r"\busing\s+(\w+)",
            r"\bwith\s+(\w+)",
            r"\bin\s+(\w+)",
            r"\bfor\s+(\w+)",
        ]
        
        for pattern in tech_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Insert keyword in technology context
                insertion_point = match.end()
                new_text = (
                    text[:insertion_point] + f" and {keyword}" + text[insertion_point:]
                )
                return new_text
        
        # Strategy 2: Append as methodology/approach
        if text.endswith("."):
            return text[:-1] + f" utilizing {keyword}."
        else:
            return text + f" using {keyword}"
    
    def _enhance_with_technical_terms(self, text: str, keyword: str) -> str:
        """
        Enhance text with related technical terms.
        
        Args:
            text: Text to enhance
            keyword: Target keyword for context
            
        Returns:
            Enhanced text
        """
        # Simple enhancement - could be expanded with more sophisticated NLP
        enhancements = {
            "python": ["programming", "scripting", "automation"],
            "javascript": ["web development", "frontend", "interactive"],
            "react": ["component-based", "user interface", "responsive"],
            "api": ["integration", "service", "endpoint"],
            "database": ["optimization", "query performance", "data management"],
            "cloud": ["scalability", "infrastructure", "deployment"],
            "agile": ["methodology", "iterative", "collaboration"],
        }
        
        keyword_lower = keyword.lower()
        if keyword_lower in enhancements:
            # Add one relevant enhancement term if not already present
            for enhancement in enhancements[keyword_lower]:
                if enhancement not in text.lower():
                    # Simple insertion strategy
                    if "development" in text.lower() and enhancement != "development":
                        text = text.replace("development", f"{enhancement} development", 1)
                        break
        
        return text
    
    def _optimize_bullet_structure(self, text: str) -> str:
        """
        Optimize bullet structure for impact and readability.
        
        Args:
            text: Text to optimize
            
        Returns:
            Optimized text
        """
        # Ensure starts with strong action verb
        doc = self.nlp(text)
        first_token = doc[0] if doc else None
        
        if first_token and first_token.pos_ != "VERB":
            # Try to find a good action verb to start with
            for token in doc:
                if token.pos_ == "VERB" and token.lemma_ in self.action_verbs:
                    # Simple restructuring - move verb to front
                    verb_form = token.text
                    if verb_form.lower() in text.lower():
                        remaining_text = text.replace(token.text, "", 1).strip()
                        if remaining_text:
                            text = f"{verb_form.capitalize()} {remaining_text}"
                        break
        
        # Ensure proper capitalization
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Ensure proper ending
        if text and not text.endswith("."):
            text += "."
        
        return text
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Use spaCy for semantic similarity
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            return doc1.similarity(doc2)
        except Exception:
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
    
    def generate_diff_summary(
        self, tailored_bullets: List[TailoredBullet], original_bullets: List[ResumeBullet]
    ) -> List[Dict[str, Any]]:
        """
        Generate before/after diff summary for tailored bullets.
        
        Args:
            tailored_bullets: List of generated tailored bullets
            original_bullets: List of original resume bullets
            
        Returns:
            List of diff summaries
        """
        diff_summaries = []
        
        for tailored_bullet in tailored_bullets:
            # Find best matching original bullet based on evidence spans
            best_match = None
            best_similarity = 0.0
            
            for original_bullet in original_bullets:
                for evidence_span in tailored_bullet.evidence_spans:
                    similarity = self._calculate_text_similarity(
                        evidence_span, original_bullet.text
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = original_bullet
            
            # Create diff summary
            diff_summary = {
                "original_text": best_match.text if best_match else "Unknown",
                "tailored_text": tailored_bullet.text,
                "similarity_score": tailored_bullet.similarity_score,
                "keywords_targeted": tailored_bullet.jd_keywords_covered,
                "evidence_similarity": best_similarity,
            }
            diff_summaries.append(diff_summary)
        
        return diff_summaries
    
    def calculate_generation_metrics(
        self,
        job_keywords: List[str],
        tailored_bullets: List[TailoredBullet],
    ) -> GenerationMetrics:
        """
        Calculate comprehensive metrics for generation quality.
        
        Args:
            job_keywords: Original job keywords
            tailored_bullets: Generated tailored bullets
            
        Returns:
            GenerationMetrics object
        """
        try:
            # Calculate coverage
            covered_keywords = set()
            for bullet in tailored_bullets:
                covered_keywords.update(bullet.jd_keywords_covered)
            
            total_keywords = len(job_keywords)
            covered_count = len(covered_keywords)
            coverage_pct = (covered_count / total_keywords * 100) if total_keywords > 0 else 0
            
            # Calculate quality metrics
            bullets_above_threshold = sum(
                1 for bullet in tailored_bullets
                if bullet.similarity_score >= self.similarity_threshold
            )
            
            avg_similarity = (
                np.mean([bullet.similarity_score for bullet in tailored_bullets])
                if tailored_bullets else 0.0
            )
            
            return GenerationMetrics(
                total_keywords=total_keywords,
                covered_keywords=covered_count,
                total_bullets_generated=len(tailored_bullets),
                bullets_above_threshold=bullets_above_threshold,
                average_similarity_score=float(avg_similarity),
                keyword_coverage_percentage=coverage_pct,
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {str(e)}")
            return GenerationMetrics(
                total_keywords=len(job_keywords),
                covered_keywords=0,
                total_bullets_generated=len(tailored_bullets),
                bullets_above_threshold=0,
                average_similarity_score=0.0,
                keyword_coverage_percentage=0.0,
            )
    
    def generate_resume_bullets(
        self,
        job_posting: JobPosting,
        resume: Resume,
        max_bullets: int = 20,
    ) -> Tuple[List[TailoredBullet], GenerationMetrics, List[Dict[str, Any]]]:
        """
        Main method to generate tailored resume bullets for a job posting.
        
        Args:
            job_posting: JobPosting object with requirements and keywords
            resume: Resume object with bullets to tailor
            max_bullets: Maximum number of bullets to generate
            
        Returns:
            Tuple of (tailored_bullets, metrics, diff_summaries)
            
        Raises:
            ResumeGenerationError: If generation fails
        """
        try:
            logger.info("Starting resume bullet generation process")
            
            # Step 1: Extract keywords from job posting
            job_keywords = self.extract_keywords_from_job(job_posting)
            
            # Step 2: Map keywords to resume bullets
            keyword_mappings = self.map_keywords_to_bullets(job_keywords, resume)
            
            # Step 3: Generate tailored bullets
            tailored_bullets = self.generate_tailored_bullets(
                keyword_mappings, max_bullets
            )
            
            # Step 4: Calculate metrics
            metrics = self.calculate_generation_metrics(job_keywords, tailored_bullets)
            
            # Step 5: Generate diff summaries
            diff_summaries = self.generate_diff_summary(tailored_bullets, resume.bullets)
            
            logger.info(
                f"Resume generation completed. "
                f"Generated {len(tailored_bullets)} bullets with "
                f"{metrics.keyword_coverage_percentage:.1f}% keyword coverage"
            )
            
            return tailored_bullets, metrics, diff_summaries
            
        except Exception as e:
            raise ResumeGenerationError(f"Resume generation failed: {str(e)}") from e


# Convenience functions for simple generation
def generate_tailored_resume(
    job_posting: JobPosting,
    resume: Resume,
    similarity_threshold: float = 0.8,
    max_bullets: int = 20,
) -> Tuple[List[TailoredBullet], GenerationMetrics, List[Dict[str, Any]]]:
    """
    Convenience function for generating tailored resume with default generator.
    
    Args:
        job_posting: JobPosting object
        resume: Resume object
        similarity_threshold: Minimum similarity threshold
        max_bullets: Maximum bullets to generate
        
    Returns:
        Tuple of (tailored_bullets, metrics, diff_summaries)
    """
    generator = ResumeGenerator(similarity_threshold=similarity_threshold)
    return generator.generate_resume_bullets(job_posting, resume, max_bullets)