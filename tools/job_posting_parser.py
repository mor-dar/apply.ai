"""
Job posting parser tool for extracting requirements and keywords from job descriptions.

This is a pure NLP processing tool that implements deterministic, stateless parsing
of job descriptions. It uses spaCy for natural language processing and TF-IDF for
keyword ranking to extract structured data for JobPosting objects.

This tool is designed to be used by the JobPostParserAgent for orchestration
within the LangGraph workflow.
"""

import re
from typing import List, Set

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.schemas.core import JobPosting, Requirement, ParserReport


class JobPostingParser:
    """
    Pure tool for parsing job descriptions into structured data.

    This tool provides stateless, deterministic parsing of job descriptions
    using NLP techniques. It extracts requirements, keywords, and metadata
    to populate JobPosting schema objects.

    Features:
    - Multi-strategy keyword extraction (entities, noun phrases, technical terms)
    - TF-IDF based keyword ranking with technical term boosting
    - Requirement sentence detection using pattern matching
    - Configurable extraction limits and filtering
    """

    def __init__(self, max_keywords: int = 30, min_keyword_length: int = 2):
        """
        Initialize the parser with NLP models and configuration.

        Args:
            max_keywords: Maximum number of keywords to extract (default: 30)
            min_keyword_length: Minimum character length for keywords (default: 2)
        """
        self.max_keywords = max_keywords
        self.min_keyword_length = min_keyword_length

        # Load spaCy model - using English model for entity recognition and POS tagging
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy English model not found. Install with: python -m spacy download en_core_web_sm"
            )

        # Initialize TF-IDF vectorizer for keyword ranking
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),  # Extract 1-3 word phrases
            max_features=1000,
            lowercase=True,
        )

        # Common job requirement keywords and patterns
        self.requirement_indicators = {
            "education": [
                "degree",
                "bachelor",
                "master",
                "phd",
                "education",
                "university",
                "college",
            ],
            "experience": ["years", "experience", "background", "proven track record"],
            "skills": [
                "knowledge of",
                "proficiency in",
                "experience with",
                "skilled in",
                "familiarity with",
            ],
            "certification": ["certification", "certified", "license", "accreditation"],
        }

        # Technology and skill-related terms (expanded list)
        self.tech_keywords = {
            "programming",
            "software",
            "development",
            "engineering",
            "python",
            "java",
            "javascript",
            "react",
            "angular",
            "vue",
            "node",
            "sql",
            "database",
            "aws",
            "cloud",
            "docker",
            "kubernetes",
            "machine learning",
            "ai",
            "data science",
            "analytics",
            "api",
            "rest",
            "graphql",
            "microservices",
            "devops",
            "ci/cd",
            "git",
            "testing",
            "automation",
        }

    def parse(
        self, job_text: str, title: str = "", company: str = "", location: str = ""
    ) -> tuple[JobPosting, ParserReport]:
        """
        Parse a job description into structured JobPosting object with quality report.

        This is the main entry point for the tool. It processes the raw job
        description text and returns a structured JobPosting object along with
        a ParserReport containing confidence metrics and quality indicators.

        Args:
            job_text: Raw job description text
            title: Job title (optional, will be auto-extracted if not provided)
            company: Company name (optional, will be auto-extracted if not provided)
            location: Job location (optional, will be auto-extracted if not provided)

        Returns:
            Tuple of (JobPosting, ParserReport) with extracted data and quality metrics

        Raises:
            ValueError: If job_text is invalid or processing fails
        """
        if not isinstance(job_text, str):
            raise ValueError("job_text must be a string")

        # Auto-extract basic info from original text (before cleaning removes line structure)
        if not title or not title.strip():
            title = self._extract_job_title(job_text)
        if not company or not company.strip():
            company = self._extract_company_name(job_text)
        if not location or not location.strip():
            location = self._extract_location(job_text)

        # Clean and preprocess text for keyword/requirement extraction
        cleaned_text = self._clean_text(job_text)

        # Extract requirements and keywords
        requirements = self._extract_structured_requirements(cleaned_text)
        keywords = self._extract_keywords(cleaned_text)

        # Handle None and empty string cases
        clean_title = title.strip() if title and title.strip() else "Unknown Position"
        clean_company = (
            company.strip() if company and company.strip() else "Unknown Company"
        )
        clean_location = location.strip() if location and location.strip() else None

        # Create JobPosting
        job_posting = JobPosting(
            title=clean_title,
            company=clean_company,
            location=clean_location,
            text=job_text,
            keywords=keywords,
            requirements=requirements,
        )

        # Generate quality report
        report = self._generate_parser_report(job_posting, cleaned_text, title, company)

        return job_posting, report

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize job description text.

        Removes common job posting artifacts, normalizes whitespace,
        and prepares text for NLP processing.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text string
        """
        # Remove excessive whitespace and normalize
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Remove common job posting artifacts
        text = re.sub(
            r"Apply now|Submit your resume|Equal opportunity employer.*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Final cleanup - remove extra spaces and strip again
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_requirements(self, text: str) -> List[str]:
        """
        Extract job requirements using NLP and pattern matching.

        Identifies sentences that contain job requirements by looking for
        specific patterns and requirement indicators.

        Args:
            text: Cleaned job description text

        Returns:
            List of extracted requirement sentences
        """
        doc = self.nlp(text)
        requirements = []

        # Split text into sentences for requirement extraction
        sentences = [sent.text.strip() for sent in doc.sents]

        for sentence in sentences:
            # Look for requirement indicators
            lower_sentence = sentence.lower()

            # Check for requirement patterns
            for indicators in self.requirement_indicators.values():
                for indicator in indicators:
                    if indicator in lower_sentence:
                        # Extract the full requirement sentence
                        if (
                            len(sentence) > 20 and len(sentence) < 200
                        ):  # Reasonable length
                            requirements.append(sentence.strip())
                            break

            # Look for specific requirement patterns
            if self._is_requirement_sentence(lower_sentence):
                if len(sentence) > 15 and len(sentence) < 250:
                    requirements.append(sentence.strip())

        # Deduplicate and return top requirements
        seen = set()
        unique_requirements = []
        for req in requirements:
            req_lower = req.lower()
            if req_lower not in seen and len(req.split()) >= 3:
                seen.add(req_lower)
                unique_requirements.append(req)

        return unique_requirements[:15]  # Limit to top 15 requirements

    def _is_requirement_sentence(self, sentence: str) -> bool:
        """
        Check if a sentence contains job requirements based on patterns.

        Uses regex patterns to identify sentences that likely contain
        job requirements or qualifications.

        Args:
            sentence: Sentence to check (lowercase)

        Returns:
            True if sentence appears to contain requirements
        """
        requirement_patterns = [
            r"\b(must|should|required|preferred|ideal|looking for|seeking)\b",
            r"\b\d+\+?\s+years?\b",  # "3+ years", "5 years"
            r"\bbachelor|master|phd|degree\b",
            r"\bexperience (with|in)\b",
            r"\bproficient (in|with)\b",
            r"\bknowledge of\b",
            r"\bfamiliarity with\b",
        ]

        return any(re.search(pattern, sentence) for pattern in requirement_patterns)

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract and rank keywords using multiple NLP strategies.

        Combines named entity recognition, noun phrase extraction,
        technical term matching, and TF-IDF ranking to identify
        the most relevant keywords from the job description.

        Args:
            text: Cleaned job description text

        Returns:
            List of top-ranked keywords (up to max_keywords limit)
        """
        # Process text with spaCy
        doc = self.nlp(text)

        # Extract candidate keywords using multiple strategies
        candidates = set()

        # Strategy 1: Named entities (companies, technologies, etc.)
        for ent in doc.ents:
            if (
                ent.label_ in ["ORG", "PRODUCT", "LANGUAGE", "PERSON", "GPE"]
                and len(ent.text) >= self.min_keyword_length
            ):
                candidates.add(ent.text.lower().strip())

        # Strategy 2: Noun phrases
        for chunk in doc.noun_chunks:
            # Filter noun phrases by length and POS patterns
            if (
                len(chunk.text) >= self.min_keyword_length
                and len(chunk.text.split()) <= 4  # Not too long
                and self._is_valid_noun_phrase(chunk)
            ):
                candidates.add(chunk.text.lower().strip())

        # Strategy 3: Technology and skill keywords
        words_in_text = set(token.text.lower() for token in doc if token.is_alpha)
        tech_matches = words_in_text.intersection(self.tech_keywords)
        candidates.update(tech_matches)

        # Strategy 4: Multi-word technical terms
        candidates.update(self._extract_technical_terms(text.lower()))

        # Filter candidates
        filtered_candidates = []
        for candidate in candidates:
            if (
                len(candidate) >= self.min_keyword_length
                and self._is_valid_keyword(candidate)
                and not self._is_stop_phrase(candidate)
            ):
                filtered_candidates.append(candidate)

        # Rank keywords using TF-IDF
        if not filtered_candidates:
            return []

        ranked_keywords = self._rank_keywords_tfidf(text, filtered_candidates)

        return ranked_keywords[: self.max_keywords]

    def _is_valid_noun_phrase(self, chunk) -> bool:
        """
        Check if a noun phrase is likely to be a meaningful keyword.

        Filters out generic phrases and validates POS patterns
        to ensure the noun phrase is relevant for keyword extraction.

        Args:
            chunk: spaCy noun chunk

        Returns:
            True if the noun phrase is valid for keyword extraction
        """
        # Check POS patterns - should contain nouns/adjectives
        pos_tags = [token.pos_ for token in chunk]

        # Must contain at least one noun
        if "NOUN" not in pos_tags and "PROPN" not in pos_tags:
            return False

        # Avoid phrases that are too generic
        text_lower = chunk.text.lower()
        generic_phrases = {
            "the company",
            "the team",
            "the role",
            "the position",
            "the candidate",
        }

        return text_lower not in generic_phrases

    def _extract_technical_terms(self, text: str) -> Set[str]:
        """
        Extract multi-word technical terms using pattern matching.

        Uses regex patterns to identify common technical phrases
        and terminology that might not be captured by other strategies.

        Args:
            text: Job description text (lowercase)

        Returns:
            Set of technical terms found
        """
        technical_patterns = [
            r"\b(machine learning|artificial intelligence|data science|deep learning)\b",
            r"\b(full stack|front end|back end|web development)\b",
            r"\b(cloud computing|distributed systems|microservices)\b",
            r"\b(agile methodology|scrum|kanban|devops)\b",
            r"\b(object oriented programming|functional programming)\b",
            r"\b(continuous integration|continuous deployment|ci/cd)\b",
            r"\b(version control|source control|git flow)\b",
            r"\b(database design|data modeling|etl)\b",
        ]

        terms = set()
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            terms.update(matches)

        return terms

    def _is_valid_keyword(self, keyword: str) -> bool:
        """
        Check if a keyword candidate meets validation criteria.

        Applies various filters to ensure keywords are meaningful
        and appropriate for job description analysis.

        Args:
            keyword: Candidate keyword string

        Returns:
            True if keyword passes validation
        """
        # Basic filtering rules
        if len(keyword) < self.min_keyword_length:  # Too short
            return False

        if len(keyword.split()) > 4:  # Too long
            return False

        if keyword.isdigit():  # Just numbers
            return False

        if re.match(r"^[^a-zA-Z]*$", keyword):  # No letters
            return False

        return True

    def _is_stop_phrase(self, phrase: str) -> bool:
        """
        Check if phrase is a stop phrase that should be filtered out.

        Identifies common job posting phrases that don't provide
        meaningful information about job requirements or skills.

        Args:
            phrase: Phrase to check

        Returns:
            True if phrase should be filtered out
        """
        stop_phrases = {
            "apply now",
            "equal opportunity",
            "click here",
            "submit resume",
            "job description",
            "we are looking",
            "you will be",
            "the ideal",
            "the successful",
            "this role",
            "this position",
            "our client",
            "the right",
            "we offer",
            "what we offer",
        }

        return phrase.lower() in stop_phrases

    def _rank_keywords_tfidf(self, text: str, candidates: List[str]) -> List[str]:
        """
        Rank keyword candidates using TF-IDF scoring with boosting.

        Calculates term frequency scores for candidates and applies
        boosting factors for technical terms and multi-word phrases.

        Args:
            text: Full job description text
            candidates: List of keyword candidates

        Returns:
            List of keywords ranked by TF-IDF score (descending)
        """
        # Calculate term frequency for each candidate
        tf_scores = {}
        text_lower = text.lower()

        for candidate in candidates:
            # Count occurrences (case-insensitive)
            count = text_lower.count(candidate.lower())
            if count > 0:
                # Simple TF score (could be enhanced with proper TF-IDF)
                tf_scores[candidate] = count

        # Sort by frequency and apply additional scoring factors
        scored_keywords = []
        for keyword, tf_score in tf_scores.items():
            # Boost score for technical terms
            boost = 1.0
            if any(tech in keyword.lower() for tech in self.tech_keywords):
                boost = 1.5

            # Boost multi-word terms slightly
            if len(keyword.split()) > 1:
                boost *= 1.2

            final_score = tf_score * boost
            scored_keywords.append((keyword, final_score))

        # Sort by score (descending) and return keywords
        scored_keywords.sort(key=lambda x: x[1], reverse=True)
        return [keyword for keyword, _ in scored_keywords]

    def _extract_structured_requirements(self, text: str) -> List[Requirement]:
        """
        Extract job requirements as structured Requirement objects.

        This method builds on the original _extract_requirements method but
        returns structured Requirement objects with must_have classification.
        """
        # First extract raw requirement strings using existing logic
        raw_requirements = self._extract_requirements(text)

        structured_requirements = []
        for req_text in raw_requirements:
            # Simple heuristics to classify must_have vs nice_to_have
            must_have = self._classify_requirement(req_text)

            requirement = Requirement(
                text=req_text,
                must_have=must_have,
                rationale=self._get_classification_rationale(req_text, must_have),
            )
            structured_requirements.append(requirement)

        return structured_requirements

    def _classify_requirement(self, req_text: str) -> bool:
        """Classify whether a requirement is must-have vs nice-to-have."""
        req_lower = req_text.lower()

        # Must-have indicators
        must_have_patterns = [
            "required",
            "must have",
            "must be",
            "essential",
            "mandatory",
            "minimum",
            "bachelor",
            "degree",
        ]

        # Nice-to-have indicators
        nice_to_have_patterns = [
            "preferred",
            "nice to have",
            "bonus",
            "plus",
            "would be nice",
            "ideally",
            "a plus",
        ]

        # Check for nice-to-have first (more specific)
        if any(pattern in req_lower for pattern in nice_to_have_patterns):
            return False

        # Check for must-have patterns
        if any(pattern in req_lower for pattern in must_have_patterns):
            return True

        # Default to must-have for unclear requirements
        return True

    def _get_classification_rationale(self, req_text: str, must_have: bool) -> str:
        """Generate rationale for requirement classification."""
        req_lower = req_text.lower()

        if not must_have:
            if "preferred" in req_lower:
                return "Contains 'preferred' indicating nice-to-have"
            elif "bonus" in req_lower or "plus" in req_lower:
                return "Contains bonus/plus language indicating optional"
            else:
                return "Language suggests optional qualification"
        else:
            if "required" in req_lower or "must" in req_lower:
                return "Contains explicit requirement language"
            elif "bachelor" in req_lower or "degree" in req_lower:
                return "Education requirement typically mandatory"
            else:
                return "Classified as core requirement based on context"

    def _generate_parser_report(
        self, job_posting: JobPosting, cleaned_text: str, title: str, company: str
    ) -> ParserReport:
        """Generate a quality report for the parsed job posting."""

        # Calculate confidence score based on multiple factors
        confidence_factors = []
        warnings = []
        missing_fields = []

        # Text length factor (0.0-1.0)
        text_length = len(cleaned_text)
        if text_length < 200:
            confidence_factors.append(0.3)
            warnings.append("Job description appears very short (<200 chars)")
        elif text_length < 500:
            confidence_factors.append(0.6)
            warnings.append("Job description is somewhat short (<500 chars)")
        else:
            confidence_factors.append(1.0)

        # Keyword extraction quality (0.0-1.0)
        keyword_count = len(job_posting.keywords)
        if keyword_count == 0:
            confidence_factors.append(0.0)
            warnings.append("No keywords extracted")
            missing_fields.append("keywords")
        elif keyword_count < 3:
            confidence_factors.append(0.4)
            warnings.append("Very few keywords extracted")
        else:
            confidence_factors.append(
                min(1.0, keyword_count / 10.0)
            )  # Scale to 10 keywords = 1.0

        # Requirements extraction quality (0.0-1.0)
        requirement_count = len(job_posting.requirements)
        if requirement_count == 0:
            confidence_factors.append(0.0)
            warnings.append("No requirements extracted")
            missing_fields.append("requirements")
        elif requirement_count < 2:
            confidence_factors.append(0.5)
            warnings.append("Very few requirements extracted")
        else:
            confidence_factors.append(
                min(1.0, requirement_count / 5.0)
            )  # Scale to 5 requirements = 1.0

        # Title/Company field quality (0.0-1.0)
        field_quality = 1.0
        if not title or title.strip() == "" or job_posting.title == "Unknown Position":
            field_quality -= 0.3
            if "title" not in missing_fields:
                missing_fields.append("title")
                warnings.append("Job title not provided or extracted")

        if (
            not company
            or company.strip() == ""
            or job_posting.company == "Unknown Company"
        ):
            field_quality -= 0.3
            if "company" not in missing_fields:
                missing_fields.append("company")
                warnings.append("Company name not provided or extracted")

        confidence_factors.append(max(0.0, field_quality))

        # Calculate overall confidence as weighted average
        # Weight text length and field quality less than content extraction
        weights = [0.15, 0.35, 0.35, 0.15]  # text, keywords, requirements, fields
        weighted_confidence = sum(f * w for f, w in zip(confidence_factors, weights))

        # Additional warnings based on content analysis
        if text_length > 5000:
            warnings.append(
                "Job description is very long (>5000 chars), may contain noise"
            )

        if keyword_count > 50:
            warnings.append(
                "Unusually high number of keywords extracted, may include noise"
            )

        return ParserReport(
            confidence=round(weighted_confidence, 3),
            missing_fields=missing_fields,
            warnings=warnings,
            keyword_count=keyword_count,
            requirement_count=requirement_count,
            text_length=text_length,
        )

    def _extract_job_title(self, text: str) -> str:
        """Extract job title from job description text."""
        lines = text.split("\n")

        # Pattern 1: Direct labels like "Job Title:", "Position:", "Role:"
        title_patterns = [
            r"(?:job\s+title|position|role|title)\s*[:]\s*(.+)",
            r"we\s+are\s+looking\s+for\s+an?\s+(.+?)\s+to\s+(?:lead|join|manage)",  # "We are looking for an X to..."
            r"looking\s+for\s+an?\s+(.+?)\s+to\s+(?:lead|join|manage)",  # "looking for an X to..."
            r"seeking\s+an?\s+(?:experienced\s+)?(?:and\s+passionate\s+)?(.+?)\s+to\s+(?:join|lead)",  # "seeking an X to join"
            r"(?:we(?:\s+are)?\s+(?:looking\s+for|seeking|hiring)|(?:recruiting\s+for|hiring))(?:\s+(?:a|an))?\s+(.+?)(?:\s+position|\s+to\s+|\s+who\s+|\.|$)",
            r"(?:the\s+role|about\s+the\s+role)\s*[:\n]\s*(.+?)(?:\n|$)",
            r"recruiting\s+for\s+(?:a|an)\s+(.+?)\s+position",  # Specific pattern for "recruiting for a/an X position"
        ]

        for pattern in title_patterns:
            for line in lines[:15]:  # Check first 15 lines
                match = re.search(pattern, line.strip(), re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    # Clean up common noise
                    title = re.sub(
                        r"\s*[-–—]\s*.+$", "", title
                    )  # Remove everything after dash
                    title = re.sub(
                        r"\s*\([^)]*\)$", "", title
                    )  # Remove parentheticals at end
                    if len(title) > 3 and len(title) < 100:  # Reasonable title length
                        return title

        # Pattern 2: First non-empty line as potential title (common format)
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) > 3 and len(line) < 100:
                # Skip lines that look like company names or contact info
                if not re.search(
                    r"@|www\.|http|\.com|Inc\.|Ltd\.|LLC|Corp\.|phone|email|address",
                    line,
                    re.IGNORECASE,
                ):
                    # Check if it looks like a job title
                    if re.search(
                        r"(?:engineer|developer|manager|director|analyst|specialist|coordinator|assistant|lead|senior|junior|head\s+of)",
                        line,
                        re.IGNORECASE,
                    ):
                        # Clean up common noise
                        cleaned_title = re.sub(
                            r"\s*[-–—]\s*.+$", "", line
                        )  # Remove everything after dash
                        cleaned_title = re.sub(
                            r"\s*\([^)]*\)$", "", cleaned_title
                        )  # Remove parentheticals at end
                        return cleaned_title.strip()

        # Pattern 3: Extract from common job posting structures
        role_section_pattern = r"(?:the\s+role|position\s+summary|job\s+summary)[:\s]*\n(.+?)(?:\n\n|\n[A-Z])"
        match = re.search(role_section_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Look for title in the content
            title_match = re.search(
                r"(?:we(?:\s+are)?\s+(?:looking\s+for|seeking|hiring)(?:\s+a)?)\s+(.+?)(?:\s+to\s+|\s+who\s+|\.|$)",
                content,
                re.IGNORECASE,
            )
            if title_match:
                title = title_match.group(1).strip()
                if len(title) > 3 and len(title) < 100:
                    return title

        return ""  # Return empty if no title found

    def _extract_company_name(self, text: str) -> str:
        """Extract company name from job description text."""
        lines = text.split("\n")

        # Pattern 1: "Who is [Company]?" or "[Company] is..." patterns (high priority)
        who_pattern = r"(?:who\s+is\s+|about\s+)([A-Z][a-zA-Z]+)(?:\?|$|\n)"
        match = re.search(who_pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            company = match.group(1).strip()
            if len(company) > 2 and len(company) < 50:
                return company

        # Pattern 1b: "[Company] is..." at start of sentence (high priority)
        company_is_pattern = r"(?:^|\n)\s*([A-Z][a-zA-Z]+)\s+is\s+(?:an?\s+)?(?:award\s+winning|leading|global|startup|start-up|company)"
        match = re.search(company_is_pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            company = match.group(1).strip()
            if len(company) > 2 and len(company) < 50:
                return company

        # Pattern 2: First few lines often contain company name
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) > 2:
                # Skip obvious non-company lines including copyright and About sections
                if not re.search(
                    r"@|phone|email|^looking\s+for|^we\s+are\s+(?:looking|hiring|seeking)|^job\s+title|^position|^role|©|copyright|all rights reserved|^about\s+",
                    line,
                    re.IGNORECASE,
                ):
                    # First priority: Check if this is the first line and extract first word as company
                    if line == lines[0].strip():
                        words = line.split()
                        if words and len(words[0]) > 2:
                            # Check if first word looks like a company name (not a common word)
                            if not re.search(
                                r"^(?:the|and|or|of|in|on|at|to|for|with|by|we|our|this|is|are|who)$",
                                words[0],
                                re.IGNORECASE,
                            ):
                                # Additional check: make sure the first word isn't itself a job title
                                if not re.search(
                                    r"^(?:engineer|developer|manager|director|position|role|job|vacancy|opportunity|senior|junior|lead)$",
                                    words[0],
                                    re.IGNORECASE,
                                ):
                                    return words[0]

                    # Second priority: Look for company indicators and extract company name before them
                    if re.search(
                        r"\b(?:Inc\.|Ltd\.|LLC|Corp\.|Company|Technologies|Systems|Solutions)\b|\bGroup\b(?:\s*Inc\.|$)",
                        line,
                        re.IGNORECASE,
                    ):
                        # Extract just the company name from the line with indicators
                        words = line.split()
                        if words:
                            # Try to extract company name before the indicator
                            for i, word in enumerate(words):
                                if re.search(
                                    r"\b(?:Inc\.|Ltd\.|LLC|Corp\.|Company|Technologies|Systems|Solutions|Group)\b",
                                    word,
                                    re.IGNORECASE,
                                ):
                                    if (
                                        i > 0
                                    ):  # There's a company name before the indicator
                                        company_words = words[:i]
                                        company_name = " ".join(company_words)
                                        if len(company_name) > 2:
                                            return company_name
                            # If we can't find the indicator in a single word, try first few words
                            if len(words) >= 2:
                                return words[0]  # Return first word as fallback

        # Pattern 3: Email domain extraction
        email_pattern = r"[\w\.-]+@([\w\.-]+\.\w+)"
        email_matches = re.findall(email_pattern, text)
        for domain in email_matches:
            if not domain.lower().endswith(
                (".gmail.com", ".yahoo.com", ".outlook.com", ".hotmail.com")
            ):
                # Extract company name from domain
                company = domain.split(".")[0]
                if len(company) > 2:
                    return company.title()

        # Pattern 3: Copyright or footer patterns
        copyright_pattern = (
            r"©.*?\d{4}\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)\s*(?:Inc\.|LLC|Corp\.|Ltd\.)"
        )
        match = re.search(copyright_pattern, text)
        if match:
            company = match.group(1).strip()
            if len(company) > 2 and len(company) < 50:
                return company + " Inc"  # Add back the indicator for completeness

        # Pattern 4: "About [Company]" sections
        about_pattern = r"about\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\n|$)"
        match = re.search(about_pattern, text, re.IGNORECASE)
        if match:
            company = match.group(1).strip()
            if (
                len(company) > 2
                and len(company) < 50
                and not re.search(
                    r"the\s+role|this\s+position|us", company, re.IGNORECASE
                )
            ):
                return company

        return ""  # Return empty if no company found

    def _extract_location(self, text: str) -> str:
        """Extract location from job description text."""
        # Common location patterns (ordered by specificity)
        location_patterns = [
            r"location\s*[:]\s*(.+?)(?:\n|$)",
            r"(?:located\s+in|based\s+in)\s+([A-Z][a-z]+-[A-Z][a-z]+)",  # Hyphenated cities like "Tel-Aviv" (high priority)
            r"position\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z][a-z]+-[A-Z][a-z]+)\.?",  # "position in Location" including hyphenated
            r"(?:group|team)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\.?",  # "group in Location" or "team in Location"
            r"(?:based\s+in|located\s+in)\s+(.+?)(?:\s+with|\s+to|\s+and|\s+in\s+|\.|$)",  # Remove comma, stop at connecting words
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})(?:\s|$)",  # City, State format
            r"(remote|hybrid|on-site)(?:\s+available|\s+work|\s+position)?",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+)(?:\s|$)",  # City, Country format (moved last)
        ]

        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    location = ", ".join(match).strip()
                else:
                    location = match.strip()

                if len(location) > 2 and len(location) < 100:
                    # Clean up common noise
                    location = re.sub(
                        r"\s*\([^)]*\)$", "", location
                    )  # Remove parentheticals
                    return location

        return ""  # Return empty if no location found
