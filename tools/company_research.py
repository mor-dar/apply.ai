"""
Company research tool for extracting and validating company facts from web sources.

This tool implements web scraping, fact extraction, and validation to compile
FactSheet objects with verified company information. It includes rate limiting,
ethical scraping practices, and source credibility assessment.
"""

import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Comment
import html2text

from src.schemas.core import Fact, FactSheet, SourceDomainClass


# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
REQUEST_TIMEOUT = 10  # seconds
REQUEST_DELAY = 1.0  # seconds between requests (polite crawling)
MAX_RETRIES = 3
MAX_FACT_LENGTH = 500
MAX_FACTS_PER_DOMAIN = 5
RECENCY_CUTOFF_DAYS = 180  # News facts must be within this many days
USER_AGENT = "ApplyAI/1.0 (Company Research Bot; +https://github.com/apply-ai/research)"


class CompanyResearchTool:
    """
    Pure tool for researching company information from web sources.

    This tool provides stateless, deterministic research of companies using
    web scraping and fact extraction. It validates sources, classifies domains,
    and filters facts by recency to produce high-quality FactSheet objects.

    Features:
    - Multi-source scraping (official websites, news outlets)
    - Source domain classification and credibility scoring
    - Fact extraction using NLP patterns and heuristics
    - Rate limiting and robots.txt compliance
    - Recency filtering for time-sensitive information
    """

    def __init__(
        self,
        max_facts: int = 10,
        request_timeout: int = REQUEST_TIMEOUT,
        request_delay: float = REQUEST_DELAY,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize the research tool with configuration.

        Args:
            max_facts: Maximum number of facts to include in FactSheet
            request_timeout: Timeout for HTTP requests in seconds
            request_delay: Delay between requests in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.max_facts = max_facts
        self.request_timeout = request_timeout
        self.request_delay = request_delay
        self.max_retries = max_retries

        # Initialize HTTP session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        # HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = False
        self.html_converter.body_width = 0  # Don't wrap lines

        # Reputable news domains for classification
        self.reputable_news_domains = {
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "ft.com",
            "forbes.com",
            "techcrunch.com",
            "venturebeat.com",
            "businessinsider.com",
            "cnbc.com",
            "cnn.com",
            "bbc.com",
            "guardian.com",
            "nytimes.com",
            "washingtonpost.com",
            "axios.com",
            "politico.com",
            "npr.org",
            "apnews.com",
            "theverge.com",
            "wired.com",
            "ars-technica.com",
        }

        # Company fact extraction patterns
        self.fact_patterns = [
            # Products and services
            (
                r"(?:offers|provides|delivers|specializes in|focuses on|develops|creates|builds)\s+([^.]{10,100})",
                "products",
            ),
            (
                r"(?:platform|solution|service|product|technology|system)\s+(?:that|which|for)\s+([^.]{10,100})",
                "products",
            ),
            # Funding and financial
            (
                r"(?:raised|secured|received|closed)\s+\$([0-9.]+\s*(?:million|billion|M|B))\s+(?:in|of)\s+([^.]{5,50})",
                "funding",
            ),
            (
                r"(?:series\s+[A-Z]|seed|pre-seed)\s+(?:funding|round|investment)\s+(?:of\s+)?\$([0-9.]+\s*(?:million|billion|M|B))",
                "funding",
            ),
            (r"valuation\s+of\s+\$([0-9.]+\s*(?:million|billion|M|B))", "funding"),
            # Company info
            (r"(?:founded|established|started)\s+in\s+([0-9]{4})", "founded"),
            (r"(?:headquartered|based|located)\s+in\s+([^.]{5,50})", "location"),
            (
                r"(?:has|employs|with)\s+(?:over\s+|more than\s+)?([0-9,]+)\s+(?:employees|people|staff)",
                "size",
            ),
            # Mission and values
            (
                r"(?:mission|vision|goal|purpose)\s+(?:is\s+)?(?:to\s+)?([^.]{10,150})",
                "mission",
            ),
            (
                r"(?:believes|committed to|dedicated to|focused on)\s+([^.]{10,100})",
                "values",
            ),
            # Recent news patterns (more specific)
            (
                r"(?:announced|launched|released|unveiled|introduced)\s+([^.]{10,100})",
                "news",
            ),
            (
                r"(?:partnership|collaboration|acquisition|merger)\s+with\s+([^.]{5,80})",
                "news",
            ),
            (r"(?:expands|expansion)\s+(?:into|to)\s+([^.]{5,80})", "news"),
        ]

    def research_company(self, company_name: str) -> FactSheet:
        """
        Research a company and compile a FactSheet with validated facts.

        This is the main entry point for the tool. It orchestrates the complete
        research workflow: source discovery, scraping, fact extraction, validation,
        and compilation into a structured FactSheet.

        Args:
            company_name: Name of the company to research

        Returns:
            FactSheet with verified facts about the company

        Raises:
            ValueError: If company_name is invalid
            RuntimeError: If research fails completely
        """
        if not isinstance(company_name, str) or not company_name.strip():
            raise ValueError("company_name must be a non-empty string")

        company_name = company_name.strip()
        logger.info(f"Starting research for company: {company_name}")

        # Discover research sources
        sources = self._discover_sources(company_name)
        logger.info(f"Discovered {len(sources)} sources to research")

        # Extract facts from all sources
        all_facts = []
        for source_url, domain_class in sources:
            try:
                facts = self._extract_facts_from_source(
                    source_url, domain_class, company_name
                )
                all_facts.extend(facts)
                logger.info(f"Extracted {len(facts)} facts from {source_url}")

                # Polite crawling delay
                time.sleep(self.request_delay)

            except Exception as e:
                logger.warning(f"Failed to extract facts from {source_url}: {str(e)}")
                continue

        # Validate and filter facts
        validated_facts = self._validate_facts(all_facts)

        # Deduplicate and prioritize facts
        final_facts = self._deduplicate_and_prioritize_facts(validated_facts)

        # Limit to max_facts
        final_facts = final_facts[: self.max_facts]

        logger.info(f"Research completed for {company_name}: {len(final_facts)} facts")

        return FactSheet(
            company=company_name,
            facts=final_facts,
            generated_at=datetime.now(timezone.utc),
        )

    def _discover_sources(
        self, company_name: str
    ) -> List[Tuple[str, SourceDomainClass]]:
        """
        Discover relevant web sources for company research.

        Uses search strategies to find official company websites and
        reputable news sources with information about the company.

        Args:
            company_name: Company name to search for

        Returns:
            List of (url, domain_class) tuples for research
        """
        sources = []

        # Strategy 1: Try common company website patterns
        company_slug = self._create_company_slug(company_name)
        potential_domains = [
            f"{company_slug}.com",
            f"{company_slug}.ai",
            f"{company_slug}.io",
            f"{company_slug}.co",
        ]

        for domain in potential_domains:
            url = f"https://{domain}"
            if self._is_valid_company_site(url, company_name):
                sources.append((url, SourceDomainClass.OFFICIAL))
                # Also try about page
                about_url = urljoin(url, "/about")
                if self._url_exists(about_url):
                    sources.append((about_url, SourceDomainClass.OFFICIAL))
                break  # Only use one official domain

        # Strategy 2: Search for news articles (simulated - in production could use search APIs)
        # For now, we'll try some known tech news sites with company searches
        news_searches = self._generate_news_search_urls(company_name)
        for search_url, domain_class in news_searches:
            sources.append((search_url, domain_class))

        return sources[:8]  # Limit total sources to avoid being blocked

    def _create_company_slug(self, company_name: str) -> str:
        """Create a URL-friendly slug from company name."""
        # Remove common business suffixes
        name = re.sub(
            r"\b(?:inc|corp|corporation|llc|ltd|limited|co|company)\b",
            "",
            company_name,
            flags=re.IGNORECASE,
        )

        # Convert to lowercase and replace spaces/punctuation with hyphens
        slug = re.sub(r"[^\w\s-]", "", name.strip()).lower()
        slug = re.sub(r"[-\s]+", "-", slug).strip("-")

        return slug

    def _is_valid_company_site(self, url: str, company_name: str) -> bool:
        """
        Check if a URL is a valid company website.

        Makes a lightweight request to verify the site exists and
        contains relevant company information.
        """
        try:
            response = self._make_request(url)
            if not response or response.status_code != 200:
                return False

            # Check if company name appears in page content or title
            soup = BeautifulSoup(response.content, "html.parser")
            page_text = soup.get_text().lower()
            title = soup.title.string.lower() if soup.title else ""

            company_lower = company_name.lower()
            return (
                company_lower in page_text[:2000]  # Check first 2000 chars
                or company_lower in title
            )

        except Exception:
            return False

    def _url_exists(self, url: str) -> bool:
        """Check if URL exists with a HEAD request."""
        try:
            response = self.session.head(url, timeout=self.request_timeout)
            return response.status_code == 200
        except Exception:
            return False

    def _generate_news_search_urls(
        self, company_name: str
    ) -> List[Tuple[str, SourceDomainClass]]:
        """
        Generate news search URLs for the company.

        In a production system, this would integrate with search APIs.
        For now, we'll create some representative URLs.
        """
        # This is a simplified implementation
        # In production, you'd integrate with Google News API, Bing News API, etc.
        return []  # Returning empty for now to focus on official site research

    def _extract_facts_from_source(
        self, url: str, domain_class: SourceDomainClass, company_name: str
    ) -> List[Fact]:
        """
        Extract facts from a single web source.

        Downloads the page content, extracts relevant text, and identifies
        factual statements about the company using pattern matching.

        Args:
            url: URL to scrape
            domain_class: Classification of the source domain
            company_name: Company name for context filtering

        Returns:
            List of extracted Fact objects
        """
        # Check robots.txt compliance
        if not self._check_robots_txt(url):
            logger.warning(f"Robots.txt disallows scraping {url}")
            return []

        # Download page content
        response = self._make_request(url)
        if not response:
            return []

        # Parse HTML and extract clean text
        soup = BeautifulSoup(response.content, "html.parser")
        clean_text = self._extract_clean_text(soup, url)

        # Extract facts using pattern matching
        facts = []
        for pattern, category in self.fact_patterns:
            matches = re.findall(pattern, clean_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    fact_text = " ".join(str(m) for m in match)
                else:
                    fact_text = str(match)

                fact_text = self._clean_fact_text(fact_text)

                if self._is_valid_fact(fact_text, company_name, category):
                    # Determine recency for news facts
                    as_of_date = self._extract_publication_date(soup, category)

                    # Skip news facts that are too old
                    if category == "news" and self._is_fact_too_old(as_of_date):
                        continue

                    confidence = self._calculate_fact_confidence(
                        fact_text, domain_class, category, url
                    )

                    fact = Fact(
                        statement=fact_text,
                        source_url=url,
                        source_domain_class=domain_class,
                        as_of_date=as_of_date,
                        confidence=confidence,
                    )
                    facts.append(fact)

        # Also try to extract general company information
        company_facts = self._extract_general_company_info(
            clean_text, url, domain_class
        )
        facts.extend(company_facts)

        return facts[:MAX_FACTS_PER_DOMAIN]  # Limit facts per source

    def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            # If we can't check robots.txt, assume it's allowed
            return True

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make a robust HTTP request with retries and error handling.

        Args:
            url: URL to request

        Returns:
            Response object or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.request_timeout)
                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request attempt {attempt + 1} failed for {url}: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(f"All retry attempts failed for {url}")
                    return None

        return None

    def _extract_clean_text(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract clean, readable text from HTML soup.

        Removes navigation, ads, and other boilerplate content to focus
        on the main textual content relevant for fact extraction.
        """
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()

        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Focus on main content areas
        main_content = None
        for selector in [
            "main",
            '[role="main"]',
            ".main-content",
            "#content",
            ".content",
            "article",
        ]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content:
            text = self.html_converter.handle(str(main_content))
        else:
            text = self.html_converter.handle(str(soup))

        # Clean up the text
        text = re.sub(r"\n\s*\n", "\n", text)  # Remove excessive newlines
        text = re.sub(r"[ \t]+", " ", text)  # Normalize whitespace
        text = text.strip()

        return text[:10000]  # Limit text length for processing

    def _clean_fact_text(self, fact_text: str) -> str:
        """Clean and normalize extracted fact text."""
        # Remove excessive whitespace
        fact_text = re.sub(r"\s+", " ", fact_text.strip())

        # Remove markdown artifacts
        fact_text = re.sub(r"[*_`#]+", "", fact_text)

        # Remove incomplete sentences (those ending with commas, etc.)
        fact_text = re.sub(r"[,;]\s*$", "", fact_text)

        # Ensure proper sentence ending
        if fact_text and not fact_text.endswith((".", "!", "?")):
            fact_text += "."

        return fact_text[:MAX_FACT_LENGTH]  # Enforce length limit

    def _is_valid_fact(self, fact_text: str, company_name: str, category: str) -> bool:
        """
        Validate that extracted text constitutes a meaningful fact.

        Checks length, content quality, and relevance to the company.
        """
        if not fact_text or len(fact_text) < 20:
            return False

        if len(fact_text) > MAX_FACT_LENGTH:
            return False

        # Must contain some meaningful content words
        content_words = re.findall(r"\b\w{3,}\b", fact_text.lower())
        if len(content_words) < 3:
            return False

        # Avoid generic or useless statements
        generic_patterns = [
            r"click here",
            r"read more",
            r"learn more",
            r"find out",
            r"contact us",
            r"get in touch",
            r"privacy policy",
            r"terms of service",
            r"all rights reserved",
        ]

        if any(
            re.search(pattern, fact_text, re.IGNORECASE) for pattern in generic_patterns
        ):
            return False

        return True

    def _extract_publication_date(self, soup: BeautifulSoup, category: str) -> datetime:
        """
        Extract publication date from page metadata.

        For news facts, tries to determine when the information was published.
        For other facts, uses current timestamp.
        """
        if category != "news":
            return datetime.now(timezone.utc)

        # Try to find publication date in meta tags
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publish-date"]',
            'meta[name="date"]',
            "time[datetime]",
            ".date",
            ".publish-date",
            ".timestamp",
        ]

        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = (
                    element.get("content")
                    or element.get("datetime")
                    or element.get_text()
                )
                if date_str:
                    parsed_date = self._parse_date_string(date_str)
                    if parsed_date:
                        return parsed_date

        # Default to current time if no date found
        return datetime.now(timezone.utc)

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats."""
        # Common date formats to try
        date_formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue

        return None

    def _is_fact_too_old(self, as_of_date: datetime) -> bool:
        """Check if a news fact is too old based on recency cutoff."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=RECENCY_CUTOFF_DAYS)
        return as_of_date < cutoff_date

    def _calculate_fact_confidence(
        self, fact_text: str, domain_class: SourceDomainClass, category: str, url: str
    ) -> float:
        """
        Calculate confidence score for a fact based on multiple factors.

        Considers source credibility, fact quality, and content specificity.
        """
        confidence = 0.5  # Base confidence

        # Source domain boost
        if domain_class == SourceDomainClass.OFFICIAL:
            confidence += 0.3
        elif domain_class == SourceDomainClass.REPUTABLE_NEWS:
            confidence += 0.2

        # Content quality factors
        if len(fact_text) > 50:  # More detailed facts
            confidence += 0.1

        if re.search(r"\d", fact_text):  # Contains specific numbers/dates
            confidence += 0.1

        # Category-specific adjustments
        if category in ["funding", "founded", "size"]:  # Quantifiable facts
            confidence += 0.1

        # URL quality indicators
        if "/about" in url or "/company" in url:
            confidence += 0.1

        return min(1.0, confidence)  # Cap at 1.0

    def _extract_general_company_info(
        self, text: str, url: str, domain_class: SourceDomainClass
    ) -> List[Fact]:
        """
        Extract general company information not covered by specific patterns.

        Uses broader heuristics to identify company descriptions, achievements, etc.
        """
        facts = []

        # Look for company description paragraphs
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            if len(paragraph) > 100 and len(paragraph) < 300:
                # Check if it's describing the company
                if any(
                    word in paragraph.lower()
                    for word in ["company", "business", "organization", "firm"]
                ):
                    if self._is_valid_fact(paragraph, "", "description"):
                        fact = Fact(
                            statement=self._clean_fact_text(paragraph),
                            source_url=url,
                            source_domain_class=domain_class,
                            as_of_date=datetime.now(timezone.utc),
                            confidence=self._calculate_fact_confidence(
                                paragraph, domain_class, "description", url
                            ),
                        )
                        facts.append(fact)
                        break  # Only take one general description

        return facts

    def _validate_facts(self, facts: List[Fact]) -> List[Fact]:
        """
        Validate facts using domain whitelisting and quality checks.

        Applies validation rules specified in the requirements to ensure
        fact quality and source credibility.
        """
        validated_facts = []

        for fact in facts:
            # Check source domain classification
            parsed_url = urlparse(str(fact.source_url))
            domain = parsed_url.netloc.lower().lstrip("www.")

            # Validate domain classification
            if fact.source_domain_class == SourceDomainClass.OFFICIAL:
                # For official sites, basic validation is sufficient
                pass
            elif fact.source_domain_class == SourceDomainClass.REPUTABLE_NEWS:
                # Verify domain is actually in our reputable list
                if not any(
                    known_domain in domain
                    for known_domain in self.reputable_news_domains
                ):
                    # Downgrade to OTHER if not actually reputable
                    fact.source_domain_class = SourceDomainClass.OTHER

            # Apply minimum confidence threshold
            if fact.confidence < 0.3:
                logger.debug(f"Dropping low-confidence fact: {fact.statement[:50]}...")
                continue

            # Validate required fields
            if not fact.source_url or not fact.as_of_date:
                logger.debug(
                    f"Dropping fact with missing required fields: {fact.statement[:50]}..."
                )
                continue

            validated_facts.append(fact)

        return validated_facts

    def _deduplicate_and_prioritize_facts(self, facts: List[Fact]) -> List[Fact]:
        """
        Remove duplicate facts and prioritize by relevance and confidence.

        Uses similarity matching to identify duplicates and ranking
        to select the best facts for the final FactSheet.
        """
        if not facts:
            return []

        # Simple deduplication based on statement similarity
        unique_facts = []
        seen_statements = set()

        for fact in facts:
            # Normalize statement for comparison
            normalized = re.sub(r"\W+", " ", fact.statement.lower()).strip()

            # Check for substantial overlap with existing facts
            is_duplicate = False
            for seen_statement in seen_statements:
                if self._are_facts_similar(normalized, seen_statement):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_facts.append(fact)
                seen_statements.add(normalized)

        # Sort by priority: official sources first, then by confidence
        def priority_key(fact: Fact) -> Tuple[int, float]:
            domain_priority = 0
            if fact.source_domain_class == SourceDomainClass.OFFICIAL:
                domain_priority = 2
            elif fact.source_domain_class == SourceDomainClass.REPUTABLE_NEWS:
                domain_priority = 1

            return (domain_priority, fact.confidence)

        unique_facts.sort(key=priority_key, reverse=True)

        return unique_facts

    def _are_facts_similar(self, statement1: str, statement2: str) -> bool:
        """Check if two fact statements are substantially similar."""
        # Simple word overlap similarity
        words1 = set(statement1.split())
        words2 = set(statement2.split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        # Consider similar if >70% word overlap
        similarity = len(intersection) / len(union)
        return similarity > 0.7

    def _classify_source_domain(self, url: str) -> SourceDomainClass:
        """
        Classify the source domain for credibility assessment.

        Determines whether a URL belongs to an official company site,
        reputable news outlet, or other source.
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower().lstrip("www.")

        # Check against reputable news domains
        for news_domain in self.reputable_news_domains:
            if news_domain in domain:
                return SourceDomainClass.REPUTABLE_NEWS

        # Simple heuristic for official sites (would be enhanced in production)
        if any(indicator in domain for indicator in [".com", ".ai", ".io", ".co"]):
            # Additional checks could be added here
            return SourceDomainClass.OFFICIAL

        return SourceDomainClass.OTHER
