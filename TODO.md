# Two-week, step-by-step implementation plan (deliverable per day)

## [ ] Scope guardrails for 14 days
- [ ] English only
- [ ] US-centric compensation (BLS)
- [ ] Single target job per run
- [ ] Resume ≤ 2 pages
- [ ] Job description ≤ 1,500 words
- [ ] Custom React UI (no auth)
- [ ] DOCX export

---

## Day 1 — Repo + scaffolding
- [x] Create monorepo with folders:
  - [x] `app/` (Streamlit)
  - [x] `agents/`
  - [x] `tools/`
  - [x] `data/`
  - [x] `eval/`
- [x] Pick orchestration (LangGraph `StateGraph`) and define states & edges as code stubs (no logic yet)
- [x] Draft reusable Pydantic schemas:
  - [x] `JobPosting {title, company, location, text, keywords[], requirements[]}`
  - [x] `Resume {raw_text, bullets[], skills[], dates[], sections[]}`
  - [x] `Fact {statement, source_url, source_domain_class, as_of_date, confidence}`
  - [x] `FactSheet {company, facts[Fact]}`
  - [x] `TailoredBullet {text, evidence_spans[], jd_keywords_covered[]}`
  - [x] `CoverLetter {intro, body_points[], closing, sources[]}`
  - [x] `CompBand {occupation_code, geography, p25, p50, p75, sources[], as_of}`
  - [x] `Metrics {jd_coverage_pct, readability_grade, evidence_mapped_ratio}`

## Day 2 — JobPostParserAgent
- [x] Implement JD splitter to extract requirements and keywords (noun phrases, skills, named entities)
- [x] Simple keyword ranking: TF-IDF + POS filter → top N (e.g., 30)
- [x] Output: `JobPosting` and set initial graph state

### Day 2.5 — Architecture Refinement (Agent/Tool Separation)

**✅ Implemented Architecture:**
```
JobPostParserAgent (agents/job_post_parser_agent.py)
├── ✅ LangGraph StateGraph workflow with conditional routing
├── ✅ TypedDict state management with full tracking
├── ✅ Error handling & retry logic (configurable, max 3 retries)  
├── ✅ Multi-node workflow orchestration (7 nodes + conditional edges)
├── ✅ Both sync/async execution methods
├── ✅ Checkpointing & debug support
└── Uses → JobPostingParser (tools/job_posting_parser.py)
              ├── ✅ Pure NLP parsing logic (stateless)
              ├── ✅ Keyword extraction & ranking (TF-IDF + boosting)
              └── ✅ JobPosting schema output (validated)
```

**Workflow Nodes:** START → initialize → parse_job_posting → validate_results → finalize → END
                  (with error handling and retry loops via handle_error node)

- [x] **Documentation Updates**
  - [x] Update all agent references to use descriptive names throughout TODO.md
  - [x] Update CLAUDE.md to reflect agent/tool architecture  
  - [x] Update docstrings and comments to use proper naming (completed in tool implementation)
- [x] **Tool Implementation** 
  - [x] Move `JobPostingParser` class to `tools/job_posting_parser.py`
  - [x] Keep as pure tool: stateless, deterministic parsing logic
  - [x] Update imports and maintain existing functionality
  - [x] Create `tools/__init__.py` with proper exports
  - [x] Update all test imports and mock patches
  - [x] Verify 100% test coverage maintained (23/23 tests passing)
  - [x] Validate code quality (ruff + black passing)
- [x] **Agent Implementation**
  - [x] Create `JobPostParserAgent` class in `agents/job_post_parser_agent.py`  
  - [x] Add LangGraph state management integration (TypedDict state with workflow tracking)
  - [x] Add error handling and retry logic (configurable retries, graceful failure)
  - [x] Add workflow orchestration (multi-node workflow with conditional routing)
  - [x] Use the JobPostingParser tool internally (clean separation of concerns)
  - [x] Create `agents/__init__.py` with proper exports
  - [x] Implement both sync and async parsing methods
  - [x] Add comprehensive state validation and metadata tracking
  - [x] Add LangGraph checkpointing and debug support
  - [x] Verify complete workflow execution (success and failure paths tested)
  - [x] Validate code quality (ruff + black passing)
- [x] **Testing Updates**
  - [x] Split tests: `tests/test_tools/test_job_posting_parser.py` for tool
  - [x] New tests: `tests/test_agents/test_job_post_parser_agent.py` for agent
  - [x] Maintain 100% coverage on both components (48 tests total: 23 tool + 25 agent)
  - [x] Add integration tests for agent-tool interaction
- [x] **Integration Updates**
  - [x] Update any existing imports to use new structure (no existing imports needed updating)
  - [x] Ensure LangGraph compatibility for agent (workflow, state management, conditional routing all working)
  - [x] Verify end-to-end functionality preserved (all 110 tests passing)

### Day 2.75 — Test Infrastructure & Code Quality Hardening

**✅ Completed Test Infrastructure Improvements:**
- [x] **Fixed All Failing Tests** - Resolved 17 failing tests caused by schema evolution:
  - [x] Updated tests to use `Requirement` objects instead of strings for JobPosting.requirements
  - [x] Fixed parser tests to handle tuple return values (JobPosting, ParserReport)
  - [x] Updated all mock configurations for new interfaces
- [x] **Eliminated All Warnings** - Achieved zero-warning test suite:
  - [x] Fixed pytest.ini configuration format ([tool:pytest] → [pytest])  
  - [x] Properly registered all custom pytest markers (integration, unit, slow, performance)
  - [x] Enabled strict marker enforcement with warning-to-error conversion
  - [x] Added comprehensive filterwarnings configuration
- [x] **Removed Duplicate Code** - Cleaned up organizational issues:
  - [x] Removed duplicate `tests/test_agents/test_job_post_parser.py` (wrong location)
  - [x] Removed duplicate `agents/job_post_parser.py` (unused JobPostingParser class)
  - [x] Maintained proper separation: tools in test_tools/, agents in test_agents/
- [x] **Code Quality Validation** - Enforced strict quality standards:
  - [x] 98% test coverage for main code (src/, tools/, agents/)
  - [x] All 123 tests passing with 0 failures, 0 warnings  
  - [x] Ruff: "All checks passed!" - no linting errors
  - [x] Black: All code properly formatted
- [x] **Documentation Updates**
  - [x] Updated CLAUDE.md with zero-warning policy and quality standards
  - [x] Added "Quality Standards" section emphasizing warning intolerance
  - [x] Documented pytest configuration and testing best practices

### Day 2.9 — JobPostParserAgent Production Issues & Fixes

**🚨 Critical Issues Identified from jd_test.py Script Output:**

- [x] **Issue #1: Title/Company Extraction Failure** ✅ **MAJOR PROGRESS COMPLETED**
  - [x] Problem: Parser fails to extract "Head of AI" and "ScaleOps" despite clear presence in text
  - [x] Root cause: No automatic extraction logic - parser expected title/company as parameters  
  - [x] Fix: Added `_extract_job_title()`, `_extract_company_name()`, and `_extract_location()` methods
  - [x] Test: Added comprehensive test cases for various title/company formats (9 new tests)
  - [x] Location: `tools/job_posting_parser.py` - implemented full auto-extraction functionality
  - [x] **Real-world Testing**: Tested on 10 diverse job description samples
  - [x] **Extraction Results**: 3/10 samples work excellently, 2/10 work partially, 5/10 need improvement
  - **Status**: Core extraction functionality implemented and working ✅
  - **Achievement**: Original ScaleOps ("Head of AI" @ "ScaleOps") and Aidoc ("AI Algorithms Team Lead" @ "Aidoc") now extract perfectly
  - **Future**: Additional pattern refinements can improve the remaining 50% of samples

- [ ] **Issue #2: Excessive Debug Output Pollution**
  - [ ] Problem: LangGraph state dumps (`[values]` and `[updates]`) make script output unreadable
  - [ ] Root cause: Debug logging enabled by default in production scripts
  - [ ] Fix: Add logging level control to agent, suppress debug output in test scripts
  - [ ] Test: Verify clean output in jd_test.py script
  - [ ] Location: `agents/job_post_parser_agent.py` + `scripts/jd_test.py`

- [ ] **Issue #3: Text Length Loss During Processing**
  - [ ] Problem: Original 2,308 characters → processed 2,245 characters (63 chars lost)
  - [ ] Root cause: Text preprocessing/cleaning removes content without tracking
  - [ ] Fix: Improve text preservation in parsing pipeline, add length validation
  - [ ] Test: Add test to verify text length preservation within acceptable threshold
  - [ ] Location: `tools/job_posting_parser.py` - review text cleaning methods

- [ ] **Issue #4: Inflated Confidence Despite Critical Failures**
  - [ ] Problem: 0.91 confidence while missing required title/company fields
  - [ ] Root cause: Confidence calculation doesn't properly penalize missing critical fields
  - [ ] Fix: Revise confidence calculation to heavily penalize missing title/company
  - [ ] Test: Add test cases verifying confidence drops appropriately for missing fields
  - [ ] Location: `tools/job_posting_parser.py` - update `_calculate_confidence()` method

- [ ] **Issue #5: Verbose Error Handling with Poor UX**
  - [ ] Problem: Retry logic generates repetitive output without meaningful progress indication
  - [ ] Root cause: Error handling focuses on debugging rather than user experience  
  - [ ] Fix: Improve error messages, add progress indicators, reduce verbosity
  - [ ] Test: Verify clean error handling in failure scenarios
  - [ ] Location: `agents/job_post_parser_agent.py` - improve error handling workflow nodes

**Implementation Priority:** Fix #1 and #2 first (critical functionality + usability), then #4, #3, #5

## Day 3 — ResumeParserAgent ✅ COMPLETED
- [x] Ingest PDF/DOCX (e.g., `pdfplumber` + `python-docx`)
- [x] Normalize structure (sections, bullets, dates)
- [x] Produce `Resume`
- [x] For each bullet, store original span offsets (for evidence links later)

### Implementation Details (Day 3)
- [x] **ResumeParser Tool**: Stateless PDF/DOCX parsing with structure normalization
  - [x] PDF parsing using `pdfplumber` with multi-page support
  - [x] DOCX parsing using `python-docx` with paragraph extraction
  - [x] Section detection using pattern matching and heuristics
  - [x] Bullet point extraction with Unicode/ASCII support and empty filtering
  - [x] Skills extraction using keyword detection and section parsing
  - [x] Date extraction with multiple format support (years, ranges, "present")
  - [x] Character span offset tracking for evidence mapping
- [x] **ResumeParserAgent**: LangGraph orchestration following established architecture
  - [x] Multi-node workflow: initialize → validate_input → parse_resume → validate_results → finalize
  - [x] Error handling with retry logic (max 3 retries) and quality validation
  - [x] Confidence scoring and validation warnings
  - [x] Both sync/async interfaces for compatibility
- [x] **Comprehensive Testing**: 148+ test cases with 99% coverage
  - [x] Unit tests for tool components (structure extraction, parsing, validation)
  - [x] Integration tests for agent workflow orchestration
  - [x] Error handling and edge case coverage
  - [x] Zero warnings achieved with strict quality enforcement
- [x] **Code Quality**: All standards met (ruff, black, pytest)

## Day 4 — Evidence index ✅ COMPLETED
- [x] Build embeddings for resume bullets + skills (SBERT or preferred model)
- [x] Implement `find_evidence(claim_text) -> top_k resume bullets + similarity scores`
- [x] Expose for Tailoring Validator and Cover-letter Validator

**✅ Implemented Architecture:**
```
EvidenceIndexer (tools/evidence_indexer.py)
├── ✅ Vector-based evidence indexing using sentence-transformers
├── ✅ ChromaDB integration for persistent vector storage
├── ✅ SBERT embeddings (all-MiniLM-L6-v2) with L2 normalization
├── ✅ Text preprocessing and bullet point normalization
├── ✅ Configurable similarity thresholds (supports ≥0.8 for evidence validation)
├── ✅ Batch processing for efficient indexing
├── ✅ Comprehensive metadata tracking for evidence provenance
├── ✅ Collection management (persistent/ephemeral, stats, cleanup)
└── ✅ Error handling and recovery mechanisms

EvidenceMatch Class: Structured evidence results with similarity scores
find_evidence() Function: Convenience API for simple evidence search
```

**Key Features:**
- **Semantic Similarity Search**: Uses SBERT for high-quality embeddings with cosine similarity
- **Resume Content Indexing**: Indexes both resume bullets and skills with metadata
- **Evidence Validation**: Configurable similarity thresholds for no-fabrication policies  
- **LangGraph Ready**: Designed for integration with validator agents
- **Production Quality**: Comprehensive error handling, logging, and performance optimization

**Implementation Statistics:**
- **36 comprehensive tests** with 100% code coverage
- **Zero warnings** compliance with project quality standards
- **Performance tested** for large-scale data (150+ items)
- **Integration verified** with existing Resume/ResumeBullet schemas
- **Tool module integration** available as `from tools import EvidenceIndexer, find_evidence`

**Ready for Day 5+:** Evidence indexer is now available for use by:
- TailoredBullet validation (≥0.8 similarity requirement)
- Cover letter evidence backing
- Anti-fabrication verification workflows

## Day 5 — CompanyResearchAgent + Validator ✅ COMPLETED
- [x] Tooling: Web search (restrict to company domain + newsroom + 1–2 major outlets)
- [x] Fetch & clean pages → cap to 10 concise facts: products, funding/filings, recent news, mission, locations
- [x] Validator:
  - [x] Require `source_url`, `as_of_date`, and a domain class whitelist (official, reputable_news, other)
  - [x] Drop facts older than N days for "recent news" (e.g., 180)
- [x] Output: `FactSheet`

### Implementation Details (Day 5)
- [x] **CompanyResearchTool**: Stateless web scraping with ethical practices (robots.txt, rate limiting)
  - [x] Source domain classification (official, reputable_news, other) with credibility scoring
  - [x] Fact extraction using NLP patterns for products, funding, news, mission, company info
  - [x] Recency filtering with 180-day cutoff for news facts
  - [x] Fact validation, deduplication, and confidence scoring
  - [x] HTML cleaning, text preprocessing, and content normalization
- [x] **CompanyResearchAgent**: LangGraph orchestration following established architecture
  - [x] Multi-node workflow: initialize → research → validate → finalize with error handling
  - [x] Quality gates: minimum facts (3), confidence thresholds (0.4), source diversity
  - [x] Retry logic with configurable max attempts and exponential backoff
  - [x] Both sync/async interfaces with comprehensive state management
- [x] **Comprehensive Testing**: 42 test cases with high coverage (1122 LOC, 739 test LOC)
  - [x] Unit tests for all tool methods (web scraping, validation, fact extraction)
  - [x] Integration tests for agent workflow orchestration and error scenarios
  - [x] Edge case coverage including network failures and malformed data
  - [x] Zero warnings compliance with strict quality standards

## Day 6 — CompensationAnalystAgent [US MVP] ✅ COMPLETED
- [x] Map job title → SOC code using O*NET crosswalk/Web Services
- [x] Query BLS OEWS tables (May 2024) for SOC in user's state (fallback: national)
- [x] Capture p25/p50/p75 with source URLs and "as-of" date
- [x] Output: `CompBand`

### Implementation Details (Day 6)
- [x] **CompensationAnalysisTool**: Stateless SOC mapping and salary analysis with geographic fallback
  - [x] Job title to SOC code mapping using fuzzy matching (38+ supported roles)
  - [x] Geographic processing for 16+ US states with abbreviation support
  - [x] BLS OEWS salary data retrieval with p25/p50/p75 percentiles
  - [x] CompBand schema compliance with source attribution and confidence scoring
  - [x] Error handling for missing data and invalid inputs
- [x] **CompensationAnalystAgent**: LangGraph orchestration following established architecture
  - [x] Multi-node workflow: initialize → validate_input → analyze_compensation → validate_results → finalize
  - [x] Error handling with retry logic (max 3 retries) and quality validation
  - [x] Both sync/async interfaces with comprehensive state management
  - [x] Workflow routing and conditional logic for error scenarios
- [x] **Comprehensive Testing**: 70 test cases with high coverage (1220+ LOC implementation)
  - [x] Unit tests for tool components (SOC mapping, geographic processing, salary analysis)
  - [x] Integration tests for agent workflow orchestration and error handling
  - [x] Edge case coverage including invalid inputs and missing data scenarios
  - [x] Zero warnings compliance with strict quality standards
- [x] **Code Quality**: All standards met (tool: 499 LOC, agent: 593 LOC, tests: 1128 LOC)

### Stretch (Day 6b–7)
- [ ] UK: ONS ASHE medians/percentiles (OGL attribution)
- [ ] EU: Eurostat SES aggregates (respect reuse terms)

## Day 7 — ResumeGeneratorAgent + ResumeValidatorAgent
- [ ] Generator: rewrite bullets to target JD keywords; must not invent facts
- [ ] Validator:
  - [ ] For each rewritten bullet, call `find_evidence`; if no match ≥ τ (e.g., 0.8 cosine) → reject/flag “needs edit”
  - [ ] Report which JD keywords were covered by that bullet
- [ ] Output: list of `TailoredBullet` + “before/after diff”

## Day 8 — CoverLetterGeneratorAgent + CoverLetterValidatorAgent
- [ ] Generator: 3–5 paragraph letter using `FactSheet` + `TailoredBullets`; include 1–2 “why this company” hooks
- [ ] Validator:
  - [ ] No claims absent from `Resume` or `FactSheet`
  - [ ] Company facts must cite a `FactSheet` item
  - [ ] Enforce length & tone constraints

## Day 9 — Metrics + gates
- [ ] JD-keyword coverage % = |covered_keywords| / |unique_JD_keywords| × 100
- [ ] Optional weighting by noun-phrase salience or TF-IDF
- [ ] Readability grade: compute Flesch-Kincaid or Gunning Fog on tailored bullets and letter; target e.g., FK ≤ 11
- [ ] Evidence-mapped ratio = (# tailored bullets passing Validator) / (total proposed bullets); pass threshold ≥ 95%

## Day 10 — React Frontend Setup
- [ ] Set up React with Vite and TypeScript
- [ ] Create FastAPI backend with CORS for frontend integration
- [ ] Implement file upload component (PDF/DOCX drag-and-drop with react-dropzone)
- [ ] Build job description input form with character counter
- [ ] Add location/geography selector component
- [ ] Create main dashboard with navigation tabs (React Router)

## Day 11 — React Components & State Management
- [ ] Implement FactSheet display component with source links
- [ ] Build compensation visualization (salary bands, percentiles with Recharts/Chart.js)
- [ ] Create tailored resume diff viewer with evidence highlighting
- [ ] Design cover letter preview with source citations
- [ ] Add metrics dashboard with progress indicators and coverage stats
- [ ] Set up state management (React Context API + useReducer) for workflow data

## Day 12 — DOCX Export & API Integration
- [ ] Add download functionality to UI components
- [ ] Implement DOCX generation endpoints in FastAPI
- [ ] Option A (code-first): `python-docx` to build Resume and Cover Letter (styles, bullets)
- [ ] Option B (template-first): render Markdown then Pandoc to DOCX (styled `reference.docx`)
- [ ] Connect frontend download buttons to backend export endpoints
- [ ] Add progress indicators for long-running operations

## Day 13 — Hardening & Real-time Features
- [ ] Add rate limiting + caching for web requests
- [ ] Add per-agent timeouts and fallback paths (e.g., national wages if state not found)
- [ ] Implement WebSocket connections for real-time progress updates
- [ ] Add error handling and user feedback throughout UI
- [ ] Log trace JSON for each run (inputs, outputs, sources, metrics)

## Day 14 — Testing & Polish
- [ ] Build a small eval set (5 JD/resume pairs)
- [ ] Track the 3 metrics and % of "red flags" from validators
- [ ] Add responsive design for mobile/tablet
- [ ] Implement accessibility features (ARIA labels, keyboard navigation)
- [ ] Record 2–3 screenshots of the custom UI
- [ ] Include UI architecture diagram in README
- [ ] Add sample DOCX outputs

---

## Nice to haves (retrieval & matching upgrades)

> These improve recall/precision while preserving the **no‑fabrication** policy. Toggle via flags and measure with the eval harness before adopting as defaults.

### A) Cross‑encoder re‑ranker (late fusion)
- [ ] Add a small cross‑encoder/NLI model to re‑rank top‑K `(requirement, bullet)` pairs from the hybrid retriever
- [ ] Set **K** (e.g., 12) and a **minimum entailment threshold**; fall back to hybrid ranks if below
- [ ] Wire entailment score into the Guard (optional) as an extra acceptance signal
- [ ] Add CLI/UI toggle: `--rerank cross_encoder` (on/off)
- [ ] Evaluate: top‑1 precision, top‑3 recall, evidence‑mapped ratio, latency delta

### B) Dual‑index + Reciprocal Rank Fusion (RRF)
- [ ] Build a tiny **lexicon index** from FactSheet terms (`company_terms`, `industry_terms`, `tech_terms`, `problem_terms`)
- [ ] Retrieve from both indices (resume index + lexicon index); fuse with **RRF**
- [ ] Ensure **evidence** still comes **only** from the resume index
- [ ] Add CLI/UI toggle: `--dual_index` (on/off) and weight parameter
- [ ] Evaluate vs. baseline; keep only if S2P and evidence‑mapped ratio improve without notable latency

### C) Query rewriting (paraphrase expansion)
- [ ] Generate **N paraphrases** per requirement (rules or LLM) using JD + FactSheet synonyms
- [ ] Retrieve with the union of rewrites; deduplicate results
- [ ] Cache rewrites per JD to control cost/latency
- [ ] Add CLI/UI toggle: `--query_rewrite` (on/off) and `--rewrites N`
- [ ] Evaluate impact on recall and end‑to‑end S2P time/cost

### Selection & defaults
- [ ] Run the eval harness with A/B/C toggles
- [ ] Pick the **best performer** subject to **p50 ≤ 3 min** and **median cost ≤ $0.25**
- [ ] Update default config and README if a nice‑to‑have becomes the new baseline
