# Two-week, step-by-step implementation plan (deliverable per day)

## [ ] Scope guardrails for 14 days
- [ ] English only
- [ ] US-centric compensation (BLS)
- [ ] Single target job per run
- [ ] Resume ≤ 2 pages
- [ ] Job description ≤ 1,500 words
- [ ] Streamlit UI (no auth)
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

## Day 4 — Evidence index
- [ ] Build embeddings for resume bullets + skills (SBERT or preferred model)
- [ ] Implement `find_evidence(claim_text) -> top_k resume bullets + similarity scores`
- [ ] Expose for Tailoring Validator and Cover-letter Validator

## Day 5 — CompanyResearchAgent + Validator
- [ ] Tooling: Web search (restrict to company domain + newsroom + 1–2 major outlets)
- [ ] Fetch & clean pages → cap to 10 concise facts: products, funding/filings, recent news, mission, locations
- [ ] Validator:
  - [ ] Require `source_url`, `as_of_date`, and a domain class whitelist (official, reputable_news, other)
  - [ ] Drop facts older than N days for “recent news” (e.g., 180)
- [ ] Output: `FactSheet`

## Day 6 — CompensationAnalystAgent [US MVP]
- [ ] Map job title → SOC code using O*NET crosswalk/Web Services
- [ ] Query BLS OEWS tables (May 2024) for SOC in user’s state (fallback: national)
- [ ] Capture p25/p50/p75 with source URLs and “as-of” date
- [ ] Output: `CompBand`

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

## Day 10 — Streamlit UI (vertical slice)
- [ ] Sidebar: upload resume (PDF/DOCX), paste JD, pick location
- [ ] Main tabs:
  - [ ] FactSheet
  - [ ] Compensation
  - [ ] Tailored Resume (diff view with evidence links)
  - [ ] Cover Letter
  - [ ] Metrics

## Day 11 — DOCX export
- [ ] Option A (code-first): `python-docx` to build Resume and Cover Letter (styles, bullets)
- [ ] Option B (template-first): render Markdown then Pandoc to DOCX (styled `reference.docx`)

## Day 12 — Hardening
- [ ] Add rate limiting + caching for web requests
- [ ] Add per-agent timeouts and fallback paths (e.g., national wages if state not found)
- [ ] Log trace JSON for each run (inputs, outputs, sources, metrics)

## Day 13 — Test set & tuning
- [ ] Build a small eval set (5 JD/resume pairs)
- [ ] Track the 3 metrics and % of “red flags” from validators

## Day 14 — Polish & submission pack
- [ ] Record 2–3 screenshots
- [ ] Include Mermaid diagram in README
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
