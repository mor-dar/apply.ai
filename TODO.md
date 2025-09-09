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

## Day 2 — Job-post parser (Agent A)
- [ ] Implement JD splitter to extract requirements and keywords (noun phrases, skills, named entities)
- [ ] Simple keyword ranking: TF-IDF + POS filter → top N (e.g., 30)
- [ ] Output: `JobPosting` and set initial graph state

## Day 3 — Resume parser (Agent B)
- [ ] Ingest PDF/DOCX (e.g., `pdfplumber` + `python-docx`)
- [ ] Normalize structure (sections, bullets, dates)
- [ ] Produce `Resume`
- [ ] For each bullet, store original span offsets (for evidence links later)

## Day 4 — Evidence index
- [ ] Build embeddings for resume bullets + skills (SBERT or preferred model)
- [ ] Implement `find_evidence(claim_text) -> top_k resume bullets + similarity scores`
- [ ] Expose for Tailoring Validator and Cover-letter Validator

## Day 5 — Researcher (Agent C) + Validator
- [ ] Tooling: Web search (restrict to company domain + newsroom + 1–2 major outlets)
- [ ] Fetch & clean pages → cap to 10 concise facts: products, funding/filings, recent news, mission, locations
- [ ] Validator:
  - [ ] Require `source_url`, `as_of_date`, and a domain class whitelist (official, reputable_news, other)
  - [ ] Drop facts older than N days for “recent news” (e.g., 180)
- [ ] Output: `FactSheet`

## Day 6 — Compensation analyst (Agent D) [US MVP]
- [ ] Map job title → SOC code using O*NET crosswalk/Web Services
- [ ] Query BLS OEWS tables (May 2024) for SOC in user’s state (fallback: national)
- [ ] Capture p25/p50/p75 with source URLs and “as-of” date
- [ ] Output: `CompBand`

### Stretch (Day 6b–7)
- [ ] UK: ONS ASHE medians/percentiles (OGL attribution)
- [ ] EU: Eurostat SES aggregates (respect reuse terms)

## Day 7 — Tailored resume generator (Agent E) + Validator (Agent F)
- [ ] Generator: rewrite bullets to target JD keywords; must not invent facts
- [ ] Validator:
  - [ ] For each rewritten bullet, call `find_evidence`; if no match ≥ τ (e.g., 0.8 cosine) → reject/flag “needs edit”
  - [ ] Report which JD keywords were covered by that bullet
- [ ] Output: list of `TailoredBullet` + “before/after diff”

## Day 8 — Cover-letter generator (Agent G) + Validator (Agent H)
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
