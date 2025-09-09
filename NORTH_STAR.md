## 🔭 North Star — apply.ai v0.1 (Job‑Application Copilot)

### One‑Sentence Outcome (by **Fri Sep 19, 2025**)
Ship **apply.ai v0.1**, a focused copilot that converts a **job description + candidate profile** into a **tailored resume, cover letter, and application Q&A**—all **ATS‑ready, citation‑linked to the user’s real experience, and never fabricated**—so a candidate can go from JD → submission‑ready package in **≤ 5 minutes**.

**Why this matters**
- Candidates waste hours tailoring materials and still risk **ATS rejections** or **inaccurate claims**.
- Teams and programs want **honest, auditable** assistance that saves time without making things up.
- apply.ai provides **speed + integrity**: fast tailoring, explicit sourcing to a user’s resume, and safety checks.

---

### Scope for v0.1 (what ships)

**Primary flows**
1. **JD → Requirements & Fit**  
   Extract must‑haves/nice‑to‑haves; compute **Fit Score**; map each requirement to **verified resume evidence**.
2. **Tailored Resume**  
   Reorder sections, rewrite bullets with JD keywords; **no new claims**; export **DOCX + PDF**.
3. **Cover Letter**  
   Structured letter with **evidence citations** (bullet ↔ resume line IDs) and tone controls.
4. **Application Q&A**  
   Draft answers to common portal questions using only **resume‑backed facts**; flag gaps to the user.
5. **Application Pack**  
   Zip download: `resume.docx/pdf`, `cover_letter.md`, `qa.json`, `fit_report.html`.

**Under the hood**
- Agents: **JD Parser / Evidence Matcher / Writer / Honesty Verifier (Guard) / ATS Formatter**.
- **Guardrails**: strict “no‑fabrication” policy; every claim links to resume evidence; PII handling.
- **Evaluation harness**: labeled JDs + synthetic resumes to test extraction, honesty, and ATS formatting.

---

### North‑Star Metric (NSM)
**S2P@T/B — Scan‑to‑Pack rate at Time/Budget.**  
% of sessions that produce a **guard‑approved, submission‑ready pack** within **T ≤ 5 min** and **B ≤ $0.25**.

> v0.1 target: **S2P@5/0.25 ≥ 35%** (median time ≤ 3 min).

---

### Success Criteria (quant + qual)

**Product & activation**
- ≥ **100 completed sessions**; **≥ 35%** reach submission‑ready pack (S2P@5/0.25).
- **Median TTV ≤ 3 min** from JD paste → downloadable pack.
- **Return use ≥ 20%** of users generate a second pack in 7 days (proxy for value).

**Quality & safety**
- **Honesty compliance 100%** on test cases (no claims outside resume/profile); all added content is explicitly marked as phrasing, not credentials.
- **Requirement extraction F1 ≥ 0.85** on a 20‑JD labeled set.
- **Mapping precision ≥ 95%** for requirement ↔ resume‑evidence links (manual audit on 50 links).
- **PII & safety**: zero critical leaks; prompt‑injection tests blocked ≥ **90%**.

**Engineering & reproducibility**
- **Clean‑clone → demo ≤ 15 min**; one‑command Docker; CI green; **coverage ≥ 80%** on core agents/tools.
- Export formats validated in common ATS (DOCX structure + PDF text layer intact).

**Credibility & publishing**
- Public repo with examples, traces, and evaluation report; short write‑up for Ready Tensor (methods, metrics, limitations).

---

### Primary Users & Stories

- **Job seeker (mid‑career IC):** “Paste JD, upload resume, get a truthful, ATS‑ready pack I can submit today.”
- **Career services / bootcamps:** “Offer students a **safe tailoring tool** that teaches evidence‑based writing.”

---

### Risks & Mitigations

- **Fabrication risk** → Honesty Verifier blocks unsupported claims; UI shows **evidence chips** for every assertion.
- **ATS formatting quirks** → Regression tests on DOCX/PDF structure; plain‑text fallback.
- **Latency/cost spikes** → caching + summarization; cheaper model fallback; per‑agent budgets.
- **Ambiguous or sparse resumes** → prompts request user confirmation or add “Gap Notes” section instead of inventing.

---

### Definition of Done (v0.1)

- Users can **paste a JD + upload a resume** and download a **zipped application pack** (resume, cover letter, Q&A, fit report).
- **Every claim is traceable** to resume/profile lines; unsupported claims blocked or flagged.
- **NSM instrumented**; baseline report captured from live usage.
- Repo meets open‑source best practices; **v0.1 tag** cut; demo and docs complete.
