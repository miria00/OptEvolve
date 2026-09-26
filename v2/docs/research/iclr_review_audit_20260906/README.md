**ICLR review evidence — accessed 6 September 2026**

This is the source ledger for the [submission attack plan](../../ICLR_SUBMISSION_ATTACK_PLAN.md).
It distinguishes official policy, reviewer opinions, author responses, decisions,
and our recommendations. Examples were selected for conceptual relevance, not as
a representative sample of submissions. Oral papers are useful comparators but
do not define a mandatory checklist for acceptance.

**Access and provenance**

The live OpenReview forum/API returned a browser challenge. Review text was
therefore read from the independently published
[public OpenReview API archive](https://github.com/qhjqhj00/iclr-openreview-reviews),
[release v1.0](https://github.com/qhjqhj00/iclr-openreview-reviews/releases/tag/v1.0).
The extractor retains selected forums and only notes whose readers include
`everyone`. It does not resolve reviewer identities. Individual forum/note IDs
are preserved so the primary records can be inspected on OpenReview. Public
decisions, rather than the sometimes stale submission venue field, determine
acceptance status below.

`archive_provenance.json` records the release URL, access date, filter and SHA-256.
`extract_public_reviews.py` reproduces the selected JSON/text files from the
release tarball. The large tarball resides in `/tmp`, not the project. Each
`<forum>.json` contains structured public notes; `<forum>.txt` is a readable
rendering. This is an archived snapshot, not a claim to have obtained live review
edits today. HTML/API error pages are not research sources.

ICLR 2026 reverted reviews/scores after a security incident and reassigned area
chairs. Some meta-reviews discuss intended or hypothetical score changes. This
audit uses review reasoning, responses and final decisions; raw averages are
not an acceptance formula.
[Official process notice](https://blog.iclr.cc/2025/12/03/iclr-2026-response-to-security-incident/).

**GEPA — ICLR 2026 Oral**

Reviewers valued sample efficiency and varied tasks, while challenging whether
reflection/evolution was novel, whether comparisons covered the closest prompt
optimizers, and whether the total LLM/search cost was accounted for. The general
response added TextGrad/Trace and other controls; the meta-review regarded the
response as strong and concerns as resolved. For OptEvolve, the lesson is to
demonstrate the claimed component's effect against the closest alternatives,
rather than attributing total system improvement to LLM guidance.

Sources: [review on novelty, budgets and clarity](https://openreview.net/forum?id=RQm2KQTM5r&noteId=lxUfvtHMEs),
[review requesting stronger controls](https://openreview.net/forum?id=RQm2KQTM5r&noteId=1exoJw5odB),
[author general response](https://openreview.net/forum?id=RQm2KQTM5r&noteId=hmGZXjcbCt),
[meta-review](https://openreview.net/forum?id=RQm2KQTM5r&noteId=KZYMhBx5Ed),
[oral decision](https://openreview.net/forum?id=RQm2KQTM5r&noteId=7SyuLBAYbC).

**AutoEP — ICLR 2026 Oral**

Reviewers appreciated linking measured landscape information to dynamic algorithm
configuration and evaluation over multiple algorithms/tasks. They questioned
inference latency, context growth, the need for LLM reasoning versus a simpler
controller, and generalization beyond familiar algorithms. The meta-review
reported unanimous suitability for publication. For OptEvolve, measured hardware
context should be tested against cheaper search/control, with actual component
latency and model calls included. Merely giving an LLM numerical context is not
self-validating.

Sources: [landscape grounding and latency review](https://openreview.net/forum?id=hit3hGBheP&noteId=jOGQSKEkZj),
[simpler-controller question](https://openreview.net/forum?id=hit3hGBheP&noteId=Okzvrzj6Ps),
[meta-review](https://openreview.net/forum?id=hit3hGBheP&noteId=XLsER09ErK),
[oral decision](https://openreview.net/forum?id=hit3hGBheP&noteId=GRuGLUDACf).

**TileLang — ICLR 2026 Oral**

The meta-review emphasized real kernels, NVIDIA/AMD evaluation, explicit
programming abstractions, and practical adoption. A critical review found a
useful system but insufficient explanation of novelty and underlying algorithms.
Other concerns involved strong backend comparisons, hardware coverage, and
absolute performance instead of speedups against a moving PyTorch baseline.
The response supplied details and additional comparisons. OptEvolve needs a
similarly concrete explanation of its research contribution and credible
compiled baselines. Cross-vendor hardware is an example from this paper, not an
ICLR requirement for every hardware submission.

Sources: [critical novelty/algorithm review](https://openreview.net/forum?id=Jb1WkNSfUB&noteId=tckQI9N6gU),
[backend-baseline review](https://openreview.net/forum?id=Jb1WkNSfUB&noteId=vYAbSmj1WN),
[meta-review](https://openreview.net/forum?id=Jb1WkNSfUB&noteId=NUdnhukzLz),
[oral decision](https://openreview.net/forum?id=Jb1WkNSfUB&noteId=SsvqEofNxr).

**Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention — ICLR 2026 Oral**

Reviewers praised targeted experiments tracing a reproducible failure to biased
rounding and shared low-rank updates, followed by a small intervention that
stabilized training. Questions addressed generalization, the apparent sharpness
of instability, and overhead. The meta-review credited expanded validation while
acknowledging remaining scale limits. This is a useful precedent for OptEvolve's
numerical analysis: a negative observation becomes valuable through a convincing
causal account and a tested intervention. A failed search plus a related
citation does not provide the same evidence.

Sources: [causal-analysis review](https://openreview.net/forum?id=0jHyEKHDyx&noteId=htPGMXsqHv),
[generality/overhead review](https://openreview.net/forum?id=0jHyEKHDyx&noteId=kq2YA2WWLF),
[meta-review](https://openreview.net/forum?id=0jHyEKHDyx&noteId=9rtf518Pgz),
[oral decision](https://openreview.net/forum?id=0jHyEKHDyx&noteId=6sTOWY1jWI).

**LLM-SR — ICLR 2025 Oral**

Reviews praised scientific template discovery and generalization, but one
explicitly asked whether adding the same numerical coefficient optimizer to
non-LLM baselines would close the gap. The reviewer also asked for the closest
LLM-based baselines and narrower novelty framing. The meta-review credited
additional experiments and clarifications. This is almost exactly the control
OptEvolve needs: give the baseline the same cadence/precision/compiler improvements
before crediting the LLM or solver search.

Sources: [matched-component baseline review](https://openreview.net/forum?id=m2nmp8P5in&noteId=RlYB0Qp5Lj),
[generalization review](https://openreview.net/forum?id=m2nmp8P5in&noteId=A7BHTPlNGP),
[meta-review](https://openreview.net/forum?id=m2nmp8P5in&noteId=KrlF09suDl),
[oral decision](https://openreview.net/forum?id=m2nmp8P5in&noteId=yZWhq1uR3A).

**AFlow — ICLR 2025 Oral**

The meta-review credited improved workflows across tasks, including small-model
cost benefits. Review discussion required clearer workflow representation,
search-stage cost, generalization, and stronger evidence for model-specific
optimal workflows. The analogy is hardware-specific recipes: a conditioning
argument needs comparative evidence, not merely a device field in the prompt.
[Meta-review](https://openreview.net/forum?id=z5uVAKwmjf&noteId=WQFZ9rKjuj),
[oral decision](https://openreview.net/forum?id=z5uVAKwmjf&noteId=3AHFNwuD6n).

**Huxley-Gödel Machine — ICLR 2026 Oral**

The meta-review credited a specific mismatch addressed by a new search-allocation
metric, theoretical grounding, and empirical comparisons. It also discussed
additional contamination checks and baseline comparisons. This is a supplementary
example of making the improvement in search itself explicit, rather than calling
an existing population loop a new framework.
[Meta-review](https://openreview.net/forum?id=T0EiEuhOOL&noteId=VqVBOByGNP),
[oral decision](https://openreview.net/forum?id=T0EiEuhOOL&noteId=toghzxSbtK).

**EvoEngineer — ICLR 2026 Reject**

This close neighbor combines LLM CUDA optimization, feedback and population
management. Reviewers questioned novelty, component ablations, reporting detail,
single-device/Level-1 scope, statistical evidence and weak or launch-bound
benchmark comparisons. A reviewer requested realistic shapes, compiled baselines
and bandwidth/FLOP sanity checks. The authors did not rebut, according to the
meta-review. These opinions should not be treated as proof that every benchmark
result was wrong; they identify concrete credibility objections OptEvolve must
answer. Its reported gains and engineering effort were not sufficient for
acceptance in this review process.

Sources: [evaluation-validity review](https://openreview.net/forum?id=LU27DiW5ik&noteId=CBiV5Ugi97),
[novelty/ablation review](https://openreview.net/forum?id=LU27DiW5ik&noteId=t1LVy0lpRg),
[scope/statistics review](https://openreview.net/forum?id=LU27DiW5ik&noteId=Mz8jC1vxt4),
[meta-review](https://openreview.net/forum?id=LU27DiW5ik&noteId=W4YeJe4pgn),
[rejection](https://openreview.net/forum?id=LU27DiW5ik&noteId=u8KHBEeNTp).

**OptiVer — ICLR 2026 Reject**

This work concerns optimization modeling rather than solver synthesis. Its
meta-review acknowledged added cost accounting, verification-reliability tests
and ablations, but retained objections about novelty, the rigor and operational
role of theoretical propositions, and the breadth of claims. For OptEvolve,
more experiments cannot substitute for a clear conceptual advance, and a generic
optimization formulation or disconnected theorem will not solve the positioning
problem.
[Meta-review](https://openreview.net/forum?id=w696Vhv5B2&noteId=qBUmXYU4Nd),
[rejection](https://openreview.net/forum?id=w696Vhv5B2&noteId=xnPl5Rmbwp).

**Large Language Models Cannot Self-Correct Reasoning Yet — ICLR 2024 Poster**

This is a poster, not an oral. Its meta-review supported the timely distinction
between intrinsic self-correction and external feedback, while noting imprecise
claims and prompt-design confounds. It supports a carefully scoped negative
study as a legitimate contribution; it does not establish that our one-model
repair result is already sufficient or novel.
[Meta-review](https://openreview.net/forum?id=IkmD3fKBPQ&noteId=T57BjkGGNR),
[poster decision](https://openreview.net/forum?id=IkmD3fKBPQ&noteId=0bWlyvMLpi).

**Official policy sources**

`reviewer2026.txt` and `reviewer2027.txt` are extracted text from the
[2026 reviewer guide](https://iclr.cc/Conferences/2026/ReviewerGuide) and
[2027 reviewer guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines).
`author2027.txt` comes from the
[2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
`ai_policy2027.txt` comes from the
[2027 AI policy for authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).
Use the [2027 call](https://iclr.cc/Conferences/2027/CallForPapers) for deadline
dates. The author's initial-submission rule is nine main-text pages; later
camera-ready wording on that page is inconsistent about the original limit.
The reviewer guide's contemporaneous-work example also contains a stale deadline.
Neither inconsistency changes the explicit September 18/25 call dates.

The [2026 retrospective](https://blog.iclr.cc/2026/03/31/a-retrospective-on-the-iclr-2026-review-process/)
reports 5,355 acceptances from 19,525 valid submissions (27.4%, counting withdrawn
submissions in that denominator). It separately reports 13,763 accept/reject
decisions. These denominators should not be mixed, and neither is an estimate of
OptEvolve's individual acceptance chance.
