# Title options, ranked

OptEvolve, ICLR 2027 draft, 10 September 2026. Ranking rule: the title must
promise the factorisation first, because that is the contribution the evidence
supports most strongly and because leading with the kernel runs into the live
scoop risk. Every number named below already appears in a drafted section with
a source comment pointing under `results/`. Style contract applies here too:
no em-dashes, no italics.

## 1. Time to a Certified Solution Factorises: Iteration Count, Cost per Iteration, and What Couples Them

Promises a measured two factor split of solve time plus an identification of
what ties the factors together; the split is delivered (eight execution and
kernel arms return identical counts in 72 of 72 instances across two
architectures and seven sizes, while run times differ by up to 2.629x), but the
coupling half is the paper's weakest claim, since precision also moves the
iteration count at K=16 and on the Blackwell at n=128 and n=496, so the
subtitle must ask what couples them and never assert a single coupling point.

## 2. Iteration Count Is Set by the Certificate Cadence, Not by the Machine

Promises that the first factor is a property of the recipe rather than the
hardware; delivered within the measured scope (per-instance vectors behind
means of 584.0, 597.3 and 682.7 at cadences 16, 64 and 256 agree element by
element on the RTX 4090 and the RTX PRO 6000 Blackwell), but two devices cannot support the general
portability reading the title invites, and the equality is only ever tested at
multiples of K, so the invariance is coarser than the phrasing suggests.

## 3. Separating Iteration Count from Cost per Iteration in Certified First-Order GPU Solvers

Promises exactly what was measured and nothing beyond it; delivered in full,
including the negative half that a reviewer can check (the wrong fusion tile
gives 0.743x, slower than not fusing, and on the Blackwell at n=496 the best
tile is still 0.925x of the static loop), and its only cost is that it promises
a decomposition rather than a result, which makes it the safe fallback rather
than the lead.

## 4. What Arithmetic Width Buys, and What Loop Structure Buys

Promises the correction that motivated the rewrite, that a speedup credited to
float32 is mostly execution structure; delivered (width buys 1.000x inside the
traced device loop and 1.576x once the loop is chunked at K=256, while a traced
rather than static inner trip count costs 2.02x at n=256), but it foregrounds
precision, where the survey records that we are not first to run reduced width
iterates under a full precision certificate and where a published fp32 against
fp64 envelope invites exactly the comparison we do not want.

## 5. A Temporally Fused PDHG Kernel and a Symbolic Construction Gate for Certified Solvers

Promises two engineering artifacts; both are delivered numerically (2.629x end
to end at n=128 against the shipped baseline, and a gate that admits 351 of 351
settings at each theta below 1 and none at 1.02), but it leads with the kernel
against the standing instruction to lead with the factorisation, and it says
nothing about the result that makes the kernel interpretable, so this is the
title with the largest gap between what it promises and what the artifact
currently proves.
