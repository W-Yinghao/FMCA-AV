# Chain pilots reading — 2026-08-25

Interpretation commit for `results/chains/20260825_pilots/` (results
committed at 417f045).  Three disjoint 8000-sample draws per cell;
coordinate rule: budget 64, variance floor 1e-2, both declared in the
records.

## Pilot A3 (depth chains): the depth profile is real

**Claim, at the disjoint-range standard.**  On every one of six cells
(resnet18/50/152 x cifar10/100), the projected closure defect falls
from `stem->endpoint` to `layer2->endpoint` in all three seeds, and the
stem and layer2 three-seed ranges are DISJOINT in all six cells.  All
values sit 2-5x above the pairing-shuffled null (<= 0.18).  Reading:
the later the chain starts, the more of what the endpoint carries is
reachable through the retained interface coordinates — a depth
profile of interface sufficiency, measured without resampling any view.

**The ordering is not an artifact of the coordinate rule.**  Sweeping
the variance floor over 1e-3 / 1e-2 / 1e-1 moves the absolute defect
(more retained directions = more measured omission, as a declared part
of the estimand must), but stem > layer1 > layer2 holds at every floor.

**The control behaves.**  Random-init resnet50: every chain is REFUSED
— pooled random activations retain 1 of 1024 directions above the
floor — and the probe curve declines with depth (39.3% stem -> 31.9%
layer4) where the pretrained curve ascends (44.7% -> 88.1%).  No
pretrained structure, no measurable chain, no ascending profile.

## Pilot A1 (self-stitch): the answer is no, and that is the result

The pilot's question: is there an interval where local metrics look
fine, the path composition does not, and the bypass intervention then
fails?  At this configuration the answer is NO, in all three seeds:

- The intervention outcome varies ~50x across intervals
  (endpoint_cka_drop 0.003 -> 0.17), highly reproducible across seeds.
- The projected path defect is nearly FLAT across the same intervals
  (0.16-0.42) and its rank agreement with the intervention flips sign
  across cells (resnet18 negative in every seed, resnet152/cifar100
  positive in every seed).  It does not order intervals by
  bridgeability.  CKA fails the same way (-1.0 to +0.8 by cell).
- The local ridge residual (relative_mse) tracks the intervention at
  spearman +1.0 in all 18 seed-cells — but this is close to
  tautological: the bypass IS the ridge map whose residual relmse
  measures, so this is a consistency check, not a discovery.

**A correction on the record.**  The 2-interval smoke run was read here
as "the bypass agrees with the defect, not CKA."  The full grid refutes
that generalization; the smoke reading was an n=1 overread, same
failure mode as the beta=16 episode.

**Why this is coherent, not a contradiction.**  The 3-state defect
measures whether what the endpoint carries passes through the 64
retained coordinates at the middle tap — suffix sufficiency of the
TAP.  The bypass measures whether the tap is linearly REACHABLE from
the earlier tap.  Different estimands; the grid demonstrates the
difference.  The depth-chain result (A3) is about the first estimand
and stands.  What A1 shows is that the path defect must not be sold as
a stitchability predictor at matched budget — the strategy's
"statistical stitchability" track needs an estimand-level redesign
before any further compute goes to it.

## Status of the framework claims after this round

- Depth-sufficiency profile: supported, three-seed disjoint, control-
  validated, floor-robust.  This is the figure-grade result.
- "Local metrics miss what the path sees" as a stitching claim: NOT
  supported at this configuration; do not use A1 as evidence for it.
- Conditioning lesson (engine-level, from the crashed first round):
  ridge whitening plus low-variance directions makes G^{-1/2} an
  amplifier; the variance floor is now a declared knob bounding
  ||I-G|| — stem deviation 1.000 -> 0.024.  Any past result computed
  on pooled features WITHOUT a floor should be treated as suspect of
  the same artifact; the gate hierarchies used 128 fixed coordinates
  and are not affected.

---

# Revision under the gram_bound rule — same day, later

The lambda_max-relative floor died on ConvNeXt (probe 92%, every chain
refused, 1 direction of 384 retained), and was replaced by the
native-anchored Gram bound: keep directions with
`lambda * f >= (1-f) * rho * s_native`, which bounds every level's
`||I-G||` by `f` by construction.  The full grid was rerun under the
new rule (results commit above this one).  Three findings survive, one
is unlocked, and one interpretation from the morning does NOT survive
its own control.

## What the full grid shows (gram_bound 0.05, 3 disjoint draws)

11 of 12 cells pass the disjoint-range standard: resnet18/50/152,
densenet121 and convnext_tiny are monotone 3/3 with first-vs-last
ranges disjoint on both datasets.  The one exception is vit_b_16 on
cifar100 (monotone 1/3), and ViT was also the weakest cell under the
old rule: the flatness is rule-invariant and consistent with a
constant-width residual stream attenuating less with depth.

## The control result that changes the caption

Under the adaptive rule the random-init network is measurable, and its
depth profile looks like everyone else's: monotone decline in all
three seeds.  Its defect LEVELS overlap the pretrained ones (stem
[0.98,1.57] vs [1.14,1.60]; layer2 [0.448,0.631] vs [0.365,0.451] —
touching at the boundary), and its endpoint dependence is full strength
(top 1.04-1.30 vs pretrained 0.91-0.95): random pooled features carry
as much input information as trained ones, so this is not a
defect-of-nothing artifact.

**Therefore: the monotone depth profile is a property of deep pooled
chain geometry under the declared coordinate rule, and is NOT by
itself evidence of learned depth structure.**  The morning's caption
("a depth profile of interface sufficiency" of the pretrained
network) overclaimed.  What separates trained from random in these
records is the probe DIRECTION (ascending 45->88% vs descending
39->29%): training changes what the dependence is about, not how much
closure the chain has.  The valid use of the chain profile is
comparisons that hold architecture and coordinate rule fixed — which
is exactly the comparison the gate experiments make (same backbone,
different objectives), and a cross-architecture depth figure must be
captioned as architecture geometry, not learning.

## Stitch, second pass: the negative got sharper

Under the new rule the rank agreement between path defect and
intervention outcome is NEGATIVE in 16 of 18 seed-cells (the two
exceptions are one mixed cell).  Lower measured closure defect on an
interval now systematically accompanies WORSE bypass outcomes,
because deeper intervals have lower defect (the depth profile) and
less linearly replaceable computation.  The 08-25 morning conclusion
stands, stronger: the path defect at matched budget must not be used
as a stitchability score.

## L-scan (E-D2): the price list has its curve

At fixed ends, refining the factorization from 2 to 15 interfaces
grows the measured defect superlinearly (c10: 0.32 -> 193) with the
shuffled null growing in proportion (0.12 -> 30), while the Tier-2
compounded radius r_P explodes to vacuity (3.4e7).  Observed growth
stays inside the predicted envelope everywhere.  The operational
reading the paper can print: do not factor a chain finer than the
calibration budget supports, and the certificate's own radii say where
that line is.

---

# ImageNet round — same day, evening

Results commit above this one: in-distribution depth chains
(calibration domain = pretraining domain) and the first corruption
paths on ImageNet-C.

## The 2x2 that gives the defect its training sensitivity back

The morning's control showed trained and random networks with
overlapping defect levels on CIFAR-10, which demoted the depth profile
to chain geometry.  The in-distribution round completes the picture:

    resnet50 stem->endpoint      imagenet val         cifar10
    pretrained                   [0.467, 0.573]       [1.137, 1.601]
    random init                  [0.789, 6.725]       [0.981, 1.570]

On the pretraining domain, trained vs random separates with DISJOINT
three-seed ranges at every chain start (layer2: [0.276,0.284] vs
[0.355,1.227]); under domain shift the trained network's defect rises
to random level and the separation vanishes.  The precise claim the
paper can make: **the closure defect measures whether the chain's
interfaces transport dependence for the calibration distribution, and
training installs that transport only for its own domain.**  Neither
"the defect detects training" (false under shift) nor "the defect is
pure geometry" (false in-distribution) survives alone; the
domain-conditional statement passes the disjoint standard on both
sides.

Also rule- and domain-invariant: ViT-B/16 stays flat in-distribution
(mono 1/3, first [0.399,0.503] vs last [0.478,0.492]) with by far the
weakest endpoint probe (23-25% vs resnet50's 52-55%) -- token-mean
states of a constant-width residual stream just do not attenuate the
way pooled CNN stages do.  Three independent settings now agree, so
this is the architecture, not noise.

## Corruption paths (first real corruption_path data)

clean -> sev1 -> sev3 -> sev5 of one corruption, exact per-image
pairing, resnet50 penultimate states, 3 disjoint 8000-image blocks:

    gaussian_noise   d_op [0.362, 0.378]   null [0.208, 0.222]
    defocus_blur     d_op [0.439, 0.485]   null [0.201, 0.205]
    fog              d_op [0.445, 0.482]   null [0.195, 0.197]

All types sit ~2x above their shuffled nulls, and the TYPE separation
is itself at the disjoint standard: the additive-noise severity ladder
is more nearly Markov in feature space than the blur or fog ladders.
Context probes agree with what the types do physically: fog keeps the
most endpoint dependence (top 0.955-0.961, probe falling only to 25%)
while gaussian noise destroys the most (probe to 5%) yet composes
best.  Composability and information preservation are different axes,
and the pilot measures them apart -- which is exactly what the
interpretation discipline wanted the corruption_path label to mean.

---

# Tranche 2 and the tin200 gate — 2026-08-26

Results commit: previous commit.  Two verdicts, one refutation.

## Corruption family prediction: REFUTED, as frozen

The frozen P1/P2 (every noise-family range below every blur-family
range) fail on two counts: speckle_noise [0.415, 0.471] sits inside
the blur band, and motion_blur's low end (0.369) dips into the noise
band.  Corruption FAMILY does not determine severity-ladder
composability.  Post-hoc observation, labeled as such: the two
i.i.d.-additive pixel-noise types (gaussian [0.362,0.378], shot
[0.363,0.385]) remain the tightest and lowest of all nine types
measured, while multiplicative (speckle) and sparse (impulse) noises
drift up.  A NARROWED hypothesis — i.i.d. additive pixel noise
composes best — is frozen prospectively in the prereg directory and
will be tested on a different encoder (resnet18); it earns no standing
from this tranche.

## tin200: the dataset ordering CONTINUES, and raw numbers would have
## said the opposite

Normalized closure defect (delta_fro / ||C_dir||_F, small-budget
protocol, 3 seeds):

    V7        [0.316, 0.357]      probe [33.96, 34.75]%
    additive  [0.353, 0.376]      probe [35.22, 35.43]%
    flat      [0.442, 0.445]      probe [39.76, 40.64]%

V7 vs flat disjoint; V7 vs additive touching.  The arm ordering
V7 < additive < flat seen on CIFAR-10 (bigcal) and CIFAR-100 holds on
the third dataset, 200 classes at 64px, under the disclosed protocol
deviations (downsample stem; only within-dataset contrasts count).
Raw delta_frobenius REVERSES this ordering (flat 2.1 vs V7 3.1-3.6)
because flat's endpoint carries a smaller ||C_dir|| -- the
normalization is not a convention here, it is the difference between
the right answer and its opposite.

The probe dissociation recurs: flat wins linear probe by ~5 points
while carrying the largest closure defect -- consistent with the c100
star control and the layerwise study (closure confines the endpoint to
the path-supported subspace; probe accuracy is not what the
certificate measures).

Caveat carried forward: these are small-budget ratios (n_cal 2500),
the regime T0 showed to be noise-floor-dominated for small-defect
arms.  The tin200 convergence sweeps (frozen prediction in the prereg
directory) will say whether the ordering widens at larger N as the
mechanism predicts.

---

# The corruption estimand, second encoder — 2026-08-26, later

## B1 CONFIRMED as frozen

On resnet18, gaussian_noise [0.465,0.519] and shot_noise [0.456,0.542]
land strictly below defocus_blur [0.647,0.753] and motion_blur
[0.631,0.683] -- disjoint at 0.542 vs 0.631.  The narrowed hypothesis
survived a frozen prediction on an independent encoder, which the
refuted family-level version never earned.

## The full nine-type table is cleaner on r18 than on r50 (descriptive)

    pixel-i.i.d. noise   impulse [0.403,0.495]  shot [0.456,0.542]  gaussian [0.465,0.519]
    digital              contrast [0.536,0.645]
    structured           zoom [0.597,0.693]  speckle [0.621,0.661]  motion [0.631,0.683]
                         fog [0.641,0.703]  defocus [0.647,0.753]

On resnet18 every pixel-independent noise sits below every spatially
structured corruption (0.542 vs 0.597, disjoint), with contrast
straddling; on resnet50 motion_blur dipped into the noise band and
speckle rose out of it.  How much of the severity-ladder Markov
structure is the corruption's and how much is the encoder's is now a
measured question -- only the i.i.d.-additive-vs-blur contrast (B1) is
encoder-stable so far, and only it carries confirmatory standing.

## Severity refinement (C, exploratory as frozen)

Refining the ladder from 3 to 5 interfaces triples the measured defect
(gaussian 0.37 -> 0.9, defocus 0.46 -> 1.12) while the shuffled null
barely moves (0.21 -> 0.23).  Unlike the depth L-scan, this is NOT
estimator compounding -- the null stays flat -- so adjacent-severity
edges carry real non-Markov structure that the coarse ladder skipped
over.  The corruption estimand depends on ladder resolution, and any
cross-type comparison must fix it (all tables above use 1,3,5).

---

# Depth ablation completion — 2026-08-26, evening

Results commit above.  Three depth questions the audit found open, now
measured.

## Network depth is NOT the variable (pure-depth series)

resnet50/101/152 share width and block design and differ only in
depth.  Their stage profiles are indistinguishable — first-start and
last-start ranges overlap across all three depths on all three domains
(imagenet last: [0.276,0.284] / [0.265,0.274] / [0.270,0.295]).  The
differences seen between resnet18 and resnet50 are therefore
width/block-design effects, not depth; captions that said "deeper
models" should say "larger stage widths".

## The fine-grained profile is a sawtooth, not a slope (depth grid)

At block resolution the profile is not monotone.  Holding edge count
fixed within a segment, the defect RISES as the start moves off a
stage boundary into mid-stage blocks (imagenet end=16: stem 0.47-0.57
vs mid-layer1 0.69-0.86, disjoint on imagenet and cifar10, direction
consistent on cifar100), and DROPS discontinuously at the next stage
boundary (disjoint on all three domains).  The network's own stage
outputs are measurably better path support than mid-stage taps.  The
stage-level "monotone decline" was real but under-sampled: it sampled
only the boundary teeth of a sawtooth.

## Endpoint depth matters non-monotonically

At fixed start, the defect peaks at end=layer3 and falls at
end=layer4 (start=1 on imagenet: 1.04 at end 13 vs 0.62 at end 16)
despite layer4 chains carrying MORE edges — the layer3->layer4 suffix
genuinely closes, consistent with the layerwise finding that closure
confines the endpoint to the path-supported subspace.  Distance from
the end and absolute position are now separated, and neither alone
explains the profile.

## ViT flatness localizes

At block resolution ViT's flatness is not uniform: early blocks (1-5)
carry real within-segment structure while the late segment (6-8) is
flat with no rise at all — late residual updates are near-identity in
the token-mean state.  The family-level "ViT does not attenuate"
refines to "ViT's depth structure lives in its early blocks".

## Still in flight (the expensive tail of the audit)

23 bigcal V1/V3/V5/V6 completion units are training; the tin200
completion (12 units), the depth-4 hierarchy QC probe, and three
stragglers sit in a slot-watching babysitter (scripts/
babysit_ablation_queue.sh).  The depth-4 FLEET is intentionally not
queued: the probe-gate decides it.
