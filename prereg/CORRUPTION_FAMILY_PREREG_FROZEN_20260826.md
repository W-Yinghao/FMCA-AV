# Frozen before compute — corruption-family prediction, 2026-08-26

Tranche 1 (committed at 1d055d0) measured three corruption types with
three disjoint draws each and found a type separation at the disjoint
standard: gaussian_noise d_op [0.362, 0.378] below defocus_blur
[0.439, 0.485] and fog [0.445, 0.482].

Tranche 2 tests whether that separation is a FAMILY property.  Frozen
predictions, before any tranche-2 job runs:

P1. Noise family (shot_noise, impulse_noise, speckle_noise): three-seed
    d_op ranges land BELOW every blur-family range (pixel-independent
    additive/multiplicative noise composes across severities the way
    gaussian noise does).
P2. Blur family (motion_blur, zoom_blur): ranges land ABOVE every
    noise-family range and overlap or exceed the defocus band.
P3. contrast is measured with NO prediction (digital family,
    exploratory).

Evidence standard: disjoint three-seed ranges, seeds = disjoint
8000-image blocks 1-3, protocol otherwise identical to tranche 1
(resnet50 IMAGENET1K_V2 penultimate, severities 1/3/5, budget 64,
gram_bound 0.05).  P1/P2 refuted by ANY overlap between a noise range
and a blur range.  Failure is reported as loudly as success.
