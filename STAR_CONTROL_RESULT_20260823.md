# The star-tree negative control, resolved

Frozen prereg: `GATE1_CIFAR10_STRUCTURE_PREREG_FROZEN_20260816.md`.
The control breaks the Markov nesting by design -- every view is drawn
from the original image (`p(Y|X_0)`) instead of from the realized
parent -- while keeping the recipe, the view count and the encoder
identical.  Only the conditional structure changes.

## Result

| dataset | arm | probe (%) | closure defect, per seed |
|---|---|---|---|
| CIFAR-100 | nested (V7) | 55.89 ± 0.32 | 0.377, 0.372, 0.378 |
| CIFAR-100 | **star (broken)** | **56.94 ± 0.24** | **0.515, 0.481** |
| CIFAR-10 | nested (V7) | 84.98 ± 0.69 | 0.316, 0.361, 0.337, 0.461, 0.377 |
| CIFAR-10 | star (broken) | 84.24 ± 0.26 | 0.434, 0.492, 0.473 |

**CIFAR-100: separated, disjoint.**  Star 0.481-0.515 against nested
0.372-0.378, with a gap wider than either range.  A third star seed is
rerunning after a node CUDA fault; the two in hand do not touch the
nested range.

**CIFAR-10: still overlapping.**  Star 0.434-0.492 against nested
0.316-0.461.  The n=1 null reported on 08-22 survives at n=3 on this
dataset.

## Why this matters

1. **The instrument is sensitive to conditional structure, and the
   sensitivity is now demonstrated rather than assumed.**  This was
   the one pillar still missing: the certificate could separate
   trained-for-closure from not-trained-for-closure, but until now it
   had never been shown to detect the structure the whole ontology is
   built on.
2. **The certificate measures something accuracy does not.**  On
   CIFAR-100 the star arm is slightly MORE accurate (56.94 against
   55.89) while its defect is 30% worse.  Breaking the Markov nesting
   costs nothing on the linear probe and wrecks the certificate.  A
   metric that moved with accuracy would have been redundant; this one
   is not.
3. **It completes the resolution story.**  CIFAR-10 resolves neither
   explicit-versus-additive nor star-versus-nested.  CIFAR-100
   resolves both.  Two independent contrasts, the same dataset
   ordering -- so "CIFAR-10 lacks resolving power" is now a claim with
   two legs, not an excuse fitted to one null.

## Standing caveat

The CIFAR-10 null is not explained away by this.  It is reported as a
null wherever the CIFAR-10 gate is reported, and the resolution
argument is offered as the reading, not as a result.
