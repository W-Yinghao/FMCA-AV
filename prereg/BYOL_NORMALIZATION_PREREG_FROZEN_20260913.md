# Frozen before compute — does the missing BatchNorm explain BYOL's collapse?

Frozen 2026-09-13, BEFORE the fixed arm runs.

**Disclosure: the collapsed runs have already been seen in full**
(`BYOL_COLLAPSE_DIAGNOSIS_20260913.md`) — three seeds, val cosine
0.9795/0.9888/0.9880, layer1 effective rank 5.1 of 64, probe accuracy
declining with depth. This wave tests a diagnosis formed after seeing
them. Nothing below is a fresh discovery; it is a controlled check of
one attribution.

## The claim under test

Our `MLP` — used for every projector and predictor in the repo —
contains no normalization layer. BYOL's projector and predictor are
specified with BatchNorm, and BYOL has no negatives and no variance or
decorrelation term, so those batch statistics are its only collapse
avoidance. The diagnosis says the collapse is ours, not BYOL's.

The four other matched baselines share the same MLP and did not
collapse, because each carries an anti-collapse term in its loss. That
selectivity is the circumstantial case. This wave is the controlled one.

## The manipulation

ONE change: `BatchNorm1d` after each non-final Linear of BYOL's
projector and predictor, which is what the method specifies. Budget,
augmentation, optimizer, schedule, backbone, batch size, epochs, seeds
and dataset are all held at the values the collapsed runs used. The
original config and its three runs are left untouched on disk.

Nothing is re-tuned. Adding a component the method specifies is not the
baseline re-tuning the matched-budget prereg forbade, and this prereg
records that reading explicitly so it cannot be relitigated later.

## Predictions

The primary endpoint is COLLAPSE or NOT, not competitiveness. A BYOL
that trains but lands below the other baselines is a confirmed
diagnosis and an uncompetitive baseline; those are separate findings.

P-B1 (primary). The fixed arm does not collapse, on all three seeds:
     layer1 effective rank above 20 of 64, against the collapsed runs'
     5.1, and final validation cosine below 0.95, against 0.98-0.99.
     REFUTED if any seed stays at effective rank below 10 or cosine
     above 0.95.

P-B2. Probe accuracy rises with depth rather than falling. The
     collapsed runs read 43.8 -> 45.8 -> 42.5 -> 37.6 across layer1-4;
     the fixed arm's layer4 exceeds its layer1 on all three seeds.
     REFUTED if layer4 is below layer1 on any seed.

P-B3. No prediction about where BYOL lands relative to simclr, vicreg,
     barlow_twins or moco_v2. It is measured and reported either way,
     and a fixed BYOL that is simply worse than the others is reported
     as that.

## What each outcome means, committed now

    P-B1 and P-B2 both hold
        The diagnosis is confirmed. BYOL's exclusion was our defect.
        The baseline table must either carry the fixed arm or state
        that BYOL was never given a faithful implementation.

    P-B1 refuted (still collapses)
        The missing BatchNorm is NOT the cause, or not the only one.
        The diagnosis document is wrong and gets a correction, and the
        original exclusion stands on different grounds that would then
        need to be found.

    P-B1 holds, P-B2 refuted
        Report as-is and do not resolve it by choosing whichever metric
        agrees. A representation that is not rank-collapsed but still
        degrades with depth is a third thing and would be named as such.

## Gates

- Probe before fleet: seed 1 alone; seeds 2 and 3 only after it
  completes.
- The checkpoint selector monitors `val_score`, which for BYOL is the
  negative cosine loss, so it MAXIMISES agreement and therefore selects
  the most collapsed checkpoint available. The collapsed runs' "best"
  checkpoint is from epoch 3. All comparisons in this wave use
  `last.ckpt`, as the collapsed runs' reported numbers did.
- Every existing baseline result, and every MAJOR result, is untouched.
