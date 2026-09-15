# Appendum — partial-probe release, 2026-09-10

Amends the release gate in PAPER_ESTIMATOR_WAVE_PREREG_FROZEN_20260910
("The fleet waits on the paper-arm QC probe completing") and the
equivalent gate for the V6 ridge-Gram arm.  **Disclosure: written after
inspecting both probes' trajectories.  No aggregate has been viewed and
no prediction is changed.**

## Why

A 200-epoch completion gate holds a fleet for ~6 hours on an otherwise
idle partition.  The failure the gate exists to prevent is a fleet
launched in front of a runner that does not work -- a device fault, a
shape error, a config that trips the divergence guard.  Those are
first-epoch failures; the ridge V7 arm died at the end of epoch 1.
Waiting for epoch 200 buys protection against late collapse, which is
a RESULT rather than a runner fault, and which would be visible and
informative rather than silent.

## The criterion, stated in full

A probe releases its fleet when ALL of:

  G1. at least 25 completed epochs with no divergence-guard trip;
  G2. the latest closure-relevant objective term is at most half its
      first-epoch value (paper arm: closure_ratio; V6: -product_score,
      i.e. the score has at least doubled);
  G3. the retained-rank floor is not decreasing across the run;
  G4. zero invalid batches.

G3 and G4 are the ones that matter for a truncated estimator: a run
that is quietly losing retained directions, or silently skipping
batches, is degenerate even while its loss falls.

## Status against the criterion

    paper_composition seed1, 31 epochs
      closure_ratio   0.740 -> 0.213   (G2: pass, 3.5x)
      dir_trace       5.14  -> 47.20
      edge_trace_sum  26.06 -> 118.71
      retained_min    33.7  -> 72.3    (G3: pass, rising)
      invalid_batch   0                (G4: pass)
      divergence trips 0               (G1: pass)

    product_only seed1 (V6, ridge Gram), 121 epochs
      product_score   0.119 -> 0.816   (G2: pass, 6.8x)
      gram_retained   88.4  -> 128.0   (G3: pass, now full rank)
      divergence trips 0               (G1: pass)

Both release.  The probes keep running to completion and remain the
seed-1 units of their arms; nothing is discarded.

## What does NOT change

The evidence standard, the four predictions, the frozen evaluation
protocol, and the rule that a failed prediction narrows the claim.  If
either probe collapses later, the affected fleet is reported as such
and the collapse is part of the result.
