# Appendum — the main run is 200 epochs, not 800

Amends `MAJOR_45K_800EPOCH_PREREG_FROZEN_20260913`, which is not edited.

**Disclosure: no unit of this wave has produced a result. The six
800-epoch units were cancelled between 5 and 40 minutes in, and their
directories removed, before any number existed.**

## The change

    max_epochs              800 -> 200
    checkpoint_milestones   [20, 200, 800] -> [20, 200]

Nothing else moves. 45000 training images, three seeds, both datasets,
`paper_composition` under the truncated estimator, every method-side
value as before.

## What it costs and what it buys

At the measured 22.7 epochs/hour a unit is 8.8 hours rather than ~35,
and the wave is ~53 GPU-hours rather than ~211.

It also removes a whole failure mode rather than relying on the fix for
it. Every GPU partition here caps at 24 hours, so an 800-epoch unit
could not finish inside one allocation and depended on being killed and
resumed correctly. A 200-epoch unit finishes inside the cap.

The checkpointing fix stays in place regardless: `last.ckpt` is now
written every 500 training steps instead of only when `fit()` returns.
It is no longer load-bearing for this wave, but a node failure or a
pre-emption at hour seven would otherwise still cost the whole run.

## Consequence for the comparisons

This is the budget that makes the wave comparable rather than the one
that breaks it. At 200 epochs the main run now matches the matched
external baselines on BOTH axes -- 45000 images and 200 epochs -- where
the 800-epoch version would have been epoch-mismatched in our favour and
required labelling or redrawing. `P-M3` of the frozen prereg recorded
that mismatch as a standing caveat; at 200 epochs it does not arise.

The ablation ladder remains at 30000 images, so a full-versus-beta0
contrast is still not drawn against this wave.

## Predictions

P-M1 and P-M2 of the frozen prereg predicted the new arm beats its
200-epoch, 30000-image counterpart. They stand unchanged in direction,
but the remaining manipulation is the data split alone -- 45000 against
30000 at equal epochs -- so the expected effect is smaller than when a
4x epoch budget was also in play. Recorded so the prediction is not
credited with an easier test than it now faces.

P-M5 (new). The epoch-20 checkpoint is a diagnostic, not a result: it
exists to show where the representation is early in training and
carries no prediction.
