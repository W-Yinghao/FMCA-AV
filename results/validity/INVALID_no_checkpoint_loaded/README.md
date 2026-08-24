# Invalid: these runs never loaded their checkpoints

`convergence_extended` was missing from the branch that loads the unit
checkpoint, so all six runs evaluated a freshly initialized backbone.
The giveaway is that all three arms returned identical curves within a
dataset.  Kept for the audit trail, not to be read as results.

Consequence: the T0 remediation babysitter released eighteen retraining
runs on this gate.  See `GRAM_STEP3_RESULTS_*` for the corrected rerun
and the decision that follows from it.

The runner now records a backbone fingerprint and a `weights_loaded`
flag in every record, so this class of failure cannot repeat silently.
