# T0 at the enlarged budget: convergence and the random floor — 2026-08-26

Interpretation commit for ebe9616.  Six convergence sweeps (three arms
x two datasets, bigcal checkpoints, Gram-corrected) reaching N=20000,
plus the two random-encoder controls at the enlarged budget.

## Criterion 1 (E-A2, frozen tolerance 0.005 per doubling)

    ratio vs N            625     1250    2500    5000    10000   20000   top-drift
    c10  flat             0.550   0.493   0.483   0.475   0.474   0.473   0.0013  PASS
    c10  additive         1.224   0.536   0.330   0.237   0.185   0.165   0.0197  FAIL
    c10  V7               1.171   0.502   0.305   0.222   0.185   0.160   0.0250  FAIL
    c100 flat             0.605   0.506   0.485   0.479   0.476   0.476   0.0002  PASS
    c100 additive         2.310   0.681   0.388   0.257   0.198   0.170   0.0283  FAIL
    c100 V7               2.658   0.621   0.342   0.235   0.193   0.167   0.0261  FAIL

The verdicts under the FROZEN criterion stand as written: flat passes,
the hierarchical arms fail at every tested N.

The mechanism is now visible, though.  The hierarchical drifts HALVE
per doubling (c10 additive: 0.206, 0.093, 0.052, 0.020) — clean 1/N
convergence toward a nonzero limit of roughly 0.15, approached from
above because the numerator of a SMALL defect is noise-floor-dominated
until N pushes the floor below it.  Flat converges instantly because
its defect (~0.47) is structural and large.  An absolute
0.005-per-doubling tolerance therefore cannot be met by a small-defect
arm on a 60k-image dataset at ANY reachable N (the halving law puts
the requirement at N ≈ 80000).  The criterion conflates "not
converged" with "small estimand"; a prospective amendment (drift-ratio
test: successive drifts halving, plus an extrapolated-limit stability
band) is proposed for the NEXT prereg — the frozen verdict above is
not retroactively changed.

## Criterion 2: the random floor at the enlarged budget

    ratio at budget       random encoder     V7        additive    flat
    c10                   0.330              0.160     0.165       0.473
    c100                  0.315              0.167     0.170       0.476

E-A1's voiding result (random encoder BEATING V7, 0.339 vs 0.370)
REVERSES at the enlarged budget: the trained hierarchical arms now sit
at half the random floor, on both datasets, while flat sits at 1.4x
above it.  The ordering hierarchical < random < flat is exactly what a
sane closure measure should produce, and it emerges once the
calibration budget pushes the estimation floor below the arm
differences.  Together with criterion 1's mechanism, the two criteria
tell one story: the 08-24 T0 failure was a failure of the SMALL-BUDGET
protocol, not of the arms — and the enlarged-budget protocol is the
one the paper must use.

## Consequences

- The bigcal reversal (V7 smallest closure defect of any arm) survives
  its deepest test to date: V7 < additive < random < flat at N=20000,
  both datasets, ordering stable from N=5000 upward.
- The ratio form remains voided for SMALL budgets (the frozen E-A1
  verdict), and regains standing at the enlarged budget where its
  random floor separates from the arms.  Any figure quoting ratios
  must state the budget.
- E-D1 side note, chain track: the budget scan under gram_bound holds
  the depth ordering at K = 16/32/64/128 on both datasets, and the
  gram_bound sweep (0.02/0.05/0.2) moves the stem defect by under 8%
  with the ordering untouched — the new coordinate rule's knobs are
  benign where the retired floor was not.
