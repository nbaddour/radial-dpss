# Reproduction check: track_D_boundary.json

Date: 2026-09-16
Script: code/task_25_4_phase3_boundary.py
Manifest: data/task_25_4_phase3/reference_mode_manifest.json
  stored SHA-256:   220b2e27157aed5624687285f786f867f224379360acf1425811707d7b7645e5
  computed SHA-256: 220b2e27157aed5624687285f786f867f224379360acf1425811707d7b7645e5  (match)

Environment of this re-run: numpy 2.4.4, scipy 1.17.1
Pinned in code/requirements.txt:  numpy 2.2.6, scipy 1.15.3   (NOT matched; see caveat)

Fresh output SHA-256: 3ee4025df2c78b17802774c9a4447006c0f35793d949972a2e5d6fb74f59398c

## Comparison against the archived data/task_25_4_phase3/track_D_boundary.json

cases 16/16, rows 128/128, P values and knee modes identical, lambda_ref identical to 1e-14.

  max relative difference, project_V_P                 1.758e-13
  max relative difference, project_V_P_plus_1          8.923e-14
  max relative difference, Boulsane_N_equals_P_plus_1  7.571e-12
  Boulsane epsilon, omega                              0.000e+00 (bit-identical)
  max relative difference, augmented_V_P_plus          1.629e-05
  max ABSOLUTE difference anywhere                     2.204e-14

The augmented column's larger relative figure is round-off at the numerical floor: the five
largest relative differences occur at the five smallest augmented errors (8.7e-11 down to
3.1e-9). At the five rows carrying the quoted claim, agreement is 1e-9 to 1e-11 relative.

Gate flags, fresh run: case_count 16, eligible 16, supported 16,
gate_D_boundary_explanation_passed true, all_augmented_matrix_audits_passed true,
all_Boulsane_matrix_audits_passed true, no_ill_conditioned_rows true.

## Conclusion

Every figure quoted in the manuscript from this file reproduces exactly:
6x / 190x / 428x / 221x / 153x at campaign sizing; 39x to 240445x with median 3087x over the
sweep; 16 of 16 cases favouring the augmented matrix.

The archived JSON was deliberately NOT replaced: it is the provenance of the published numbers
and was produced under the pinned library versions. This log records that it regenerates.

## Caveat

The re-run library versions differ from the pin. A re-run under numpy 2.2.6 / scipy 1.15.3 would
remove the only caveat on this check.
