# Recorded deviations and engineering fixes

All entries below precede the new model protocol freeze.

1. Reproduce the published aggregation expressions, not a complete champion
   training/ensemble recipe. The public inference scripts depend on absent final
   feature masks and fitted model files. Therefore exact original column counts
   and champion scores are not claimed.
2. Restrict raw tables to the existing A/B application IDs before aggregation.
   Match `index_in_full` against raw base row order; verify all dates and A/B
   labels exactly. Check unique application IDs and shard duplication. These IDs
   are application identifiers, not established customer identifiers.
3. Fit missing-rate and categorical-cardinality filters only on A. Preserve
   source aggregation expressions, calendar features and relative-date formulas.
   Use table prefixes instead of opaque join suffixes. Omit all-data numerical
   downcasting and CatBoost sentinel filling; both model families receive the
   same A-fitted median/unknown/scaling/missing-indicator information.
4. First/last aggregates preserve the author's ordering. Depth-1 tables sort by
   `num_group1`; this is not proof that every first/last feature is chronological.
   Depth-2 table order is not reinterpreted as a verified event timestamp.
5. Empty A/B-filtered shards caused the old Polars UDF to fail dtype inference.
   Such shards are skipped before aggregation. This does not change nonempty
   application values. The first failure log is retained.
6. Polars 0.20.31 raised an internal multi-chunk gather assertion during joins.
   Explicit rechunking passed a probe over all 15 saved LightGBM source tables.
   No aggregation formula was changed. The second failure log is retained.
7. The Cat kernel casts `case_id` to Int32 whereas mapping uses Int64. Both join
   keys are now explicitly Int64; row order and coverage are asserted. The third
   failure log and join regression tests are retained.

Missing indicators expose information already present in the source missing
mask. They are an explicit transfer hypothesis motivated by the prior audit,
not a claim that the champion specifically used the same appended columns.
The original anonymous TabReD representation retains its inherited transductive
preprocessing limitation. Raw-derived features avoid that specific global
OrdinalEncoder but do not establish strict point-in-time data availability or
repair unknown label maturity. The follow-up does not revise the frozen MVP.
