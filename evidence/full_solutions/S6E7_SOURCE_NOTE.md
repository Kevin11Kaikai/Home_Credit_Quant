# S6E7 source verification

Read on 2026-10-14 from the [official fourth-place writeup](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/4th-place-from-414-to-4-trusting-oof-when-the), by **Ricky / rickyds97**, which credits the model and original probability files to **Masaya Kawamata**.

The writeup confirms seven stratified folds, four independently initialized FT-Transformer members per fold, 13 raw features plus 39 cross-fitted multiclass exact-value TE features, PLR-lite, 16 fixed epochs, and class-prior correction for final balanced-accuracy decisions. No trees are blended into the final submission. The reported private balanced accuracy is 0.95084; it is a source result, not a target or result for Home Credit.

The mechanism has two distinct parts. Prior correction changes class decisions for balanced accuracy and does not improve a probability estimate merely by rescaling it. Exact-value TE exploits repeated numeric values in the synthetic source data. Its usefulness on real credit aggregates must be measured rather than presumed. `reports/exact_value_coverage_AB.csv` audits repetition and later-period coverage without reading any labels or changing the frozen features.

Our transfer retains the complete 28-member training structure and explicit fold isolation. The binary task uses one target mean per raw feature and one sigmoid output. The original source has three classes; this adaptation is not an exact reproduction of the student-health leaderboard score. The source notebook's fixed beta grid is retained only as an A-OOF classification diagnostic. It does not modify probabilities used for Home Credit log loss/Brier.

