# IMDB Sentiment Classification — Combined Model Comparison (CountVectorizer / Bag-of-Words)

# 1. Executive Summary

Eleven models were evaluated on Bag-of-Words features across this series: Decision Tree, Random Forest, XGBoost, ANN, Logistic Regression, LinearSVC, KNN, and four Naive Bayes variants (Bernoulli, Multinomial, Complement, Gaussian).

* **Best model: ANN** (30,000 features, 128→32 dense network) — **90.02% accuracy**, **0.9006 ROC-AUC**. No cross-validation estimate exists for this result, so it should be read as "best on a single split," not "best with statistical confidence."
* **Best model *with* full cross-validation confidence: Logistic Regression** (30,000 features) — **88.78% accuracy**, CV mean 88.09%, tight std (0.0037). This is the most defensible "best model" claim in the series if reliability matters as much as raw score.
* **Headline pattern: simple models are far more competitive under BoW than ensembles.** Logistic Regression and BernoulliNB both outperform Random Forest. BernoulliNB even edges out LinearSVC and XGBoost. Tree ensembles do not dominate the way they're often expected to.
* **Second pattern: models split cleanly into "peaks around 10-20K features" vs. "keeps improving to 30K."** Trees, forests, and Naive Bayes plateau early; linear models, KNN, and the ANN keep gaining all the way to 30K.

---

# 2. Research Questions Answered

**Which model performs best on BoW features?**
By raw accuracy, the ANN (90.02%). By accuracy backed with cross-validation, Logistic Regression (88.78%, CV mean 88.09%). Confidence: high for the ranking of the bottom 9 models; medium for whether ANN genuinely beats Logistic Regression, since the ANN has no CV estimate to confirm its single-split result isn't optimistic.

**Does increasing vocabulary size always help?**
No — it depends on the model family. Decision Tree, Random Forest, XGBoost (untuned), and all Naive Bayes variants peak between 10K-20K features and then plateau or mildly decline. Logistic Regression, LinearSVC, KNN, and the ANN kept improving all the way to 30K, the largest vocabulary tested. See Section 5.

**Is hyperparameter tuning worth the computational cost?**
Mostly yes, but not unconditionally. Decision Tree tuning was a clean win (+2.97 points). XGBoost's first tuning attempt actually *hurt* performance by ~5.8 points due to an overly conservative learning rate, before a redesigned search recovered and then improved on the untuned baseline. See Section 6.

**Which model should I use if...**
— *false positives and false negatives matter equally, and I want the single best score:* ANN.
— *I want a fast, reliable, well-validated model:* Logistic Regression.
— *I need near-instant training and still want ~87-88% accuracy:* BernoulliNB or MultinomialNB.
— *I need to explain individual predictions to a non-technical stakeholder:* Logistic Regression (coefficient-based) or Decision Tree (path-based, though it trades off ~14 points of accuracy for that clarity).
— *I have limited compute and can't afford a 3M+ parameter neural network:* Logistic Regression or Naive Bayes — both land within 2-3 points of the ANN at a tiny fraction of the cost.

---

# 3. Full Leaderboard

| Rank | Model | Best Config | Accuracy | ROC-AUC | CV Mean | CV Std |
| ---- | ----- | ------------ | -------- | ------- | ------- | ------ |
| 1 | **ANN** | 30K features, 128→32 dense, dropout | **90.02%** | **0.9006** | *not performed* | — |
| 2 | **Logistic Regression** | 30K features, max_iter=1000 | **88.78%** | 0.8880 | 88.09% | 0.0037 |
| 3 | BernoulliNB | 25K features | 87.61% | 0.8763 | 87.37% | 0.0069 |
| 4 | LinearSVC | 30K features, min_df=2 | 87.51% | 0.8753 | 86.56% | 0.0024 |
| 5 | XGBoost | 10K features, tuned (gbtree) | 87.45% | 0.8747 | 86.80% | 0.0026 |
| 6 | MultinomialNB | 20K features, min_df=2 | 87.19% | 0.8719 | 86.62% | 0.0035 |
| 7 | ComplementNB | 20K features, min_df=2 | 87.17% | 0.8717 | 86.63% | 0.0037 |
| 8 | Random Forest | 10K features, tuned | 86.13% | 0.8614 | 85.72% | 0.0027 |
| 9 | GaussianNB | 15K features | 81.01% | 0.8092 | 79.37%* | 0.0063 |
| 10 | KNN | 30K features, k=15, cosine, distance-weighted | 76.79% | 0.7671 | 74.26% | 0.0016 |
| 11 | Decision Tree | 5K features, tuned | 74.34% | 0.7441 | 72.97% | 0.0059 |

*GaussianNB's CV mean (79.37%) is *lower* than its test accuracy (81.01%) and even lower than 10K's own CV mean (80.53%) — see Section 7.

## 3a. Leaderboard by Cross-Validation Mean (Alternate View)

Section 3's ranking uses single-split test accuracy, matching how every individual report declared its "best model." But sorting the same nine CV-backed models by **CV mean** instead tells a slightly different story:

| Rank (CV Mean) | Model | CV Mean | Test Accuracy Rank (Section 3) |
| --------------- | ----- | ------- | -------------------------------- |
| 1 | Logistic Regression | 88.09% | #2 (unchanged) |
| 2 | BernoulliNB | 87.37% | #3 (unchanged) |
| 3 | **XGBoost** | **86.80%** | #5 → **moves up 2 spots** |
| 4 | ComplementNB | 86.63% | #7 → moves up 1 spot |
| 5 | MultinomialNB | 86.62% | #6 → moves down 1 spot |
| 6 | **LinearSVC** | **86.56%** | #4 → **moves down 2 spots** |
| 7 | Random Forest | 85.72% | #8 (unchanged) |
| 8 | GaussianNB | 79.37% | #9 (unchanged) |
| 9 | KNN | 74.26% | #10 (unchanged) |
| 10 | Decision Tree | 72.97% | #11 (unchanged) |

The one meaningful swap: **XGBoost's CV mean (86.80%) is actually higher than LinearSVC's (86.56%)**, even though LinearSVC edges it out on single-split test accuracy (87.51% vs. 87.45%). Since CV mean is averaged over 5 folds rather than one split, it's the more trustworthy signal when the two disagree — by that measure, XGBoost is arguably the stronger model of the two, not LinearSVC.

## 3b. Are the Top Rankings Statistically Meaningful?

Using each model's CV mean ± 1 standard deviation as a rough confidence band:

| Model | CV Mean | ±1 Std Range |
| ----- | ------- | -------------- |
| Logistic Regression | 88.09% | [87.72%, 88.46%] |
| BernoulliNB | 87.37% | [86.68%, 88.06%] |
| XGBoost | 86.80% | [86.54%, 87.06%] |
| ComplementNB | 86.63% | [86.26%, 87.00%] |
| MultinomialNB | 86.62% | [86.27%, 86.97%] |
| LinearSVC | 86.56% | [86.32%, 86.80%] |
| Random Forest | 85.72% | [85.45%, 85.99%] |

Reading down this list, **every adjacent pair from Logistic Regression through LinearSVC overlaps** — meaning the exact rank order among these six is not strongly distinguishable from noise, band to neighboring band. What *is* clearly distinguishable: **Random Forest's range does not overlap with LinearSVC's** (85.99% vs. 86.32%), so Random Forest is a genuine, statistically real step down from that top cluster — not just a coincidence of one split.

**Practical reading:** treat Logistic Regression, BernoulliNB, XGBoost, ComplementNB, MultinomialNB, and LinearSVC as a **statistical tie for the top tier** rather than a strict ranking, with Random Forest as a clear notch below them, and ANN as an unconfirmed (no-CV) claim to the very top.

---

# 4. Model Family Comparison

| Family | Models | Best in Family | Takeaway |
| ------- | ------ | ---------------- | -------- |
| Neural | ANN | **90.02%** | Strongest raw performance, but the only model with no CV confidence estimate — and needed the largest vocabulary (30K) and by far the most parameters (3.8M) to get there. |
| Linear / Probabilistic | Logistic Regression, LinearSVC, 4× Naive Bayes | **88.78% (Logistic Regression)** | The strongest *family* overall — five of the top seven models belong here. Naive Bayes in particular delivers ~87% accuracy at negligible training cost. |
| Tree Ensembles | Random Forest, XGBoost | **87.45% (XGBoost)** | Solidly mid-table — beats the single tree and KNN comfortably, but does **not** outperform the best linear/probabilistic models on this dataset, contrary to the usual expectation that ensembles dominate. |
| Single Tree | Decision Tree | 74.34% | Weakest model in the series by a wide margin (12+ points behind the next-weakest). Confirms trees alone are poorly suited to high-dimensional sparse BoW features; ensembling (Random Forest, XGBoost) closes most — but not all — of the gap to linear models. |
| Distance-based | KNN | 76.79% | Second-weakest. Even after switching from Euclidean to cosine distance (its single biggest lever), it never closes the gap to the linear/probabilistic models. |

The standout finding: **ensembling trees only gets you to "solid mid-table," not to the top.** On this dataset, a plain Logistic Regression and even a Naive Bayes variant beat both tree ensembles.

## Computational Cost by Family

Accuracy alone doesn't tell you what a model costs to run. A rough qualitative comparison, since exact training times weren't logged in this series:

| Family | Training Cost | Inference Cost | Notes |
| ------- | -------------- | ---------------- | ----- |
| Naive Bayes (4 variants) | Very low — closed-form counting | Very low | Fastest model to train in the entire series by a wide margin; no hyperparameter search needed to be competitive. |
| Linear (Logistic Regression, LinearSVC) | Low-moderate — convex optimization | Very low | Cheap in both directions; scales predictably with feature count. |
| Single Tree (Decision Tree) | Low | Very low | Cheap but the weakest performer — the accuracy trade-off isn't worth the savings here. |
| Tree Ensembles (Random Forest, XGBoost) | Moderate-high | Moderate | The hyperparameter search overhead (`HalvingRandomSearchCV`) added real cost in this series, on top of already training hundreds of trees. |
| KNN | None (lazy learner — no training phase) | **High** | The trade-off is inverted: free to "train," but every single prediction requires scanning the full training set, making it the most expensive model in the series at serving time despite having no training cost at all. |
| ANN | High (needs multiple dense layers, ideally GPU) | Low (single forward pass) | Priciest to train (3.8M parameters at its best configuration), cheapest of the "expensive-to-train" models to actually serve. |

If deployment cost matters as much as accuracy, **Naive Bayes is the standout value pick** in this series — it lands within 1-3 points of the top models at a small fraction of the training and infrastructure cost.

---

# 5. Vocabulary Size: Cross-Model Trends

| Model | 5K | 10K | 15K | 20K | 25K | 30K | Pattern |
| ----- | -- | --- | --- | --- | --- | --- | ------- |
| Decision Tree | 74.34% | 74.08% | *not run* | — | — | — | Peaks at 5K (see caveat, Section 7) |
| Random Forest | 85.46% | 86.13% | 85.48% | — | — | — | Peaks at 10K, declines |
| XGBoost (untuned) | 85.55% | 86.39% | 86.01% | — | — | — | Peaks at 10K, declines |
| BernoulliNB | 85.63% | 86.99% | 87.33% | 87.33% | **87.61%** | — | Still climbing at 25K, but slowing |
| MultinomialNB | 85.00% | 86.21% | 86.69% | **87.19%** | 86.95% | — | Peaks at 20K, declines |
| ComplementNB | 84.96% | 86.17% | 86.69% | **87.17%** | 86.95% | — | Peaks at 20K, declines |
| GaussianNB | 79.48% | 80.49% | **81.01%** | 80.73% | — | — | Peaks at 15K, declines |
| Logistic Regression | 85.79% | *not run* | 87.85% | 88.26% | 88.44% | **88.78%** | Monotonic increase (gap at 10K) |
| LinearSVC | 83.46% | 85.36% | 86.00% | 86.67% | 87.25% | **87.51%** | Monotonic increase, no plateau yet |
| KNN | 72.62% | 74.44% | 74.84% | 75.91% | 76.39% | **76.79%** | Monotonic increase, no plateau yet |
| ANN | 86.65%† | 88.92% | 89.08% | 89.32% | — | **90.02%** | Monotonic increase, no plateau yet |

†ANN's 5K figure uses the regularized [64,16] architecture, not the earlier minimal [6,6] run, for a fair comparison to the other 5K rows.

**The pattern is clean and consistent**: capacity-constrained models (single trees, forests, untuned boosting, Naive Bayes) hit a ceiling around 10K-20K features and gain little or lose ground afterward. Models that scale smoothly with more input dimensions — linear models, KNN, and the ANN — kept improving through every vocabulary size tested, with none of them showing signs of plateauing even at 30K. This suggests **the next natural experiment is testing vocabularies beyond 30K for those four models specifically**, since their ceiling hasn't been found yet.

---

# 6. Hyperparameter Tuning: What Worked, What Didn't

| Model | Untuned | Tuned | Gain | Notes |
| ----- | ------- | ------ | ---- | ----- |
| Decision Tree | 71.37% (default CV, no restriction) | 74.34% (5K, tuned) | **+2.97 pts** | Clean, straightforward win from RandomizedSearchCV. |
| XGBoost, Round 1 | 86.39% (10K, untuned gbtree) | 80.63% (10K, tuned) | **-5.76 pts** | Tuning *hurt* — the search pinned `learning_rate` at 0.01, undertraining the ensemble. |
| XGBoost, manual fix | 80.63% | 87.03% (lr raised to 0.08) | +6.40 pts (recovery) | Confirmed the low learning rate was the cause. |
| XGBoost, Round 2 (redesigned search) | 86.39% (untuned) | 87.45% (10K, `n_estimators`=1000) | **+1.06 pts** | Net positive once the search itself was redesigned (continuous ranges, `resource='n_estimators'`). |
| Random Forest | *(no directly comparable untuned baseline was run)* | 86.13% (10K) | N/A | HalvingRandomSearchCV was used from the start; still beat Decision Tree by ~12 points via ensembling alone. |

**Takeaway:** tuning delivered a net positive in every case it was properly executed, but XGBoost's Round 1 is a genuine cautionary tale — a poorly-designed search grid can actively make a model worse than doing nothing, and the only way this was caught was by keeping an untuned baseline in the results table for comparison. That practice paid for itself directly in this series.

---

# 7. Known Issues & Caveats Ledger

| # | Issue | Model(s) | Status | Impact on Conclusions |
| - | ------ | --------- | ------- | ----------------------- |
| 1 | Missing 15K tuned evaluation | Decision Tree | **Open** | Minor — leaves a gap in the vocabulary-size table (Section 5) |
| 2 | Missing 10K run | Logistic Regression | **Open** | Minor — same kind of gap, different model |
| 3 | No cross-validation performed | ANN | **Open** | Significant — the #1 ranking in the leaderboard rests on a single train/test split with no stability estimate |
| 4 | Default-vectorizer run failed with an out-of-memory error | GaussianNB | **Open, and likely not worth fixing** | Minor — GaussianNB is the weakest Naive Bayes variant regardless; not competitive even with the runs that did complete |
| 5 | SVC (RBF kernel) abandoned for excessive training time | — | **By design, not an error** | None — documented as a deliberate scope decision |
| 6 | ComplementNB and MultinomialNB scores coincide almost exactly at several vocab sizes | Naive Bayes | **Explained, not a bug** | None — mathematically expected for binary classification |
| 7 | KNN's `k`, distance metric, and vocabulary size were changed together across most of the sweep | KNN | **Open (methodological)** | Minor — makes it hard to isolate which change drove KNN's improvement, aside from the clean 30K-only `k` comparison |

---

# 8. Practical Recommendation

* **Best raw accuracy:** ANN — use if squeezing out the last fraction of a percent matters more than validated confidence or interpretability, and you can afford ~3.8M parameters and GPU/CPU training time.
* **Best accuracy with full statistical confidence:** Logistic Regression — the strongest CV-backed result in the series, with a tight standard deviation (0.0037) and a fraction of the ANN's computational cost.
* **Best accuracy-per-compute:** BernoulliNB or MultinomialNB — both land within 1-1.5 points of LinearSVC and XGBoost, train in a fraction of a second, and require no GPU, no hyperparameter search, and almost no tuning.
* **Most interpretable:** Logistic Regression (per-feature coefficients) if you need global interpretability, or Decision Tree if you specifically need to trace individual prediction paths (at a real accuracy cost).
* **Most stable across folds among the top performers:** LinearSVC (CV std = 0.0024, the tightest of any model scoring above 85% accuracy).
* **Avoid for this dataset:** Decision Tree alone (worst accuracy in the series) and KNN (second-worst, plus the slowest inference time of any model here since it must compare against the full training set at prediction time).

## Precision/Recall Trade-off (By the Numbers)

The generic advice above is backed by the actual class-level metrics of the top five models (Class 0 = negative, Class 1 = positive):

| Model | Class 0 Precision | Class 0 Recall | Class 1 Precision | Class 1 Recall |
| ----- | -------------------- | ----------------- | -------------------- | ----------------- |
| ANN | **0.93** | 0.87 | **0.88** | **0.93** |
| Logistic Regression | 0.90 | **0.87** | 0.87 | 0.91 |
| BernoulliNB | 0.89 | 0.86 | 0.86 | 0.89 |
| LinearSVC | 0.89 | 0.86 | 0.86 | 0.89 |
| XGBoost | 0.89 | 0.86 | 0.86 | 0.89 |

Reading this directly:

* **If minimizing false negatives on positive sentiment matters most** (i.e., you'd rather over-flag negative reviews than miss a genuinely positive one) — ANN has the highest Class 1 recall (0.93), with Logistic Regression a clear second (0.91).
* **If minimizing false positives matters most in either direction** — ANN has the highest precision on *both* classes (0.93 / 0.88), making it the safest single choice if you can't decide which error type costs more.
* **Among the CV-validated models**, Logistic Regression is consistently the runner-up to ANN on every one of these four numbers — reinforcing Section 3's recommendation that it's the most defensible "second-best" choice if ANN's unvalidated lead makes you want a fallback.
* **BernoulliNB, LinearSVC, and XGBoost are essentially tied** on class-level metrics (identical to two decimal places) — so among *those* three, the choice should come down to Section 4's cost comparison (Naive Bayes wins on cost) rather than any accuracy or error-type difference.

---

# 9. Next Steps

1. **Cross-validate the ANN** (via a `scikeras`-wrapped `cross_val_score`) to confirm its #1 ranking isn't an artifact of one favorable train/test split — this is the single most important open item, since it's the current headline result.
3. **Fill the two vocabulary-size gaps**: Decision Tree's missing 15K run and Logistic Regression's missing 10K run, so the Section 5 table is complete across all models.
4. **Test vocabulary sizes beyond 30K** for the four models that hadn't plateaued yet (Logistic Regression, LinearSVC, KNN, ANN) — their true ceiling on this dataset is still unknown.
5. **Move to the TF-IDF combined report** using the same structure, then synthesize a BoW-vs-TF-IDF comparison across all eleven models — early signs from the individual reports (e.g., Decision Tree, ANN) suggest TF-IDF has a modest but real edge, worth confirming at the full-leaderboard level.
6. **Proceed to the next roadmap phase** (Word2Vec/GloVe embeddings, then RNN/LSTM) now that the classical ML phase has a complete, cross-validated-where-possible baseline to compare future architectures against.

---

# Appendix: Individual Model Reports

Full per-model detail — architecture/hyperparameter grids, precision/recall by class, ROC-AUC breakdowns, and model-specific caveats — is available in the individual reports already produced for: Logistic Regression, LinearSVC, KNN, and the four Naive Bayes, Decision Tree, Random Forest, XGBoost, and ANN. 