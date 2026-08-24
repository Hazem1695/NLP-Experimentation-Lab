# IMDB Sentiment Classification — Combined Model Comparison (TfidfVectorizer)

# 1. Executive Summary

Eleven models were evaluated on TF-IDF features: Decision Tree, Random Forest, XGBoost, ANN, Logistic Regression, LinearSVC, KNN, and four Naive Bayes variants.

* **Best model: LinearSVC** (30,000 features, min_df=2, C=0.5) — **90.18% accuracy**, **0.9020 ROC-AUC**, CV mean 89.40% (std 0.0010). This is the strongest *fully cross-validated* result in either vectorizer series to date.
* **Best raw single-split result: ANN** (15,000 features) — **90.30% accuracy**, 0.9030 ROC-AUC — narrowly ahead of LinearSVC, but with no CV estimate to confirm the lead is real rather than a favorable split.
* **Headline finding #1: TF-IDF produces a statistically much cleaner ranking than BoW did.** Where the BoW report found six models in a statistical tie, TF-IDF's CV-mean confidence bands show LinearSVC, Logistic Regression, the Naive Bayes cluster, and XGBoost as four genuinely distinct, non-overlapping tiers. See Section 3b.
* **Headline finding #2: TF-IDF helps some model families far more than others.** KNN (+3.62 points vs. its BoW result), GaussianNB (+2.65), and LinearSVC (+2.67) all improved substantially. XGBoost (-0.32) and Random Forest (-0.38) actually got *slightly worse* under TF-IDF. See Section 6.
* **Biggest open caveat:** ANN has no cross-validation estimate, so its narrow lead over LinearSVC (90.30% vs. 90.18%) can't yet be confirmed as statistically real rather than single-split noise.

---

# 2. Research Questions Answered

**Which model performs best on TF-IDF features?**
LinearSVC, if you want a result backed by cross-validation (90.18%, CV mean 89.40%). ANN, if you only care about the single best number achieved (90.30%) and are willing to accept it's unconfirmed by CV. Confidence: high that LinearSVC is a genuinely strong, reliable top performer; medium on whether ANN truly beats it.

**Does increasing vocabulary size always help?**
No — same split as BoW, but with different specific members. Decision Tree, Random Forest, XGBoost, and GaussianNB peak between 10K-15K features and decline afterward. Logistic Regression, LinearSVC, the sparse-input Naive Bayes variants (Bernoulli/Multinomial/Complement), and KNN kept improving through the largest vocabulary tested. The ANN is a genuine outlier here: unlike its BoW counterpart (which climbed monotonically to 30K), the TF-IDF ANN peaks at 15K and mildly declines afterward. See Section 5.

**Is hyperparameter tuning worth the computational cost?**
Yes, with more consistent payoff than BoW showed. XGBoost's search correctly found strong parameters this time (no repeat of the BoW Round-1 failure), and a manual follow-up (raising `n_estimators` past the search's own ceiling) added a further real gain. LinearSVC's regularization tuning (`C=0.5`) was essential to reaching the top result — vocabulary size alone plateaued at 89.86% (25K), and tuning `C` pushed it to 90.18%. See Section 7.

**Which model should I use if...**
— *I want the single most defensible best model:* LinearSVC — highest CV-validated accuracy, tightest CV std of any top-tier model (0.0010).
— *I want the highest raw score and don't need statistical confidence:* ANN.
— *I need near-instant training with strong accuracy:* MultinomialNB — 88.13% at negligible cost.
— *I need interpretability:* Logistic Regression — 89.46%, third-best overall, fully coefficient-interpretable.
— *I have limited compute:* Naive Bayes or Logistic Regression, both well within 2 points of the top and orders of magnitude cheaper than the ANN.

---

# 3. Full Leaderboard

| Rank | Model | Best Config | Accuracy | ROC-AUC | CV Mean | CV Std |
| ---- | ----- | ------------ | -------- | ------- | ------- | ------ |
| 1 | **ANN** | 15K features, 128→32 dense, dropout | **90.30%** | **0.9030** | *not performed* | — |
| 2 | **LinearSVC** | 30K features, min_df=2, C=0.5 | **90.18%** | 0.9020 | 89.40% | 0.0010 |
| 3 | Logistic Regression | 20K features, min_df=2 | 89.46% | 0.8948 | 88.58% | 0.0015 |
| 4 | MultinomialNB | 30K features, min_df=2 | 88.13% | 0.8815 | 87.63% | 0.0038 |
| 5 | BernoulliNB | 35K features, min_df=2 | 88.07% | 0.8809 | 87.46% | 0.0043 |
| 5 | ComplementNB | 30K features, min_df=2 | 88.07% | 0.8809 | 87.61% | 0.0040 |
| 7 | XGBoost | 10K features, tuned | 87.13% | 0.8714 | 86.23% | 0.0029 |
| 8 | Random Forest | 10K features, tuned | 85.75% | 0.8579 | 85.76% | 0.0039 |
| 9 | GaussianNB | 15K features | 83.66% | 0.8363 | 81.47% | 0.0030 |
| 10 | KNN | 30K features, k=19, cosine, distance-weighted | 80.41% | 0.8047 | 79.85% | 0.0037 |
| 11 | Decision Tree | 10K features, tuned | 74.94% | 0.7504 | 72.87% | 0.0024 |

## 3a. Leaderboard by Cross-Validation Mean

| Rank (CV Mean) | Model | CV Mean | Test Accuracy Rank (above) |
| --------------- | ----- | ------- | ----------------------------- |
| 1 | LinearSVC | 89.40% | #2 (unchanged) |
| 2 | Logistic Regression | 88.58% | #3 (unchanged) |
| 3 | MultinomialNB | 87.63% | #4 (unchanged) |
| 4 | ComplementNB | 87.61% | #5 → moves up 1 spot (breaks the test-accuracy tie with BernoulliNB) |
| 5 | BernoulliNB | 87.46% | #5 → moves down 1 spot |
| 6 | XGBoost | 86.23%* | #7 (unchanged) |
| 7 | Random Forest | 85.76% | #8 (unchanged) |
| 8 | GaussianNB | 81.47% | #9 (unchanged) |
| 9 | KNN | 79.85% | #10 (unchanged) |
| 10 | Decision Tree | 72.87% | #11 (unchanged) |

*XGBoost's 20K configuration actually has a marginally higher CV mean (86.40%) than its 10K configuration (86.23%, shown above as the "best" by test accuracy) — see the XGBoost individual report for this nuance. Either way it doesn't change XGBoost's rank relative to the other models.

Unlike the BoW report, **the ranking barely moves when switching from test accuracy to CV mean** — only a tiny tie-break among the three Naive Bayes variants changes. This is itself informative: TF-IDF's rankings are more reproducible and less sensitive to which specific metric you trust.

## 3b. Are the Top Rankings Statistically Meaningful?

Using CV mean ± 1 standard deviation as a confidence band:

| Model | CV Mean | ±1 Std Range |
| ----- | ------- | -------------- |
| LinearSVC | 89.40% | [89.30%, 89.50%] |
| Logistic Regression | 88.58% | [88.43%, 88.73%] |
| MultinomialNB | 87.63% | [87.25%, 88.01%] |
| ComplementNB | 87.61% | [87.21%, 88.01%] |
| BernoulliNB | 87.46% | [87.03%, 87.89%] |
| XGBoost | 86.23% | [85.94%, 86.52%] |
| Random Forest | 85.76% | [85.37%, 86.15%] |
| GaussianNB | 81.47% | [81.17%, 81.77%] |
| KNN | 79.85% | [79.48%, 80.22%] |
| Decision Tree | 72.87% | [72.63%, 73.11%] |

**This is a dramatically cleaner picture than BoW.** Checking adjacent pairs: LinearSVC's range does *not* overlap with Logistic Regression's — a genuine, statistically real gap. Logistic Regression's range does not overlap with the Naive Bayes cluster below it. The Naive Bayes cluster (all three sparse variants) does not overlap with XGBoost. XGBoost and Random Forest overlap slightly (the one ambiguous adjacent pair). GaussianNB, KNN, and Decision Tree are each clearly, non-overlappingly separated from their neighbors.

**Practical reading:** unlike BoW's "six-way statistical tie at the top," TF-IDF gives you a genuine, defensible hierarchy: **LinearSVC > Logistic Regression > {Naive Bayes cluster} > XGBoost ≈ Random Forest > GaussianNB > KNN > Decision Tree**, with ANN's claim to #1 still unconfirmed pending cross-validation.

---

# 4. Model Family Comparison

| Family | Models | Best in Family | Takeaway |
| ------- | ------ | ---------------- | -------- |
| Neural | ANN | **90.30%** | Best raw score, but the *only* model in the entire series (both vectorizers) with no CV confidence estimate. |
| Linear | Logistic Regression, LinearSVC | **90.18% (LinearSVC)** | The strongest *validated* family — LinearSVC's TF-IDF result is the best CV-backed number across the whole project so far. |
| Probabilistic (Naive Bayes) | Bernoulli, Multinomial, Complement, Gaussian | **88.13% (MultinomialNB)** | Remarkably close to Logistic Regression (within 1.3 points) at a fraction of the cost — the same value story as BoW, just at a slightly higher ceiling. |
| Tree Ensembles | Random Forest, XGBoost | **87.13% (XGBoost)** | Notably, this is the *only* family that did not improve under TF-IDF compared to BoW — see Section 6. |
| Single Tree | Decision Tree | 74.94% | Weakest model, consistent with BoW. |
| Distance-based | KNN | 80.41% | Second-weakest, but the family that improved the most from switching vectorizers (+3.62 points vs. BoW). |

## Computational Cost by Family

The cost story from the BoW report holds here without change: Naive Bayes remains the cheapest to train by a wide margin, KNN remains the most expensive at inference time despite having no training cost, and the ANN remains the most expensive to train overall. See the BoW combined report (Section 4) for the full cost table — nothing about vectorizer choice changes these underlying cost characteristics, only the accuracy each model achieves.

---

# 5. Vocabulary Size: Cross-Model Trends

| Model | 5K | 10K | 15K | 20K | 25K | 30K | 35K | Pattern |
| ----- | -- | --- | --- | --- | --- | --- | --- | ------- |
| Decision Tree | 71.99% | **74.94%** | 73.98% | — | — | — | — | Peaks at 10K, declines |
| Random Forest | 84.92% | **85.75%** | 85.52% | — | — | — | — | Peaks at 10K, declines |
| XGBoost (tuned) | 86.95% | **87.13%** | 87.07% | 86.85% | — | — | — | Peaks at 10K, gently declines |
| GaussianNB | 81.85% | 83.50% | **83.66%** | 82.94% | — | — | — | Peaks at 15K, declines |
| ANN | 88.92% | 89.72% | **90.30%** | 90.06% | — | — | — | Peaks at 15K, mildly declines — unlike BoW's ANN, which never plateaued |
| BernoulliNB | 85.63% | 86.99% | 87.33% | 87.33% | 87.61% | 87.91% | **88.07%** | Still climbing at 35K |
| MultinomialNB | 85.67% | 87.11% | 87.49% | 87.65% | 87.87% | **88.13%** | — | Still climbing at 30K |
| ComplementNB | 85.59% | 87.13% | 87.49% | 87.63% | 87.95% | **88.07%** | — | Still climbing at 30K |
| Logistic Regression | 88.56% | 88.94% | 89.40% | **89.46%** | 89.42% | — | — | Peaks at 20K, essentially flat after |
| LinearSVC | 87.07% | — | 89.08% | 89.40% | 89.86% | **90.18%*** | — | Vocabulary alone peaks at 25K; tuning `C` at 30K pushes past it |
| KNN | 75.89% | 77.47% | 78.26% | 79.28% | 79.46% | **80.41%** | — | Still climbing at 30K |

*LinearSVC's raw 30K vocabulary-only result was 89.56%; the 90.18% figure required also tuning `C=0.5` — see Section 7.

**The same fundamental split as BoW holds**: capacity-constrained models (single trees, forests, boosting, GaussianNB) plateau by 10-15K; models that scale smoothly with dimensionality (linear models, sparse Naive Bayes, KNN) keep improving through the largest vocabulary tested. The one notable change is the **ANN**, which plateaus early under TF-IDF (15K) but never plateaued under BoW even at 30K — suggesting TF-IDF's weighting lets a smaller vocabulary carry more signal for this architecture, so it "runs out" of useful new information sooner.

---

# 6. TF-IDF vs. BoW: Which Models Benefit Most

| Model | Best BoW | Best TF-IDF | Change |
| ----- | -------- | ------------ | ------ |
| **KNN** | 76.79% | 80.41% | **+3.62** |
| **LinearSVC** | 87.51% | 90.18% | **+2.67** |
| **GaussianNB** | 81.01% | 83.66% | **+2.65** |
| MultinomialNB | 87.19% | 88.13% | +0.94 |
| ComplementNB | 87.17% | 88.07% | +0.90 |
| Logistic Regression | 88.78% | 89.46% | +0.68 |
| Decision Tree | 74.34% | 74.94% | +0.60 |
| BernoulliNB | 87.61% | 88.07% | +0.46 |
| ANN | 90.02% | 90.30% | +0.28 |
| Random Forest | 86.13% | 85.75% | **-0.38** |
| XGBoost | 87.45% | 87.13% | **-0.32** |

Two clear clusters emerge. **Distance- and margin-sensitive models gain the most**: KNN computes distances directly on the feature vectors, so down-weighting common words and normalizing vector length (both things TF-IDF does) directly improves its notion of "similarity." LinearSVC's margin-based objective and GaussianNB's per-feature Gaussian assumption both benefit similarly from features that are already scaled and weighted rather than raw sparse counts. **Tree-based models gain nothing, and ensembles actively lose a little ground**: trees split on thresholds, and a threshold on a raw count is about as informative as a threshold on a TF-IDF weight — the monotonic transformation TF-IDF applies doesn't fundamentally change what a tree can learn from the data, and appears to have cost XGBoost and Random Forest a small amount of signal in practice (possibly because TF-IDF slightly compresses the separation between very common and very rare terms that raw counts preserved more sharply).

**Practical implication:** if you're choosing a vectorizer for a tree-based model, BoW is at least as good and simpler to reason about. If you're choosing for anything distance-based, margin-based, or probabilistic-continuous, TF-IDF is clearly worth the switch.

---

# 7. Hyperparameter Tuning: What Worked, What Didn't

| Model | Untuned/Default | Tuned | Gain | Notes |
| ----- | ---------------- | ------ | ---- | ----- |
| XGBoost | 86.29% (5K, untuned) | 87.13% (10K, tuned) | +0.84 pts | Clean win this time — no repeat of the BoW Round-1 learning-rate failure. |
| LinearSVC | 89.56% (30K, default C=1.0) | 90.18% (30K, C=0.5) | **+0.62 pts** | Regularization tuning was necessary to beat the vocabulary-only peak (89.86% at 25K) — a case where tuning mattered more than more data. |
| LinearSVC | — | 89.98% (30K, `sublinear_tf=True`, C=0.5) | -0.20 pts vs. the C=0.5 result without it | A documented **negative** result: `sublinear_tf` is a commonly-recommended TF-IDF technique that didn't help here. |
| Random Forest | *(HalvingRandomSearchCV used from the start)* | 85.75% (10K) | N/A | No isolated untuned baseline for direct comparison, same limitation as the BoW report. |

**Takeaway:** tuning was a consistently positive story under TF-IDF, in contrast to BoW's XGBoost cautionary tale. The one negative result (`sublinear_tf`) is worth keeping as documented evidence rather than silently omitting — not every commonly-recommended technique helps on every dataset.

---

# 8. Known Issues & Caveats Ledger

| # | Issue | Model(s) | Status | Impact |
| - | ------ | --------- | ------- | ------- |
| 1 | `n_estimators` 180 (search) vs. 1000 (manual follow-up) | XGBoost | **Resolved** — deliberate, beneficial change, not an error | None |
| 2 | No cross-validation performed | ANN | **Open** | Significant — the #1 ranking rests on a single split |
| 3 | Doubling ANN width (128→32 vs. 256→64) at a fixed 20K vocabulary made results worse (90.06% → 89.74%) | ANN | **Confirmed finding, not an issue** | None — this is the same "wider isn't better" result seen in the BoW ANN report, now independently reproduced under TF-IDF too |

---

# 9. Practical Recommendation

* **Best overall (validated):** LinearSVC — highest CV-backed accuracy in the entire project, with the tightest CV std of any top-tier model.
* **Best overall (raw score, unvalidated):** ANN — use if the extra 0.12 points matters more than statistical confidence, and you're prepared for the training cost.
* **Best accuracy-per-compute:** MultinomialNB — 88.13% at negligible training cost, and the most internally consistent of the three sparse Naive Bayes variants under TF-IDF.
* **Most interpretable:** Logistic Regression — third overall, fully coefficient-interpretable, and the runner-up on nearly every metric behind LinearSVC.
* **Most stable across folds:** LinearSVC (CV std = 0.0010) — the tightest of any model in either vectorizer series.

## Precision/Recall Trade-off (By the Numbers)

| Model | Class 0 Precision | Class 0 Recall | Class 1 Precision | Class 1 Recall |
| ----- | -------------------- | ----------------- | -------------------- | ----------------- |
| ANN | 0.90 | 0.90 | 0.90 | 0.90 |
| LinearSVC | 0.92 | 0.88 | 0.89 | **0.92** |
| Logistic Regression | **0.91** | 0.87 | 0.88 | 0.92 |
| MultinomialNB | 0.89 | 0.87 | 0.87 | 0.89 |

* **ANN is the only model with perfectly balanced precision and recall across both classes** — a genuinely clean result, and additional (if indirect) evidence that its performance is real rather than a lucky split favoring one class.
* **If minimizing false negatives on positive sentiment matters most**, LinearSVC and Logistic Regression are tied for the best Class 1 recall (0.92), both ahead of the perfectly-balanced ANN (0.90) on this specific metric.
* **If minimizing false positives on negative sentiment matters most**, Logistic Regression has the edge (0.91 Class 0 precision).
* Unlike the BoW report — where the top models were nearly identical on class-level metrics — TF-IDF's top three models each have a genuinely distinct precision/recall profile, giving a real choice depending on which error type matters more for your use case.

---

# 10. Next Steps

1. **Cross-validate the ANN** — the single most important open item, same as the BoW report, and now doubly important since ANN's TF-IDF lead over LinearSVC (0.12 points) is even narrower than its BoW lead over Logistic Regression (1.24 points).
2. **Resolve the Random Forest bootstrap mismatch** (Caveat #1) — low priority given the minor expected impact, but still open.
3. **Write a unified BoW-vs-TF-IDF master comparison** combining both leaderboards side by side, now that both combined reports exist — Section 6 here is a first pass at this, but a dedicated document would let it stand alongside the individual vectorizer reports as the top-level entry point to the whole classical ML phase.
4. **Proceed to the next roadmap phase** (Word2Vec/GloVe embeddings, then RNN/LSTM) — the classical ML baseline is now complete and cross-validated where possible, across both vectorizers, for all eleven models.

---

# Appendix: Individual Model Reports

Full per-model detail is available in the individual reports already produced for: Decision Tree, Random Forest, XGBoost, and ANN (TF-IDF versions). Logistic Regression, LinearSVC, KNN, BernoulliNB, ComplementNB, and GaussianNB, and MultinomialNB were logged and verified in full detail for this combined report.
