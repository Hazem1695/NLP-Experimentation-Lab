# Bag of Words Experiments

Welcome to the **Bag of Words (BoW) Experiment Notebooks**.

This directory contains all Jupyter notebooks used during the experimentation phase of the **Bag of Words** pipeline for IMDb Binary Sentiment Analysis.

Rather than presenting only the final model, these notebooks document the complete research process—from feature extraction and model training to hyperparameter optimization, evaluation, and comparative analysis.

Every experiment is fully reproducible and follows a consistent workflow.

---

# Purpose

The goal of these notebooks is to systematically investigate how different machine learning algorithms perform when trained on **Bag of Words** features.

Each notebook explores:

- Model implementation
- Feature engineering
- Hyperparameter optimization
- Performance evaluation
- Experimental observations
- Model comparison

This research-driven approach ensures that every modeling decision is supported by empirical evidence rather than assumptions.

---

# Experimental Workflow

```text
Raw Text
    │
    ▼
Data Cleaning
    │
    ▼
Bag of Words (CountVectorizer)
    │
    ▼
Feature Engineering
    │
    ▼
Model Training
    │
    ▼
Hyperparameter Optimization
    │
    ▼
Cross Validation
    │
    ▼
Performance Evaluation
    │
    ▼
Model Comparison
    │
    ▼
Final Model Selection
```

---

# What These Notebooks Contain

Depending on the experiment, notebooks may include:

- Data preprocessing
- CountVectorizer configuration
- Vocabulary analysis
- N-gram experiments
- Feature selection
- Hyperparameter tuning
- Model training
- Cross-validation
- Error analysis
- Performance visualization
- Experimental conclusions

---

# Models Evaluated

The following machine learning algorithms were experimentally evaluated using Bag of Words features:

- Logistic Regression
- Linear Support Vector Machine (LinearSVC)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

Each model was trained, optimized, and evaluated under multiple experimental settings.

---

# Feature Engineering

The Bag of Words representation was extensively investigated using different configurations, including:

- Vocabulary size (`max_features`)
- Unigrams
- Bigrams
- Rare-word filtering (`min_df`)
- Frequent-word filtering (`max_df`)
- Binary feature representation
- Vocabulary optimization

The objective was to understand how feature engineering influences classification performance.

---

# Hyperparameter Optimization

Several optimization strategies were explored depending on computational complexity.

These include:

- Manual tuning
- GridSearchCV
- RandomizedSearchCV
- HalvingRandomSearchCV

Each strategy was selected based on the trade-off between search quality and computational efficiency.

---
# Evaluation Metrics

Every experiment is evaluated using multiple performance metrics to provide a comprehensive assessment.

Metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Cross-Validation Mean Accuracy
- Cross-Validation Standard Deviation

Using multiple evaluation metrics ensures that model performance is analyzed from different perspectives rather than relying solely on accuracy.

---

# Experimental Philosophy

These notebooks emphasize **experimentation over implementation**.

Rather than asking:

> *"Which model gives the highest accuracy?"*

The experiments investigate questions such as:

- How does vocabulary size affect performance?
- Which classifier is most suitable for sparse text?
- How much does hyperparameter tuning improve results?
- What are the trade-offs between different algorithms?
- Which model generalizes best?
- Which feature engineering choices have the greatest impact?

This methodology transforms the repository from a collection of notebooks into a structured NLP experimentation laboratory.

---

---

# Typical Notebook Structure

Each notebook follows a consistent workflow:

```text
1. Import Libraries

2. Load Dataset

3. Text Preprocessing

4. Feature Extraction (Bag of Words)

5. Model Construction

6. Hyperparameter Optimization

7. Model Training

8. Model Evaluation

9. Experimental Analysis

10. Conclusions
```

---

# Reproducibility

All notebooks are designed to be reproducible.

Running the notebooks allows anyone to recreate the experiments, validate the reported results, and understand the reasoning behind every modeling decision.

---

# Note

These notebooks represent the **research and experimentation phase** of the Bag of Words pipeline.

The final optimized models are stored separately in the **`models/`** directory, while detailed experiment reports summarize the findings and compare the performance of all evaluated algorithms.

---

## Experiment Progress
To maintain consistency and focus on understanding each model deeply, this project is structured as a daily release series, where each model is implemented, analyzed, and published separately.

| Day    | Model Name                                    | Publish Date    | Status               |
| ------ | --------------------------------------------  | --------------- | -------------------- |
| 1      | Logistic Regression Classification            | 2026-07-05      | Completed (Updated)  |
| 2      | K-Nearest Neighbors (KNN) Classification      | 2026-07-05      | Completed (Updated)  |
| 3      | Support Vector Machine (SVM) Classification   | 2026-07-07      | Completed (Updated)  |
| 4      | Naive Bayes Classification                    | 2026-06-25      | Completed            |
| 5      | Decision Trees Classification                 | 2026-07-01      | Completed            |
| 6      | Random Forest Classification                  | 2026-07-02      | In Progress          |

- In Progress – Currently being worked on
- Planned – Not uploaded yet
- Completed – Uploaded and documented
- TBD – To Be Determined
