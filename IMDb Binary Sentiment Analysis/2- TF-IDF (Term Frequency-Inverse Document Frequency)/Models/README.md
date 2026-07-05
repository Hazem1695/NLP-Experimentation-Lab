# Models Directory

This folder contains the final trained models generated during the NLP experiments.

Each model has a corresponding vectorizer that was fitted during training.

Example:

- `tf-idf_logistic_regression.pkl`
- `tf-idf_logistic_regression_vectorizer.pkl`

The vectorizer must always be loaded together with its matching model, as it transforms raw text into the numerical feature representation expected by the model during inference.
