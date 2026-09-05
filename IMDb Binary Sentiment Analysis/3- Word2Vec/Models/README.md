## Pre-trained Model Files

The pre-trained **Word2Vec** model files are **not included in this repository** because they exceed GitHub's recommended file size limit.

To make the project easy to reproduce, I have documented the configuration used for each model (such as vector size, window size, `min_count`, and other training parameters). Using these settings, you can retrain the corresponding Word2Vec model and use it together with the provided `.pkl` classifier files.

Each classifier (`.pkl`) is associated with its documented Word2Vec configuration, allowing you to reproduce the experiments and obtain the required embeddings for inference.

---

## Word2Vec Configurations

### 1. Logistic Regression

Use the following configuration to retrain the Word2Vec model used with the Logistic Regression classifier:

```python
from gensim.models import Word2Vec
model = Word2Vec(
    sentences=X_train,
    vector_size=300,
    window=10,
    min_count=2,
    sg=1,
    epochs=20,
    seed=0,
    workers=1,
)

def get_average_vector(tokens, model, vector_size):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(valid_vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(valid_vectors, axis=0)

vector_size = 300
X_train_w2v = np.array([get_average_vector(tokens, model, vector_size) for tokens in X_train])
X_test_w2v = np.array([get_average_vector(tokens, model, vector_size) for tokens in X_test])
```

Matching classifier: `Word2Vec_Logistic_Regression_model.pkl`
