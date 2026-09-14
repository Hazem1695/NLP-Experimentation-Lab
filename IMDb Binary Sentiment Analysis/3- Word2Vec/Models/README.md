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

---

### 1. KNearest Neighbors (KNN)

Use the following configuration to retrain the Word2Vec model used with the KNearest Neighbors (KNN) classifier:

```python
from gensim.models import Word2Vec
model = Word2Vec(
    sentences=X_train,
    vector_size=100,
    window=10,
    min_count=2,
    sg=1,
    epochs=20,
    seed=0,
    workers=1,
)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit([' '.join(tokens) for tokens in X_train])
tfidf_weights = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

def get_tfidf_w2v_vector(tokens, model, vector_size):
    valid_tokens = [w for w in tokens if w in model.wv]
    if not valid_tokens:
        return np.zeros(vector_size)
    weights = [tfidf_weights.get(w, 1.0) for w in valid_tokens]
    vectors = [model.wv[w] for w in valid_tokens]
    return np.average(vectors, axis=0, weights=weights)

X_train_vec = np.array([get_tfidf_w2v_vector(t, model, 100) for t in X_train])
X_test_vec = np.array([get_tfidf_w2v_vector(t, model, 100) for t in X_test])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_vec = scaler.fit_transform(X_train_vec)
X_test_vec = scaler.transform(X_test_vec)
```

Matching classifier: ` `
