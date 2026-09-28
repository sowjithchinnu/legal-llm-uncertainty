# Legal LLM Uncertainty

Utilities for generating, storing, and measuring consistency across multiple legal LLM responses.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The semantic similarity functions use the local
`sentence-transformers/all-MiniLM-L6-v2` model. Its weights are downloaded by
`sentence-transformers` the first time the model is used.

## Similarity

The project provides three response-consistency measures:

- `src.similarity.lexical`: TF-IDF cosine similarity.
- `src.similarity.semantic`: cosine similarity between MiniLM embeddings.
- `src.similarity.combined`: weighted lexical and semantic similarity.

Each module provides a similarity matrix and a helper for extracting unique
upper-triangle pairwise scores. The combined module also provides a mean score.

Example:

```python
from src.similarity.combined import (
    combined_similarity_matrix,
    mean_combined_similarity,
)

responses = [
    "The respondent is the Union of India.",
    "The Union of India is the respondent.",
]

matrix = combined_similarity_matrix(responses, lexical_weight=0.5)
mean_score = mean_combined_similarity(responses, lexical_weight=0.5)
```

## Tests

Run the test suite with:

```bash
.venv/bin/python -m unittest discover -s tests
```
