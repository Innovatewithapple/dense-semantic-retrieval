# 🚀 Dense Semantic Retrieval using BERT

A dense retrieval model fine-tuned on the **MS MARCO** dataset for semantic search. This project transforms natural language queries and passages into dense vector embeddings, enabling efficient similarity search using **FAISS**.

The project includes the complete pipeline from data preprocessing and model fine-tuning to evaluation and Hugging Face model packaging.

---

## ✨ Features

- 🔍 Dense semantic retrieval using **BERT-base-uncased**
- 🧠 Fine-tuned dual-encoder architecture
- 📚 Trained on the **MS MARCO** passage ranking dataset
- ⚡ Mean Pooling + L2 Normalization
- 🚀 FAISS-based retrieval
- 📊 Evaluation using Recall@10, MRR and nDCG@10
- 🤗 Hugging Face compatible model
- 💾 Best model checkpoint saving based on validation Recall@10

---

# 🏗 Model Architecture

```
                 Query
                   │
                   ▼
         BERT-base-uncased
                   │
            Mean Pooling
                   │
          L2 Normalization
                   │
             Query Embedding
                   │
                   │
            Cosine Similarity
                   │
                   │
            Passage Embedding
                   ▲
          L2 Normalization
                   ▲
            Mean Pooling
                   ▲
         BERT-base-uncased
                   ▲
                Passage
```

The model is trained using **CrossEntropy Loss** over the similarity matrix (InfoNCE-style objective), encouraging matching query-passage pairs to have higher similarity than other pairs within the batch.

---

# 📊 Results

| Model | Recall@10 | MRR | nDCG@10 |
|--------|----------:|----:|---------:|
| BERT-base-uncased (Original) | 0.4793 | 0.2837 | 0.3301 |
| **Fine-tuned Dense Retriever** | **0.9693** | **0.8521** | **0.8810** |

The fine-tuned model substantially improves retrieval quality on the evaluation set compared with the untuned BERT-base encoder.

---

# ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | bert-base-uncased |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Epochs | 10 |
| Weight Decay | 0.01 |
| Temperature | 0.05 |
| Pooling | Mean Pooling |
| Embedding Normalization | L2 |
| Similarity | Cosine Similarity |
| Loss Function | CrossEntropy Loss (InfoNCE-style) |

---

# 📂 Project Structure

```text
dense-semantic-retrieval/
│
├── huggingface/              # Hugging Face compatible model
├── images/
├── models/                   # Saved checkpoints
├── notebooks/                # Training notebook
├── src/                      # Model and configuration classes
│
├── requirements.txt
├── LICENSE
├── README.md
├── test_model.py             # Export model
└── test_huggingface.py       # Verify Hugging Face package
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/dense-semantic-retrieval.git

cd dense-semantic-retrieval
```

Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 💻 Usage

Load the tokenizer and model

```python
from transformers import AutoTokenizer
from huggingface.modeling_bert_retriever import BertRetriever

tokenizer = AutoTokenizer.from_pretrained("huggingface")

model = BertRetriever.from_pretrained("huggingface")
```

Generate an embedding

```python
inputs = tokenizer(
    "What is deep learning?",
    return_tensors="pt"
)

embedding = model(**inputs)

print(embedding.shape)
```

---

# 📈 Evaluation

Retrieval performance is measured using:

- **Recall@10** – Measures whether the correct passage appears within the top-10 retrieved results.
- **MRR (Mean Reciprocal Rank)** – Rewards higher ranking of the first relevant passage.
- **nDCG@10** – Measures ranking quality by considering the position of relevant passages within the top-10 results.

Similarity search is performed using **FAISS** over normalized dense embeddings.

---

# 🤗 Hugging Face Model

The model has been exported using the Hugging Face `PreTrainedModel` API and includes:

- model.safetensors
- config.json
- tokenizer files
- custom model implementation
- custom configuration

(Repository link will be added after publishing.)

---

# 🔮 Future Improvements

- Support additional encoder backbones (RoBERTa, DeBERTa, ModernBERT)
- Hard negative mining
- Larger retrieval benchmarks
- Multi-vector retrieval architectures
- Hybrid dense + sparse retrieval

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Mihir Vyas**

If you found this project useful, consider giving it a ⭐ on GitHub.