import torch

from transformers import AutoTokenizer

from huggingface.modeling_bert_retriever import BertRetriever

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggingface")

# Load model
model = BertRetriever.from_pretrained("huggingface")

model.eval()

text = "What is deep learning?"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
)

with torch.no_grad():
    embedding = model(**inputs)

print("Embedding Shape:", embedding.shape)