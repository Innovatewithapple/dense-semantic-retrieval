import torch

from src.configuration_bert_retriever import BertRetrieverConfig
from src.modeling_bert_retriever import BertRetriever

# ---------------- Create Model ---------------- #

config = BertRetrieverConfig(
    backbone_name="bert-base-uncased",
    hidden_size=768,
    dropout=0.2,
)

model = BertRetriever(config)

# ---------------- Load Weights ---------------- #

state_dict = torch.load(
    "models/bert_retriever_weights.pth",
    map_location="cpu",
)

missing_keys, unexpected_keys = model.load_state_dict(
    state_dict,
    strict=False,
)

print("\n========== LOAD REPORT ==========")
print("Missing Keys   :", missing_keys)
print("Unexpected Keys:", unexpected_keys)

# ---------------- Test Forward Pass ---------------- #

model.eval()

dummy_input_ids = torch.randint(0, 1000, (1, 16))
dummy_attention_mask = torch.ones((1, 16), dtype=torch.long)

with torch.no_grad():
    embeddings = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
    )

print("\nEmbedding Shape:", embeddings.shape)

from transformers import AutoTokenizer

# ---------------- Save Hugging Face Model ---------------- #

tokenizer = AutoTokenizer.from_pretrained(
    config.backbone_name
)

model.save_pretrained("huggingface")

tokenizer.save_pretrained("huggingface")

print("\n✅ Hugging Face model exported successfully!")