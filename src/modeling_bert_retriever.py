import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
)

from .configuration_bert_retriever import BertRetrieverConfig

class BertRetriever(PreTrainedModel):
    config_class = BertRetrieverConfig
    base_model_prefix = "autoModel"

    def __init__(self, config):
        super().__init__(config)

        backbone_config = AutoConfig.from_pretrained(
            config.backbone_name
        )

        self.autoModel = AutoModel.from_config(backbone_config)

        self.dropout = nn.Dropout(config.dropout)

        self.post_init()

    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()

        embeddings = (last_hidden_state * mask).sum(dim=1)
        embeddings = embeddings / mask.sum(dim=1).clamp(min=1e-9)

        return embeddings

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,**kwargs):
        outputs = self.autoModel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        embeddings = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask,
        )

        embeddings = self.dropout(embeddings)

        return embeddings