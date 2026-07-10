from transformers import PretrainedConfig

class BertRetrieverConfig(PretrainedConfig):
    model_type = "bert_retriever"

    def __init__(self,backbone_name="bert-base-uncased",hidden_size=768,dropout=0.2,**kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.hidden_size = hidden_size
        self.dropout = dropout