def load_bert(model_name):
    if model_name:
        # such as: "vinai/phobert-base"
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install transformers library for BERT support! Try `pip install transformers`.")
        bert_model = AutoModel.from_pretrained(model_name)
        bert_tokenizer = load_tokenizer(model_name)
        return bert_model, bert_tokenizer
    return None, None

