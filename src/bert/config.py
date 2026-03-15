"""Bert Config"""

bert_config = {
    "model_name": "bert-base-uncased",
    "data_path": "data/guardian_headlines.csv",
    "max_length": 80,
    "device": "cuda:0",
}
