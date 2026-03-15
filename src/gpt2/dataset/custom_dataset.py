"""Custom DataSet for Fine-Tuning GPt2"""

import torch
import torch.nn as nn
import torch.nn.functional as F




class CustomDataset(torch.utils.data.Dataset):
    """Custom Dataset for fine-tuning gpt"""
    
    def __init__(self, headlines, tokenizer, max_length):
        self.headlines = headlines
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.headlines)
    
    def __getitem__(self, idx):
        headline = (
            self.tokenizer.bos_token + 
            self.headlines[idx] + 
            self.self.tokenizer.eos_token
        )
        
        return headline
    
    def collate_fn(self, batch):
        inputs = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length = self.max_length,
            return_tensors="pt"
        )
        
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels
        return inputs
