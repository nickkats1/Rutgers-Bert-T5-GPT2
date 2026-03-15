"""Train and eval epoch

user-defined functions to train
gpt2
"""

import torch
import torch.nn as nn
import numpy as np


def train_epoch(epoch, model, device, loader, optimizer):
    """Runs one training epoch.

    Args:
        epoch: Current epoch number.
        model: GPT2LMHeadModel.
        device: Torch device.
        loader: Training DataLoader.
        optimizer: Optimizer instance.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        if step % 50 == 0:
            print(f"  Epoch {epoch + 1} | Step {step} | Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    avg_loss = total_loss / len(loader)
    print(f"  Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
    return avg_loss


# --- eval epoch ---

def eval_epoch(model, device, loader):
    """Evaluates the model on a validation set.

    Args:
        model: GPT2LMHeadModel.
        device: Torch device.
        loader: Validation DataLoader.

    Returns:
        Average validation loss and perplexity.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(loader)
    perplexity = np.exp(avg_loss)

    print(f"  Val Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
    return avg_loss, perplexity




