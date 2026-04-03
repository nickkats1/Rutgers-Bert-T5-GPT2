"""Model building and saving utilities for GPT-2 fine-tuning."""

import os

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from src.gpt2.config import MODEL_NAME, OUTPUT_DIR, DEVICE


def build_model(device=DEVICE):
    """Load and configure the GPT-2 tokenizer and language model.

    Configures the tokenizer to use the EOS token as padding and sets the
    model's pad token id accordingly.

    Args:
        device: Torch device to move the model to. Defaults to DEVICE from config.

    Returns:
        A tuple of (tokenizer, model).
    """
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    return tokenizer, model


def save_model(model, tokenizer, output_dir=OUTPUT_DIR):
    """Save the fine-tuned model and tokenizer to disk.

    Args:
        model: The GPT-2 language model to save.
        tokenizer: The GPT-2 tokenizer to save.
        output_dir: Directory to save model files. Defaults to OUTPUT_DIR from config.

    Raises:
        OSError: If saving fails due to a filesystem error.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"models saved to: {output_dir}")
    except OSError:
        print("Output Directory Does Not Exist")
        raise
