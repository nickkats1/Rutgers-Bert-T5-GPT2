"""Generate output"""


def generate_text(model, tokenizer, device, prompts, max_length=128):
    """Generates headline text from a list of prompts.

    Args:
        model: Fine-tuned GPT2LMHeadModel.
        tokenizer: GPT-2 tokenizer.
        device: Torch device.
        prompts: List of prompt strings to seed generation.
        max_length: Maximum length of generated sequences.

    Returns:
        List of generated headline strings.
    """
    model.eval()
    generated = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated.append(text)

    return generated
