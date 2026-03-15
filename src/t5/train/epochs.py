"""Training and validation functions for T5 Model"""

import torch


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(loader):
        target_ids = batch["target_ids"].to(device, dtype=torch.long)
        y_ids = target_ids[:, :-1].contiguous()

        lm_labels = target_ids[:, 1:].clone().detach()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100

        input_ids = batch["source_ids"].to(device, dtype=torch.long)
        attention_mask = batch["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )

        loss = outputs.loss

        if step % 50 == 0:
            print(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["source_ids"].to(device, dtype=torch.long)
            attention_mask = batch["source_mask"].to(device, dtype=torch.long)
            target_ids = batch["target_ids"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            targets = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

            if step % 10 == 0:
                print(f"Validation step: {step}")

            predictions.extend(preds)
            actuals.extend(targets)

    return predictions, actuals
