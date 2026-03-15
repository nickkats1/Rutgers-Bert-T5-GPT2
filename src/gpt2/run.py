"""Main Script to run GPT2"""

from src.gpt2 import config
from src.gpt2.dataset.custom_dataset import CustomDataset
from src.gpt2.eval.generate import generate_text
from src.gpt2.eval.train_eval_epoch import train_epoch, eval_epoch
from src.gpt2.helpers import (
    split_data,
    load_data,
    set_seed,
    build_model,
    build_dataloaders,
    save_model
)



def main():
    """Main Function to run script"""
    
    
    device = config.DEVICE
    
    # set seed
    set_seed()
    
    # load data
    
    df = load_data()
    
    # split dataframe
    
    train_headlines, val_headlines = split_data(df)
    
    # load tokenizer and model
    
    tokenizer, model = build_model(config.DEVICE)
    
    
    # load in train_loader and val_loader
    
    train_loader, val_loader = build_dataloaders(
        train_headlines,
        val_headlines,
        tokenizer
    )
    
    # load in optimizer
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE
    )
    
    print("\nTraining\n")
    
    for epoch in range(config.EPOCHS):
        train_epoch(
            epoch,
            model,
            device,
            train_loader,
            optimizer
        )
        
    # save model
    save_model(model, tokenizer)
    
    print("\nValidation\n")
    
    eval_epoch(model, device, val_loader)
    
    print("Generating HeadLines")
    
    
    sample_prompts = val_headlines[:30]
    
    prompts = [" ".join(h.split()[:5]) for h in sample_prompts]
    
    predictions = generate_text(model, tokenizer, device, prompts)
    
    
    for prompt, pred in zip(prompts, predictions):
        print(f"Prompt: {prompt}")
        print(f"Generated: {pred}")
    
            
    
if __name__ == "__main__":
    main()
    