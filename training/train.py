# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoConfig, AutoTokenizer
from tqdm.auto import tqdm  # For progress bars
import os  # For saving model

# Import from local modules
from models import TransformerForSequenceClassification
from utils.helpers import (
    parse_train_args,
    get_device,
)  # Import argument parser and device helper
from data.preparation import prepare_data  # Import data preparation function


# --- Main Training Function ---
def main():
    args = parse_train_args()  # Use the imported argument parser
    device = get_device()  # Use the imported device helper
    print(f"Using device: {device}")

    # --- Load Config and Tokenizer ---
    print(f"Loading config for {args.model_ckpt}...")
    try:
        config = AutoConfig.from_pretrained(args.model_ckpt)
        config.num_labels = args.num_labels  # Set from arguments
    except Exception as e:
        print(f"Failed to load config '{args.model_ckpt}'. Error: {e}")
        exit()

    print(f"Loading tokenizer for {args.model_ckpt}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    except Exception as e:
        print(f"Failed to load tokenizer '{args.model_ckpt}'. Error: {e}")
        exit()

    # --- Prepare Data ---
    try:
        train_dataloader, eval_dataloader = prepare_data(
            args, tokenizer
        )  # Use imported function
    except Exception as e:
        print(f"Data preparation failed. Error: {e}")
        exit()

    # --- Instantiate Model ---
    print("Instantiating the custom Transformer model...")
    model = TransformerForSequenceClassification(config)
    model.to(device)

    # --- Optimizer and Loss Function ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()  # Standard for classification

    # --- Training Loop ---
    num_training_steps = args.epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    print("Starting training...")
    model.train()  # Ensure model starts in training mode
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_dataloader:
            # Move batch to device
            # Filter batch to only include keys the model expects in its forward pass
            # Currently, our model only needs 'input_ids'. Labels are needed for loss.
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids)  # Pass only input_ids
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"Epoch": epoch + 1, "Loss": loss.item()})

        avg_train_loss = total_loss / len(train_dataloader)
        print(
            f"\nEpoch {epoch + 1}/{args.epochs} - Average Training Loss: {avg_train_loss:.4f}"
        )

        # --- Validation ---
        if eval_dataloader:
            model.eval()  # Set model to evaluation mode
            total_eval_loss = 0
            correct_predictions = 0
            total_predictions = 0

            print("Running validation...")
            with torch.no_grad():
                for batch in eval_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids)
                    loss = loss_fn(outputs, labels)
                    total_eval_loss += loss.item()

                    # Calculate accuracy (example metric)
                    logits = outputs
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)

            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            accuracy = (
                correct_predictions / total_predictions
                if total_predictions > 0
                else 0.0
            )
            print(
                f"Validation Loss: {avg_eval_loss:.4f} - Validation Accuracy: {accuracy:.4f}"
            )
            model.train()  # Set back to train mode for next epoch
        else:
            print("Skipping validation as no evaluation dataloader is available.")

    # --- Save Model ---
    try:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model to {output_dir}")
        # Save model state dict
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        # Save config
        config.save_pretrained(output_dir)
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Error saving model/config/tokenizer: {e}")

    print("Training finished.")


if __name__ == "__main__":
    main()
