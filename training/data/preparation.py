import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


# --- Dataset Loading and Tokenization ---
def prepare_data(args, tokenizer):
    """Loads, tokenizes, and prepares dataset splits for training and evaluation."""
    # Example using Hugging Face datasets
    print(f"Loading dataset: {args.dataset_name}")
    # Replace with your dataset loading logic if custom
    # Make sure splits ('train', 'validation'/'test') are appropriate
    try:
        raw_datasets = load_dataset(args.dataset_name)
    except Exception as e:
        print(
            f"Failed to load dataset '{args.dataset_name}'. Please check the name or provide a path."
        )
        print(f"Error: {e}")
        # Consider raising the exception instead of exiting for better integration
        raise

    # --- Input Validation ---
    # Check if train split exists
    if "train" not in raw_datasets:
        print(f"Error: Dataset '{args.dataset_name}' must contain a 'train' split.")
        exit()

    # Ensure the necessary columns exist before mapping
    required_columns = {args.text_column, args.label_column}
    # Check train split first, assume others have similar structure
    if not required_columns.issubset(raw_datasets["train"].column_names):
        print(
            f"Error: Dataset 'train' split missing required columns. Found: {raw_datasets['train'].column_names}, Required: {required_columns}"
        )
        exit()

    # --- Tokenization ---
    print("Tokenizing dataset...")

    # Define tokenization function using provided args
    def tokenize_function(examples):
        # add_special_tokens=False matches the model's Embeddings layer expectation
        return tokenizer(
            examples[args.text_column],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # --- Formatting ---
    # Remove original text columns, set format to PyTorch tensors
    columns_to_remove = [
        col
        for col in raw_datasets["train"].column_names
        if col
        not in ["input_ids", "attention_mask", "token_type_ids", args.label_column]
    ]
    # Ensure label column is not accidentally removed if it shares name with another column
    if args.label_column in columns_to_remove:
        columns_to_remove.remove(args.label_column)

    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
    # Rename the label column only if it's not already 'labels'
    if args.label_column != "labels":
        tokenized_datasets = tokenized_datasets.rename_column(
            args.label_column, "labels"
        )
    tokenized_datasets.set_format("torch")

    # --- Create DataLoaders ---
    train_dataset = tokenized_datasets["train"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size
    )

    # Use validation or test split as needed - adjust split name if necessary
    eval_split = "test" if "test" in tokenized_datasets else "validation"
    if eval_split not in tokenized_datasets:
        print(
            f"Warning: No 'test' or 'validation' split found in dataset. Skipping evaluation."
        )
        eval_dataloader = None
    else:
        eval_dataset = tokenized_datasets[eval_split]
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)

    return train_dataloader, eval_dataloader


# --- Custom Dataset Class (if not using Hugging Face datasets) ---
class CustomTextDataset(Dataset):
    """A custom Dataset class for text classification.

    Args:
        texts (list[str]): A list of text samples.
        labels (list[int]): A list of corresponding labels.
        tokenizer: A Hugging Face tokenizer instance.
        max_length (int): Maximum sequence length for padding/truncation.
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        # Squeeze to remove the batch dimension added by return_tensors='pt'
        # Prepare item dictionary expected by the training loop
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)  # Ensure label is a tensor
        return item
