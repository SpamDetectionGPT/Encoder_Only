#!/usr/bin/env python3
import json
import random


def clean_text(text):
    """
    Remove newline and tab characters and trim extra whitespace.
    """
    cleaned_text = text.replace("\n", " ").replace("\t", " ").replace("\\", " ").strip()
    return cleaned_text


def sample_emails(file_path, sample_size=20):
    """
    Load the phishing emails dataset and randomly sample 'sample_size' emails.
    """
    with open(file_path, "r") as f:
        # Assuming COMBINED.json is a JSON array of email objects
        data = json.load(f)

    # Ensure we have enough emails in the dataset
    if len(data) < sample_size:
        raise ValueError("Not enough emails in the dataset to sample from.")

    samples = random.sample(data, sample_size)
    cleaned_samples = [clean_text(email.get("text", "")) for email in samples]
    return cleaned_samples


def main():
    file_path = "/home/voldie/nasbak/datasets/COMBINED.json"
    samples = sample_emails(file_path)

    # Print out each sample prefixed with "Example X:" so you can easily copy them
    for idx, sample in enumerate(samples, start=1):
        print(f"Example {idx}:")
        print(sample)
        print()  # blank line for readability


if __name__ == "__main__":
    main()
