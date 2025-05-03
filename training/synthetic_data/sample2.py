import json
import random
import os
import hashlib

def sample_emails_without_replacement(file_path, sample_size=20, history_file='sampling_history.json'):
    """
    Memory-efficient sampling without replacement from a large JSON file.
    """
    # Load previously sampled fingerprints
    previously_sampled = set()
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                previously_sampled = set(json.load(f))
            except:
                print(f"Warning: Could not load history from {history_file}, starting fresh")
    
    # Count total emails using file streaming
    total_count = 0
    with open(file_path, 'r') as f:
        data = json.load(f)
        total_count = len(data)
    
    # Generate random indices to check
    all_indices = list(range(total_count))
    random.shuffle(all_indices)
    
    # Sample emails
    sampled_emails = []
    sampled_fingerprints = []
    
    with open(file_path, 'r') as f:
        all_emails = json.load(f)
        
        for idx in all_indices:
            if len(sampled_emails) >= sample_size:
                break
                
            email = all_emails[idx]
            text = email.get('text', '')
            fingerprint = hashlib.md5(text.encode()).hexdigest()
            
            if fingerprint not in previously_sampled:
                sampled_emails.append(email)
                sampled_fingerprints.append(fingerprint)
    
    # Update history file
    previously_sampled.update(sampled_fingerprints)
    with open(history_file, 'w') as f:
        json.dump(list(previously_sampled), f)
    
    # Format the emails
    formatted_samples = []
    for i, email in enumerate(sampled_emails):
        text = email.get('text', '')
        formatted_samples.append(f"Example {i+1}:\n{text}")
    
    if len(formatted_samples) < sample_size:
        print(f"Warning: Could only sample {len(formatted_samples)} unique emails")
    
    return "\n\n".join(formatted_samples)

def main():
    """Main function to sample emails and print them."""
    # Use relative path to the combined spam dataset
    input_file = "../../datasets/combined_spam.json"
    formatted_samples = sample_emails_without_replacement(input_file)
    print(formatted_samples)

if __name__ == "__main__":
    main()