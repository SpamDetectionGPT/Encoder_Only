import json
import random
import os

def sample_emails_with_replacement(input_file, output_file, sample_size=547689):
    """
    Sample emails with replacement from input file to reach the desired sample size.
    Writes the sampled emails to the output file.
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    total_available = len(data)
    print(f"Source file contains {total_available} emails")
    print(f"Target sample size: {sample_size}")
    
    if sample_size <= total_available:
        # Simple sampling without replacement if we have enough
        sampled_data = random.sample(data, sample_size)
    else:
        # Need to sample with replacement
        print(f"Sampling with replacement (requested {sample_size} from {total_available})")
        # First take all available emails
        sampled_data = data.copy()
        
        # Then randomly sample the rest
        additional_needed = sample_size - total_available
        print(f"Need {additional_needed} additional samples")
        
        for _ in range(additional_needed):
            # Add a random email from the original data
            random_email = random.choice(data)
            sampled_data.append(random_email)
            
            # Progress tracking
            if (_ + 1) % 50000 == 0:
                print(f"Sampled {_ + 1} of {additional_needed} additional emails")
    
    # Shuffle the final dataset
    random.shuffle(sampled_data)
    print(f"Writing {len(sampled_data)} emails to {output_file}...")
    
    # Create the output structure
    output_data = sampled_data
    
    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
    
    print("Sampling completed successfully!")
    return len(sampled_data)

def main():
    """Main function to sample emails and write to output file."""
    # Use relative path for input dataset
    input_file = "../../datasets/combined_spam.json" 
    # Keep output file relative to the script's directory or use an argument
    output_file = "sampled_spam.json"
    
    sample_size = 72209  # The target number of samples
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    sampled_count = sample_emails_with_replacement(input_file, output_file, sample_size)
    print(f"Successfully sampled {sampled_count} emails to {output_file}")

if __name__ == "__main__":
    main() 