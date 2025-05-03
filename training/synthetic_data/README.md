# Synthetic Email Data Generator

This system generates synthetic phishing emails using Ollama LLM models and real-world email samples. The system periodically samples examples from a dataset, feeds them to an Ollama model, and collects the synthetically generated emails.

## Overview

The synthetic email generation pipeline works as follows:

1. The system samples real phishing emails from `../../datasets/combined_spam.json`
2. These examples are inserted into an Ollama model definition file (`Modelfile2`)
3. An Ollama model instance is created using this definition
4. The system queries the model to generate synthetic phishing emails
5. The generated emails are processed and saved to `synthetic_emails_output.json`
6. This process repeats at regular intervals, cycling through new examples

## Prerequisites

- [Ollama](https://ollama.ai/) installed and running as a service
- Python 3.7+
- Go (if recompiling the main executable)

## Components

- **sampler.sh**: Main orchestration script that automates the entire process
- **main**: Compiled Go executable that handles querying the Ollama API
- **sample2.py**: Python script that samples examples from the dataset
- **Modelfile2**: Ollama model definition file with prompts and examples
- **sampling_history.json**: Keeps track of which examples have been used
- **synthetic_emails_output.json**: Contains the generated synthetic emails

## Running the Generator

### 1. Basic Usage

To run the generator with default settings:

```bash
cd /path/to/Encoder_Only/training/synthetic_data
chmod +x sampler.sh  # Ensure it's executable
./sampler.sh
```

### 2. Run with Custom Ollama Port

If your Ollama instance is running on a non-default port:

```bash
./sampler.sh 11435  # Use port 11435 instead of default 11434
```

### 3. Run in Background (Production)

For long-running background execution:

```bash
nohup ./sampler.sh > sampler.log 2>&1 &
```

This will:
- Run the script in the background
- Continue running after you log out
- Redirect output to sampler.log
- Return the process ID for later management

## How it Works

### The sampler.sh Script

The `sampler.sh` script is the main orchestrator. It:

1. Runs the `main` executable to query Ollama for a fixed time period (120 seconds by default)
2. Terminates that process when the time limit is reached
3. Runs `sample2.py` to get new email examples from the dataset
4. Updates `Modelfile2` with these new examples using awk
5. Ensures the base model `llama3.2:1b-instruct-q4_K_M` is available
6. Creates/updates the Ollama model named `syntheticgenerator`
7. Restarts the `main` executable
8. Repeats this cycle indefinitely

### The sample2.py Script

The `sample2.py` script handles example sampling:

1. Reads from `../../datasets/combined_spam.json`
2. Tracks which examples have been used in `sampling_history.json`
3. Randomly selects examples that haven't been used yet
4. Formats them for inclusion in the Modelfile
5. Returns the formatted examples to `sampler.sh`

### The main Executable

The `main` executable (compiled from Go source):

1. Connects to the Ollama API
2. Sends generation requests to the `syntheticgenerator` model
3. Processes the responses
4. Writes the generated synthetic emails to `synthetic_emails_output.json`

## Customization

### Adjust Sampling Size

Edit `sample2.py` to change the number of examples sampled (default is 20):

```python
# In sample2.py
def main():
    input_file = "../../datasets/combined_spam.json"
    # Change 20 to desired number of examples
    formatted_samples = sample_emails_without_replacement(input_file, sample_size=20)
    print(formatted_samples)
```

### Adjust Cycling Period

Edit `sampler.sh` to change how often the examples are refreshed:

```bash
# In sampler.sh
# Change time limit in seconds
TIME_LIMIT=120  # Default is 2 minutes
```

### Ollama Model Parameters

Adjust generation parameters in `Modelfile2`:

```
# In Modelfile2
PARAMETER temperature 0.8  # Adjust for more/less creative outputs
PARAMETER num_ctx 4096     # Adjust context window size
```

## Troubleshooting

- **Ollama Model Creation Fails**: Check that `Modelfile2` doesn't have any invalid syntax
- **No Output Generated**: Verify Ollama is running with `ollama list`
- **Permission Issues**: Make sure `sampler.sh` and `main` are executable
- **Memory Issues**: If the system runs out of memory, consider lowering `num_ctx` in `Modelfile2`

## Notes

- The system uses a fingerprinting mechanism to avoid reusing the same examples
- Generated emails are stored in JSON format in `synthetic_emails_output.json`
- The script will run indefinitely until manually terminated 