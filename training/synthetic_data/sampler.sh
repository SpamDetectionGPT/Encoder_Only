#!/bin/bash

# Use a passed-in port or default to 11434 if not provided
OLLAMA_PORT=${1:-11434}

# Configuration variables:
TIME_LIMIT=120                                  
EXECUTABLE="./main"                            
MODELFILE="./Modelfile2"                        
SAMPLER_SCRIPT="sample2.py"                     
# Update the OLLAMA command to use the port
OLLAMA_CMD="ollama create syntheticgenerator --port $OLLAMA_PORT -f"

while true; do
  $EXECUTABLE &
  PID=$!
  echo "Started $EXECUTABLE with PID $PID. Running for $TIME_LIMIT seconds..."
  sleep $TIME_LIMIT
  echo "Time limit reached. Terminating process $PID..."
  kill $PID
  wait $PID 2>/dev/null

  echo "Sampling new email examples..."
  NEW_EXAMPLES=$(python3 "$SAMPLER_SCRIPT")
  echo "New examples generated:"
  echo "$NEW_EXAMPLES"

  echo "Updating modelfile with new examples..."
  awk -v new_examples="$NEW_EXAMPLES" '
    BEGIN { in_replace=0; replaced=0 }
    /Example 1:/ {
      print new_examples;
      replaced=1;
      in_replace=1;
      next;
    }
    in_replace && /Example 20:/ {
      in_replace=0;
      next;
    }
    !in_replace { print }
    END { if (!replaced) print new_examples }
  ' "$MODELFILE" > Modelfile.tmp && mv Modelfile.tmp "$MODELFILE"
  echo "Updated modelfile stored in $MODELFILE"

  echo "Pulling base model if needed..."
  ollama pull llama3.2:1b-instruct-q4_K_M --port $OLLAMA_PORT

  echo "Loading new syntheticgenerator instance with updated modelfile..."
  $OLLAMA_CMD "$MODELFILE"

  echo "Restarting $EXECUTABLE..."
done

