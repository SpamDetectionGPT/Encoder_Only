#!/bin/bash
EXECUTABLE="./main"   # Path to your Go executable
TIME_LIMIT=60               # Run the process for 600 seconds (10 minutes)

while true; do
  $EXECUTABLE &              # Start the executable in the background
  PID=$!                     # Capture its PID
  echo "Started $EXECUTABLE with PID $PID. Running for $TIME_LIMIT seconds..."
  
  sleep $TIME_LIMIT          # Wait for the specified duration
  
  echo "Time limit reached. Terminating process $PID..."
  kill $PID                  # Terminate the process
  wait $PID 2>/dev/null      # Wait for the process to exit, suppressing errors if already terminated
done
