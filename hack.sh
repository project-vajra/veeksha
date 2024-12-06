#!/bin/bash

# Repeat the process 6 times
for i in {1..6}
do
  # Wait for 1.5 hours
  echo "waiting for 1.5 hours"
  sleep 5400
  
  # Execute the command sequence
  ps aux | grep etalon | grep 'run_benchmark' | sed 's/|/ /' | awk '{print $2}' > processes.txt
  sudo kill -9 $(cat processes.txt)
done

