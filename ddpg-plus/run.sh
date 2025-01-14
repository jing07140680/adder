#!/bin/bash
# Loop over the tmux sessions
for i in $(seq 1 1 15); do
  echo $i
  # Run taskset -c i in the ith tmux session
  tmux new-session -d "taskset -c $i python3 train.py -n train$i -a $((5*i)) -c $((2*i))"
done


