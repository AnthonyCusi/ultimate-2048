#!/bin/bash

# To run model multiple times:
# 1. change the letter below to desired model
# 2. run chmod +x src/run_script.sh
# 3. run ./src/run_script.sh

for i in {1..10}
do
    printf "\nRunning simulation $i...\n"
    echo -e "a" | python3 src/run_simulation.py
done
