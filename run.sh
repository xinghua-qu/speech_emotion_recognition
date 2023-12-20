#!/bin/bash

model_name_list=("openai/whisper-large-v3" "openai/whisper-large-v2" "openai/whisper-large" "openai/whisper-medium" "openai/whisper-base" "openai/whisper-small" "openai/whisper-tiny")
gpu_ids=(0 1 2 3 4 5 6 7)

for i in ${!model_name_list[@]}; do
    python main.py --model_name ${model_name_list[$i]} --gpu_id ${gpu_ids[$i]}
done

# Wait for all background processes to finish
wait