#!/bin/bash

model_name_list=("openai/whisper-large-v3" "openai/whisper-large-v2" "openai/whisper-large" "openai/whisper-medium" "openai/whisper-base" "openai/whisper-small" "openai/whisper-tiny")

for i in ${!model_name_list[@]}; do
    python main.py --model_name ${model_name_list[$i]} --epoch 100
done

# Wait for all background processes to finish
wait
