#!/bin/bash

#SBATCH --job-name=train_vistral
#SBATCH -o /data2/npl/luannt/IHSD/implicit-hatespeech-detection/script/vistral_finetune.out
#SBATCH --gres=gpu:1
#SBATCH --mem=35G
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1000:00:00

source /data2/npl/luannt/IHSD/bin/activate
export PATH='/data2/npl/luannt/IHSD/bin:$PATH'
export PYTHONPATH='/data2/npl/luannt/IHSD/lib/python3.9/site-packages:$PYTHONPATH'
export PATH="/data2/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/Harmful-Videos-Detections/ffmpeg/bin:$PATH"
export CUDA_VISIBLE_DEVICES=6

python /data2/npl/luannt/IHSD/chatbot/src/models/finetuning.py

############################# ZERO-SHOT #############################

python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
  --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
  --model   qwen_2.5_7B \
  --prompt-type zero \
  --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
  --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/zs/qwen_2.5_7B_5.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model llama_3.1_8B \
#   --prompt-type zero \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/zs/llama_3.1_8B_2.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.3 \
#   --prompt-type zero \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/zs/mistral_7B_v0.3.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.2 \
#   --prompt-type zero \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/zs/mistral_7B_v0.2.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type zero \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/zs/qwen_2.5_14B.jsonl 

python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
  --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
  --model gemma_3_12B \
  --prompt-type zero \
  --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/official_test_gemma_4.jsonl \
  --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/zs/gemma_3_12B_4.jsonl 



# ############################# FEW-SHOT #############################

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model   qwen_2.5_7B \
#   --prompt-type few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/fs/qwen_2.5_7B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model llama_3.1_8B \
#   --prompt-type few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/fs/llama_3.1_8B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/fs/qwen_2.5_14B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.3 \
#   --prompt-type few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/fs/mistral_7B_v0.3.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.2 \
#   --prompt-type few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/fs/mistral_7B_v0.2.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/fs/qwen_2.5_14B.jsonl 

python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
  --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
  --model gemma_3_12B \
  --prompt-type few \
  --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/official_test_gemma_fs.jsonl \
  --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/fs/gemma_3_12B_2.jsonl 



# ############################# SELF-FEW-SHOT #############################
# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model llama_3.1_8B \
#   --prompt-type self_few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sfs/llama_3.1_8B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model   qwen_2.5_7B \
#   --prompt-type self_few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sfs/qwen_2.5_7B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.3 \
#   --prompt-type self_few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sfs/mistral_7B_v0.3.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.2 \
#   --prompt-type self_few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sfs/mistral_7B_v0.2.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type self_few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/official_test_qwen14.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sfs/qwen_2.5_14B_2.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model gemma_3_12B \
#   --prompt-type self_few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sfs/gemma_3_12B.jsonl 


############################# COT #############################
# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model llama_3.1_8B \
#   --prompt-type cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/300_official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot/300_llama_3.1_8B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model   qwen_2.5_7B \
#   --prompt-type cot \
#   --scenario cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot/qwen_2.5_7B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.3 \
#   --prompt-type cot \
#   --scenario cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot/mistral_7B_v0.3.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.2 \
#   --prompt-type cot \
#   --scenario cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot/mistral_7B_v0.2.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type cot \
#   --scenario cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot/qwen_2.5_14B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type cot \
#   --scenario cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/300_official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot_scen2/300_qwen_2.5_14B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type cot \
#   --scenario cot_scen3 \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/300_official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot_scen3/300_qwen_2.5_14B.jsonl 



# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model gemma_3_12B \
#   --prompt-type cot \
#   --scenario cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot/gemma_3_12B.jsonl 

# ##################################### COT 2 #####################################

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_7B \
#   --prompt-type cot2 \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot2/qwen_2.5_7B.jsonl 


# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.3 \
#   --prompt-type cot2 \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot2/mistral_7B_v0.3.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.2 \
#   --prompt-type cot2 \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot2/mistral_7B_v0.2.jsonl 


python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
  --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
  --model qwen_2.5_14B \
  --prompt-type cot2 \
  --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
  --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot2/qwen_2.5_14B_3.jsonl 

python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
  --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
  --model gemma_3_12B \
  --prompt-type cot2 \
  --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
  --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot2_new/gemma_3_12B_5.jsonl 




# ################################# SELF-CONSISTENT #############################



# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_7B \
#   --prompt-type self_consistent \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sc/qwen_2.5_7B.jsonl 


# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.3 \
#   --prompt-type self_consistent \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sc/mistral_7B_v0.3.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.2 \
#   --prompt-type cot2 \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot2/mistral_7B_v0.2.jsonl 


# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type cot2 \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot2/qwen_2.5_14B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model gemma_3_12B \
#   --prompt-type self_consistent \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sc/gemma_3_12B.jsonl 
















# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type zero \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/zs/qwen_2.5_14B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/fs/qwen_2.5_14B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type self_few \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sfs/qwen_2.5_14B.jsonl 


######## SCEN #####

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model   qwen_2.5_7B \
#   --prompt-type self_consistent \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sc/qwen_2.5_7B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model qwen_2.5_14B \
#   --prompt-type cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/300_official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sc/300_qwen_2.5_14B.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.3 \
#   --prompt-type cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/300_official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sc/300_mistral_7B_v0.3.jsonl 

# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model mistral_7B_v0.2 \
#   --prompt-type cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/300_official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sc/300_mistral_7B_v0.2.jsonl 


# python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
#   --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
#   --model gemma_3_12B \
#   --prompt-type cot \
#   --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
#   --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/sc/gemma_3_12B.jsonl 


python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
  --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
  --model qwen_2.5_7B \
  --prompt-type cot2_random_fewshot \
  --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/official_test_qwen7B.jsonl \
  --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot2_random_fewshot/qwen_2.5_7B_2.jsonl 

python /data2/npl/luannt/IHSD/implicit-hatespeech-detection/src/main.py \
  --config /data2/npl/luannt/IHSD/implicit-hatespeech-detection/config/config_LLM.yaml \
  --model qwen_2.5_14B \
  --prompt-type cot2_random_fewshot \
  --input-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/official_test.jsonl \
  --output-file /data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/cot2_random_fewshot/qwen_2.5_14B.jsonl 


