#!/bin/bash

# CUDA_VISIBLE_DEVICES=5 python inference-f1-dfl.py > f1/event-only2label-easy2hard.txt

# CUDA_VISIBLE_DEVICES=1 python finetune_glm3.py /home/juguoyang/finetune/A100/event-finetune/event-only2label /home/juguoyang/finetune/THUDM/chatglm3-6b configs/lora.yaml >logfile/glm3-event-only2label-lossval.txt &
# CUDA_VISIBLE_DEVICES=3 python finetune_glm3.py /home/juguoyang/finetune/A100/event-finetune/event-only2label-prompt2 /home/juguoyang/finetune/THUDM/chatglm3-6b configs/lora.yaml  > logfile/glm3-event-only2label-pepf.txt &
CUDA_VISIBLE_DEVICES=1 python event-f1.py > f1/event-chushiprompt-pepf.txt
# CUDA_VISIBLE_DEVICES=3 python event-f1-2.py > f1/event-only2label-pepf-400000.txt
# CUDA_VISIBLE_DEVICES=3 python finetune_glm3.py /home/juguoyang/finetune/A100/event-finetune/event-only2label /home/juguoyang/finetune/THUDM/chatglm3-6b configs/lora.yaml  > logfile/glm3-event-only2label-pepf-chushiprompt.txt &
