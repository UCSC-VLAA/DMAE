#!/bin/bash
EXP_NAME=distill_finetuned_base_model
GPUS=8
SAVE_DIR1="./work_dirs/${EXP_NAME}_e1/"

IMAGENET_DIR='data/imagenet'

FINETUNE_EXP_FOLDER=''
FINETUNE_MODEL_NAME='checkpoint-99.pth'


OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} main_finetune.py \
    --output_dir ${SAVE_DIR1} \
    --log_dir ${SAVE_DIR1} \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune "./work_dirs/${FINETUNE_EXP_FOLDER}/${FINETUNE_MODEL_NAME}" \
    --epochs 100 \
    --blr 5e-4 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval \
    --data_path ${IMAGENET_DIR} \
    --seed 0
