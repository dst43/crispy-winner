#!/bin/bash

#module load gcc/8.3.0 cuda/11.6 cudampi/openmpi-3.1.5 conda/pytorch_1.13
eval "$(conda shell.bash hook)"

module load gcc/8.3.0
module load cuda/11.6

# export PATH="/usr/local/cuda-11.6/bin:$PATH"
conda info --envs
conda activate moe

# pushd /scratch/$USER/fairseq/DeepSpeed
# pip install .

# pushd /scratch/$USER/fairseq/expertserver/shared_pinned_memory
# python setup.py install

pushd /scratch/$USER/fairseq

EMBED_DIM_LIST="512 1024 2048"
BATCH_SIZE_LIST="4 8"
NUM_EXPERTS_LIST="4 8 16"

NUM_LAYER=12
NUM_UPDATE=500

for EMBED_DIM in $EMBED_DIM_LIST; do
    for BATCH_SIZE in $BATCH_SIZE_LIST; do
        for NUM_EXPERTS in $NUM_EXPERTS_LIST; do

            echo "----------------------------------------------------------------------------------------------"
            echo "EXP_SIZE: $EMBED_DIM, BATCH_SIZE: $BATCH_SIZE, NUM_EXPERTS: $NUM_EXPERTS"
            echo "----------------------------------------------------------------------------------------------"
            
            FFN_DIM=$((EMBED_DIM*4))

            rm -rf checkpoints

            python fairseq_cli/train.py ../data/wikitext-103-vocab/ \
            --ddp-backend pytorch_ddp --fp16 --task language_modeling --tokens-per-sample 1024 \
            --arch transformer_lm_gpt --share-decoder-input-output-embed --decoder-layers $NUM_LAYER \
            --decoder-embed-dim $EMBED_DIM --decoder-ffn-embed-dim $FFN_DIM --decoder-attention-heads 32 \
            --moe-expert-count $NUM_EXPERTS --moe-freq 1 --moe-gating-use-fp32 --moe-top1-expert --required-batch-size-multiple 1 \
            --moe-normalize-expert-grad sqrt_world_size   --moe-eval-capacity-token-fraction -1.0 \
            --max-sentences-valid 1 --num-workers-valid 0 --criterion moe_cross_entropy \
            --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum   --optimizer adam --adam-betas '(0.9, 0.98)' \
            --clip-norm 0.0 --lr 0.0005 --warmup-updates 750 --dropout 0.1 --attention-dropout 0.1 \
            --batch-size $BATCH_SIZE --update-freq 1 --max-update $NUM_UPDATE --disable-validation --log-format json --log-interval 100 \
            --fp16-no-flatten-grads --save-interval 1000 --activation-fn relu --seed 1 \
            --opt-type deepspeed --gate-type baseline

            rm -rf checkpoints
            bash expertserver/kill_ipcs.sh

            python fairseq_cli/train.py ../data/wikitext-103-vocab/ \
            --ddp-backend pytorch_ddp --fp16 --task language_modeling --tokens-per-sample 1024 \
            --arch transformer_lm_gpt --share-decoder-input-output-embed --decoder-layers $NUM_LAYER \
            --decoder-embed-dim $EMBED_DIM --decoder-ffn-embed-dim $FFN_DIM --decoder-attention-heads 32 \
            --moe-expert-count $NUM_EXPERTS --moe-freq 1 --moe-gating-use-fp32 --moe-top1-expert --required-batch-size-multiple 1 \
            --moe-normalize-expert-grad sqrt_world_size   --moe-eval-capacity-token-fraction -1.0 \
            --max-sentences-valid 1 --num-workers-valid 0 --criterion moe_cross_entropy \
            --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum   --optimizer adam --adam-betas '(0.9, 0.98)' \
            --clip-norm 0.0 --lr 0.0005 --warmup-updates 750 --dropout 0.1 --attention-dropout 0.1 \
            --batch-size $BATCH_SIZE --update-freq 1 --max-update $NUM_UPDATE --disable-validation --log-format json --log-interval 100 \
            --fp16-no-flatten-grads --save-interval 1000 --activation-fn relu --seed 1 \
            --opt-type deepspeed --gate-type ours --moe-cpu
        done
    done
done