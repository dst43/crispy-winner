#!/bin/bash

killall -9 python
rm -rf checkpoints
bash expertserver/kill_ipcs.sh

OMP_NUM_THREADS=16

CUDA_VISIBLE_DEVICES=0,1 python fairseq_cli/train.py /home/ykim/data/wikitext-103/wikitext-103-vocab/ \
--ddp-backend pytorch_ddp --fp16 --task language_modeling --tokens-per-sample 1024 --arch transformer_lm_gpt \
--share-decoder-input-output-embed --decoder-layers 6 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 \
--decoder-attention-heads 32 --moe-expert-count 4 --moe-freq 2 --moe-gating-use-fp32 --moe-top1-expert \
--moe-normalize-expert-grad sqrt_world_size   --moe-eval-capacity-token-fraction -1.0   --max-sentences-valid 1 \
--num-workers-valid 0   --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum  \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0   --lr 0.0005 --warmup-updates 750   --dropout 0.1 -\
-attention-dropout 0.1 --batch-size 1 --update-freq 1  --max-update 5 --disable-validation   --log-format json \
--log-interval 10 --fp16-no-flatten-grads --save-interval 1000 --seed 1 \
--opt-type deepspeed --gate-type ours --moe-cpu

#--empty-cache-freq \

# CUDA_VISIBLE_DEVICES=0,1 python fairseq_cli/train.py \
# --ddp-backend pytorch_ddp --fp16 --task dummy_lm --tokens-per-sample 4 --arch transformer_lm_gpt \
# --share-decoder-input-output-embed --decoder-layers 1 --decoder-embed-dim 6 --decoder-ffn-embed-dim 32 \
# --decoder-attention-heads 1 --moe-expert-count 4 --moe-freq 1 --moe-gating-use-fp32 --moe-top1-expert \
# --moe-normalize-expert-grad sqrt_world_size   --moe-eval-capacity-token-fraction -1.0   --max-sentences-valid 1 \
# --num-workers-valid 0   --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum  \
# --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0   --lr 0.0005 --warmup-updates 750   --dropout 0.1 -\
# -attention-dropout 0.1 --batch-size 2 --update-freq 1  --max-update 10 --disable-validation   --log-format json \
# --log-interval 1 --fp16-no-flatten-grads --save-interval 1000 --seed 1 \
# --opt-type deepspeed --gate-type ours --moe-cpu