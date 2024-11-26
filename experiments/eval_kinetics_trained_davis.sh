CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --nnodes 1 /home/zhaochenzhi/CRW_TAP/test_on_tapvid.py \
--eval_dataset davis \
--setting forward_backward_time_independent_query_point \
--upsample_factor 1 \
--model_path /home/zhaochenzhi/CRW_TAP/experiments/results/train_kinetics/checkpoints/model_170000.pth

# --model_path /home/zhaochenzhi/CRW_TAP/results/kinetics_144k.pth

# python /home/zhaochenzhi/CRW_TAP/test_on_tapvid.py \

# --setting forward_backward_query_point \

# --model_path results/kinetics_144k.pth

# --upsample_factor 4 \