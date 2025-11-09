#!/bin/bash  
export CUDA_VISIBLE_DEVICES=2
# python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py
# python train_futureframe.py --config-name=flow_video_mmdit  # flow_video_uvit
# zip -rj submissions_16x.zip /pub0/qasim/1xgpt/humanoid_world_model/submissions/diffusion/*
python create_val_results.py --config-name=flow_video_mmdit # flow_video_mmdit
