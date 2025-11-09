#!/bin/bash  
export CUDA_VISIBLE_DEVICES=0,1,2
# python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py

python -m torch.distributed.run --nproc_per_node=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train_futureframe.py --config-name=flow_video_mmdit # flow_video_uvit

# python train_futureframe.py --config-name=flow_video_mmdit # flow_video_uvit