# Generating Humanoid Futures: Multimodal Joint Attention Between Past and Future for Action-Guided Video Prediction

Author: **Qasim Ali**  
License: **MIT License**  

---

## Overview

This folder contains the implementation of Flow Humanoid World Model.

---

## Usage

The repository includes several scripts for training, evaluation, and debugging. Below are the instructions for running these scripts. The configurations for debugging and training are defined in `.vscode/launch.json`.

### 1. **Training the Model**
   To train the model, use the `train.py` script:
   ```bash
   python train.py
   ```
   Example debug configuration:
   ```jsonc
   {
       "name": "train (one sample)",
       "type": "debugpy",
       "request": "launch",
       "program": "train.py",
       "console": "integratedTerminal",
       "args": ["one_sample=True"],
       "justMyCode": false,
       "env": {
           "CUDA_VISIBLE_DEVICES": "0"
       }
   }
   ```

### 2. **Evaluating the Model**
   To evaluate the model, use the `eval_diffusion.py` script:
   ```bash
   python eval_diffusion.py
   ```
   Example debug configuration:
   ```jsonc
   {
       "name": "eval",
       "type": "debugpy",
       "request": "launch",
       "program": "eval_diffusion.py",
       "console": "integratedTerminal"
   }
   ```

### 3. **Profiling the Model**
   To profile the model, use the `profile_model.py` script:
   ```bash
   python profile_model.py
   ```
   Example debug configuration:
   ```jsonc
   {
       "name": "profiler",
       "type": "debugpy",
       "request": "launch",
       "program": "profile_model.py",
       "console": "integratedTerminal"
   }
   ```

### 4. **Distributed Training**
   For distributed training, use the `torch.distributed.run` module:
   ```bash
   python -m torch.distributed.run --nproc_per_node=2 train.py
   ```
   Example debug configuration:
   ```jsonc
   {
       "name": "train (distributed + debug)",
       "type": "python",
       "request": "launch",
       "module": "torch.distributed.run",
       "console": "integratedTerminal",
       "args": [
           "--nproc_per_node=1",
           "--rdzv_backend", 
           "c10d",
           "--rdzv_endpoint", 
           "localhost:29500",
           "--nnodes=1",
           "train.py",
           "debug=True"
       ],
       "env": {
           "CUDA_VISIBLE_DEVICES": "1,2",
           "TORCH_NCCL_DEBUG": "INFO",
           "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
           "NCCL_P2P_DISABLE": "0",
           "NCCL_ASYNC_ERROR_HANDLING": "1"
       }
   }
   ```

---

## Project Structure

- **`models/`**: Contains the implementation of the video generation models, including joint attention mechanisms and parameter reduction schemes.
- **`data/`**: Handles data loading and preprocessing for the 1xgpt dataset.
- **`configs/`**: YAML configuration files for training and evaluation.
- **`train.py`**: Main training script.
- **`generate_videos.py`**: Script for generating videos from a trained model.
- **`profile_model.py`**: Script for profiling the model.
- **`.vscode/launch.json`**: Debug configurations for various scripts.