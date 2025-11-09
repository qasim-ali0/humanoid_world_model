# 🧠 Masked-HWM: Masked Video Modeling for World Model Learning

Authors: **Qasim Ali, Aditya Sridhar**  
License: **MIT License**  

---

## Overview

**Masked-HWM** is a **masked video modeling framework** for learning world models.
Inspired by **MaskGIT** and **MAGVIT**, it leverages *masking and noising strategies* on **discrete video tokens** to learn robust spatiotemporal dynamics.
The model operates on pre-tokenized video data produced by the **Cosmos DV-8×8×8 Tokenizer**.

---

## Dataset Setup

Masked-HWM uses the **1xGPT dataset**, which contains pre-tokenized frames (Cosmos DV-8×8×8).

To download:

```bash
huggingface-cli download 1x-technologies/worldmodel --repo-type dataset --local-dir data
```

---

## Cosmos Tokenizer Setup

1. Clone and set up the **Cosmos Tokenizer** repository:
   [https://github.com/NVIDIA/Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer)

2. Download the **DV-8×8×8 tokenizer** using:

   ```bash
   python download_cosmos.py --variant DV8x8x8 --token <your_hf_token>
   ```

---

## Usage

This repository includes scripts for **training**, **evaluation**, and **distributed training**.
Configuration files for debugging and experiment setup can be managed through `.vscode/launch.json`.

---

### 1. **Training the Model**

To train Masked-HWM:

```bash
python train.py \
    --with_act \
    --genie_config genie/configs/cosmos_24_512.json \
    --train_data_dir <path_to_train_data> \
    --val_data_dir <path_to_val_data> \
    --log_name cosmos_24_512 \
    --output_dir logs \
    --max_eval_steps 1000 \
    --seed 5 \
    --eval_every_n_steps 8000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 60000 \
    --learning_rate 0.00003
```

> **Note:**
>
> * By default, the *Base Block* variant is trained.
> * To use another variant, uncomment the corresponding block lines (883–885) in `genie/st_transformer.py`.

#### Example Debug Configuration

```jsonc
{
    "name": "train (Masked-HWM)",
    "type": "debugpy",
    "request": "launch",
    "program": "train.py",
    "console": "integratedTerminal",
    "args": ["--with_act", "--genie_config", "genie/configs/cosmos_24_512.json"],
    "justMyCode": false,
    "env": {
        "CUDA_VISIBLE_DEVICES": "0"
    }
}
```

---

### 2. **Distributed Training**

For multi-GPU training, use:

```bash
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    train.py
```

#### Example Debug Configuration

```jsonc
{
    "name": "train (distributed)",
    "type": "python",
    "request": "launch",
    "module": "torch.distributed.run",
    "console": "integratedTerminal",
    "args": [
        "--nproc_per_node=2",
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", "localhost:29500",
        "train.py"
    ],
    "env": {
        "CUDA_VISIBLE_DEVICES": "0,1"
    }
}
```

---

### 3. **Inference and Evaluation**

To run inference:

```bash
python genie/evaluate.py \
    --checkpoint_dir <checkpoint_directory> \
    --save_outputs_dir <output_directory> \
    --max_examples <optional_number_of_samples> \
    --batch_size 6
```

> Update **line 252** in `evaluate.py` with your decoder path before running inference.

Currently, generation is configured to predict **8 future frames (1 latent)**.
To enable **autoregressive generation**, modify the `predict_zframe_logits` method to:

* Predict continuously using a `WINDOW_SIZE` latent window, and
* Feed back previous predictions for iterative rollout.

---

## Project Structure

* **`genie/`** — Core model and transformer implementation.
* **`configs/`** — Configuration files for training and inference.
* **`train.py`** — Main training entry point.
* **`evaluate.py`** — Video generation and evaluation script.
* **`.vscode/launch.json`** — Debug configurations for local development.
* **`data/`** — Directory for pre-tokenized dataset storage.

---

## Citation

If you find Masked-HWM useful in your research, please consider citing it (BibTeX coming soon).

