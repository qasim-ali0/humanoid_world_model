# 🧠 Masked-HWM

**Masked-HWM** is a masked video modeling method for world model learning. Inspired by MaskGIT and MAGVIT, it uses a masking + noising training technique with discrete tokens to learn spatiotemporal dynamics.

---

## Setup

### Dataset

We use the 1xGPT dataset, which contains frames which have already been tokenized using the DV-8x8x8 Cosmos Tokenizer.

`huggingface-cli download 1x-technologies/worldmodel -repo-type dataset --local-dir data`

### Cosmos Tokenizer

First download and perform setup for https://github.com/NVIDIA/Cosmos-Tokenizer.git. Then, download the DV 8x8x8 tokenizer using `download_cosmos.py` with your HF token.

## Training

Below is an example training command. Note that some arguments must be filled in manually.
```
python train.py --with_act --genie_config genie/configs/cosmos_24_512.json --train_data_dir <path to train data> --val_data_dir <path to val data> --log_name cosmos_24_512 --output_dir logs --max_eval_steps 1000  --seed 5 --eval_every_n_steps 8000 --per_device_train_batch_size 4 --per_device_eval_batch_size 4  --gradient_accumulation_steps 2 --max_train_steps 60000 --learning_rate 0.00003
```

Distributed training is supported using `python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py`.

## Inference

In `evaluate.py`, add your decoder path on line 252. You can run generation as follows:

```python genie/evaluate.py --checkpoint_dir <checkpoint directory> --save_outputs_dir <where to save outputs> --max_examples <(optional) how many examples to process> --batch_size 6```

Note that as of now, this is configured to generate 8 new frames (1 latent). A more configurable generation script is coming soon, but if you are interested in implementing it manually it would mainly require changing the `predict_zframe_logits` method to work in a autoregressive fashion, continuously predicting using `WINDOW_SIZE` latents and feeding old predictions back in.