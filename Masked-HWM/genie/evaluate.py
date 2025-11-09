"""
Example usage:
`python genie/evaluate.py --checkpoint_dir 1x-technologies/GENIE_35M`
"""

import argparse
import time
import os
import sys
from collections import defaultdict
from pathlib import Path
from safetensors.torch import load_file
import lpips
import torch
import transformers
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator

import matplotlib.pyplot as plt
# 1xgpt imports
sys.path.append(os.getcwd())
from data import RawTokenDataset
from visualize import decode_latents_wrapper
from eval_utils import decode_tokens, compute_lpips, AvgMetric, compute_loss
from genie.st_mask_git import STMaskGIT, GenieConfig

from data import get_maskgit_collator_evaluate

from cosmos_tokenizer.utils import tensor2numpy
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from torchvision.utils import save_image
import cv2

# Hardcoded values for the v1.1 dataset
WINDOW_SIZE = 3
STRIDE = 17  # Data is 30 Hz so with stride 15, video is 2 Hz


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GENIE-style models.")
    parser.add_argument(
        "--val_data_dir", type=str, default="/pub0/qasim/1xgpt/data/data_v2/val_v2.0",
        help="Directory containing tokenized data, should have a `video.bin`, `metadata.json` and `segment_ids.json`."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        help="Path to a HuggingFace-style checkpoint."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size, current script only supports a single GPU."
    )
    parser.add_argument(
        "--maskgit_steps", type=int, default=2, help="Number of MaskGIT sampling steps."
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Sampling temperature. If `temperature` <= 1e-8, will do greedy sampling."
    )
    parser.add_argument(
        "--save_outputs_dir", type=str,
        help="Debug option. If specified, will save model predictions and ground truths to this directory. "
             "Specifically, will save `{pred_frames,pred_logits,gtruth_frames,gtruth_tokens}.pt`"
    )
    parser.add_argument(
        "--max_examples", type=int,
        help="If specified, will stop evaluation early after `max_examples` examples."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default='outputs/evaluate.txt',
        help="If specified, will stop evaluation early after `max_examples` examples."
    )

    return parser.parse_args()


class GenieEvaluator:
    def __init__(self, args, config, decode_latents, device="cuda"):
        super().__init__()

        # self.model = STMaskGIT.from_pretrained(args.checkpoint_dir)
        self.model = STMaskGIT(config)
        checkpoint_path = os.path.join(args.checkpoint_dir, "model.safetensors")
        state_dict = load_file(checkpoint_path)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        self.with_act = config.with_act

        self.model = self.model.to(device=device)
        self.model.eval()

        self.decode_latents = decode_latents
        self.device = device
        self.args = args
        self.latent_h, self.latent_w = args.latent_h, args.latent_w
        self.maskgit_steps = args.maskgit_steps
        self.temperature = args.temperature

    def predict_zframe_logits(self, input_ids: torch.LongTensor, act: torch.LongTensor = None, start_frame=2) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Conditioned on each prefix: [frame_0], [frame_0, frame_1], ..., [frame_0, frame_1, ... frame_{T-1}],
        predict the tokens in the following frame: [pred_frame_1, pred_frame_2, ..., pred_frame_T].

        Image logits are denoised in parallel across spatial dimension and teacher-forced
        across the time dimension. To compute logits, we save both the samples and logits as we do MaskGIT generation.

        Total number of forward passes is (T-1) * maskgit steps.

        Args:
            input_ids: LongTensor of size (B, T*H*W) corresponding to flattened, tokenized images.

        Returns: (samples_THW, factored_logits)
            samples_THW:
                size (B, T, H, W) corresponding to the token ids of the predicted frames.
                May differ from the argmax of `factored_logits` if not greedy sampling.
            factored_logits:
                size (B, 512, 2, T-1, H, W) corresponding to the predicted logits.
                Note that we are factorizing the 2**18 vocabulary into two separate vocabularies of size 512 each.
        """
        inputs_THW = rearrange(input_ids, "b (t h w) -> b t h w", t=WINDOW_SIZE,
            h=self.latent_h, w=self.latent_w).to(self.device)
        
        if self.with_act:
            act = act.to(self.device)

        all_samples = []
        all_logits = []
        for timestep in range(start_frame, WINDOW_SIZE):
            print(f"Generating frame {timestep}")
            inputs_masked = inputs_THW.clone()

            if timestep > start_frame:
                inputs_masked[:, start_frame:timestep] = torch.stack(all_samples, dim=1)

            inputs_masked[:, timestep:] = self.model.mask_token_id

            # MaskGIT sampling
            samples_HW, factored_logits = self.model.maskgit_generate(
                inputs_masked, 
                out_t=timestep, 
                maskgit_steps=self.maskgit_steps,
                temperature=self.temperature,
                act=act,
            )

            all_samples.append(samples_HW)
            all_logits.append(factored_logits)

        samples_THW = torch.stack(all_samples, dim=1)
        return samples_THW, torch.stack(all_logits, dim=3)

    def predict_next_frames(
        self,
        gt_latents: torch.Tensor,     # (B, 6, H, W)
        pred_latents: torch.Tensor,   # (B, 3, H, W)
        decoder,
        output_dir: Path,
    ):
        output_dir = Path(output_dir)
        context_dir = output_dir / "context"
        gt_future_dir = output_dir / "gt_future"
        pred_future_dir = output_dir / "pred_future"

        context_dir.mkdir(parents=True, exist_ok=True)
        gt_future_dir.mkdir(parents=True, exist_ok=True)
        pred_future_dir.mkdir(parents=True, exist_ok=True)

        B, T_total, H, W = gt_latents.shape
        assert T_total == 3, f"Expected gt_latents with 6 time steps, got {T_total}"
        assert pred_latents.shape[1] == 1, "pred_latents should have 3 time steps"

        with torch.no_grad():
            for i in tqdm(range(B)):

                context_subdir = output_dir / "context" / f"{i}"
                gt_future_subdir = output_dir / "gt_future" / f"{i}"
                pred_future_subdir = output_dir / "pred_future" / f"{i}"
                
                context_subdir.mkdir(parents=True, exist_ok=True)
                gt_future_subdir.mkdir(parents=True, exist_ok=True)
                pred_future_subdir.mkdir(parents=True, exist_ok=True)
                
                context_and_gt_future = gt_latents[i].unsqueeze(0).cuda()
                context_and_pred_future = torch.concat([gt_latents[i, :2].cuda(), pred_latents[i].cuda()], dim=0).unsqueeze(0).cuda()

                context_and_gt_future_decoded = decoder.decode(context_and_gt_future).float().cpu().reshape(-1, 3, 17, 256, 256)
                context_and_pred_future_decoded = decoder.decode(context_and_pred_future).float().cpu().reshape(-1, 3, 17, 256, 256)

                pred_decoded = context_and_pred_future_decoded[:, :, 9:]
                context_decoded = context_and_gt_future_decoded[:, :, :9]
                gt_future_decoded = context_and_gt_future_decoded[:, :, 9:]

                # Decode and convert to numpy
                context_np = tensor2numpy(context_decoded)      # (1, T, H, W, C)
                gt_future_np = tensor2numpy(gt_future_decoded)
                pred_np = tensor2numpy(pred_decoded)

                # Save each frame
                for t in range(context_np.shape[1]):
                    frame = context_np[0, t]  # (H, W, C)
                    save_path = context_dir / f"{i}" / f"{t}.png"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                for t in range(gt_future_np.shape[1]):
                    frame = gt_future_np[0, t]
                    save_path = gt_future_dir / f"{i}" / f"{t}.png"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                for t in range(pred_np.shape[1]):
                    frame = pred_np[0, t]
                    save_path = pred_future_dir / f"{i}" / f"{t}.png"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

@torch.no_grad()
def main():
    transformers.set_seed(42)
    args = parse_args()

    # val_dataset = RawTokenDataset(args.val_data_dir, is_eval=True, with_act=False)
    val_dataset = RawTokenDataset(args.val_data_dir, is_eval=True, with_act=True)
    args.latent_h = args.latent_w = 32

    config = GenieConfig.from_pretrained(os.path.join(args.checkpoint_dir, "config.json"))
    config.with_act = True

    # decode_latents = decode_latents_wrapper()
    decode_latents = None

    lpips_alex = lpips.LPIPS(net="alex")  # Calculate LPIPS w/ AlexNet, which is the fastest model out of their options

    if args.max_examples is not None:
        val_dataset.valid_start_inds = val_dataset.valid_start_inds[:args.max_examples]

    print (len(val_dataset))
    
    dataloader = DataLoader(val_dataset, collate_fn=get_maskgit_collator_evaluate(config), batch_size=args.batch_size, shuffle=False)

    evaluator = GenieEvaluator(args, config, decode_latents)
    metrics = defaultdict(AvgMetric)
    loss_values = []

    if args.save_outputs_dir is not None:
        outputs_to_save = defaultdict(list)

    try:
        decoder_path = "YOUR DECODER PATH"
        decoder = CausalVideoTokenizer(checkpoint_dec=str(decoder_path))
        if decoder._dec_model is None:
            raise RuntimeError(f"Failed to load decoder model from {decoder_path}")
        print("Decoder initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading decoder: {str(e)}") from e

    f = open(args.log_file, 'w')
    all_input_ids = []
    all_pred_samples = []

    for batch in tqdm(dataloader):
        batch_size = batch["input_ids"].size(0)
        reshaped_input_ids = rearrange(batch["input_ids"], "b (t h w) -> b t h w", t=WINDOW_SIZE,
                                       h=args.latent_h, w=args.latent_w)

        start_time = time.time()
        samples, factored_logits = evaluator.predict_zframe_logits(batch["input_ids"], act=batch["actions"])

        frames_per_batch = (WINDOW_SIZE - 1) * batch["input_ids"].size(0)
        metrics["gen_time"].update((time.time() - start_time) / frames_per_batch, batch_size)

        loss = compute_loss(batch["labels"], factored_logits, num_factored_vocabs=1, factored_vocab_size=64000)
        loss_values.append(loss)

        acc = (reshaped_input_ids[:, 2:].to("cuda") == samples).float().mean().item()

        metrics["loss"].update(loss, batch_size)
        metrics["acc"].update(acc, batch_size)

        all_input_ids.append(reshaped_input_ids)
        all_pred_samples.append(samples)

        # start_time = time.time()
        # pred_frames = evaluator.predict_next_frames(samples)
        # metrics["dec_time"].update((time.time() - start_time) / frames_per_batch, batch_size)

        # decoded_gtruth = decode_tokens(reshaped_input_ids, decode_latents)
        # metrics["pred_lpips"].update_list(compute_lpips(decoded_gtruth[:, 1:], pred_frames, lpips_alex))
        
        print({key: f"{val.mean():.4f}" for key, val in metrics.items()})
        # if args.save_outputs_dir is not None:
        #     # outputs_to_save["pred_frames"].append(pred_frames)
        #     outputs_to_save["pred_tokens"].append(samples)
        #     # outputs_to_save["pred_logits"].append(factored_logits)
        #     # outputs_to_save["gtruth_frames"].append(decoded_gtruth)
        #     outputs_to_save["gtruth_tokens"].append(reshaped_input_ids)
    
    for key,val in metrics.items():
        f.write(key + ': ' + str(val.mean()) + '\n')
    f.close()

    evaluator.predict_next_frames(torch.concat(all_input_ids, dim=0), torch.concat(all_pred_samples, dim=0), decoder, args.save_outputs_dir)

    plt.figure(figsize=(8, 5))
    plt.hist(loss_values, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Evaluation Loss")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/eval_loss_histogram.png")
    plt.close()


if __name__ == "__main__":
    main()
