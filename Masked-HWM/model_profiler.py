import torch
import torch.nn as nn
import os
import argparse
import time
from torch.utils.data import DataLoader
from torchvision import models
from genie.st_mask_git import STMaskGIT, GenieConfig
from safetensors.torch import load_file
from data import get_maskgit_collator_evaluate, RawTokenDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Profile GENIE-style models.")
    parser.add_argument(
        "--checkpoint_dir", type=str,
        help="Path to a HuggingFace-style checkpoint."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for profiling."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run the profiling on (default is 'cuda')."
    )
    parser.add_argument(
        "--num_batches", type=int, default=10,
        help="Number of batches to run for throughput profiling."
    )
    return parser.parse_args()


class ModelProfiler:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def report_params(self):
        """Reports the number of parameters in the model."""
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of Parameters: {num_params / 1e6:.2f}M")

    def report_peak_memory_usage(self, dataloader):
        """Reports the peak memory usage during a forward pass."""
        torch.cuda.empty_cache()
        self.model.eval()

        with torch.no_grad():
            ex = next(iter(dataloader))
            for key, val in ex.items():
                if val is not None:
                    ex[key] = val.to(self.device)
            self.model(ex["input_ids"], ex["labels"], ex["actions"])

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            ex = next(iter(dataloader))
            for key, val in ex.items():
                if val is not None:
                    ex[key] = val.to(self.device)
            self.model(ex["input_ids"], ex["labels"], ex["actions"])

        peak_memory = torch.cuda.max_memory_allocated(device=self.device) / 1e9  # in GB
        print(f"Peak Memory Usage: {peak_memory:.2f} GB")

    def report_throughput(self, dataloader, num_batches):
        """Reports model throughput in samples per second."""
        self.model.eval()
        total_samples = 0
        total_time = 0.0

        with torch.no_grad():
            for i, ex in enumerate(dataloader):
                if i >= num_batches:
                    break
                for key, val in ex.items():
                    if val is not None:
                        ex[key] = val.to(self.device)

                start = time.time()
                self.model(ex["input_ids"], ex["labels"], ex["actions"])
                torch.cuda.synchronize()
                end = time.time()

                total_time += end - start
                total_samples += ex["input_ids"].size(0)

        throughput = total_samples / total_time
        print(f"Throughput: {throughput:.2f} samples/sec")

        # Optional: dummy EC = total_time * 100 (placeholder energy cost in Joules)
        dummy_ec = total_time * 100
        print(f"Estimated Samples per EC (fake EC=100J/sec): {total_samples / dummy_ec:.4f} samples/J")


def main():
    args = parse_args()
    
    config = GenieConfig.from_pretrained(os.path.join(args.checkpoint_dir, "config.json"))
    config.with_act = True

    val_dataset = RawTokenDataset("/pub0/qasim/1xgpt/data/data_v2/val_v2.0", is_eval=True, with_act=True)
    dataloader = DataLoader(
        val_dataset,
        collate_fn=get_maskgit_collator_evaluate(config),
        batch_size=args.batch_size,
        shuffle=True
    )

    model = STMaskGIT(config)
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.safetensors")
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)

    profiler = ModelProfiler(model, args.device)
    profiler.report_params()
    profiler.report_peak_memory_usage(dataloader)
    profiler.report_throughput(dataloader, args.num_batches)


if __name__ == "__main__":
    main()
