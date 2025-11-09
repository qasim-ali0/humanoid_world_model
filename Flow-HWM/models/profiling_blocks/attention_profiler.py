import torch
import torch.nn as nn

from models.common_blocks import *


class AttentionProfiler:
    def __init__(self, device: str = "cuda"):
        """
        Initialize the profiler with the specified device.

        Args:
            device: Device to run the profiling on ('cuda' or 'cpu')
        """
        self.device = device
        self.attention_types = ["naive", "torch", "xformers", "flash"]
        self.results = {}

    def _generate_inputs(
        self, batch_size: int, seq_len: int, dim: int, dtype=torch.float32
    ) -> torch.Tensor:
        """Generate random input tensors for profiling."""
        return torch.randn(batch_size, seq_len, dim, device=self.device, dtype=dtype)

    def _create_attention_module(
        self, attn_type: str, dim: int, num_heads: int, dtype=torch.float32
    ) -> Attention:
        """Create an attention module with the specified configuration."""
        return Attention(
            dim=dim, num_heads=num_heads, qkv_bias=True, attn_type=attn_type
        ).to(self.device)

    def profile_single_config(
        self,
        batch_size: int,
        seq_len: int,
        dim: int,
        num_heads: int,
        num_repeats: int = 100,
        warmup: int = 10,
        dtype=torch.float32,
    ) -> Dict[str, float]:
        """
        Profile all attention types for a single configuration.

        Args:
            batch_size: Batch size for input
            seq_len: Sequence length for input
            dim: Hidden dimension
            num_heads: Number of attention heads
            num_repeats: Number of times to repeat the forward pass for timing
            warmup: Number of warmup runs before timing

        Returns:
            Dictionary of average execution times for each attention type
        """
        results = {}
        inputs = self._generate_inputs(batch_size, seq_len, dim, dtype)

        for attn_type in self.attention_types:
            try:
                # Skip flash attention for CPU
                if attn_type == "flash" and self.device == "cpu":
                    print(f"Skipping flash attention on CPU as it's not supported")
                    continue

                model = self._create_attention_module(attn_type, dim, num_heads).to(
                    dtype
                )
                model.eval()  # Set to evaluation mode to disable dropout

                # Warmup
                for _ in range(warmup):
                    with torch.no_grad():
                        _ = model(inputs)

                # Perform timing
                start_time = time.time()
                for _ in range(num_repeats):
                    with torch.no_grad():
                        _ = model(inputs)

                # Synchronize if using CUDA
                if self.device == "cuda":
                    torch.cuda.synchronize()

                end_time = time.time()
                avg_time = (end_time - start_time) / num_repeats * 1000  # Convert to ms
                results[attn_type] = avg_time

                print(
                    f"Config [B={batch_size}, L={seq_len}, D={dim}, H={num_heads}] "
                    f"- {attn_type}: {avg_time:.3f} ms"
                )

            except Exception as e:
                print(f"Error profiling {attn_type} attention: {e}")
                results[attn_type] = float("nan")

        return results

    def profile_varying_sequence_length(
        self,
        batch_size: int,
        seq_lengths: List[int],
        dim: int,
        num_heads: int,
        num_repeats: int = 100,
        dtype=torch.float32,
    ):
        """Profile with varying sequence lengths."""
        results = {attn_type: [] for attn_type in self.attention_types}

        for seq_len in seq_lengths:
            print(f"\nProfiling with sequence length {seq_len}")
            config_result = self.profile_single_config(
                batch_size, seq_len, dim, num_heads, num_repeats, dtype=dtype
            )

            for attn_type, time_ms in config_result.items():
                results[attn_type].append(time_ms)

        self.results["seq_length"] = {"seq_lengths": seq_lengths, "times": results}
        return results

    def profile_varying_hidden_dim(
        self,
        batch_size: int,
        seq_len: int,
        dims: List[int],
        num_heads_list: Optional[List[int]] = None,
        num_repeats: int = 100,
        dtype=torch.float32,
    ):
        """Profile with varying hidden dimensions."""
        if num_heads_list is None:
            # Default to dim / 64 heads
            num_heads_list = [dim // 64 for dim in dims]

        assert len(dims) == len(
            num_heads_list
        ), "dims and num_heads_list must have same length"

        results = {attn_type: [] for attn_type in self.attention_types}

        for dim, num_heads in zip(dims, num_heads_list):
            print(f"\nProfiling with dimension {dim} and {num_heads} heads")
            config_result = self.profile_single_config(
                batch_size, seq_len, dim, num_heads, num_repeats, dtype=dtype
            )

            for attn_type, time_ms in config_result.items():
                results[attn_type].append(time_ms)

        self.results["hidden_dim"] = {
            "dims": dims,
            "num_heads": num_heads_list,
            "times": results,
        }
        return results

    def profile_varying_batch_size(
        self,
        batch_sizes: List[int],
        seq_len: int,
        dim: int,
        num_heads: int,
        num_repeats: int = 100,
        dtype=torch.float32,
    ):
        """Profile with varying batch sizes."""
        results = {attn_type: [] for attn_type in self.attention_types}

        for batch_size in batch_sizes:
            print(f"\nProfiling with batch size {batch_size}")
            config_result = self.profile_single_config(
                batch_size, seq_len, dim, num_heads, num_repeats, dtype=dtype
            )

            for attn_type, time_ms in config_result.items():
                results[attn_type].append(time_ms)

        self.results["batch_size"] = {"batch_sizes": batch_sizes, "times": results}
        return results

    def profile_memory_usage(
        self,
        batch_size: int,
        seq_len: int,
        dim: int,
        num_heads: int,
        dtype=torch.float32,
    ):
        """Profile memory usage for each attention type."""
        if self.device != "cuda":
            print("Memory profiling is only available on CUDA devices")
            return {}

        memory_usage = {}
        inputs = self._generate_inputs(batch_size, seq_len, dim, dtype)

        for attn_type in self.attention_types:
            try:
                # Skip flash attention for CPU
                if attn_type == "flash" and self.device == "cpu":
                    continue

                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                model = self._create_attention_module(attn_type, dim, num_heads).to(
                    dtype
                )

                # Forward pass
                with torch.no_grad():
                    _ = model(inputs)

                # Get memory stats
                memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                memory_usage[attn_type] = memory_used

                print(
                    f"Config [B={batch_size}, L={seq_len}, D={dim}, H={num_heads}] "
                    f"- {attn_type}: {memory_used:.2f} MB"
                )

            except Exception as e:
                print(f"Error profiling memory for {attn_type} attention: {e}")
                memory_usage[attn_type] = float("nan")

        self.results["memory"] = memory_usage
        return memory_usage

    def plot_results(self, save_path: Optional[str] = None):
        """Plot the profiling results."""
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # Plot 1: Sequence Length vs. Time
        if "seq_length" in self.results:
            data = self.results["seq_length"]
            seq_lengths = data["seq_lengths"]
            times = data["times"]

            for attn_type, time_list in times.items():
                if any(not np.isnan(t) for t in time_list):
                    axs[0].plot(seq_lengths, time_list, marker="o", label=attn_type)

            axs[0].set_xlabel("Sequence Length")
            axs[0].set_ylabel("Time (ms)")
            axs[0].set_title("Attention Performance vs. Sequence Length")
            axs[0].legend()
            axs[0].grid(True)

        # Plot 2: Hidden Dimension vs. Time
        if "hidden_dim" in self.results:
            data = self.results["hidden_dim"]
            dims = data["dims"]
            times = data["times"]

            for attn_type, time_list in times.items():
                if any(not np.isnan(t) for t in time_list):
                    axs[1].plot(dims, time_list, marker="o", label=attn_type)

            axs[1].set_xlabel("Hidden Dimension")
            axs[1].set_ylabel("Time (ms)")
            axs[1].set_title("Attention Performance vs. Hidden Dimension")
            axs[1].legend()
            axs[1].grid(True)

        # Plot 3: Batch Size vs. Time
        if "batch_size" in self.results:
            data = self.results["batch_size"]
            batch_sizes = data["batch_sizes"]
            times = data["times"]

            for attn_type, time_list in times.items():
                if any(not np.isnan(t) for t in time_list):
                    axs[2].plot(batch_sizes, time_list, marker="o", label=attn_type)

            axs[2].set_xlabel("Batch Size")
            axs[2].set_ylabel("Time (ms)")
            axs[2].set_title("Attention Performance vs. Batch Size")
            axs[2].legend()
            axs[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Results saved to {save_path}")
        else:
            plt.show()

    def summarize_results(self):
        """Print a summary of the profiling results."""
        print("\n===== ATTENTION PROFILING SUMMARY =====")

        if "memory" in self.results:
            print("\nMemory Usage (MB):")
            # memory_data = self.


if __name__ == "__main__":
    profiler = AttentionProfiler("cuda")

    print("vary seq len", "torch.float32")
    print(
        profiler.profile_varying_sequence_length(
            16, [256, 512, 768, 1024, 2048], 512, 8, dtype=torch.float32
        )
    )

    print("vary seq len", "torch.bfloat16")
    print(
        profiler.profile_varying_sequence_length(
            16, [256, 512, 768, 1024, 2048], 512, 8, dtype=torch.bfloat16
        )
    )

    print("vary hidden dim", "torch.float32")
    print(
        profiler.profile_varying_hidden_dim(
            16, 512, [256, 512, 768, 1024, 2048], [8, 8, 8, 8, 8], dtype=torch.float32
        )
    )

    print("vary hidden dim", "torch.bfloat16")
    print(
        profiler.profile_varying_hidden_dim(
            16, 512, [256, 512, 768, 1024, 2048], [8, 8, 8, 8, 8], dtype=torch.bfloat16
        )
    )

    print("vary batch size", "torch.float32")
    print(
        profiler.profile_varying_batch_size(
            [16, 16, 24, 32], 512, 512, 8, dtype=torch.float32
        )
    )

    print("vary batch size", "torch.bfloat16")
    print(
        profiler.profile_varying_batch_size(
            [16, 16, 24, 32], 512, 512, 8, dtype=torch.bfloat16
        )
    )

    print("memory", "torch.float32")
    print(profiler.profile_memory_usage(16, 1024, 512, 8, torch.float32))

    print("memory", "torch.bfloat16")
    print(profiler.profile_memory_usage(16, 1024, 512, 8, torch.bfloat16))
