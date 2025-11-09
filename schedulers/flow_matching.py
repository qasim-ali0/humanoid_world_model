import numpy as np
import torch


class FlowMatching:
    def __init__(self, num_train_timesteps):
        self.set_timesteps(num_train_timesteps)

    def step(self, model_output, t, latents, generator=None):
        prev_latents = latents + self.dt * model_output
        return prev_latents

    def add_noise(self, latents, noise, timesteps):
        target_dim = (timesteps.shape[0],) + (1,) * (noise.ndim - 1)
        timesteps = timesteps.view(target_dim) / (self.n_timesteps - 1)
        return (1 - timesteps) * noise + timesteps * latents

    def get_target(self, latents, noise, timesteps):
        return latents - noise  # noise - latents

    def set_timesteps(self, num_timesteps):
        self.n_timesteps = num_timesteps
        self.dt = 1.0 / self.n_timesteps
        self.timesteps = (
            torch.from_numpy(
                (
                    np.linspace(0, num_timesteps - 1, num_timesteps)
                    .round()
                    .copy()
                    .astype(np.int64)
                )
            )
            / num_timesteps
        )
