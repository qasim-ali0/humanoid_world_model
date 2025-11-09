from diffusers import DDPMScheduler, DDIMScheduler, FlowMatchEulerDiscreteScheduler
from .flow_matching import FlowMatching
def get_scheduler(type, num_steps):
    if type.lower() == 'ddpm':
        return HF_Wrapper(DDPMScheduler(num_train_timesteps=num_steps))
    elif type.lower() == 'ddim':
        return HF_Wrapper(DDIMScheduler(num_train_timesteps=num_steps))
    elif type.lower() == 'flow':
        return FlowMatching(num_train_timesteps=num_steps)
    else:
        raise Exception('Unknown scheduler type')
    
class HF_Wrapper:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.timesteps = scheduler.timesteps
    def step(self, model_output, t, latents, generator=None):
        return self.scheduler.step(model_output, t, latents, generator=generator)
    def add_noise(self, latents, noise, timesteps):
        return self.scheduler.add_noise(latents, noise, timesteps)
    def get_target(self, latents, noise, timesteps):
        return noise
    def set_timesteps(self, num_timesteps):
        self.scheduler.set_timesteps(num_timesteps)
        self.timesteps = self.timesteps
