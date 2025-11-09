import torch.nn as nn
import torch
import math

class FFT2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.proj = nn.Linear(embed_dim, embed_dim)  # Learnable frequency weighting
        
    def forward(self, x):
        x_freq_embed = torch.fft.fft(x, dim=-1)
        x_freq_both = torch.fft.fft(x_freq_embed, dim=-2)
        return x_freq_both.real / math.sqrt(x.size(-2) * x.size(-1))
        
        # return self.freq_proj(x_freq_both.real)

class InverseFFT2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        x_ifft_seq = torch.fft.ifft(x, dim=-2)
        x_ifft_both = torch.fft.ifft(x_ifft_seq, dim=-1)

        return x_ifft_both.real * math.sqrt(x.size(-2) * x.size(-1))