from transformers import T5Tokenizer, T5EncoderModel
import torch
import abc
from transformers import CLIPTextModel, CLIPTokenizer
from .tokenizer import Embedder        
        
class T5TextEmbedder(Embedder):
    def __init__(self, device='cpu'):
        self.t5_model = T5EncoderModel.from_pretrained("google-t5/t5-small").to(device)
        for param in self.t5_model.parameters():
            param.requires_grad = False
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        self._device = device
        
    def tokenize(self, captions):
        return self.t5_tokenizer(captions, padding="max_length", truncation=True, return_tensors="pt", max_length=200)
    
    def to(self, device):
        self.t5_model.to(device)
        self._device = device
    def get_embeddding(self, captions):
        # Tokenize and process captions
        inputs = {key: value.to(self.t5_model.device).squeeze(1) for key, value in captions.items()}
        # inputs = [{key: inputs[key][i, 0, :].clone() for key in inputs} for i in range(batch_size)]

        # Get model embeddings
        with torch.no_grad():
            outputs = self.t5_model.encoder(**inputs)
        
        # Extract the hidden states
        embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

        # Compute mean pooling (ignoring padding tokens)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # Expand for broadcasting
        pooled_embeddings = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        return pooled_embeddings  # Shape: (batch_size, hidden_dim)

class CLIPTextEmbedder(Embedder):
    def __init__(self, device='cpu'):
        # Load the CLIP text model and tokenizer
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        # Freeze model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self._device = device

    def tokenize(self, captions):
        # Use a fixed maximum length (77 is typical for CLIP)
        return self.clip_tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=77
        )

    def to(self, device):
        self.clip_model.to(device)
        self._device = device

    def get_embeddding(self, captions):
        # Ensure that all tensor inputs are on the proper device
        # and remove extra dimensions if they exist.
        inputs = {}
        for key, value in captions.items():
            tensor = value.to(self._device)
            # Only squeeze if there's a singleton dimension in position 1
            if tensor.ndim == 3 and tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            inputs[key] = tensor

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        # Use the pooler output as the embedding representation
        embeddings = outputs.pooler_output  # Shape: (batch_size, hidden_dim)
        return embeddings