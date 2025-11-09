from .text_tokenizers import T5TextEmbedder, CLIPTextEmbedder

class ConditioningManager:
    """Handles various conditioning mechanisms dynamically."""
    
    def __init__(self, conditioning_params):
        """
        Args:
            conditioning_types (list): List of conditioning types (e.g., ['text', 'action', 'past_frame'])
        """
        self.conditioning_modules = {}
        self.type = conditioning_params.type
        self.no_condition = False
        # Initialize required conditioning modules
        if 'text' == self.type or 'text' in self.type:
            if 't5' in conditioning_params.text_tokenizer.lower():
                self.conditioning_modules['text'] = T5TextEmbedder() 
            else:
                self.conditioning_modules['text'] = CLIPTextEmbedder() 
            self.conditioning_modules['context_dim'] = 512
        elif 'action' == self.type or 'action' in self.type:
            self.conditioning_modules['action'] = {}
        else:
            self.no_condition = True
    def __getitem__(self, key):
        return self.conditioning_modules[key]
    def get_module(self):
        """Returns the conditioning module if it exists, else None."""
        return self.conditioning_modules
    def get_tokenizer(self):
        """Returns the conditioning module if it exists, else None."""
        return self.conditioning_modules[self.type]
    def to(self, device):
        if not self.no_condition and hasattr(self.conditioning_modules[self.type], 'to'):
            self.conditioning_modules[self.type].to(device)