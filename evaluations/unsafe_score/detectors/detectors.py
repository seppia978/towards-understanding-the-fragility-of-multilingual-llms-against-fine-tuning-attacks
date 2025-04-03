from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Detector:
    def __init__(self, name, from_language='en', device='auto', dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map=device)
        self.from_language = from_language
        
    def to(self, device):
        return self.model.to(device)