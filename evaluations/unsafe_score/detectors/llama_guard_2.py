from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# import translators
from .detectors import Detector


name = "meta-llama/Meta-Llama-Guard-2-8B"
device = "cuda"
dtype = torch.bfloat16

class LlamaGuard2(Detector):
    def __init__(self, name=name, from_language='en', device=device, dtype=dtype):
        super().__init__(name=name, from_language=from_language, device=device, dtype=dtype)
        
        if self.from_language != 'en':
            self.translate = lambda x: x #translators.translate_text(x[:1000], from_language=self.from_language)
        else:
            self.translate = lambda x:x
    def moderate(self, chat):

        for idx, _ in enumerate(chat):
            try:
                chat[idx]['content'] = self.translate(chat[idx]['content'])
            except Exception as e:
                print(e)
                chat[idx]['content'] = chat[idx]['content']

        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        ret = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        
        return 1 if 'unsafe' in ret else 0


