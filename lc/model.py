#Use this file to load the model (WizardLM)

import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig

def load_model():
    tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")

    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

    model = LlamaForCausalLM.from_pretrained("TheBloke/wizardLM-7B-HF",
                                              device_map='auto',
                                              quantization_config=quantization_config,
                                              torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True
                                              )
    
    return model, tokenizer