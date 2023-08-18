#Use this file to load the model (WizardLM)

import torch
import transformers
from langchain.llms import HuggingFacePipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

def load_model():
    # CUDA out of memory. Model is too big.
    tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")

    #Use BitsAndBytesConfig to enble CPU offload
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

    model = LlamaForCausalLM.from_pretrained("TheBloke/wizardLM-7B-HF",
                                              device_map='auto',
                                              quantization_config=quantization_config,
                                              torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True
                                              )
    
    return model, tokenizer


def build_pipeline(model, tokenizer):

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15
    )

    hf_pipe = HuggingFacePipeline(pipeline=pipe)

    return hf_pipe

def load_t5_model():
    tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

    model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

    return model, tokenizer

def load_incite():
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")

    model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")

    return model, tokenizer

def build_t5(model, tokenizer):
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256
    )

    t5 = HuggingFacePipeline(pipeline=pipe)
    return t5
