#Use this file to load the model (WizardLM)

import torch
import transformers
from langchain.llms import HuggingFacePipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


def load_t5_model():
    tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

    model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

    return model, tokenizer

def build_t5(model, tokenizer):
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048
    )

    t5 = HuggingFacePipeline(pipeline=pipe)
    return t5
