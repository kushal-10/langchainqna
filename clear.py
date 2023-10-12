import gc
import torch
# def report_gpu():
# print(torch.cuda.list_gpu_processes())
gc.collect()
torch.cuda.empty_cache()