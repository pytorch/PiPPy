import torch
from llama2 import Llama
import torch.distributed as dist

import fire

def convert_checkpoints(
    ckpt_dir: str,
    tokenizer_path: str,
    model_parallel_size: int,
    max_seq_len: int=512,
    max_batch_size: int=4,
    ):
    dist.init_process_group("nccl")
    llama_model_and_tok = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path= tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )
    model = llama_model_and_tok.model
    tokenizer = llama_model_and_tok.tokenizer
    print("model is done")
    #plan to pass the model to convert checkpoints 
if __name__ == "__main__":
    fire.Fire(convert_checkpoints)
    