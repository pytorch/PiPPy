import os 

import torch
import torch.distributed as dist
from benchmark_utils import load_model, compile_model, display_memory_stats, display_model_size, benchmark_text_completion, prepare_prompts, write_to_csv
import logging
import fire

# Initialize logging for PyTorch Dynamo
torch._logging.set_logs(dynamo=logging.INFO, aot=logging.INFO, inductor=logging.INFO)



def main(model_args: str,
        converted_ckpt_dir: str,
        tokenizer_path:str,
        num_trials: int = 5,
        warmup: int = 2,
        batch_size:int=1,
        max_prompt_len: int = 100,
        max_gen_len: int = 10,
        
        # benchmark_mode: str ="text_completion", # chat_completion
         ):
    """Main function to run the llama benchmark."""
    seed = 40
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Load the model
    model, tokenizer = load_model(model_args= model_args,
                                  converted_ckpt_dir=converted_ckpt_dir,
                                  tokenizer_path=tokenizer_path)
    
    if rank==0:
        print("model is loaded")
        print("==============================")
    model_name="llama2"
    if rank==0:
        model_size = display_model_size(model, model_name)
        peak_active_gb, peak_reserved_gb = display_memory_stats("model_loading")
    
    # Compile the model
    # compile_model(model, )
    
    # Run inference
    prompt = "Generative AI has emerged as a groundbreaking domain, reshaping creative and technological landscapes. Leveraging deep learning, it can produce content, from images to text, that's often indistinguishable from human-generated work. Generative Adversarial Networks (GANs) stand out, where two neural networks contest to generate realistic outputs. OpenAI's DALL-E creates stunning visual art, while GPT models craft coherent text. Beyond art, generative AI aids in drug discovery, simulating molecular structures. However, it's not without challenges. Ethical concerns arise, especially with deepfakes and misinformation. As generative AI evolves, striking a balance between its potential and responsible use becomes paramount."
    print("prompts are ready")
    batch_prompts= prepare_prompts(prompt=prompt, tokenizer=tokenizer, max_prompt_len=max_prompt_len, batch_size=batch_size)
    print(f"batched prompts are ready and size is {len(batch_prompts)} and each element size is {len(batch_prompts[0])}")
   
        
    tokens_per_second, bandwidth = benchmark_text_completion(model= model, tokenizer=tokenizer,prompts=batch_prompts, max_gen_len=max_gen_len,num_trials=num_trials, warmup=warmup)
    print("benchmark is done")
    
    if local_rank==0:
        print(f"the avg token/second {tokens_per_second}")
        print("===================================================")
        peak_active_gb, peak_reserved_gb = display_memory_stats("end_of_inference")
        metrics = {}
        metrics["model_size"] = model_size
        metrics["batch_size"] = batch_size
        metrics["tokens/second"] = tokens_per_second
        metrics["bandwidth"] = bandwidth
        metrics["GPUs"] = world_size
    
        # Write values to CSV
        write_to_csv("llama_benchmarks.csv",metrics)
if __name__ == "__main__":
    fire.Fire(main)