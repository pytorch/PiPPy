import torch
from llama2 import Llama
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional, Tuple
import abc 
import fire

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@torch.inference_mode()
def generate(model,
    tokenizer,
    prompt_tokens: List[List[int]],
    max_gen_len: int,
    temperature: float = 0.6,
    top_p: float = 0.9,
    logprobs: bool = False,
    echo: bool = False,
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    """
    Generate text sequences based on provided prompts using the language generation model.

    Args:
        prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
        max_gen_len (int): Maximum length of the generated text sequence.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    Returns:
        Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

    Note:
        This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
        If logprobs is True, token log probabilities are computed for each generated token.

    """
    bsz = len(prompt_tokens)
    assert bsz <= model.max_batch_size, (bsz, model.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= model.max_seq_len
    total_len = min(model.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    if logprobs:
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id
    if min_prompt_len == total_len:
        logits = model.forward(tokens, prev_pos)
        token_logprobs = -F.cross_entropy(
            input=logits.transpose(1, 2),
            target=tokens,
            reduction="none",
            ignore_index=pad_id,
        )

    for cur_pos in range(min_prompt_len, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        if logprobs:
            token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens[:, prev_pos + 1 : cur_pos + 1],
                reduction="none",
                ignore_index=pad_id,
            )
        eos_reached |= (~input_text_mask[:, cur_pos]) & (
            next_token == tokenizer.eos_id
        )
        prev_pos = cur_pos
        if all(eos_reached):
            break

    if logprobs:
        token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        probs = None
        if logprobs:
            probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        if tokenizer.eos_id in toks:
            eos_idx = toks.index(tokenizer.eos_id)
            toks = toks[:eos_idx]
            probs = probs[:eos_idx] if logprobs else None
        out_tokens.append(toks)
        out_logprobs.append(probs)
    return (out_tokens, out_logprobs if logprobs else None)


def token_generation(
    ckpt_dir: str,
    converted_ckpt_dir: str,
    tokenizer_path: str,
    model_parallel_size: int,
    max_seq_len: int=512,
    max_batch_size: int=4,
    ):
    dist.init_process_group("nccl")
    
      
    llama_model_and_tok = Llama.build(
            ckpt_dir=ckpt_dir,
            converted_ckpt_dir=converted_ckpt_dir,
            tokenizer_path= tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )
    model = llama_model_and_tok.model
    
    print(model)
    tokenizer = llama_model_and_tok.tokenizer
    print("model is done")
    #plan to pass the model to convert checkpoints 
    
    prompt =[tokenizer.encode("how does it feel to be the first test case", bos=True,eos=False)]
    
    generation_tokens, generation_logprobs = generate(model,
                                                        tokenizer,
                                                        prompt_tokens=prompt,
                                                        max_gen_len= 100,
                                                        temperature = 0.6,
                                                        top_p= 0.9,
                                                    )
    print([{"generation": tokenizer.decode(t)} for t in generation_tokens])
if __name__ == "__main__":
    fire.Fire(token_generation)
    