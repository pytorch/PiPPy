```
$ torchrun --nproc-per-node 2 pippy_llama.py
```
```
$ torchrun --nproc-per-node 4 pippy_llama.py
```
```
$ torchrun --nproc-per-node 8 pippy_llama.py
```
```
prompts = (
    "How do you", "I like to", "Can I help", "You need to",
    "The weather is", "I found a", "What is your", "You are so",
)
Outputs:
['make', 'think', 'you', 'be', 'getting', 'great', 'favorite', 'right']
```
