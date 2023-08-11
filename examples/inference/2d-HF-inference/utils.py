import re
import torch
import torch.nn as nn

from torch.distributed.tensor.parallel import (
        PairwiseParallel,
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )


def format_to_gb(item, precision=4):
    gigabyte_size = 1024 ** 3
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


def print_mem_usage():
    memory_reserved = format_to_gb(torch.cuda.memory_reserved())
    memory_allocated = format_to_gb(torch.cuda.memory_allocated())
    print(
        f"memory_reserved: {memory_reserved} GB, "
        f"memory_allocated: {memory_allocated} GB"
    )
    
def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_submodules(model):
        for name, module in model.named_modules():
            print(f"Module name: {name}")
            # print(module)
            print()