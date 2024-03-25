# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Owner(s): ["oncall: stl_multimodal"]

import os
import sys

import torch

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.testing._internal.common_fsdp import FSDPTest

from llama2 import Llama

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


class TestLLaMA2Native(FSDPTest):
    def setUp(self) -> None:
        super().setUp()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    def _init_model(self):
        os.environ["LOCAL_RANK"] = str(dist.get_rank())
        _DEFAULT_CKPT_DIR = os.environ.get("DEFAULT_CKPT_DIR", None)
        _DEFAULT_TOKENIZER_PATH = os.environ.get("DEFAULT_TOKENIZER_PATH", None)
        if not _DEFAULT_CKPT_DIR or not _DEFAULT_TOKENIZER_PATH:
            raise RuntimeError(
                "Users need to set DEFAULT_CKPT_DIR and DEFAULT_TOKENIZER_PATH env variables for test to run."
            )
        llama_model_and_tok = Llama.build(
            ckpt_dir=_DEFAULT_CKPT_DIR,
            tokenizer_path=_DEFAULT_TOKENIZER_PATH,
            max_seq_len=512,
            max_batch_size=4,
            model_parallel_size=2,
        )
        model = llama_model_and_tok.model
        tok = llama_model_and_tok.tokenizer
        return model, tok

    def test_init(self):
        model, _ = self._init_model()
        self.assertTrue(isinstance(model, FSDP))
        for tformer_block in model.layers:
            self.assertTrue(isinstance(tformer_block, FSDP))

    def test_fwd_bwd(self):
        model, tok = self._init_model()
        prompts = [
            "I believe the meaning of life is",
        ]
        prompt_tokens = [
            torch.tensor(
                tok.encode(x, bos=True, eos=False),
                dtype=torch.long,
                device="cuda",
            )
            for x in prompts
        ]

        for tokens in prompt_tokens:
            out = model(tokens.unsqueeze(0), 0)
            loss = out.sum()
            loss.backward()


if __name__ == "__main__":
    run_tests()