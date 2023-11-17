# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import logging
import os
import types
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed
import transformers
import transformers.utils.fx as fx
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.utils import cached_property, is_torch_available

from pippy.PipelineDriver import PipelineDriverBase


logger = logging.getLogger(__name__)


@dataclass
class PiPPyTrainingArguments(TrainingArguments):
    dp_group_size: int = field(default=-1, metadata={"help": "DP group size."})

    pp_group_size: int = field(
        default=-1, metadata={"help": "Pipeline group size."}
    )

    rank: int = field(
        default=int(os.getenv("RANK", -1)), metadata={"help": "Rank."}
    )

    driver_index: int = field(
        default=-1,
        metadata={
            "help": "Index of current pipeline driver in all pipeline drivers."
        },
    )

    local_driver_index: int = field(
        default=-1,
        metadata={
            "help": "Index of current pipeline driver in local pipeline drivers."
        },
    )

    master_addr: str = field(
        default=os.getenv("MASTER_ADDR", "localhost"),
        metadata={"help": "Master address."},
    )

    master_port: str = field(
        default=os.getenv("MASTER_PORT", "29500"),
        metadata={"help": "Master port."},
    )

    exclude_master: int = field(
        default=0,
        metadata={"help": "Exclude master.", "choices": [0, 1]},
    )

    # TODO: use `no_cuda` instead?
    cuda: int = field(
        default=int(torch.cuda.is_available()),
        metadata={"help": "Exclude master.", "choices": [0, 1]},
    )

    chunks: Optional[int] = field(
        default=None, metadata={"help": "Number of Chunks."}
    )

    record_mem_dumps: int = field(
        default=0, metadata={"help": "Record memory dumps flag."}
    )

    checkpoint: int = field(default=1, metadata={"help": "Checkpoint flag."})

    _device: Optional[torch.device] = None

    @property
    def device(self):
        if self.rank == -1:
            if self.cuda and torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return super().device

    @device.setter
    def device(self, value):
        self._device = value

    # Process Group including all drivers
    _driver_group = None

    @property
    def driver_group(self):
        return self._driver_group

    @driver_group.setter
    def driver_group(self, value):
        self._driver_group = value

    @cached_property
    def _setup_devices(self) -> "torch.device":
        if self.cuda:
            n_devs = torch.cuda.device_count()
            if n_devs > 0:
                dev_id = self.rank % n_devs
                self._device = torch.device(f"cuda:{dev_id}")
            else:
                self.cuda = 0
                self._device = torch.device("cpu")
        else:
            self._device = torch.device("cpu")
        return self._device

    # Overriding property `world_size` in TrainingArguments
    # Here it means number of pipelines
    @property
    def world_size(self):
        return self.dp_group_size

    # Overriding property `process_index` in TrainingArguments
    # Here it means the index of current pipeline driver in all pipeline drivers
    @property
    def process_index(self):
        return self.driver_index

    # Overriding property `local_process_index` in TrainingArguments
    # Here it means the index of current pipeline driver in local pipeline drivers
    @property
    def local_process_index(self):
        return self.local_driver_index

    def __post_init__(self):
        super().__post_init__()
        self.local_rank = (
            -1
        )  # must be -1 to disable automatic DDP in the HF trainer

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        if is_torch_available() and self.world_size > 1:
            main_process_desc = "main process"
            if local:
                is_main_process = self.local_process_index == 0
                main_process_desc = "main local process"
            # elif is_sagemaker_mp_enabled():
            #     is_main_process = smp.rank() == 0
            else:
                is_main_process = self.process_index == 0

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(
                        f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}"
                    )
                    # if is_torch_tpu_available():
                    #     xm.rendezvous(desc)
                    # elif is_sagemaker_dp_enabled():
                    #     dist.barrier()
                    # else:
                    torch.distributed.barrier(group=self.driver_group)
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    # logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    # if is_torch_tpu_available():
                    #     pass  # TODO xm.rendezvous(desc)
                    # elif is_sagemaker_dp_enabled():
                    #     pass  # TODO dist.barrier()
                    # else:
                    torch.distributed.barrier(group=self.driver_group)
        else:
            yield


@dataclass
class PiPPySeq2SeqTrainingArguments(
    PiPPyTrainingArguments, Seq2SeqTrainingArguments
):
    pass


def _backward(
    self, gradient=None, retain_graph=None, create_graph=False, inputs=None
):
    # No-op backward for pipe mode, because otherwise HF Trainer will call loss.backward second time and will crash
    pass


class PiPPyTrainer(Trainer):
    def create_optimizer(self):
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = self.model.instantiate_optimizer(  # type: ignore[operator]
            optimizer_cls, **optimizer_kwargs
        )
        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        self.lr_scheduler = self.model.instantiate_lr_scheduler(  # type: ignore[operator]
            transformers.optimization.TYPE_TO_SCHEDULER_FUNCTION[
                self.args.lr_scheduler_type
            ],
            num_warmup_steps=self.args.get_warmup_steps(self.args.max_steps),
            num_training_steps=self.args.max_steps,
        )
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            loss, outputs = Trainer.compute_loss(
                self, model, inputs, return_outputs
            )
            loss.backward = types.MethodType(_backward, loss)
            return loss, outputs
        else:
            loss = Trainer.compute_loss(self, model, inputs, return_outputs)
            loss.backward = types.MethodType(_backward, loss)
            return loss


class PiPPySeq2SeqTrainer(PiPPyTrainer, Seq2SeqTrainer):
    pass


def torch_ones_wrapper(*args, **kwargs):
    return torch.ones(*args, **kwargs)


def torch_arange_wrapper(*args, **kwargs):
    return torch.arange(*args, **kwargs)


def torch_full_like_wrapper(*args, **kwargs):
    return torch.full_like(*args, **kwargs)


def torch_create_extended_attention_mask_for_decoder_wrapper(*args, **kwargs):
    return ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
        *args, **kwargs
    )


def torch_zeros_wrapper(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


class PiPPyHFTracer(fx.HFTracer):
    def trace(self, *args, **kwargs):
        graph = super().trace(*args, **kwargs)
        for node in graph.nodes:
            if node.op == "call_function":
                if getattr(node.target, "_orig", None) == torch.ones:
                    node.target = torch_ones_wrapper
                elif getattr(node.target, "_orig", None) == torch.arange:
                    node.target = torch_arange_wrapper
                elif getattr(node.target, "_orig", None) == torch.full_like:
                    node.target = torch_full_like_wrapper
                elif (
                    getattr(node.target, "_orig", None)
                    == ModuleUtilsMixin.create_extended_attention_mask_for_decoder
                ):
                    node.target = (
                        torch_create_extended_attention_mask_for_decoder_wrapper
                    )
                elif getattr(node.target, "_orig", None) == torch.zeros:
                    node.target = torch_zeros_wrapper
        return graph


# The `DotDict` class adds dot notation access to dictionary attributes.
class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, item):
        self.__delitem__(item)


# This is an experimental utility function that replaces the original model's forward method with PiPPy's PipelineDriver
# forward method.  It is used to support HuggingFace's `generate()` method, which is defined in a `GenerationMixin`
# class that `PreTrainedModel` inherits from.  We choose this replacement path instead of writing our own `generate()`
# method because the `generate()` method would call into many `GenerationMixin` APIs that may be implemented differently
# by each model.
def inject_pipeline_forward(
    model: torch.nn.Module,
    pipe_driver: PipelineDriverBase,
):
    logger.info(
        f"Inserting PiPPy pipeline forward into model {model._get_name()}"
    )
    # Inject pipeline driver as a member object of original model
    setattr(model, "pippy_pipeline_driver", pipe_driver)

    # Define a new forward method that uses PiPPy's pipeline driver
    def pippy_forward(self, *args, **kwargs):
        output = self.pippy_pipeline_driver(*args, **kwargs)
        if isinstance(output, dict):
            # Add dot access if output is a dictionary. The output of a traced HF model is a traditional dict which has
            # only [`key`] access.  The wrapping is needed for Transformer versons >= 4.28 which access attributes of
            # output via dot notation, such as `output.logits`.  See for example the `generate()` method and
            # `modeling_output.py`.
            output = DotDict(output)
        return output

    # Replace the forward method in original model
    setattr(model, "forward", types.MethodType(pippy_forward, model))
