import logging
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.fsdp._fsdp_extensions import (
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
)


@dataclass
class FairScaleFSDPManagedParam:
    """
    Information about an original parameter managed by (fairscale) FSDP.
    Attributes:
        flat_param_key: FQN of the flat_param in FSDP this original param belongs to. This is the key in the fairscale state_dict.
        fqn: full original param FQN, with no FSDP prefixing, starting from root module.
        full_shape: full, unsharded parameter shape
        local_numels: numels value from the fairscale state_dict, unused for now
        data_tensor: Union[ShardedTensor, DTensor] - data tensor sharded in the PT-D style
    """

    flat_param_key: str  # this is the key in the fairscale state_dict
    fqn: str  # full FQN starting from root module, with no FSDP prefixing
    full_shape: torch.Size  # full, unsharded shape (original parameter shape)
    local_numels: int  # numels value from the fairscale state_dict
    data_tensor: torch.Tensor  # actual data tensor


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _verify_fqn_across_ranks(fqn, grp_gloo):
    olist = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(olist, fqn, group=grp_gloo)
    assert len(set(olist)) == 1
    assert olist[0] == fqn


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _all_gather_into_list(data_tensor, model_parallel_group):
    tensor_list = [
        torch.zeros_like(data_tensor).cuda()
        for _ in range(dist.get_world_size(model_parallel_group))
    ]
    dist.all_gather(tensor_list, data_tensor.cuda(), group=model_parallel_group)
    return tensor_list


def _is_tp_sharded(fqn: str) -> bool:
    """
    Returns whether a tensor given by the fqn is tensor parallel sharded.
    NOTE: this is currently done by inspection of the MF model and is quite
    brittle and would need to be updated if the MF sharding changes.
    """
    return (
        "attention" in fqn
        or "feed_forward" in fqn
        or "output" in fqn
        or "tok_embeddings" in fqn
    )


# pyre-fixme[3]: Return type must be annotated.
def _unshard_param(
    # pyre-fixme[2]: Parameter must be annotated.
    ref_state_dict,
    # pyre-fixme[2]: Parameter must be annotated.
    fqn,
    # pyre-fixme[2]: Parameter must be annotated.
    model_parallel_group,
    # pyre-fixme[2]: Parameter must be annotated.
    grp_gloo,
    # pyre-fixme[2]: Parameter must be annotated.
    data_tensor,
    # pyre-fixme[2]: Parameter must be annotated.
    tp_sharded_shape,
):
    """
    Unshards the row or col-wise sharded parameter.
    For rowwise, this is done by reshaping into the local shape, allgathering,
    and stacking rows. For colwise, the only difference is we stack columns.
    This is done via vstack and column_stack respectively.
    """
    mp_size = dist.get_world_size(model_parallel_group)
    ref_shape = ref_state_dict[fqn].shape
    assert (
        ref_shape[0] == tp_sharded_shape[0] or ref_shape[1] == tp_sharded_shape[1]
    ), f"Expected sharded shape to match either row or col-wise, but does not: {ref_shape} {tp_sharded_shape}"
    _verify_fqn_across_ranks(fqn, grp_gloo)
    if ref_shape[0] != tp_sharded_shape[0]:
        assert ref_shape[0] == tp_sharded_shape[0] * mp_size
        # reshape the flat data_tensor into the rowwise shape
        data_tensor = data_tensor.reshape(tp_sharded_shape)
        # now, all_gather such tensors
        tensor_list = _all_gather_into_list(data_tensor, model_parallel_group)
        # stack rowwise to produce the final unsharded tensor
        data_tensor = torch.vstack(tensor_list).cpu()
        assert data_tensor.shape == ref_shape
        full_shape = data_tensor.shape
    elif (
        len(ref_shape) > 1
        and len(tp_sharded_shape) > 1
        and ref_shape[1] != tp_sharded_shape[1]
    ):
        assert ref_shape[1] == mp_size * tp_sharded_shape[1]
        # first, reshape the flat data_tensor into the colwise shape
        data_tensor = data_tensor.reshape(tp_sharded_shape)
        tensor_list = _all_gather_into_list(data_tensor, model_parallel_group)
        data_tensor = torch.column_stack(tensor_list).cpu()
        assert data_tensor.shape == ref_shape, f"{data_tensor.shape} vs {ref_shape}"
        full_shape = data_tensor.shape
    else:
        assert ref_shape == tp_sharded_shape  # not tensor parallel sharded
        full_shape = tp_sharded_shape
        logging.warning(f"{fqn} {ref_shape} {full_shape} - not sharded")
    return data_tensor, full_shape


def _build_fsdp_managed_param_dict(
    # pyre-fixme[2]: Parameter must be annotated.
    model,
    # pyre-fixme[2]: Parameter must be annotated.
    fs_state_dict,
    # pyre-fixme[2]: Parameter must be annotated.
    use_dtensor,
) -> Dict[str, List[FairScaleFSDPManagedParam]]:
    grp_gloo = dist.new_group(backend="gloo")
    # TODO: this should be the FSDP device mesh
    mesh = (
        DeviceMesh(
            device_type="cuda",
            mesh=list(range(dist.get_world_size())),
        )
        if use_dtensor
        else None
    )
    model_parallel_group, _ = dist.new_subgroups()
    state_dict = fs_state_dict
    model_weights = state_dict["weights"]
    param_metadatas = state_dict["meta"]["param_metadata"]
    ref_state_dict = (
        model.state_dict()
    )  # TODO: only need a "meta" state dict w/ shape info
    managed_param_dict = {}
    for param_metadata in param_metadatas:
        # TODO - shared_param_info, no_broadcast_optim_state
        fsdp_managed_params = param_metadata["params"]
        prefix = param_metadata["fsdp_path"]
        for key in fsdp_managed_params:
            assert "flat_param" in key, f"Got key {key} for {param_metadata}"
            model_weights_key = f"{prefix}.{key}" if prefix != "" else key
            assert (
                model_weights_key in model_weights
            ), f"{model_weights_key} not in model state dict {model_weights.keys()}"
            orig_param_dict = fsdp_managed_params[key]
            # names, shape, numels
            # TODO - handle padding?
            names_info = orig_param_dict["names"]
            shape_info = orig_param_dict["shapes"]
            numels_info = orig_param_dict["numels"]
            offset = 0
            overall_tensor = model_weights[model_weights_key]
            assert len(overall_tensor.shape) == 1, f"Expected 1d sharded tensor"
            overall_numels = overall_tensor.shape[0]
            managed_params_for_this_flat_param = []
            for i, name in enumerate(names_info):
                fqn = f"{prefix}.{name}" if prefix != "" else name
                flat_param_key = model_weights_key
                tp_sharded_shape = shape_info[i]
                local_numels = numels_info[i]
                data_tensor = overall_tensor[offset : offset + local_numels]
                # RESHARDING PART
                # attention, feed_forward, and output linear layers are all model parallel sharded.
                # this is a hacky way of figuring things out, fix it
                if (
                    "attention" in fqn
                    or "feed_forward" in fqn
                    or "output" in fqn
                    or "tok_embeddings" in fqn
                ):
                    data_tensor, full_shape = _unshard_param(
                        ref_state_dict,
                        fqn,
                        model_parallel_group,
                        grp_gloo,
                        data_tensor,
                        tp_sharded_shape,
                    )
                else:
                    full_shape = tp_sharded_shape
                offset += local_numels
                assert offset <= overall_numels
                # make it a shardedtensor/dtensor
                # NOTE: this chunks the sharded tensor across the entire world, for 1D FSDP style parallelism. This
                # should change once we support 2D parallelism.
                if use_dtensor:
                    assert mesh is not None
                    data_tensor = _ext_chunk_dtensor(
                        tensor=data_tensor.contiguous(),
                        rank=dist.get_rank(),
                        device_mesh=mesh,
                    )
                else:
                    data_tensor = _ext_chunk_tensor(
                        tensor=data_tensor.contiguous(),
                        rank=dist.get_rank(),
                        world_size=dist.get_world_size(),
                        num_devices_per_node=torch.cuda.device_count(),  # TODO: this is not accurate if user set CUDA_VISIBLE_DEVICES
                        pg=dist.distributed_c10d._get_default_group(),  # TODO: this should be the FSDP process group
                    )
                managed_param = FairScaleFSDPManagedParam(
                    flat_param_key,
                    fqn,
                    full_shape,
                    local_numels,
                    data_tensor,
                )
                managed_params_for_this_flat_param.append(managed_param)
            assert (
                flat_param_key not in managed_param_dict  # pyre-ignore[61]
            ), f"Overwriting {flat_param_key}!"  # pyre-ignore[61]
            managed_param_dict[
                flat_param_key  # pyre-ignore[61]
            ] = managed_params_for_this_flat_param
        assert (
            offset == overall_numels  # pyre-ignore[61]
        ), f"{offset} vs {overall_numels} for {orig_param_dict} with tensor {data_tensor.shape} and key {model_weights_key}"  # pyre-ignore[61]
    return managed_param_dict


def build_distributed_state_dict_from_consolidated(
    model: nn.Module,
    consolidated_state_dict: Dict[str, Tensor],
    offload_to_cpu: bool = False,
    use_dtensor: bool = False,
    model_parallel_world_size: int = 8,
) -> Dict[str, Union[Tensor, DTensor, ShardedTensor]]:
    """
    Main API that takes a model (with no parallelism applied) and a fairscale checkpoint
    and builds a PT-D compliant distributed state dict. Note that this expects a consolidated
    checkpoint.

    Args:
        model (torch.nn.Module): module with no parallelism applied (i.e. result of `build_model` with parallel_impl=ParallelImpl.NONE)
        fs_state_dict (Dict[str, Any]): Fairscale consolidated
        offload_to_cpu (bool): Whether to offload the resulting state_dict to CPU (default: False)
        use_dtensor (bool): Whether to use PyTorch Distributed Tensor instead of ShardedTensor (default: False)
            (this will eventually default to True)
        model_parallel_world_size: Model parallel world size that was used to create the consolidated checkpoint.
            This can be obtained by checking the number of consolidated0x.pth files in the checkpoint directory.

    Example usage::
        ```
        
        MODEL_PARALLEL_SIZE = 8
        ckpt_path = get_consolidated_ckpt_path(
            ckpt_dir=PTH_65b, mp_rank=local_rank, mp_size=MODEL_PARALLEL_SIZE
        )
        state_dict = torch.load(ckpt_path)
        # Build a local LLaMA with no parallelism
        model = build_model(...)
        sharded_state_dict = build_distributed_state_dict_from_consolidated(
            model, state_dict, model_parallel_world_size=MODEL_PARALLEL_SIZE,
        )
        # Wrap model with PT-native APIs + load
        model = FSDP(model)
        FSDP.set_state_dict_type(StateDictType.SHARDED_STATE_DICT)
        model.load_state_dict(sharded_state_dict)
        ```

    Note: Please make sure to pass an unsharded model as the model arg! Otherwise, things will not
    work.

    This distributed state dict is a mapping of FQN: ShardedTensor/DTensor. It will be replaced with
    DTensor once DTensor 2D checkpoint format is fully rolled out.

    Note: This has only been tested for loading state_dict into PT-D FSDP sharded_state_dict for now.
    """
    torch._C._log_api_usage_once("build_distributed_state_dict")
    dist_state_dict = {}
    ref_state_dict = model.state_dict()
    grp_gloo = dist.new_group(backend="gloo")
    # TODO: this should be the FSDP device mesh
    mesh = (
        DeviceMesh(
            device_type="cuda",
            mesh=list(range(dist.get_world_size())),
        )
        if use_dtensor
        else None
    )
    input_dtypes = {v.dtype for v in consolidated_state_dict.values()}
    logging.warning(f"input_dtypes {input_dtypes}")
    model_parallel_group, _ = dist.new_subgroups(group_size=model_parallel_world_size)
    for fqn, tensor in consolidated_state_dict.items():
        # Hack for buffer
        if "rope.freqs" in fqn:
            dist_state_dict[fqn] = tensor.clone()
            continue
        if _is_tp_sharded(fqn):
            tensor, _ = _unshard_param(
                ref_state_dict,
                fqn,
                model_parallel_group,
                grp_gloo,
                tensor,
                tensor.shape,
            )
        if use_dtensor:
            assert mesh is not None
            tensor = _ext_chunk_dtensor(
                tensor=tensor.contiguous(),
                rank=dist.get_rank(),
                device_mesh=mesh,
            )
        else:
            tensor = _ext_chunk_tensor(
                tensor=tensor.contiguous(),
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                num_devices_per_node=torch.cuda.device_count(),  # TODO: this is not accurate if user set CUDA_VISIBLE_DEVICES
                pg=dist.distributed_c10d._get_default_group(),  # TODO: this should be the FSDP process group
            )
        dist_state_dict[fqn] = tensor

    dtypes = {v.dtype for v in dist_state_dict.values()}
    logging.warning(f"Made dist_state_dict with dtypes {dtypes}")
    return dist_state_dict


# pyre-fixme[3]: Return type must be annotated.
def build_distributed_state_dict_from_unconsolidated(
    # pyre-fixme[2]: Parameter must be annotated.
    model,
    # pyre-fixme[2]: Parameter must be annotated.
    fs_state_dict,
    offload_to_cpu: bool = True,
    use_dtensor: bool = False,
):
    """
    Main API that takes a model (with no parallelism applied) and a fairscale checkpoint
    and builds a PT-D compliant distributed state dict.

    Args:
        model (torch.nn.Module): module with no parallelism applied (i.e. result of `build_model` with parallel_impl=ParallelImpl.NONE)
        fs_state_dict (Dict[str, Any]): Fairscale checkpoint
        offload_to_cpu (bool): Whether to offload the resulting state_dict to CPU (default: True)
        use_dtensor (bool): Whether to use PyTorch Distributed Tensor instead of ShardedTensor (default: False)
            (this will eventually default to true)

    
    {
        'weights': <mapping of FSDP param to flat tensor>
        'metadata': <per parameter sharding info and FSDP prefix>
    }

    Note: Please make sure to pass an unsharded model as the model arg! Otherwise, things will not
    work.

    This distributed state dict is a mapping of FQN: ShardedTensor/DTensor. It will be replaced with
    DTensor once DTensor 2D checkpoint format is fully rolled out.

    Note: This has only been tested for loading state_dict into PT-D FSDP sharded_state_dict for now.
    TODO: The current implementation is hardcoded to assume a fairscale tensor parallel size of 8.
    """
    torch._C._log_api_usage_once("build_distributed_state_dict")
    managed_param_dict = _build_fsdp_managed_param_dict(
        model, fs_state_dict, use_dtensor
    )
    buffer_names = fs_state_dict["meta"]["buffer_names"]
    state_dict = {}
    for key in fs_state_dict["weights"]:
        if key not in managed_param_dict:
            assert key in buffer_names
            logging.warning(f"assigning buffer {key}")
            state_dict[key] = fs_state_dict["weights"][key]
    for param in managed_param_dict:
        assert (
            param in fs_state_dict["weights"]
        ), f"FSDP flat_param key {param} not found in original state_dict!"
        managed_params = managed_param_dict[param]
        for managed_param in managed_params:
            fqn = managed_param.fqn
            shaped_tensor = managed_param.data_tensor
            assert fqn not in state_dict, f"Unexpected Overwriting {fqn}"
            state_dict[fqn] = shaped_tensor
    # TODO: implement offload_to_cpu properly - offloading the sharded tensor after _ext_chunk_tensor.
    if offload_to_cpu:
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
    return state_dict
