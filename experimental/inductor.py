import torch
from torch import nn, Tensor
from torch.fx.experimental.proxy_tensor import (
    make_fx,
    maybe_disable_fake_tensor_mode,
)
from torch._functorch.eager_transforms import functionalize
from torch._inductor.compile_fx import compile_fx_inner
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from typing import Callable, Dict, Tuple, TypeVar, Union
from torch._inductor.decomposition import select_decomp_table
from functools import partial
from torch.utils._pytree import tree_flatten




inductor_decomps = select_decomp_table()

DIST = True
DEBUG = True

if DIST:
    from torch.distributed._tensor.ops.utils import register_prop_rule
    from torch.distributed._tensor import (
        Replicate,
        Shard,
        DeviceMesh,
    )
    from torch.distributed._tensor.dispatch import (
        operator_dispatch,
        OpSchema,
        OutputSharding,
    )
    from torch.distributed._tensor.placement_types import DTensorSpec
    from spmd.compiler.distribute import Schema

    from torch.distributed._tensor import DTensor, ops, distribute_tensor
    DTensor._op_to_rules['aten.lift_fresh_copy.default'] = ops.tensor_ops.default_prop_rule

    # the rule below may have issues issues:
    #  - likely doens't cover the case when a shape constant is passed
    DTensor._op_to_rules['aten.zero.default'] = ops.tensor_ops.prop_create_like

    @register_prop_rule("aten.zeros.default")
    def prop_aten_zeros_default(op_schema: OpSchema) -> OutputSharding:
        # rule for tensor creation operators that take no tensor as input.
        # we create the output tensor as replicated for now.
        assert op_schema.default_mesh is not None, (
            'Default mesh required for aten.zeros sharding propagation.')
        return OutputSharding(
            output_spec=DTensorSpec(
                mesh=op_schema.default_mesh,
                placements=(Replicate(), ) * op_schema.default_mesh.ndim,
                shape=op_schema.args_schema[0],
                ndim=len(op_schema.args_schema[0]),
            ),
        )

    # required sharded parameter initialization
    # we assume IID so it's fine to just run uniform_ on each shard
    # and keep the sharding as long as its not _Partial()
    # (TODO: whanchao -- ideally if we wanted reproducibility we'd need to do random seed distribution)
    DTensor._op_to_rules['aten.uniform_.default'] = ops.tensor_ops.prop_create_like


"""
* We are passing in:
1) objects with state (most likely with tensors that will be updated over the iterations)
2) list of tensors

* We want all tensors to be fake tensors when passed in.
  - this will lead to fake optimizer state as well.

* Next time user calls our functions
  - we will not take the state again
"""


from copy import copy
all_decomps = copy(inductor_decomps)

from torch._decomp import get_decompositions

all_decomps.update(get_decompositions([
    # used in Adam optimizer
    # TODO: add here https://github.com/pytorch/pytorch/blob/66b324cf06bce720887dabac710aee11b522450f/torch/_inductor/decomposition.py#L20
    torch.ops.aten.addcdiv.default,
]))


def prepare_args(attr_vals, unused_inputs_idx, args):
    """
    modify args to pass into a function compiled with make_functional_fx
    args: original arguments
    unused_inputs_idx, attr_vals: data returned from make_functional_fx
    """
    all_args = list(args) + list(attr_vals)
    return [a for i, a in enumerate(all_args) if i not in unused_inputs_idx]



def print_graph(name, graph):
    if DEBUG:
        print(f'===== {name} ====')
        graph.print_readable()
        print(f'===== end {name} ====')


def print_value(name, value):
    if DEBUG:
        print(name, value)


def dprint(msg):
    if DIST:
        print(f'rank {torch.distributed.get_rank()}: {msg}')
    else:
        print(msg)


def mark_placements(tensor, placements):
    tensor._placements = placements


def get_placements(tensor):
    if hasattr(tensor, '_placements'):
        return tensor._placements
    elif isinstance(tensor, DTensor):
        return tensor.placements
    else:
        return None


def get_example_shard(tensor, mesh, default=None, fn=torch.randn):
    if not isinstance(tensor, Tensor):
        return tensor
    elif isinstance(tensor, DTensor):
        return tensor.to_local()
    elif hasattr(tensor, '_placements'):
        inp_spec = DTensorSpec(mesh, tensor._placements, tensor.shape, tensor.ndim)
        return fn(inp_spec.local_shape, device=tensor.device)
    elif default is not None:
        inp_spec = DTensorSpec(mesh, default, tensor.shape, tensor.ndim)
        return fn(inp_spec.local_shape, device=tensor.device)
    raise NotImplementedError(f'Dont know how to get a shard of tensor {tensor}')


def convert_getattrs_to_inputs(gm):
    """
    Turn getattr into inputs
        - This is needed for functionalize to work. Functionalize expects
          all tensors to be passed as inputs to the function being functionalized.
        - We collect all concrete getattr inputs into "attr_vals"
    """
    from collections import defaultdict
    attr_vals_map = dict()
    attr_nodes = defaultdict(lambda: [])

    for p in gm.graph.nodes:
        if p.op == 'get_attr':
            val = getattr(gm, p.target)
            attr_vals_map[p.target] = val
            attr_nodes[p.target].append(p)

    after_last_placeholder = None
    for i, p in enumerate(gm.graph.nodes):
        if p.op != 'placeholder':
            after_last_placeholder = p
            break

    attr_vals = []
    with gm.graph.inserting_before(after_last_placeholder):
        for attr, nodes in attr_nodes.items():
            placeholder = gm.graph.placeholder(attr)
            attr_vals.append(attr_vals_map[attr])
            for node in nodes:
                node.replace_all_uses_with(placeholder)
                gm.graph.erase_node(node)

    return attr_vals


def make_functional_fx(fn):
    """
    Given a function, does make_fx(functionalize(make_fx(fn))).
    Some input-related transformations need to be done before functionalize.
    Returns an GraphModule to be fed to inductor.
    """
    def call(*args, **kwargs):
        # 0. hack: run one iteration to initialize optimizer states
        for inp in args:
            if isinstance(inp, torch.optim.Optimizer):
                warmup_optimizer(inp)

        # 1. call make_fx for the first time
        k1 = make_fx(fn)
        k2 = k1(*args, **kwargs)
        print_graph('first_fx', k2)

        attr_vals = convert_getattrs_to_inputs(k2)
        print_value('attr_vals', attr_vals)

        # 3. Remove unused inputs
        #    - This is needed in order to remove non-tensor inputs before passing this
        #      graph to inductor. Inductor expects only tensors as inputs
        #    - This works since all the non-tensor inputs should be unused at this point,
        #      since in principle we should have converted all getattrs into inputs.
        #    - We keep the indices of unused inputs in "unused_inputs_idx"
        unused_inputs = []
        unused_inputs_idx = []

        for i, p in enumerate(k2.graph.nodes):
            if p.op != 'placeholder':
                break
            if len(p.users) == 0:
                unused_inputs.append(p)
                unused_inputs_idx.append(i)

        for p in unused_inputs:
            k2.graph.erase_node(p)

        k2.recompile()
        print_graph('before_functionalize', k2)
 
        # 4. call make_fx a second time with functionalization
        with torch.no_grad():
            used_args_and_attrs = prepare_args(attr_vals, unused_inputs_idx, args)
            k3 = make_fx(functionalize(k2), decomposition_table=all_decomps)
            k4 = k3(*used_args_and_attrs, **kwargs)

        # 5. remove useless "alias" nodes
        for p in k4.graph.nodes:
            if p.target == torch.ops.aten.alias.default:
                p.replace_all_uses_with(p.args[0])

        # 6. functionalization adds getattrs sometimes, so move them to inputs again
        #    (hoping they'll be deterministic constants...)
        attr_vals += convert_getattrs_to_inputs(k4)
        print_value('attr_vals_after_functionalize', attr_vals)

        print_graph('after functionalize before DCE', k4)

        # 7. eliminate dead code but make sure we're still keeping the final copy_
        torch.fx.node._side_effectful_functions.add(torch.ops.aten.copy_.default)
        k4.graph.eliminate_dead_code()
        k4.graph.lint()
        k4.recompile()

        print_graph('after functionalize after DCE', k4)

        return k4, attr_vals, unused_inputs_idx

    return call

DEFAULT_SCHEMA = 'default' 


def track_state_dict_materialization(map: Dict[FakeTensor, Tensor], fake_state_dict, real_state_dict):
    for k, v in fake_state_dict.items():
        real_k = map[k] if isinstance(k, Tensor) else k
        if isinstance(v, Tensor):
            map[v] = real_state_dict[real_k]
        elif isinstance(v, dict):
            track_state_dict_materialization(map, v, real_state_dict[real_k])


def warmup_optimizer(optim):
    # unfortunatelly, in order to initialize the optimizer states
    # we need to call optimizer.step(). In order to do that, we will initialize
    # parameter gradients with zero tensors just so we can run a step.
    for group in optim.param_groups:
        for param in group['params']:
            param.grad = torch.zeros_like(param)
    # run an optimizer step just to initialize the optimizer states
    # multiple issues here:
    # 1. this will make the optimizer think we're at the 2nd iteration
    # 2. doesn't support optimizers that require callables passed in
    optim.step()
    # now we can delete those zero gradients
    optim.zero_grad(set_to_none=True)


def materialize_optimizer(optim, tracking_map):
    # reset this optimizer state so that we can properly materialize
    # the optimizer state next.
    from collections import defaultdict
    saved_fake_state = optim.state
    optim.state = defaultdict(dict)

    # replace parameters in the param groups with the materialized
    # parameters taken from the tracking materializer
    # this will throw if any of the parameters is not found in one of the models
    for group in optim.param_groups:
        group['params'] = [tracking_map[param] for param in group['params']]

    # this warms up the optimizer with real state dict now
    warmup_optimizer(optim)

    # we will then map the fake optimizer states to the real ones
    track_state_dict_materialization(
        tracking_map,
        saved_fake_state,
        optim.state,
    )


def _materialize_tensor(t: torch.Tensor, param=None):
    if True: # if isinstance(t, FakeTensor):
        return torch.zeros(t.shape, device=t.device)

	
T = TypeVar('T')


def _materialize_dtensor(mesh, default, tensor, param=None):
    if isinstance(tensor, DTensor):
        return tensor
    else:
        # do we have a way to make this into a DTensor?
        placements = get_placements(tensor)
        # the placement tag could be either in the param.data or in param
        # we are prioritizing param.data first, and then falling back to param
        if placements is None and param is not None:
            placements = get_placements(param)

        # we then fallback to default if all else fails.
        # TODO we should probably just error out instead of falling back to default.
        if placements is None:
            placements = default

        # we don't have a way to make into DTensor, return either
        # the original tensor if it's a materialized one, or a new example with
        # zeros.
        if placements is None:
            return (
                torch.zeros(tensor.shape, device=tensor.device)
                if isinstance(tensor, FakeTensor) else tensor
            )

        if isinstance(tensor, FakeTensor):
            shard = get_example_shard(tensor, mesh, placements, fn=torch.zeros)
            return DTensor.from_local(shard, device_mesh=mesh, placements=placements)
        else:
            # assume the input is a full real tensor, we distribute it.
            return distribute_tensor(tensor, device_mesh=mesh, placements=placements)


def materialize_module(
    mod: nn.Module,
    tensor_materializer: Callable[[FakeTensor], torch.Tensor] = _materialize_tensor,
) -> Tuple[nn.Module, dict[torch.Tensor, torch.Tensor]]:
    def to_real_tensor(m: nn.Module, key: str, t: T) -> Union[T, torch.Tensor, nn.Parameter]:
        if isinstance(t, nn.Parameter):
            return nn.Parameter(tensor_materializer(t.data, param=t))
        else:
            assert isinstance(t, torch.Tensor)
            return tensor_materializer(t)

    def materialize_this_module(m: nn.Module) -> None:
        for key, param in m._parameters.items():
            assert isinstance(param, nn.Parameter)
            if True:  # if isinstance(param.data, FakeTensor):
                dparam = to_real_tensor(m, key, param)
                assert isinstance(dparam, nn.Parameter)
                m.register_parameter(key, dparam)
        for key, buffer in m._buffers.items():
            if True:  # if isinstance(buffer, FakeTensor):
                m._buffers[key] = to_real_tensor(m, key, buffer)

    materialize_this_module(mod)
    for name, submod in mod.named_modules():
        materialize_this_module(submod)

    return mod


def get_example_shard(tensor, mesh, default=None, fn=torch.randn):
    """
    Return a "real" and "local" tensor.
    """
    if not isinstance(tensor, Tensor):
        return tensor
    elif isinstance(tensor, DTensor):
        return tensor.to_local()
    elif hasattr(tensor, '_placements'):
        inp_spec = DTensorSpec(mesh, tensor._placements, tensor.shape, tensor.ndim)
        return fn(inp_spec.local_shape, device=tensor.device)
    elif default is not None:
        inp_spec = DTensorSpec(mesh, default, tensor.shape, tensor.ndim)
        return fn(inp_spec.local_shape, device=tensor.device)
    raise NotImplementedError(f'Dont know how to get a shard of tensor {tensor}')


from torch.utils._python_dispatch import TorchDispatchMode


class DTensorMode(TorchDispatchMode):
    """
    This is used for when new tensors get created using a no-input-tensor constructor
    such as torch.zeros, and we want the output to be a DTensor.

    Note that for ops that take regular tensors (but no DTensors) as inputs, we are 
    still falling back to the non DTensor dispatch.
    """
    def __init__(self, default_mesh):
        self.default_mesh = default_mesh

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        # unfortunatelly we have these non-standard aten calls that we'll have to ignore
        # they happen in torch.optim.Optimizer base-class.
        non_tensor_ops = (
            torch.ops.profiler._record_function_enter_new.default,
            torch.ops.profiler._record_function_exit._RecordFunction,
        )

        # if we do have tensor inputs that are non-DTensor then we should fallback as well
        # as this is a regular non-dtensor op.
        # this is needed to handle stuff like print(my_tensor)
        arg_list, _ = tree_flatten(args)
        has_regular_tensors = False
        for arg in arg_list:
            if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                has_regular_tensors = True

        if func in non_tensor_ops or has_regular_tensors:
            return func(*args, **kwargs)

        if kwargs is None:
            kwargs = {}

        return operator_dispatch(
            func,
            args,
            kwargs,
            DTensor._op_to_rules,
            DTensor._custom_dispatch_ops,
            default_mesh=self.default_mesh,
        )


def make_inductor_fn(fn, example_inputs, default_shard_schema=None):
    """
    Use inductor to compile the function fn. 
    Uses make_functional_fx to trace it passing example_inputs.
    """
    fnfx, attr_vals, unused_input_idx = make_functional_fx(fn)(*example_inputs)

    used_args = prepare_args(attr_vals, unused_input_idx, example_inputs)

    # materialize models + SPMD expansion. must be outside of fake mode
    with maybe_disable_fake_tensor_mode():
        if default_shard_schema is None:
            tensor_materializer = _materialize_tensor
        else:
            tensor_materializer = partial(
                _materialize_dtensor,
                default_shard_schema.mesh,
                default_shard_schema.placements,
            )
        # In the following lines we perform utterly black magic in order to
        # materialize our models. It special cases models and optimizers
        # and assumes that all parameters contained in optimziers are model
        # parameters.
        tracking_materializer = TrackingMaterializer(tensor_materializer)
        # 1. materialize models
        for inp in example_inputs:
            if isinstance(inp, nn.Module):
                materialize_module(inp, tensor_materializer=tracking_materializer)
        # 2. "materialize" optimizers
        for inp in example_inputs:
            if isinstance(inp, torch.optim.Optimizer):
                if default_shard_schema is not None:
                    # we need DTensorMode because our optimizer used here (Adam)
                    # instantiates a few tensors using torch.zeros() which take no
                    # tensors in the input. We need to ensure the tensor created is a DTensor.
                    with DTensorMode(default_shard_schema.mesh):
                        materialize_optimizer(inp, tracking_materializer.map)
                else:
                    materialize_optimizer(inp, tracking_materializer.map)


        # (re-)create attr_vals with concrete inputs
        real_attr_vals = [
            tracking_materializer.map[val] if isinstance(val, FakeTensor) else val
            for val in attr_vals
        ]

        # from now on, we use the real attr_vals as inputs.
        # in particular this is important because in the previous step we warmed up
        # optimizers -- this means that the optimizer states should have the same sharding
        # as their respective parameters.
        used_args = prepare_args(real_attr_vals, unused_input_idx, example_inputs)
 
        def get_schema(inp):
            placements = get_placements(inp)
            if placements is not None:
                return Schema(default_shard_schema.mesh, placements)
            else:
                return default_shard_schema

        if default_shard_schema != None:
            schemas = [get_schema(inp) for inp in used_args]

            # from now on we'll pass local shards as example inputs
            used_args = [
                get_example_shard(
                    arg,
                    schema.mesh,
                    schema.placements
                ) for arg, schema in zip(used_args, schemas)
            ]

            from spmd.compiler.distribute import _convert_to_distributed, TrainingPhase
            fnfx, _, output_schemas = _convert_to_distributed(TrainingPhase.FORWARD, fnfx, used_args, schemas, _allow_partial=True)

            print_graph('after_spmd_expansion', fnfx)
        else:
            output_schemas = None

        compiled_f = compile_fx_inner(fnfx, used_args)

        # temporary hack to reset state since we're actually updating the model and our eager collective implementation is incorrect
        # this shouldn't be needed once either convert_to_distributed accepts fake mode or functional collectives work in eager mode.
        # also this is strictly incorrect in multiple levels, because:
        #  1. parametres should be properly initialized, not initialized to zero.
        #  2. optimizer states may or may not bei nitialized with zero, it should be up to the optimizer to decide. 
        # 
        # (NOTE: the immediate problem i'm solving here is that when sharding parameters we need to do an all_gather in the forward
        #        step to gather the parametres ,but collectives are not working in eager mode. Not sure why I'm getting "nan"s for the
        #        loss, however.
        # for t in real_attr_vals:
        #     t.zero_()

    def inductor_fn(*args):
        with torch.no_grad():
            used_args = prepare_args(real_attr_vals, unused_input_idx, args)

            # if args are DTensors, get a local shard. But we also need to check
            # that their placements match the intended placement.
            for i in range(len(used_args)):
                arg = used_args[i]
                if isinstance(arg, DTensor):
                    assert arg.device_mesh == schemas[i].mesh, (
                        f'Input device meshes passed to compiled function '
                        f'do not match: got {arg.device_mesh}, expected {schemas[i].mesh}'
                    )
                    # TODO(wanchao) we should normalize whether we're returning tuples or lists for placements
                    assert tuple(arg.placements) == tuple(schemas[i].placements), (
                        f'Input placements passed to compiled function '
                        f'do not match: got {arg.placements}, expected {schemas[i].placements}'
                    )
                    used_args[i] = arg.to_local()

            outputs = compiled_f(used_args)

            return outputs if output_schemas is None else [
                output if schema is None else DTensor.from_local(
                    local_tensor=output,
                    device_mesh=schema.mesh,
                    placements=schema.placements,
                    run_check=False,
                )
                for output, schema in zip(outputs, output_schemas)
            ]


    return inductor_fn


# =====================================
# 1. Minimal SGD example, no nn.Module
# =====================================

def inplace_test(x, p):
    p.grad = None  # zero_grad
    loss = (torch.sigmoid(torch.matmul(x, p)) ** 2).sum()
    # pg, = torch.autograd.grad(loss, p)
    loss.backward()
    with torch.no_grad():
        p -= 0.1 * p.grad
    return (loss, )


def main_simple():
    fake_mode = FakeTensorMode()
    torch.set_default_device('cuda')

    if DIST:
        mesh = DeviceMesh('cuda', range(torch.distributed.get_world_size()))
        default_shard_schema = Schema(mesh, [Replicate()])

    # with fake_mode:
    with fake_mode:
        x_fake = torch.randn(10, 2, device='cuda')
        p_fake = torch.randn(2, 1, requires_grad=True, device='cuda')

        if DIST:
            mark_placements(x_fake, [Shard(0)])
            mark_placements(p_fake, [Replicate()])

            # x_shard = get_example_shard(x, mesh)
            # dprint(f'Input shape: {x.shape}; shard shape: {x_shard.shape}')
        else:
            # x_shard = x
            default_shard_schema = None

        compiled_f = make_inductor_fn(
            inplace_test,
            [x_fake, p_fake],
            default_shard_schema=default_shard_schema,
        )

    x = torch.randn(10, 2, device='cuda')
    p = torch.randn(2, 1, requires_grad=True, device='cuda')

    if DIST:
        x = distribute_tensor(x, mesh, [Shard(0)])
        p = distribute_tensor(p, mesh, [Replicate()])

    if DIST:
        # make sure our parameter is the same across devices initially
        # doing this after compile because during compilation we're running the training step a few times
        # TODO: fix this. we should run compilation with fake tensors.
        # torch.distributed.all_reduce(p)
        pass

    dprint(f'after compile, param norm = {(p*p).sum()}')

    with torch.no_grad():
        for _ in range(10):
            loss, = compiled_f(x, p)
            dprint(f'iteration: param_norm={(p*p).sum()}, local_loss={loss}')


# =====================================
# 2. Simple example with nn.Module and Optim
# =====================================

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_step(m, o, x, y):
    # if we dont set to none, the gradients will be copied into the buffers at the end
    # that's alright but probably a perf regression
    o.zero_grad(set_to_none=True)  

    y_hat = m.forward(x)
    loss = torch.sum((y_hat - y) ** 2)
    loss.backward()
    o.step()
    return (loss, )


def main(optim_cls):
    torch.set_default_device('cuda')

    x = torch.randn(10, 4, device='cuda')
    y = torch.randn(10, 2, device='cuda')
    m = MyModule().to('cuda')
    sgd = optim_cls(m.parameters(), lr=0.01)

    # from copy import deepcopy
    # mc = deepcopy(m)

    print ('==== Eager ====')
    # eager
    for i in range(10): 
        loss = train_step(m, sgd, x, y)
        print(loss)

    if DIST:
        mesh = DeviceMesh('cuda', range(torch.distributed.get_world_size()))
        default_shard_schema = Schema(mesh, [Replicate()])
    else:
        default_shard_schema = None

    with FakeTensorMode():
        mc = MyModule()
        sgd = optim_cls(mc.parameters(), lr=0.01)

        fake_x = torch.ones(10, 4)
        fake_y = torch.ones(10, 2)

        if DIST:
            mark_placements(fake_x, [Shard(0)])
            mark_placements(fake_y, [Shard(0)])

            # we are sharding the weight of our linear layer
            mark_placements(mc.linear.weight, [Shard(0)])
            # (but we are replicating the bias)
            # mark_placements(mc.linear.bias, [Replicate()])

        # warning: even though we'ree passing mc and sgd as "example" inputs,
        #    the tracer actually takes its tensor states and saves them as part of
        #    the compiled_f. So mc and sgd cannot be mutated directly afterwards.
        compiled_f = make_inductor_fn(train_step, [mc, sgd, fake_x, fake_y], default_shard_schema=default_shard_schema)

    # now that we compiled our function all of our parameters are already DTensors, but since we were in fake mode 
    # when we first initialized it, we don't have real values for these parametesr (they are all initialized as zero)
    # 
    # We'll need to figure out how to do standard lazy-initialization.
    # We can use any inplace initialization functions (pretty much all of torch.nn.init)
    # as well as reset_parameters() which is present in a bunch of nn.modules.
    # 
    # Note that this needs inplace random functions to be properly exposed as DTensor rules.
    mc.linear.reset_parameters()

    if DIST:
        # make sure our parameter is the same across devices initially
        # doing this after compile because during compilation we're running the training step a few times
        # TODO: fix this. we should run compilation with fake tensors.
        # for p in mc.parameters(): 
        #    torch.distributed.all_reduce(p)
        x = distribute_tensor(x, mesh, placements=[Shard(0)])
        y = distribute_tensor(y, mesh, placements=[Shard(0)])

    with torch.no_grad():
        for i in range(10): 
            # FIXME attention -- mc and sgd are actualy _not_ being used!!
            #    this is because we collected parameter/tensors from mc and sgd
            #    during the first tracing, and we're reusing those here instead of
            #    actually taking them from mc and sgd.
            loss_shard,  = compiled_f(mc, sgd, x, y)
            dprint(f'iteration: local_loss={loss_shard}')



class TrackingMaterializer:
    def __init__(self, materializer):
        self.map: Dict[torch.Tensor, torch.Tensor] = dict()
        self.materializer = materializer

    def __call__(self, tensor, param=None):
        if tensor in self.map:
            return self.map[tensor]
        elif param is not None and param in self.map:
            return self.map[param]
        else:
            materialized = self.materializer(tensor, param=param)
            self.map[tensor] = materialized
            self.map[param] = materialized
            return materialized


if __name__ == '__main__':
    # Need to allocate at least one real cuda tensor to avoid bug:
    # https://github.com/pytorch/pytorch/issues/92627
    torch.zeros(1, device='cuda')

    if DIST:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(torch.distributed.get_rank() % 8)

    # print('================ main_simple =================')
    # main_simple()

    # print('================ main with SGD =================')
    # main(torch.optim.SGD)

    print('================ main with Adam =================')
    # NOTE: we get multiple all_reduce for each parameter but this is because
    #       the parameter gradient is used multiple times by adam.
    #       we need to run CSE for this
    # fused needs to be false because the fused ops don't support "meta" device.
    from functools import partial
    main(partial(torch.optim.Adam, capturable=True, fused=False))