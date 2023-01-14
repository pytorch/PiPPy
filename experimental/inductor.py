import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.aot_autograd import run_functionalized_fw_and_collect_metadata, aot_function
import functorch
from functorch.compile import min_cut_rematerialization_partition

from torch._functorch.eager_transforms import functionalize

from torch import autograd
from torch._inductor.compile_fx import compile_fx_inner


from torch._inductor.decomposition import select_decomp_table
inductor_decomps = select_decomp_table()



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


DEBUG = True

def print_graph(name, graph):
    if DEBUG:
        print(f'===== {name} ====')
        graph.print_readable()
        print(f'===== end {name} ====')


def print_value(name, value):
    if DEBUG:
        print(name, value)


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
        fn(*args, **kwargs)

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

def make_inductor_fn(fn, example_inputs, shard_map=None, sharded_example_inputs=None):
    """
    Use inductor to compile the function fn. 
    Uses make_functional_fx to trace it passing example_inputs.
    """
    fnfx, attr_vals, unused_input_idx = make_functional_fx(fn)(*example_inputs)

    used_args = prepare_args(attr_vals, unused_input_idx, example_inputs)

    if shard_map != None:
        schemas = [shard_map.get(inp, shard_map[DEFAULT_SCHEMA]) for inp in used_args]

        # pass the sharded inputs to spmd expander and to inductor afterwards
        used_args = prepare_args(attr_vals, unused_input_idx, sharded_example_inputs)

        from spmd.compiler.distribute import _convert_to_distributed, TrainingPhase
        fnfx, output_schemas = _convert_to_distributed(TrainingPhase.FORWARD, fnfx, used_args, schemas, _allow_partial=True)

        print_graph('after_spmd_expansion', fnfx)


    compiled_f = compile_fx_inner(fnfx, used_args)

    def inductor_fn(*args):
        used_args = prepare_args(attr_vals, unused_input_idx, args)
        return compiled_f(used_args)

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


def main_simple(dist=False):
    x = torch.randn(10, 2, device='cuda')
    p = torch.randn(2, 1, requires_grad=True, device='cuda')

    print('before copmile: ', (p*p).sum())
    if dist:
        from torch.distributed._tensor import (
            Replicate,
            Shard,
            DeviceMesh,
            DTensor,
        )

        # register rule for this op
        from torch.distributed._tensor.ops.tensor_ops import default_prop_rule
        DTensor._op_to_rules['aten.lift_fresh_copy.default'] = default_prop_rule

        from spmd.compiler.distribute import Schema
        mesh = DeviceMesh('cuda', range(torch.distributed.get_world_size()))
        shard_map = {
            x: Schema(mesh, [Shard(0)]),
            p: Schema(mesh, [Replicate()]),
            # default (TODO figure this out)
            # currently this is needed for when functionalization adds constants
            # so assuming its safe to make them replicated.
            DEFAULT_SCHEMA: Schema(mesh, [Replicate()]),
        }
        # we run the SPMD expanded with a sharded version of the input tensor
        x_shard = torch.randn(10 // torch.distributed.get_world_size(), 2, device='cuda')
    else:
        shard_map = None
    compiled_f = make_inductor_fn(inplace_test, [x, p], shard_map=shard_map, sharded_example_inputs=[x_shard, p])
    print('after compile: ', (p*p).sum())

    with torch.no_grad():
        for _ in range(10):
            loss, = compiled_f(x_shard, p)
            print('after iteration: ', (p*p).sum(), loss)



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
    o.zero_grad()
    y_hat = m.forward(x)
    loss = torch.sum((y_hat - y) ** 2)
    loss.backward()
    o.step()
    return (loss, )


def main(optim_cls):
    x = torch.ones(10, 4, device='cuda')
    y = torch.ones(10, 2, device='cuda')
    m = MyModule().to('cuda')
    sgd = optim_cls(m.parameters(), lr=0.01)

    from copy import deepcopy
    mc = deepcopy(m)

    print ('==== Eager ====')
    # eager
    for i in range(10): 
        loss = train_step(m, sgd, x, y)
        print(loss)

    print('==== Compiled ====')

    sgd = optim_cls(mc.parameters(), lr=0.01)

    # warning: even though we'ree passing mc and sgd as "example" inputs,
    #    the tracer actually takes its tensor states and saves them as part of
    #    the compiled_f. So mc and sgd cannot be mutated directly afterwards.
    compiled_f = make_inductor_fn(train_step, [mc, sgd, x, y])

    with torch.no_grad():
        for i in range(10): 
            # FIXME attention -- mc and sgd are actualy _not_ being used!!
            #    this is because we collected parameter/tensors from mc and sgd
            #    during the first tracing, and we're reusing those here instead of
            #    actually taking them from mc and sgd.
            loss,  = compiled_f(mc, sgd, x, y)
            print(loss)


DIST = True
if __name__ == '__main__':
    if DIST:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(torch.distributed.get_rank() % 8)
        main_simple(dist=True)
    else:
        main_simple(dist=False)
        # from functools import partial
        # note we need capturable=True otherwise Adam implementation uses a scalar training step
        # which doesn't work with make_fx
        # main(partial(torch.optim.Adam, capturable=True))