import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from . import inductor

DIST = True
inductor.DIST = DIST


# BUG: Returning mutated input generate 2 issues:
# 1) functionalization doesn't properly return the same initial buffer but rather the new one its copied into
# 2) inductor doesn't emit the buffer materialization for that input.

# SURPRISING BEHAVIOR: Creating DeviceMesh in FakeMode fails because mesh and rank_coordinatees are stored as tensors

if DIST:
    from torch.distributed._tensor import (
        Replicate,
        Shard,
        DeviceMesh,
    )
    from torch.distributed._tensor.placement_types import DTensorSpec
    from spmd.compiler.distribute import Schema

    from torch.distributed._tensor import DTensor, ops
    # the rule below has two issues:
    #  - likely doens't cover the case when a shape constant is passed
    #  - correctness issues since it will use different random seeds in different shards
    DTensor._op_to_rules['aten.uniform.default'] = ops.tensor_ops.prop_create_like

    DTensor._op_to_rules['aten.maximum.default'] = ops.common_rules.pointwise_rule
    DTensor._op_to_rules['aten.minimum.default'] = ops.common_rules.pointwise_rule


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def main():
    # need to create device mesh outside Fake mode. see above
    if DIST:
        mesh = DeviceMesh('cuda', range(torch.distributed.get_world_size()))
        default_shard_schema =  Schema(mesh, [Replicate()])
    else:
        default_shard_schema = None


    fake_mode = FakeTensorMode()

    # with fake_mode:
    # fake mode failing during SPMD expansion.
    if True:
        my_mod = MyModule().to('cuda')
        inductor.mark_placements(my_mod.linear.weight, [Shard(0)])
        inductor.mark_placements(my_mod.linear.bias, [Shard(0)])


        def init_model(my_mod, input):
            # nn.init.trunc_normal_(input, 0., 1., -2., 2.)
            nn.init.trunc_normal_(my_mod.linear.weight, 0., 1., -2., 2.)
            return (my_mod.linear.weight, )

        ffx = inductor.make_inductor_fn(
            init_model,
            [my_mod, my_mod.linear.weight],
            default_shard_schema=default_shard_schema,
        )

        # errors out because we need to materialize the sharding
        # TODO
        ffx(my_mod)




if __name__ == '__main__':
    if DIST:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(torch.distributed.get_rank() % 8)
 
    main()