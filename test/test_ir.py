import torch
import copy
import unittest
from typing import NamedTuple
from torch.utils._pytree import tree_flatten, tree_unflatten

from pippy.IR import Pipe, pipe_split, MultiUseParameterConfig, annotate_split_points, PipeSplitWrapper

class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(512, 512))
        self.mm_param2 = torch.nn.Parameter(torch.randn(512, 512))
        self.lin = torch.nn.Linear(512, 512)
        self.register_buffer('buffer', 0.001 * torch.randn(512))

    def forward(self, x):
        x = torch.mm(x, self.mm_param)
        skip_connection = x
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param) + self.buffer
        x = self.lin(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        x = self.lin(x)
        return x

def check_qualname_mapping(old, new):
    seen_old_qns = {}
    for _, old_qn in new.new_to_old_qualname_mapping.items():
        seen_old_qns.setdefault(old_qn)

    for param_name, _ in old.named_parameters():
        assert param_name in seen_old_qns, f'Expected parameter {param_name} in {seen_old_qns}'

class TestIR(unittest.TestCase):
    def setUp(self):
        mods = [torch.nn.Linear(512, 512) for _ in range(5)]
        mods += [mods[0]]
        self.seq = torch.nn.Sequential(*mods)
        self.ec = ExampleCode()

    def test_sequential(self):
        seq_pipe = Pipe.from_sequential(self.seq)
        assert seq_pipe.replicated_params == [{'submod_0': 'weight', 'submod_5': 'weight'}, {'submod_0': 'bias', 'submod_5': 'bias'}]

        x = torch.randn(50, 512)
        torch.testing.assert_allclose(self.seq(x), seq_pipe(x))

        check_qualname_mapping(old=self.seq, new=seq_pipe)

    def test_tracing_transmit(self):
        ec_pipe = Pipe.from_tracing(self.ec, MultiUseParameterConfig.TRANSMIT)
        x = torch.randn(5, 512)
        torch.testing.assert_allclose(self.ec(x), ec_pipe(x))
        assert ec_pipe.replicated_params == [
            {'submod_1': 'lin.weight', 'submod_2': 'lin.weight'}, {'submod_1': 'lin.bias', 'submod_2': 'lin.bias'}]
        check_qualname_mapping(old=self.ec, new=ec_pipe)

    def test_tracing_replicate(self):
        ec_pipe_replicated = Pipe.from_tracing(self.ec, MultiUseParameterConfig.REPLICATE)
        x = torch.randn(5, 512)
        torch.testing.assert_allclose(self.ec(x), ec_pipe_replicated(x))
        assert ec_pipe_replicated.replicated_params == [
            {'submod_0': 'moved_mm_param', 'submod_1': 'moved_mm_param'},
            {'submod_1': 'lin.weight', 'submod_2': 'lin.weight'},
            {'submod_1': 'lin.bias', 'submod_2': 'lin.bias'}]
        check_qualname_mapping(old=self.ec, new=ec_pipe_replicated)

    def test_tracing_shared_non_leaf_mod(self):
        class ShareMe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(5, 3))

            def forward(self, x):
                return x + self.param

        class PipeMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = ShareMe()

            def forward(self, x):
                x = self.shared(x)
                pipe_split()
                x = self.shared(x)
                pipe_split()
                return torch.relu(x)

        pm = PipeMod()
        pm_pipe = Pipe.from_tracing(pm, MultiUseParameterConfig.TRANSMIT)
        x = torch.randn(5, 3)
        torch.testing.assert_allclose(pm_pipe(x), pm(x))
        assert pm_pipe.replicated_params == []
        check_qualname_mapping(old=pm, new=pm_pipe)

    def test_loss_backward_sequential(self):
        mse_loss = torch.nn.MSELoss()
        seq_pipe_with_loss = Pipe.from_sequential(self.seq, mse_loss)
        check_qualname_mapping(old=self.seq, new=seq_pipe_with_loss)

        test_optim = torch.optim.SGD(seq_pipe_with_loss.parameters(), lr=0.01, momentum=0.9)
        ref_optim = torch.optim.SGD(self.seq.parameters(), lr=0.01, momentum=0.9)

        x = torch.randn(5, 512)
        target = torch.zeros(5, 512)

        test_optim.zero_grad()
        test_out = seq_pipe_with_loss(x, target)
        test_grads = {seq_pipe_with_loss.remap_qualname(name): copy.copy(val.grad) for name, val in seq_pipe_with_loss.named_parameters()}
        torch.testing.assert_allclose(test_out, mse_loss(self.seq(x), target))

        ref_optim.zero_grad()
        ref_out = mse_loss(self.seq(x), target)
        ref_out.backward()
        ref_grads = {name: copy.copy(val.grad) for name, val in self.seq.named_parameters()}

        for name, ref_grad in ref_grads.items():
            assert name in test_grads
            torch.testing.assert_allclose(test_grads[name], ref_grad)

    def test_loss_backward_tracing(self):
        mse_loss = torch.nn.MSELoss()
        ec_pipe_with_loss = Pipe.from_tracing(self.ec, loss_fn=mse_loss)
        check_qualname_mapping(old=self.ec, new=ec_pipe_with_loss)

        test_optim = torch.optim.SGD(ec_pipe_with_loss.parameters(), lr=0.01, momentum=0.9)
        ref_optim = torch.optim.SGD(self.ec.parameters(), lr=0.01, momentum=0.9)

        x = torch.randn(5, 512)
        target = torch.zeros(5, 512)

        test_optim.zero_grad()
        test_out = ec_pipe_with_loss(x, target)
        test_grads = {ec_pipe_with_loss.remap_qualname(name): copy.copy(val.grad) for name, val in ec_pipe_with_loss.named_parameters()}
        torch.testing.assert_allclose(test_out, mse_loss(self.ec(x), target))

        ref_optim.zero_grad()
        ref_out = mse_loss(self.ec(x), target)
        ref_out.backward()
        ref_grads = {name: copy.copy(val.grad) for name, val in self.ec.named_parameters()}

        for name, ref_grad in ref_grads.items():
            assert name in test_grads
            torch.testing.assert_allclose(test_grads[name], ref_grad)

    def test_grad_accumulation(self):
        # TODO: test grad accumulation in runtime
        class Code(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(50, 50)

            def forward(self, x):
                x = self.linear(x)
                pipe_split()
                y = torch.relu(x)
                pipe_split()
                z = torch.sigmoid(x)
                pipe_split()
                return y + z

        c = Code()
        c.train()
        mse_loss = torch.nn.MSELoss()
        accum_pipe = Pipe.from_tracing(c, loss_fn=mse_loss)
        print(accum_pipe.split_gm)
        assert any(n.target == torch.add for n in accum_pipe.split_gm.graph.nodes)
        accum_pipe(torch.randn(5, 50), torch.randn(5, 50))

    def test_invoke_pipeline_error(self):
        ec_pipe = Pipe.from_tracing(self.ec, MultiUseParameterConfig.TRANSMIT)
        with self.assertRaisesRegex(RuntimeError, 'To run pipeline locally, invoke the Pipe object directly'):
            ec_pipe.split_gm(torch.randn(5, 50))

    def test_loss_is_a_function(self):
        def mse_loss(output, target):
            return torch.mean((output - target) ** 2)

        ec_pipe_with_loss = Pipe.from_tracing(self.ec, loss_fn=mse_loss)
        check_qualname_mapping(old=self.ec, new=ec_pipe_with_loss)

        test_optim = torch.optim.SGD(ec_pipe_with_loss.parameters(), lr=0.01, momentum=0.9)
        ref_optim = torch.optim.SGD(self.ec.parameters(), lr=0.01, momentum=0.9)

        x = torch.randn(5, 512)
        target = torch.zeros(5, 512)

        test_optim.zero_grad()
        test_out = ec_pipe_with_loss(x, target)
        test_grads = {ec_pipe_with_loss.remap_qualname(name): copy.copy(val.grad) for name, val in ec_pipe_with_loss.named_parameters()}
        torch.testing.assert_allclose(test_out, mse_loss(self.ec(x), target))

        ref_optim.zero_grad()
        ref_out = mse_loss(self.ec(x), target)
        ref_out.backward()
        ref_grads = {name: copy.copy(val.grad) for name, val in self.ec.named_parameters()}


    def test_deeply_nested_parameter(self):
        class Nest(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(5, 5))

            def forward(self, x):
                return x + self.param

        class TestCode(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nest = Nest()

            def forward(self, x):
                x = self.nest(x)
                pipe_split()
                return torch.relu(x)

        tc = TestCode()
        tc_pipe = Pipe.from_tracing(tc)
        torch.testing.assert_allclose(tc_pipe(torch.ones(5, 5)), 2 * torch.ones(5, 5))

    def test_deeply_nested_shared_param(self):
        class Nest(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(5, 5))

            def forward(self, x):
                return x + self.param

        class TestCode(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nest = Nest()

            def forward(self, x):
                x = self.nest(x)
                pipe_split()
                return self.nest(x)

        tc = TestCode()
        tc_pipe = Pipe.from_tracing(tc)
        torch.testing.assert_allclose(tc_pipe(torch.ones(5, 5)), 3 * torch.ones(5, 5))

    def test_multi_use_param_error(self):
        with self.assertRaisesRegex(ValueError, 'multi_use_param_spec must be MultiUseParamSpec enum or dict'):
            Pipe.from_tracing(self.ec, multi_use_param_spec=3)

    def test_multi_use_param_dict_error(self):
        with self.assertRaisesRegex(ValueError, 'Unknown multi-use config value 3 specified for mm_param'):
            Pipe.from_tracing(self.ec, multi_use_param_spec={'mm_param': 3})

    def test_annotate_split_points_beginning(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.l(x) + 5

        class Bar(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.f = Foo()
                self.b = Bar()

            def forward(self, x):
                return self.b(self.f(x))

        b = Base()
        annotate_split_points(b, {'b': PipeSplitWrapper.SplitPoint.BEGINNING})
        pipe = Pipe.from_tracing(b)
        x = torch.randn(5, 5)
        torch.testing.assert_allclose(pipe(x), b(x))
        assert 'submod_1' in dict(pipe.split_gm.named_modules())

    def test_from_tracing_preserves_buffer(self):
        ec = ExampleCode()
        pipe = Pipe.from_tracing(ec)
        assert 'moved_buffer' in dict(pipe.split_gm.submod_1.named_buffers())
        # NB: identity comparison
        assert ec.buffer is pipe.split_gm.submod_1.moved_buffer

    def test_annotate_split_points_end(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.l(x) + 5

        class IndirectionForTesting(torch.nn.Module):
            def __init__(self, submod):
                super().__init__()
                self.submod = submod

            def forward(self, x):
                return self.submod(x)

        class Bar(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.f = IndirectionForTesting(Foo())
                self.b = Bar()

            def forward(self, x):
                return self.b(self.f(x))

        b = Base()
        annotate_split_points(b, {'f.submod': PipeSplitWrapper.SplitPoint.END})
        pipe = Pipe.from_tracing(b)
        x = torch.randn(5, 5)
        torch.testing.assert_allclose(pipe(x), b(x))
        assert 'submod_1' in dict(pipe.split_gm.named_modules())

    def test_annotate_split_points_nonexistent_module(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        m = Module()
        with self.assertRaisesRegex(AttributeError, 'Specified target foo.bar.baz referenced nonexistent module foo'):
            annotate_split_points(m, {'foo.bar.baz': PipeSplitWrapper.SplitPoint.END})

    def test_pipe_forward_kwargs(self):
        class ModuleWithArgsAndKWArgs(torch.nn.Module):
            def forward(self, x, y=2):
                return x + y

        module = ModuleWithArgsAndKWArgs()
        pipe = Pipe.from_tracing(module)
        self.assertEqual(pipe(1), 3)
        self.assertEqual(pipe(1, y=3), 4)
        self.assertEqual(pipe(1, 4), 5)
        with self.assertRaisesRegex(TypeError, "got an unexpected keyword argument 'z'"):
            pipe(1, z=100)
        with self.assertRaisesRegex(TypeError, "got an unexpected keyword argument 'a'"):
            pipe(1, y=3, a='a', b=True, c='c')
        with self.assertRaisesRegex(TypeError, "multiple values for argument 'y'"):
            pipe(1, 4, y=100)

        class ModuleWithOnlyKWArgs(torch.nn.Module):
            def forward(self, x=1, y=2):
                return x + y

        module = ModuleWithOnlyKWArgs()
        pipe = Pipe.from_tracing(module)
        self.assertEqual(pipe(), 3)
        self.assertEqual(pipe(2), 4)
        self.assertEqual(pipe(x=3), 5)
        self.assertEqual(pipe(y=4), 5)
        self.assertEqual(pipe(3, 3), 6)
        self.assertEqual(pipe(3, y=4), 7)
        self.assertEqual(pipe(x=4, y=4), 8)
        with self.assertRaisesRegex(TypeError, "got an unexpected keyword argument 'b'"):
            pipe(b=True, a='a', c='c')
        with self.assertRaisesRegex(TypeError, "multiple values for argument 'x'"):
            pipe(2, x=3)
        with self.assertRaisesRegex(TypeError, "multiple values for argument 'x'"):
            pipe(1, 2, x=3, y=4)

    def test_pipe_generalized_chunking(self):
        class SpecialInputNamedTuple(NamedTuple):
            my_tensor : torch.Tensor
            my_integer : int

        class TheModel(torch.nn.Module):
            def forward(self, x : torch.Tensor, input_nt : SpecialInputNamedTuple):
                added = x + input_nt.my_tensor + input_nt.my_integer
                pipe_split()
                multiplied = x * input_nt.my_integer
                return {'added' : added, 'multiplied' : multiplied}

        tm = TheModel()
        x = torch.randn(5, 3)
        input_nt = SpecialInputNamedTuple(torch.randn(5, 3), 42)
        ref_out = tm(x, input_nt)

        pipe = Pipe.from_tracing(tm)
        pipe_out = pipe(x, input_nt)

        torch.testing.assert_allclose(pipe_out['added'], ref_out['added'])
        torch.testing.assert_allclose(pipe_out['multiplied'], ref_out['multiplied'])

        class TensorChunkSpec:
            def __init__(self, split_dim):
                self.split_dim = split_dim

            split_dim : int

        def shard_dict_of_args(args_dict, args_chunk_spec, chunks):
            # Stage 1+2: flatten and shard/replicate

            # args_sharded_replicated : [num args, num flat values, num chunks]
            args_sharded_replicated = {}

            arg_specs = []

            assert len(args_dict) == len(args_chunk_spec)
            for arg_key, arg in args_dict.items():
                chunk_spec = args_chunk_spec[arg_key]
                flat, spec = tree_flatten(arg)
                arg_specs.append(spec)
                chunk_spec_flat, _ = tree_flatten(chunk_spec)

                if len(flat) != len(chunk_spec_flat):
                    raise ValueError(f'Argument value {arg} did not have the same number of '
                                    f'values as as chunk spec {chunk_spec}')

                sharded_arg_flat = []

                for v, chunk_v in zip(flat, chunk_spec_flat):
                    if isinstance(chunk_v, TensorChunkSpec):
                        # TODO: check type of v. If it's a tensor, use chunk (or debug mask).
                        # If it's a collection type, split it as you would expect. Otherwise,
                        # Throw an error
                        sharded_arg_flat.append(torch.chunk(v, chunks, chunk_v.split_dim))
                    else:
                        sharded_arg_flat.append([v] * chunks)

                args_sharded_replicated[arg_key] =sharded_arg_flat

            # chunks_flat : [num chunks, num args, num flat values]
            chunks_flat = []
            for chunk_idx in range(chunks):
                chunk_args = {}
                for key, arg in args_sharded_replicated.items():
                    arg_single_chunk = []
                    for v_flat in arg:
                        arg_single_chunk.append(v_flat[chunk_idx])
                    chunk_args[key] = arg_single_chunk
                chunks_flat.append(chunk_args)


            # args_split : [num chunks, num args]
            args_split = []

            for chunk in chunks_flat:
                per_chunk_args = {}
                assert len(arg_specs) == len(chunk)
                for (key, arg), arg_spec in zip(chunk.items(), arg_specs):
                    per_chunk_args[key] = tree_unflatten(arg, arg_spec)
                args_split.append(per_chunk_args)

            return args_split

        def split_args_kwargs_into_chunks(args, kwargs, args_chunk_spec, kwargs_chunk_spec, chunks):
            # Given `args` and `kwargs`, we want to yield a set of `chunks` args and kwargs such that
            # the constituent Tensor values have been sharded/replicated according to the `args_chunk_spec`
            # and `kwargs_chunk_spec` specifications. The steps are as follows:
            #
            # 1. Use pytree.tree_flatten to flatten each arg and its spec into nto a 1d array of values.
            #    To use a running example: suppose our inputs look like
            #
            #       args = ([A, [B, C]], D) args_spec = ([None, [None, TensorChunkSpec]], None)
            #       (kwargs not shown but it's a similar process)
            #
            #    Then for this step we would end up with
            #
            #       args = ([A, B, C], D) args_spec = ([None, None, TensorChunkSpec], None)
            #
            # 2. Shard or replicate the arguments subject to the policy in the spec. Suppose chunks = 2
            #
            #       args = ([[A, A], [B, B], [C_1, C_2]], [D, D])
            #
            # 3. Rotate the nesting order such that chunks are the outer dimension
            #
            #       args_chunks = [
            #           ([A, B, C_1], D),
            #           ([A, B, C_2], D),
            #       ]
            #
            # 4. Unflatten each chunk according to the spec
            #
            #       args_chunks = [
            #           ([A, [B, C_1]], D),
            #           ([A, [B, C_2]], D),
            #       ]

            # TODO: _debug_mask_minibatches

            args_split_dict = shard_dict_of_args(dict(enumerate(args)), dict(enumerate(args_chunk_spec)), chunks)
            kwargs_split = shard_dict_of_args(kwargs, kwargs_chunk_spec, chunks)

            args_split = []
            for chunk_args in args_split_dict:
                args_split.append(tuple(chunk_args[i] for i in range(len(chunk_args))))

            return args_split, kwargs_split

        NCHUNKS = 5

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            (x,), {'input_nt': input_nt}, (None,), {'input_nt': SpecialInputNamedTuple(TensorChunkSpec(0), None)},
            chunks=NCHUNKS)

        # Args should have 5 chunks that all contain the single replicated `x` value
        assert len(args_split) == NCHUNKS
        for arg in args_split:
            assert isinstance(arg, tuple) and len(arg) == 1
            torch.testing.assert_allclose(arg[0], x)

        # kwargs should have 5 chunks that contain a SpecialInputNamedTuple at the 'input_nt' key
        # `input_nt` should have rows of `input_nt.my_tensor` split along dimension 0.
        # `my_integer` should be the replicated value 42

        assert len(kwargs_split) == NCHUNKS
        for i, kwarg in enumerate(kwargs_split):
            assert isinstance(kwarg, dict) and len(kwarg) == 1 and 'input_nt' in kwarg
            input_nt_to_test = kwarg['input_nt']
            assert isinstance(input_nt_to_test, SpecialInputNamedTuple)
            torch.testing.assert_allclose(input_nt_to_test.my_tensor, input_nt.my_tensor[i:i+1, :])
            assert input_nt_to_test.my_integer == 42

        # Above test case is actually not valid for the example program, since it compels a
        # broadcast across the sharded dimension

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            (x,), {'input_nt': input_nt}, (TensorChunkSpec(0),),
            {'input_nt': SpecialInputNamedTuple(TensorChunkSpec(0), None)}, chunks=NCHUNKS)

        assert len(args_split) == NCHUNKS
        for chunk_idx, arg in enumerate(args_split):
            assert isinstance(arg, tuple) and len(arg) == 1
            torch.testing.assert_allclose(arg[0], x[chunk_idx:chunk_idx + 1, :])

        # Test execution
        model_out_chunks = []
        for chunk_idx in range(NCHUNKS):
            model_out_chunks.append(pipe(*args_split[chunk_idx], **kwargs_split[chunk_idx]))

        assert len(model_out_chunks) == NCHUNKS
        catted_added = torch.cat(tuple(v['added'] for v in model_out_chunks))
        catted_multiplied = torch.cat(tuple(v['multiplied'] for v in model_out_chunks))

        torch.testing.assert_allclose(catted_added, ref_out['added'])
        torch.testing.assert_allclose(catted_multiplied, ref_out['multiplied'])

        def merge_chunks(chunks, chunk_spec):
            # Given a list of chunks and a chunk specification, merge the chunks
            # into a single value according to that chunk spec. This is essentially
            # the inverse of `split_args_kwargs_into_chunks`, so the steps are
            # similar to the steps in that function but in reverse. Given the
            # input values:
            #
            #       chunks = [
            #           ([A, [B, C_1]], D),
            #           ([A, [B, C_2]], D),
            #       ]
            #       args_spec = ([None, [None, TensorChunkSpec]], None)
            #
            # 1. Flatten the chunks according to the chunk_spec
            #
            #       chunks_flat = [
            #           ([A, B, C_1], D),
            #           ([A, B, C_2], D),
            #       ]
            #
            # 2. Rotate the nesting order such that chunks are the inner dimension
            #
            #       value_inner = ([A, B, [C_1, C_2]], D)
            #
            # 3. Concatenate sharded arguments
            #
            #       value_combined = ([A, B, C], D)
            #
            # 4. Unflatten the combined args given the spec
            #
            #       value = ([A, [B, C]], D)

            # Preliminary: flatten the chunk spec
            spec_flattened, flatten_spec = tree_flatten(chunk_spec)

            # Stage 1: flatten chunks
            # chunks_flattened : [num chunks, num args]
            chunks_flattened = []

            for chunk in chunks:
                chunk_flattened, _ = tree_flatten(chunk)
                if len(chunk_flattened) != len(spec_flattened):
                    raise ValueError(f'Chunk {chunk} did not match chunk spec {chunk_spec}')

                chunks_flattened.append(chunk_flattened)

            # Stage 2 and 3: Rotate nesting order s.t. chunks are inner dimension and
            #                concatenate sharded operands
            # args_flattened : [num args]
            args_flattened = []
            for arg_idx, arg in enumerate(spec_flattened):
                if isinstance(arg, TensorChunkSpec):
                    # TODO: cat or stack?
                    partial_values = [chunks_flattened[chunk_idx][arg_idx] for chunk_idx in range(len(chunks_flattened))]
                    args_flattened.append(torch.cat(partial_values))
                else:
                    value = chunks_flattened[0][arg_idx]
                    for chunk_idx in range(1, len(chunks_flattened)):
                        assert chunks_flattened[chunk_idx][arg_idx] == value
                    args_flattened.append(value)

            # Stage 4: Unflatten combined args
            return tree_unflatten(args_flattened, flatten_spec)

        chunks_merged = merge_chunks(model_out_chunks, {'added': TensorChunkSpec(0), 'multiplied': TensorChunkSpec(0)})

        torch.testing.assert_allclose(chunks_merged['added'], ref_out['added'])
        torch.testing.assert_allclose(chunks_merged['multiplied'], ref_out['multiplied'])


if __name__ == '__main__':
    unittest.main()
