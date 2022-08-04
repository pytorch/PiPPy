from dataclasses import dataclass
from typing import Set, Tuple, Union

Dim = str
Pattern = Union['DimSpec', str]
DimPattern = Union[Tuple[Dim], str]

@dataclass
class DimSpec:
  inputs: Tuple[Tuple[Dim]]
  outputs: Tuple[Tuple[Dim]]
  _outputs_has_new_dims: bool = False

  def __post_init__(self):
    assert isinstance(self.inputs, tuple), 'inputs must be a tuple'
    assert isinstance(self.outputs, tuple), 'outputs must be a tuple'
    for dim_tuple in self.inputs + self.outputs:
        assert isinstance(dim_tuple, tuple), f'all inputs and outputs must be a tuple of dims, got: {dim_tuple}'
        for dim in dim_tuple:
            assert isinstance(dim, Dim), f'dim must be an instance of str, got: {dim} for input {dim_tuple}'
    
    # check that all output dims are in input dims
    all_inputs = set()
    for x in self.inputs:
      all_inputs |= set(x)
    all_outputs = set()
    for y in self.outputs:
      all_outputs |= set(y)
    if self._outputs_has_new_dims:
      assert len(all_outputs - all_inputs) > 0, 'output pattern should contain new dimensions.'
    else:
      assert len(all_outputs - all_inputs) == 0, f'output pattern contains unknown dimensions: {self}'

  @staticmethod
  def parse(pattern: Pattern, allow_new_dims: bool = False) -> 'DimSpec':
    if isinstance(pattern, DimSpec):
      return pattern

    before, after = pattern.split('->')
    return DimSpec(
      inputs=tuple(DimSpec.parse_dim(x) for x in before.split(',')),
      outputs=tuple(DimSpec.parse_dim(x) for x in after.split(',')),
      _outputs_has_new_dims=allow_new_dims,
    )

  @staticmethod
  def find_new_dim(pattern: Pattern, input_idx: int = 0, output_idx: int = 0) -> Tuple[Dim, Dim, int]:
    """
    Find the single dimension that gets replaced from the input to the output.
    All other dimensions must match
    For a pattern like abc->abd, return (c, d, 2)
    """
    pattern = DimSpec.parse(pattern, allow_new_dims=True)
    inputs = pattern.inputs[input_idx]
    outputs = pattern.outputs[output_idx]
    input_set = set(inputs)
    output_set = set(outputs)
    new_dims = output_set - input_set
    old_dims = input_set - output_set

    assert(len(new_dims) == 1)
    assert(len(old_dims) == 1)
 
    old_dim = list(old_dims)[0]
    new_dim = list(new_dims)[0]
    return old_dim, new_dim, inputs.index(old_dim)

  # reduction followed by broadcast
  # abc -> abd does
  # abc -> ab -> abd
  @staticmethod
  def replace_dim(pattern: Pattern, input_idx: int, output_idx: int) -> Tuple['DimSpec', 'DimSpec']:
    pattern = DimSpec.parse(pattern, allow_new_dims=True)
    old_dim, new_dim, index = DimSpec.find_new_dim(pattern, input_idx, output_idx)
    middle = tuple(dim for dim in pattern.inputs[input_idx] if dim is not old_dim)
    return (
      DimSpec(pattern.inputs, (middle, )),
      DimSpec((middle, ), pattern.outputs, _outputs_has_new_dims=True),
    )

  @staticmethod
  def gen_replace_dim(num_dims: int, dim: int) -> 'DimSpec':
    """
    Generate a pattern where a single dimension is replaced
    Example:  gen_replace_dim(4, 2) -> 'abcd->abed'
    """
    dims = DimSpec.gen_dims(num_dims + 1)
    dims_in = dims[:-1]
    dims_out = dims_in[:dim] + (dims[-1], ) + dims_in[dim+1:]
    return DimSpec((dims_in, ), (dims_out, ), _outputs_has_new_dims=True)

  @staticmethod
  def parse_dim(pattern: Union[str, Tuple[Dim]]) -> Tuple[Dim]:
      if isinstance(pattern, tuple):
          return pattern
      return tuple(pattern) 

  @staticmethod
  def remove_dim(dims: Tuple[Dim], to_remove: Tuple[Dim]) -> Tuple[Dim]:
    return tuple(d for d in dims if d not in to_remove)

  @staticmethod
  def gen_dims(num: int) -> Tuple[Dim]:
      assert num >= 0 and num <= 26, 'cannot generate more than 26 dims'
      return tuple('abcdefghijklmnopqrstuvwxyz'[:num])

  def __str__(self):
    return '->'.join((
      ','.join(''.join(x) for x in self.inputs),
      ','.join(''.join(x) for x in self.outputs)
    ))

if __name__ == '__main__':
    print(DimSpec.gen_replace_dim(4, 2))