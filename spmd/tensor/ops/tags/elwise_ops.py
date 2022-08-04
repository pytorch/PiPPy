from torch import *
from tags import *

op_infos = {
# Dunders
'__imul__': Op(),
'__matmul__': Op(),
'__radd__': Op(),
'__rand__': Op(),
'__rdiv__': Op(),
'__rmatmul__': Op(),
'__rmod__': Op(),
'__rmul__': Op(),
'__ror__': Op(),
'__rpow__': Op(),
'__rrshift__': Op(),
'__rsub__': Op(),
'__rxor__': Op(),
# Elementwise Binary Ops
add: Op(elwise=AllDims(), linear=True, cls=(LINALG, ADD)),
atan2: ELWISE,
bitwise_and: Op(elwise=AllDims(), cls=(BITWISE, MUL)),
bitwise_left_shift: Op(elwise=AllDims(), cls=(BITWISE, OTHER)),
bitwise_or: Op(elwise=AllDims(), cls=(BITWISE, OTHER)),
bitwise_right_shift: Op(elwise=AllDims(), cls=(BITWISE, OTHER)),
bitwise_xor: Op(elwise=AllDims(), cls=(BITWISE, OTHER)),
complex: Op(elwise=AllDims(), cls=(BITWISE, OTHER)),
copysign: ELWISE,
div: ELWISE,
eq: ELWISE,
float_power: ELWISE,
floor_divide: ELWISE,
fmax: ELWISE,
fmin: ELWISE,
fmod: ELWISE,
gcd: ELWISE,
ge: ELWISE,
gt: ELWISE,
heaviside: ELWISE,
hypot: ELWISE,
igamma: ELWISE,
igammac: ELWISE,
isclose: ELWISE,
lcm: ELWISE,
ldexp: ELWISE,
le: ELWISE,
logical_and: ELWISE,
logical_or: ELWISE,
logical_xor: ELWISE,
lt: ELWISE,
maximum: ELWISE,
minimum: ELWISE,
mul: ELWISE,
ne: ELWISE,
nextafter: ELWISE,
polar: ELWISE,
pow: ELWISE,
remainder: ELWISE,
rsub: ELWISE,
special.xlog1py: ELWISE,
special.zeta: ELWISE,
sub: ELWISE,
true_divide: ELWISE,
xlogy: ELWISE,
# Elementwise Unary Ops
abs: ELWISE,
acos: ELWISE,
acosh: ELWISE,
angle: ELWISE,
asin: ELWISE,
asinh: ELWISE,
atan: ELWISE,
atanh: ELWISE,
bitwise_not: ELWISE,
ceil: ELWISE,
clamp: ELWISE,
conj: ELWISE,
conj_physical: ELWISE,
cos: ELWISE,
cosh: ELWISE,
deg2rad: ELWISE,
digamma: ELWISE,
erf: ELWISE,
erfc: ELWISE,
erfinv: ELWISE,
exp: ELWISE,
exp2: ELWISE,
expm1: ELWISE,
floor: ELWISE,
frac: ELWISE,
frexp: ELWISE,
i0: ELWISE,
imag: ELWISE,
isfinite: ELWISE,
isinf: ELWISE,
isnan: ELWISE,
isneginf: ELWISE,
isposinf: ELWISE,
isreal: ELWISE,
lgamma: ELWISE,
log: ELWISE,
log10: ELWISE,
log1p: ELWISE,
log2: ELWISE,
logical_not: ELWISE,
logit: ELWISE,
special.multigammaln: ELWISE,
nan_to_num: ELWISE,
neg: ELWISE,
nn.functional.celu: ELWISE,
nn.functional.elu: ELWISE,
nn.functional.hardsigmoid: ELWISE,
nn.functional.logsigmoid: ELWISE,
nn.functional.mish: ELWISE,
nn.functional.rrelu: ELWISE,
nn.functional.selu: ELWISE,
nn.functional.silu: ELWISE,
nn.functional.softsign: ELWISE,
nn.functional.tanhshrink: ELWISE,
polygamma: ELWISE,
positive: ELWISE,
rad2deg: ELWISE,
real: ELWISE,
reciprocal: ELWISE,
round: ELWISE,
rsqrt: ELWISE,
sgn: ELWISE,
sigmoid: ELWISE,
sign: ELWISE,
signbit: ELWISE,
sin: ELWISE,
sinc: ELWISE,
sinh: ELWISE,
special.entr: ELWISE,
special.erfcx: ELWISE,
special.i0e: ELWISE,
special.i1: ELWISE,
special.i1e: ELWISE,
special.log_ndtr: ELWISE,
special.ndtr: ELWISE,
special.ndtri: ELWISE,
special.polygamma: ELWISE,
sqrt: ELWISE,
square: ELWISE,
tan: ELWISE,
tanh: ELWISE,
trunc: ELWISE,
'fft_fftfreq': Op(),
'fft_rfftfreq': Op(),
fft.fft: Op(elwise=AllDBut(kw='dim', default=-1)),
fft.fft2: Op(elwise=AllDBut(kw='dim', default=(-2, -1))),
fft.fftn: Op(elwise=AllDBut(kw='dim', default=lambda s: list(range(-len(s), 0)) if s is not None else ALL)),
fft.fftshift: Op(elwise=AllDBut(kw='dim', default=ALL)),
fft.hfft: Op(elwise=AllDBut(kw='dim', default=-1)),
fft.hfft2: Op(elwise=AllDBut(kw='dim', default=(-2, -1))),
fft.hfftn: Op(elwise=AllDBut(kw='dim', default=lambda s: list(range(-len(s), 0)) if s is not None else ALL)),
fft.ifft: Op(elwise=AllDBut(kw='dim', default=-1)),
fft.ifft2: Op(elwise=AllDBut(kw='dim', default=(-2, -1))),
fft.ifftn: Op(elwise=AllDBut(kw='dim', default=lambda s: list(range(-len(s), 0)) if s is not None else ALL)),
fft.ifftshift: Op(elwise=AllDBut(kw='dim', default=ALL)),
fft.ihfft: Op(elwise=AllDBut(kw='dim', default=-1)),
fft.ihfft2: Op(elwise=AllDBut(kw='dim', default=(-2, -1))),
fft.ihfftn: Op(elwise=AllDBut(kw='dim', default=lambda s: list(range(-len(s), 0)) if s is not None else ALL)),
fft.irfft: Op(elwise=AllDBut(kw='dim', default=-1)),
fft.irfft2: Op(elwise=AllDBut(kw='dim', default=(-2, -1))),
fft.irfftn: Op(elwise=AllDBut(kw='dim', default=lambda s: list(range(-len(s), 0)) if s is not None else ALL)),
fft.rfft: Op(elwise=AllDBut(kw='dim', default=-1)),
fft.rfft2: Op(elwise=AllDBut(kw='dim', default=(-2, -1))),
fft.rfftn: Op(elwise=AllDBut(kw='dim', default=lambda s: list(range(-len(s), 0)) if s is not None else ALL)),
istft: Op(elwise=AllDBut(fixed=(-3, -2, -1))),  # input (channel, fft_size, n_frame, 2) with channel and 2 is for real inputs (real input deprecated)
stft: Op(elwise=AllDBut(fixed=-1)),
adjoint: Op(transpose_dims=[(-2, -1)]),
as_strided: Op(),  # too general, 
atleast_1d: Op(inputwise=True, elwise=AllDims()),
atleast_2d: Op(inputwise=True, dim_map=(-1, ) * max(0, len(x.shape) - 1 move_dims=lambda x: ((0, 1) if len(x.shape) == 1 else []),
atleast_3d: Op(inputwise=True, move_dims=labmda x: ())
broadcast_shapes: None,
broadcast_tensors: ELWISE,
broadcast_to: Op(dim_map=lambda input, shape: broadcast_dim_map(input, shape)),
cat
chunk
column_stack
contiguous
dsplit
dstack
expand
expand_as
flatten
flip
fliplr
flipud
H
hsplit
hstack
mH
movedim
mT
narrow
narrow_copy
permute
ravel
repeat
repeat_interleave
reshape
reshape_as
resize_
resize_as_
roll
rot90
split
split_with_sizes
squeeze
stack
swapaxes
T
t
tensor_split
tile
transpose
unbind
unflatten
unfold
unsqueeze
view
view_as
vsplit
vstack
addbmm
addmm
addmv
addr
baddbmm
block_diag
bmm
cartesian_prod
chain_matmul
cholesky
cholesky_inverse
cholesky_solve
cross
diag
diag_embed
diagflat
diagonal
dot
eig
einsum
frobenius_norm
geqrf
ger
inner
inverse
kron
linalg.cholesky
linalg.cholesky_ex
linalg.cond
linalg.cross
linalg.det
linalg.eig
linalg.eigh
linalg.eigvals
linalg.eigvalsh
linalg.householder_product
linalg.inv
linalg.inv_ex
linalg.lstsq
linalg.lu_factor
linalg.lu_factor_ex
linalg.matrix_norm
linalg.matrix_power
linalg.matrix_rank
linalg.multi_dot
linalg.norm
linalg.pinv
linalg.qr
linalg.slogdet
linalg.solve
linalg.solve_triangular
linalg.svd
linalg.svdvals
linalg.tensorinv
linalg.tensorsolve
linalg.vector_norm
lobpcg
logaddexp
logaddexp2
logdet
lu
lu_solve
lu_unpack
matmul
matrix_exp
mm
mv
norm
nuclear_norm
orgqr
ormqr
outer
pca_lowrank
pinverse
qr
renorm
solve
svd
svd_lowrank
symeig
tensordot
trace
triangular_solve
tril
tril_indices
triu
triu_indices
vdot
all
amax
amin
nanquantile
quantile
sum
nansum
aminmax
any
argmax
std
var
var_mean
std_mean
argmin
cummax
cummin
cumprod
cumsum
logcumsumexp
logsumexp
max
mean
median
min
mode
nanmean
nanmedian
nanquantile
nansum
quantile
std
std_mean
sum
var
var_mean
cumulative_trapezoid
diff
gradient
grid_sampler
grid_sampler_2d
grid_sampler_3d
lerp
trapezoid
trapz
arange
as_tensor
asarray
bartlett_window
blackman_window
empty
empty_like
empty_strided
eye
from_numpy
full
full_like
hamming_window
hann_window
kaiser_window
linspace
logspace
meshgrid
new
new_empty
new_empty
new_empty_strided
new_full
new_full
new_ones
new_ones
new_tensor
new_zeros
new_zeros
normal
ones
ones_like
rand
rand_like
randint
randint_like
randn
randn_like
randn_like
random_
randperm
range
tensor
vander
zeros
zeros_like
adaptive_max_pool1d_with_indices
adaptive_max_pool2d_with_indices
adaptive_max_pool3d_with_indices
affine_grid
alpha_dropout
batch_norm_gather_stats_with_counts
batch_norm_stats
binary_cross_entropy
binary_cross_entropy_with_logits
cdist
channel_shuffle
constant_pad_nd
conv3d
convolution
dist
dropout3d
feature_alpha_dropout
feature_dropout
fractional_max_pool2d_with_indices
fractional_max_pool3d_with_indices
gru_cell
gumbel_softmax
kl_div
l1_loss
log_softmax
lp_pool1d
lp_pool2d
lstm_cell
margin_ranking_loss
max_pool1d_with_indices
max_pool2d_with_indices
max_pool3d_with_indices
max_unpool1d
max_unpool2d
max_unpool3d
multi_head_attention_forward
multi_margin_loss
multilabel_margin_loss
multilabel_soft_margin_loss
native_batch_norm
native_dropout
native_layer_norm
native_norm
nn.functional.adaptive_avg_pool1d
nn.functional.adaptive_avg_pool2d
nn.functional.adaptive_avg_pool3d
nn.functional.adaptive_max_pool1d
nn.functional.adaptive_max_pool2d
nn.functional.adaptive_max_pool3d
nn.functional.avg_pool1d
nn.functional.avg_pool2d
nn.functional.avg_pool3d
nn.functional.batch_norm
nn.functional.bilinear
nn.functional.celu
nn.functional.conv_transpose1d
nn.functional.conv_transpose2d
nn.functional.conv_transpose3d
nn.functional.conv1d
nn.functional.conv2d
nn.functional.cosine_embedding_loss
nn.functional.cosine_similarity
nn.functional.cross_entropy
nn.functional.ctc_loss
nn.functional.dropout
nn.functional.dropout2d
nn.functional.elu
nn.functional.embedding
nn.functional.embedding_bag
nn.functional.feature_alpha_dropout
nn.functional.fractional_max_pool2d
nn.functional.fractional_max_pool3d
nn.functional.gaussian_nll_loss
nn.functional.gelu
nn.functional.glu
nn.functional.grid_sample
nn.functional.group_norm
nn.functional.hardshrink
nn.functional.hardsigmoid
nn.functional.hardswish
nn.functional.hardtanh
nn.functional.hinge_embedding_loss
nn.functional.huber_loss
nn.functional.instance_norm
nn.functional.interpolate
nn.functional.kl_div
nn.functional.layer_norm
nn.functional.leaky_relu
nn.functional.linear
nn.functional.local_response_norm
nn.functional.logsigmoid
nn.functional.max_pool1d
nn.functional.max_pool2d
nn.functional.max_pool3d
nn.functional.mish
nn.functional.mse_loss
nn.functional.nll_loss
nn.functional.normalize
nn.functional.one_hot
nn.functional.pad
nn.functional.pairwise_distance
nn.functional.pixel_shuffle
nn.functional.pixel_unshuffle
nn.functional.poisson_nll_loss
nn.functional.prelu
nn.functional.relu
nn.functional.relu6
nn.functional.selu
nn.functional.silu
nn.functional.softmin
nn.functional.softplus
nn.functional.softshrink
nn.functional.softsign
nn.functional.tanhshrink
nn.functional.threshold
nn.functional.unfold
nn.functional.upsample_bilinear
nn.functional.upsample_nearest
pdist
rnn_relu
rnn_relu_cell
rnn_tanh
rnn_tanh_cell
smooth_l1_loss
soft_margin_loss
softmax
triplet_margin_loss
triplet_margin_with_distance_loss
argsort
argwhere
bincount
bucketize
corrcoef
count_nonzero
cov
histc
histogram
histogramdd
isin
kthvalue
msort
nonzero
searchsorted
sort
topk
unique
unique_consecutive
clamp
clamp_max
clamp_min
where
diagonal_scatter
gather
scatter
scatter_add
scatter_reduce
select_scatter
slice_scatter
index_add
index_copy
index_fill
index_put
index_select
item
put
select
take
take_along_dim
bfloat16
bool
byte
char
clone
cpu
cuda
double
fill_
float
half
int
long
short
type_as
zero_
coalesce
dense_dim
sparse_coo_tensor
sparse_csr_tensor
to_dense
to_sparse
to_sparse_coo
complex
conj
conj_physical
real
resolve_conj
resolve_neg
view_as_complex
view_as_real
bernoulli
binomial
cauchy
exponential_
geometric_
multinomial
poisson
__getitem__
combinations
is_complex
is_floating_point
is_signed
sum_to_size
_masked.amax
_masked.amin
_masked.log_softmax
_masked.mean
_masked.norm
_masked.normalize
_masked.prod
_masked.softmax
_masked.softmin
_masked.std
_masked.sum
_masked.var
masked_fill
masked_scatter
masked_select
addcdiv
addcmul
allclose
equal
}