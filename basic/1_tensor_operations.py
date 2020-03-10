# --------------------------------------------------------
# Basic pytorch tensor operations
# Written by Huy Thanh Nguyen (kaizen0890@gmail.com - github:)
# These example is created following to "Deep Learning with Pytorch: Quick start guide" book and pytorch tutorial documents
# --------------------------------------------------------

import torch
import numpy as np

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Default value initialization of Tensor
# 2. Convert tensor to numpy and revert
# 3. Tensor basic operations
# 4. Tensor with distribution
# 5. Math operations
# 6. Reduction operations
# 7. Comparison operation
# 8. Matrix, vector multiplication
# 9. Other available operations


# ================================================================== #
#                 1. Default value initialization of Tensor          #
# ================================================================== #

shape = [3,4]
# create zero tensor with shape
zero_tensor = torch.zeros(shape)
# create ones tensor with shape
ones_tensor = torch.ones(shape)
# create random value tensor with shape
random_tensor = torch.rand(shape)
# random int value tensor with range (low,hight,shape)
random_int_tensor = torch.randint(1,12,shape)
# random normal distributed value tensor
random_float_tensor = torch.randn(shape)

# create fixed random value of tensor every run time
torch.manual_seed(42)
fixed_random_tensor = torch.rand(shape)
# print(fixed_random_tensor)

# create a Tensor initialized with a specific array
v = torch.Tensor([[1,2],[4,5]])
# create a Tensor of type Long
v = torch.LongTensor([1,2,3])

# Create random permutation of integers from 0 to 3
v = torch.randperm(4)


# Initialize Tensor with a range of value
v = torch.arange(5)
v = torch.arange(0, 5, step=1)
v = torch.arange(9)
v = v.view(3, 3)

# Create a Tensor with 10 linear points for (1, 10) inclusively
v = torch.linspace(1, 10, steps=10)

# Create a Tensor with logspace values from [-10,10] and step= 5 (1.0e-10 1.0e-05 1.0e+00, 1.0e+05, 1.0e+10)
v = torch.logspace(start=-10, end=10, steps=5)


"""
Other options of tensor creation:
+ torch.eye  
+ torch.ones_like
+ torch.range                   
+ torch.zeros_like
"""

# ================================================================== #
#                 2. Convert tensor to numpy and revert
# ================================================================== #

# numpy to tensor
np_array = np.array([[1,3,4],[0,5,6]])
torch_tensor = torch.from_numpy(np_array)
# create tensor from numpy with defined type of value
torch_tensor = torch.from_numpy(np_array.astype(np.int32))
# print(torch_tensor)

# tensor to numpy
np_array = torch_tensor.numpy()
# print(np_array)


# ================================================================== #
#                 3. Tensor basic operations
# ================================================================== #

x_tensor = torch.tensor([[1,3,4],[0,5,6]])
# Slicing
x_1 = x_tensor[0]
x_2 = x_tensor[1]

# indexing
x_3 = x_tensor[0][0:2]
# print(x_3)

# reshape tensor
y = x_tensor.view(6)
z = x_tensor.view(-1, 2)
u = x_tensor.view(3,2)

"""
Note: Indicating –1  here is telling PyTorch to calculate the number of rows required.
Using it without another dimension simply creates a tensor of a single row. You could
rewrite example two mentioned previously, as follows, if you did not know the input
tensor's shape but know that it needs to have three rows:
"""
v = x_tensor.view(3,-1)

# Swapping axes or transposing
# print(x_tensor)
tranpose = x_tensor.transpose(0,1)
# print(tranpose)

"""
Note: transpose()  can only swap two axes at once. We could use  transpose  in
multiple steps; however, a more convenient way is to use  permute()
"""
a = torch.ones(1,2,3,4)
permuted_tensor = a.permute(3,0,2,1) # swap all axis at once
permuted_tensor = x_tensor.permute(1,0)

# Concatenate multi-tensors
concatenated_tensor_0 = torch.cat((x_tensor, x_tensor, x_tensor), dim= 0)
concatenated_tensor_1 = torch.cat((x_tensor, x_tensor, x_tensor), dim= 1)
# print(concatenated_tensor_0)
# print(concatenated_tensor_1)

# Stack
stacked_tensor = torch.stack((x_tensor, x_tensor))
# print(stacked_tensor)

"""
Note: difference between concatenate and stack is output shape
print(concatenated_tensor_0.shape)
print(stacked_tensor.shape)
"""

# Split a Tensor
splited_tensor = torch.split(x_tensor, 1)

# Create tensor with Index select
indices_1 = torch.tensor([0,2])
indices_2 = torch.tensor([0,1])
indices_3 = torch.tensor([0])

tensor_index_1 = torch.index_select(x_tensor, 1, indices_1) # Select element 0 and 2 for each dimension 1.
tensor_index_2 = torch.index_select(x_tensor, 1, indices_2) # Select element 0 and 1 for each dimension 1.
tensor_index_3 = torch.index_select(x_tensor, 0, indices_3) # Select element 0 for dimension 0.

# Create mask and tensor with selected value by that mask
x = torch.randn(3, 4)
mask = x.ge(0.5)
mask_tensor = torch.masked_select(x, mask)

# Squeeze and unsqueeze

"""
Squeeze: Returns a tensor with all the dimensions of input of size 1 removed.
For example, if input is of shape: (A×1×B×C×1×D) then the out tensor will be of shape: (A×B×C×D)
When dim is given, a squeeze operation is done only in the given dimension. 
For example, If input is of shape: (A×1×B) , squeeze(input, 0) leaves the tensor unchanged, 
but squeeze(input, 1) will squeeze the tensor to the shape (A×B) .
"""
t = torch.ones(2,1,2,1) # Size 2x1x2x1
r = torch.squeeze(t)     # Size 2x2
r = torch.squeeze(t, 1)  # Squeeze dimension 1: Size 2x2x1

# Un-squeeze a dimension
x = torch.Tensor([1, 2, 3])
r = torch.unsqueeze(x, 0)       # Size: 1x3
r = torch.unsqueeze(x, 1)       # Size: 3x1

# Non-zero elements
r = torch.nonzero(v)

# Create new tensor by taking some value indexes from
new_tensor = torch.take(x_tensor, torch.LongTensor([0,2]))

# Remove tensor dimension
removed_tensor_1 = x_tensor.unbind(dim=0)
removed_tensor_2 = x_tensor.unbind(dim=1)


# ================================================================== #
#                 4. Tensor with distribution
# ================================================================== #


# 2x2: A uniform distributed random matrix with range [0, 1]
uniform = torch.Tensor(2, 2).uniform_(0, 1)

# bernoulli
bernoulli = torch.bernoulli(uniform)   # Size: 2x2. Bernoulli with probability p stored in elements of r


# Multinomial
w = torch.Tensor([0, 4, 8, 2]) # Create a tensor of weights
multinomial = torch.multinomial(w, 4, replacement=True) # Size 4: 3, 2, 1, 2

# Normal distribution
# From 10 means and SD
normal_dist_1 = torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
normal_dist_2 = torch.normal(mean=0.5, std=torch.arange(1., 6.))
normal_dist_3 = torch.normal(mean=torch.arange(1., 6.))
normal_dist_4 = torch.normal(2, 3, size=(1, 4))



# ================================================================== #
#                        5. Math operations
# ================================================================== #

f= torch.FloatTensor([-1, -2, 3])
r = torch.abs(f)      # 1 2 3

# Add x, y and scalar 10 to all elements
x = torch.Tensor(2, 3)
y = torch.rand(2, 3)

r = torch.add(x, 10)
r = torch.add(x, 10, y)

# Clamp the value of a Tensor
r = torch.clamp(v, min=-0.5, max=0.5)

# Element-wise divide
r = torch.div(v, v+0.03)

# Element-wise multiple
r = torch.mul(v, v)
"""
Other math operations
**********************************************
torch.abs
torch.acos           - arc cosine
torch.add
torch.addcdiv        - element wise: t1 + s * t2/t3
torch.addcmul        - element wise: t1 + s * t2 * t3
torch.asin           - arc sin
torch.atan
torch.atan2
torch.ceil           - ceiling
torch.clamp          - clamp elements into a range
torch.cos
torch.cosh
torch.div            - divide
torch.erf            - Gaussian error functiom
torch.erfinv         - Inverse
torch.exp
torch.expm1          - exponential of each element minus 1
torch.floor          
torch.fmod           - element wise remainder of division
torch.frac           - fraction part 3.4 -> 0.4
torch.lerp           - linear interpolation
torch.log            - natural log
torch.log1p          - y = log(1 + x)
torch.mul            - multiple
torch.neg 
torch.pow
torch.reciprocal     - 1/x
torch.remainder      - remainder of division
torch.round
torch.rsqrt          - the reciprocal of the square-root 
torch.sigmoid        - sigmode(x)
torch.sign
torch.sin
torch.sinh
torch.sqrt
torch.tan
torch.tanh
torch.trunc          - truncated integer
"""

# ================================================================== #
#                        6. Reduction operations
# ================================================================== #

r = torch.cumsum(v, dim=0)

# L-P norm
r = torch.dist(v, v+3, p=2)  # L-2 norm: ((3^2)*9)^(1/2) = 9.0

# Mean
# 1 4 7
r = torch.mean(v, 1)         # Size 3: Mean in dim 1

r = torch.mean(v, 1, True)   # Size 3x1 since keep dimension = True

# Sum
# 3 12 21
r = torch.sum(v, 1)          # Sum over dim 1

# 36
r = torch.sum(v)

"""
Other reduction operations
*********************************************
torch.cumprod        - accumulate product of elements x1*x2*x3...
torch.cumsum
torch.dist           - L-p norm
torch.mean
torch.median
torch.mode
torch.norm           - L-p norm
torch.prod           - accumulate product
torch.std            - compute standard deviation
torch.sum
torch.var            - variance of all elements

"""

# ================================================================== #
#                        7. Comparison operation
# ================================================================== #

r = torch.eq(v, v)

# Max element with corresponding index
r = torch.max(v, 1)

# sort
r = torch.sort(v, 1)

# k-th and top k
x = torch.arange(1., 6.)
r = torch.kthvalue(x, 4)
r = torch.topk(v, 1)

"""
other compare operations
*****************************************************

Comparison Ops
~~~~~~~~~~~~~~~~~~~~~~
torch.eq             - Compare elements
torch.equal          - True of 2 tensors are the same 
torch.ge             - Element-wise greater or equal comparison
torch.gt
torch.kthvalue       - k-th element
torch.le
torch.lt
torch.max
torch.min
torch.ne
torch.sort
torch.topk           - top k

"""

# ================================================================== #
#                        8. Matrix, vector multiplication
# ================================================================== #

# Dot product
dot_product_result = torch.dot(torch.Tensor([4, 2]), torch.Tensor([3, 1]))

# Matrix X vector
mat = torch.randn(2, 4)
vec = torch.randn(4)
result_1 = torch.mv(mat, vec)

# Matrix + Matrix X vector
M = torch.randn(2)
mat = torch.randn(2, 3)
vec = torch.randn(3)
result_2 = torch.addmv(M, mat, vec)

# Matrix, Matrix products
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 4)
result = torch.mm(mat1, mat2)

# Outer product of vectors
v1 = torch.arange(1, 4)    # Size 3
v2 = torch.arange(1, 3)    # Size 2
result = torch.ger(v1, v2)

"""
Other matrix Operations
**************************************** 
torch.cross           - cross product
torch.diag            - convert vector to diagonal matrix
torch.histc           - histogram
torch.renorm          - renormalize a tensor
torch.trace           - tr(M)
torch.tril            - lower triangle of 2-D matrix 
torch.triu            - uppser triangle
"""

# ================================================================== #
#                       9. Other available operations
# ================================================================== #

"""

Tensors
----------------------------------
torch.is_tensor
torch.is_storage
torch.set_default_tensor_type
torch.numel
torch.set_printoptions

Serialization
----------------------------------
torch.save          - Saves an object to a disk file
torch.load          - Loads an object saved with torch.save() from a file

Parallelism
----------------------------------
torch.get_num_threads - Gets the number of OpenMP threads used for parallelizing CPU operations
torch.set_num_threads

Spectral Ops
~~~~~~~~~~~~~~~~~~~~~~
torch.stft          - Short-time Fourier transform 
torch.hann_window   - Hann window function
torch.hamming_window  - Hamming window function
torch.bartlett_window - Bartlett window function


BLAS and LAPACK Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.addbmm          - Batch add and mulitply matrices nxp + b×n×m X b×m×p -> bxnxp
torch.addmm           - Add and mulitply matrices nxp + n×m X m×p -> nxp
torch.addmv           - Add and matrix, vector multipy n + nxm X m -> n
torch.addr            - Outer product of vectors
torch.baddbmm         - Batch add and mulitply matrices
torch.bmm             - Batch mulitply matrices b×n×m X b×m×p -> b×n×p
torch.btrifact        - LU factorization
torch.btrifact_with_info
torch.btrisolve
torch.btriunpack
torch.dot             - Dot product of 2 tensors
torch.eig             - Eigenvalues and eigenvectors ofsquare matrix
torch.gels            - Solution for least square or p-norm(AX - B)
torch.geqrf
torch.er             - Outer product of 2 vectors
torch.gesv            - Solve linear equations
torch.inverse         - Inverse of square matrix
torch.det             - Determinant of a 2D square Variable
torch.matmul          - Matrix product of tensors
torch.mm				- Matrix multiplication
torch.mv              - Matrix vector product
torch.orgqr           - Orthogal matrix Q 
torch.ormqr           - Multiplies matrix by the orthogonal Q matrix
torch.potrf           - Cholesky decomposition
torch.potri           - Inverse of a positive semidefinite matrix with Cholesky
torch.potrs           - Solve linear equation with positive semidefinite
torch.pstrf           - Cholesky decomposition of a positive semidefinite matrix
torch.qr              - QR decomposition
torch.svd             - SVD decomposition
torch.symeig          - Eigenvalues and eigenvectors
torch.trtrs           - Solves a system of equations with a triangular coefficient
"""


