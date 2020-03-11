# --------------------------------------------------------
# Basic pytorch nn modules APIs
# Written by Huy Thanh Nguyen (kaizen0890@gmail.com)
# github:
# --------------------------------------------------------


import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Convolution layers
# 2. Pooling layers
# 3. Padding layers
# 4. Non- Linear activations
# 5. Normalization layers
# 6. Recurrent layers
# 7. Linear Layer
# 8. Vision layers
# 9. Initialization of parameters
# 10. Utilities



# ================================================================== #
#                   1. Convolution layers
# ================================================================== #

# Convolution 1D
conv1D_layer = nn.Conv1d(16, 33, 3, stride=2)
input_tensor = Variable(torch.randn(20, 16, 50))
output = conv1D_layer(input_tensor)

# Convolution 2D
conv2D_layer = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input_tensor = Variable(torch.randn(20, 16, 50, 100))
output = conv2D_layer(input_tensor)

# Convolution 3D
conv3D_layer = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
input_tensor = Variable(torch.randn(20, 16, 10, 50, 100))
output = conv3D_layer(input)


"""

Convolution Layers
----------------------------------
torch.nn.functional.Conv1d`
~~~~~~~~~~~~~~~~
torch.nn.functional.Conv2d`
~~~~~~~~~~~~~~~~
torch.nn.functional.Conv3d`
~~~~~~~~~~~~~~~~
torch.nn.functional.ConvTranspose1d`
~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.ConvTranspose2d`
~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.ConvTranspose3d`
~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# ================================================================== #
#                      2. Pooling layers
# ================================================================== #


# Max pooling layer
m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = Variable(torch.randn(20, 16, 50, 32))
output = m(input)

# Max Unpooling layer
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)
input = Variable(torch.Tensor([[[[ 1,  2,  3,  4],
                                 [ 5,  6,  7,  8],
                                 [ 9, 10, 11, 12],
                                 [13, 14, 15, 16]]]]))
output, indices = pool(input)
unpool(output, indices)

# Average pooling layer
m = nn.AvgPool2d((3, 2), stride=(2, 1))
input = Variable(torch.randn(20, 16, 50, 32))
output = m(input)


"""
Other pooling layer functions

----------------------------------
torch.nn.functional.MaxPool1d`
~~~~~~~~~~~~~~~~~~~
torch.nn.functional.MaxPool2d`
~~~~~~~~~~~~~~~~~~~
torch.nn.functional.MaxPool3d`
~~~~~~~~~~~~~~~~~~~
torch.nn.functional.MaxUnpool1d`
~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.MaxUnpool2d`
~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.MaxUnpool3d`
~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.AvgPool1d`
~~~~~~~~~~~~~~~~~~~
torch.nn.functional.AvgPool2d`
~~~~~~~~~~~~~~~~~~~
torch.nn.functional.AvgPool3d`
~~~~~~~~~~~~~~~~~~~
torch.nn.functional.FractionalMaxPool2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.LPPool2d`
~~~~~~~~~~~~~~~~~~
torch.nn.functional.AdaptiveMaxPool1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.AdaptiveMaxPool2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.AdaptiveMaxPool3d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.AdaptiveAvgPool1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.AdaptiveAvgPool2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.AdaptiveAvgPool3d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""


# ================================================================== #
#                          3. Padding layers
# ================================================================== #


# Replication padding
m = nn.ReplicationPad2d(3)
input = Variable(torch.randn(16, 3, 320, 480))
output = m(input)

# Zero-Padding
m = nn.ZeroPad2d(3)
input = Variable(torch.randn(16, 3, 320, 480))
output = m(input)


"""

Padding Layers
--------------

torch.nn.functional.ReflectionPad2d`
~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.ReplicationPad2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.ReplicationPad3d`
~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.ZeroPad2d`
~~~~~~~~~~~~~~~~~~~
torch.nn.functional.ConstantPad2d`
~~~~~~~~~~~~~~~~~~~~~~~

"""


# ================================================================== #
#                     4. Non-Linear activations
# ================================================================== #

# Relu
m = nn.ReLU()
input = Variable(torch.randn(2))
output = m(input)

# Leaky ReLU
m = nn.LeakyReLU(0.1)
input = Variable(torch.randn(2))
output = (m(input))

# Sigmoid
m = nn.Sigmoid()
input = Variable(torch.randn(2))
output = (m(input))

# Softplus
m = nn.Softplus()
input = Variable(torch.randn(2))
output = (m(input))

# Softmax
m = nn.Softmax()
input = Variable(torch.randn(2, 3))
output = (m(input))


"""
Other Non-linear activation functions
-------------------------------
nn.functional.threshold`
~~~~~~~~~~~~~~~~~~~
nn.functional.relu`
~~~~~~~~~~~~~~
nn.functional.hardtanh`
~~~~~~~~~~~~~~~~~~
nn.functional.relu6`
~~~~~~~~~~~~~~~
nn.functional.elu`
~~~~~~~~~~~~~
nn.functional.selu`
~~~~~~~~~~~~~~
nn.functional.leaky_relu`
~~~~~~~~~~~~~~~~~~~~
nn.functional.prelu`
~~~~~~~~~~~~~~~
nn.functional.rrelu`
~~~~~~~~~~~~~~~
nn.functional.glu`
~~~~~~~~~~~~~~~
nn.functional.logsigmoid`
~~~~~~~~~~~~~~~~~~~~
nn.functional.hardshrink`
~~~~~~~~~~~~~~~~~~~~
nn.functional.tanhshrink`
~~~~~~~~~~~~~~~~~~~~
nn.functional.softsign`
~~~~~~~~~~~~~~~~~~
nn.functional.softplus`
~~~~~~~~~~~~~~~~~~
nn.functional.softmin`
~~~~~~~~~~~~~~~~~
nn.functional.softmax`
~~~~~~~~~~~~~~~~~
nn.functional.softshrink`
~~~~~~~~~~~~~~~~~~~~
nn.functional.log_softmax`
~~~~~~~~~~~~~~~~~~~~~
nn.functional.tanh`
~~~~~~~~~~~~~~
nn.functional.sigmoid`
~~~~~~~~~~~~~~~~~

"""

# ================================================================== #
#                       5. Normalization layers
# ================================================================== #

# BatchNorm 1D
m = nn.BatchNorm1d(100)
# Without Learnable Parameters
m = nn.BatchNorm1d(100, affine=False)
input = torch.autograd.Variable(torch.randn(20, 100))
output = m(input)

# BatchNorm 2D
# With Learnable Parameters
m = nn.BatchNorm2d(100)
# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=False)
input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
output = m(input)

"""
Normalization layers
----------------------------------
torch.nn.functional.BatchNorm1d`
~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.BatchNorm2d`
~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.BatchNorm3d`
~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.InstanceNorm1d`
~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.InstanceNorm2d`
~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.InstanceNorm3d`
~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.functional.LocalResponseNorm`
~~~~~~~~~~~~~~~~~~~~~~~~

"""


# ================================================================== #
#                       6. Recurrent layers
# ================================================================== #

# RNN
rnn = nn.RNN(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, h0)
# LSTM
rnn = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, (h0, c0))
# GRU
rnn = nn.GRU(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, h0)
# RNNCell
rnn = nn.RNNCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)
# LSTMCell
rnn = nn.LSTMCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
cx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)
# GRUCell
rnn = nn.GRUCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)


"""

Recurrent layers
----------------------------------
torch.nn.functional.RNN`
~~~~~~~~~~~~~
torch.nn.functional.LSTM`
~~~~~~~~~~~~~~
torch.nn.functional.GRU`
~~~~~~~~~~~~~
torch.nn.functional.RNNCell`
~~~~~~~~~~~~~~~~~
torch.nn.functional.LSTMCell`
~~~~~~~~~~~~~~~~~~
torch.nn.functional.GRUCell`
~~~~~~~~~~~~~~~~~

"""

# ================================================================== #
#                            7. Linear Layer
# ================================================================== #

# Linear
m = nn.Linear(20, 30)
input = Variable(torch.randn(128, 20))
output = m(input)
print(output.size())
# Dropout
m = nn.Dropout(p=0.2)
input = torch.autograd.Variable(torch.randn(20, 16))
output = m(input)





"""

Linear layers
----------------------------------
torch.nn.functional.Linear`
~~~~~~~~~~~~~~~~
torch.nn.functional.Bilinear`
~~~~~~~~~~~~~~~~~~



Dropout layers
----------------------------------
torch.nn.functional.Dropout`
~~~~~~~~~~~~~~~~~
torch.nn.functional.Dropout2d`
~~~~~~~~~~~~~~~~~~~
torch.nn.functional.Dropout3d`
~~~~~~~~~~~~~~~~~~~
torch.nn.functional.AlphaDropout`
~~~~~~~~~~~~~~~~~~~~~~


"""
# ================================================================== #
#                       8. Vision layers
# ================================================================== #


# pixel shuffle
ps = nn.PixelShuffle(3)
input = torch.autograd.Variable(torch.Tensor(1, 9, 4, 4))
output = ps(input)

# Upsampling
m = nn.Upsample(scale_factor=2, mode='bilinear')

# Upsampling Bilinear 3D
m = nn.UpsamplingBilinear2d(scale_factor=2)



"""

Other Vision functions
----------------
nn.functional.pixel_shuffle`
~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.pad`
~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.upsample`
~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.upsample_nearest`
~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.upsample_bilinear`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.grid_sample`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.affine_grid`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# ================================================================== #
#                 9. Initialization of parameters (torch.nn.init)
# ================================================================== #

# uniform init
w = torch.Tensor(3, 5)
nn.init.uniform(w)

# normal init
w = torch.Tensor(3, 5)
nn.init.normal(w)

# Constant init
w = torch.Tensor(3, 5)
nn.init.constant(w, 0.3)

# xavier uniform init
w = torch.Tensor(3, 5)
nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('relu'))

# xavier normal init
w = torch.Tensor(3, 5)
nn.init.xavier_normal(w)

# Kaiming normal init
w = torch.Tensor(3, 5)
nn.init.kaiming_normal(w, mode='fan_out')

"""

Other initialization

***********************************************
torch.nn.init.calculate_gain
torch.nn.init.uniform
torch.nn.init.normal
torch.nn.init.constant
torch.nn.init.eye
torch.nn.init.dirac
torch.nn.init.xavier_uniform
torch.nn.init.xavier_normal
torch.nn.init.kaiming_uniform
torch.nn.init.kaiming_normal
torch.nn.init.orthogonal

"""




# ================================================================== #
#                       10. Utilities
# ================================================================== #


"""

Utilities
---------
torch.nn.utils.clip_grad_norm`
~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.utils.weight_norm`
~~~~~~~~~~~~~~~~~~~~~
torch.nn.utils.remove_weight_norm`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.utils.PackedSequence`
~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.utils.pack_padded_sequence`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.utils.pad_packed_sequence`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.nn.utils.pad_sequence`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:hidden:`pack_sequence`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""







