# --------------------------------------------------------
# Basic pytorch distance and loss funtion
# Written by Huy Thanh Nguyen (kaizen0890@gmail.com)
# github:
# --------------------------------------------------------


import torch
import torch.nn as nn
from torch.autograd import Variable

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Distance function
# 2. Loss function


# ================================================================== #
#                       1. Distance function
# ================================================================== #


# Cosine Similarity
input1 = torch.Variable(torch.randn(100, 128))
input2 = Variable(torch.randn(100, 128))
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)


# ================================================================== #
#                       2. Loss function
# ================================================================== #


# L1 Loss
loss = nn.L1Loss()
input = torch.autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = torch.autograd.Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()

# Mean Square Error loss
loss = nn.MSELoss()
input = Variable(torch.randn(3, 5), requires_grad=True)
target = Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()

# Cross Entropy loss
loss = nn.CrossEntropyLoss()
input = torch.autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = torch.autograd.Variable(torch.LongTensor(3).random_(5))
output = loss(input, target)
output.backward()

# Negative Log-likehood loss
m = nn.LogSoftmax()
loss = nn.NLLLoss()
# input is of size N x C = 3 x 5
input = Variable(torch.randn(3, 5), requires_grad=True)
# each element in target has to have 0 <= value < C
target = torch.autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(m(input), target)
output.backward()

# Kullback-Leibler Divergence loss
a2 = torch.FloatTensor([0.3, 0.6, 0.1])
a1 = torch.FloatTensor([0.4, 0.5, 0.1])
c1 = torch.nn.KLDivLoss(size_average=False)(a1.log(),a2) # (a2*(a2.log()-a1.log())).sum()




"""
Other loss functions

************************************************

nn.functional.binary_cross_entropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.poisson_nll_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.cosine_embedding_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.cross_entropy
~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.hinge_embedding_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.kl_div
~~~~~~~~~~~~~~~~
nn.functional.l1_loss
~~~~~~~~~~~~~~~~~
nn.functional.mse_loss
~~~~~~~~~~~~~~~~~~
nn.functional.margin_ranking_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.multilabel_margin_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.multilabel_soft_margin_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.multi_margin_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.nll_loss
~~~~~~~~~~~~~~~~~~
nn.functional.binary_cross_entropy_with_logits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.smooth_l1_loss
~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.soft_margin_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~
nn.functional.triplet_margin_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
