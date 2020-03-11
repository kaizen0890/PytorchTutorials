# --------------------------------------------------------
# checkpoint example with pytorch
# Written by Huy Thanh Nguyen (kaizen0890@gmail.com)
# github:
# --------------------------------------------------------
import torch
import torchvision
import logging
from torchvision.models import resnet18, mobilenet_v2
import torchsummary.torchsummary



PRETRAIN_PATH = "./resnet18-5c106cde.pth"


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Load pretrain model
# 2. Unfreeze or freeze some layer of pretrained model
# 3. Print state_dict of each layer of the model
# 4. Customize pretrained-model
# 5. Summary model with torchsummary



# ================================================================== #
#                      1. Load pretrain model
# ================================================================== #


logger = logging.getLogger('global')

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    if not torch.cuda.is_available():
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        check_keys(model, pretrained_dict)
    except:
        logger.info('[Warning]: using pretrain as features. Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


resnet18_model = resnet18(pretrained=True,progress=True)
load_pretrain(resnet18_model, PRETRAIN_PATH)

######### print struct of model #########
# for child in resnet18_model.children():
#     print(child)



# ================================================================== #
#        2. Unfreeze or freeze some layer of pretrained model
# ================================================================== #

######### Freeze all layer of the pretrained-model #########

for child in resnet18_model.children():
    for param in child.parameters():
        param.requires_grad = False


######### Freeze selected layer of the pretrained-model #########

## Freeze layers from 5 to end layer of resnet18

for i, child in enumerate(resnet18_model.children()):
    if i > 5:
        for param in child.parameters():
            param.requires_grad = True

# Print to check
# for child in resnet18_model.children():
#     for param in child.parameters():
#         print(param.requires_grad)


#### Freeze from 0th to 13th layers of mobilenetV2 pretrained model

mobilenetV2_model = mobilenet_v2(pretrained=True, progress=True)
for i, child in enumerate(mobilenetV2_model.children()):
    for param in child[:-5].parameters():
        param.requires_grad = False

# # Print to check
# for child in resnet18_model.children():
#     for param in child.parameters():
#         print(param.requires_grad)


"""
NOTE: We even freeze or unfreeze weight or bias of layers:
net.fc2.weight.requires_grad = True
net.fc2.bias.requires_grad = False

"""


# ================================================================== #
#        3. Print state_dict of each layer of the model
# ================================================================== #

"""
NOTE:
What is a state_dict?
In PyTorch, the learnable parameters(i.e. weights and biases) of an torch.nn.Module model 
are contained in the model’s parameters (accessed with model.parameters()). 
A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor. 
Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) 
and registered buffers (batchnorm’s running_mean) have entries in the model’s state_dict. 
Optimizer objects (torch.optim) also have a state_dict, which contains information about the 
optimizer’s state, as well as the hyperparameters used.
Because state_dict objects are Python dictionaries, they can be easily saved, updated, altered, 
and restored, adding a great deal of modularity to PyTorch models and optimizers.

https://pytorch.org/tutorials/beginner/saving_loading_models.html

"""


## Print model's state_dict
for child in mobilenetV2_model.children():
    for param_tensor in child.state_dict():
        print(param_tensor, "\t", child.state_dict()[param_tensor].size())

"""


# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

"""

# ================================================================== #
#                4. Customize pretrained-model
# ================================================================== #

##### Remove some last layers from pretrained model:

# Example with resNet18
### Remove two last layers from Original ResNet18
customized_resNet18 = torch.nn.Sequential(*list(resnet18_model.children())[:-2])

## Replace last layer by Linear layer
num_ftrs = resnet18_model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
resnet18_model.fc = torch.nn.Linear(num_ftrs, 2)


"""
Another way to replace classify layer:
*****************************************************************
num_ftrs = resnet18_model.classifier[1].in_features
resnet18_model.classifier[1] = nn.Linear(model_ft.last_channel, 2)

"""

# ================================================================== #
#                5. Summary model with torchsummary
# ================================================================== #

print(torchsummary.summary(resnet18_model,input_size=(3,127,127)))

