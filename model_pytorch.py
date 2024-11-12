from ultralytics import YOLO
import torch
import torch.nn as nn
from torchinfo import summary
from collections import OrderedDict

model = YOLO('weights/yolo11x-cls.pt')

unfrozen_layers = ['model.10.linear.weight', 'model.10.linear.bias']

for name, param in model.model.named_parameters():
    # if not name in unfrozen_layers:
    param.requires_grad = False
    # else:
    #     param.requires_grad = True

# layers = list(model.model.children())[:-1]

# # model.model = nn.Sequential(*layers, nn.Sequential(
# #     nn.Linear(1280, 512),
# #     nn.ReLU(),
# #     nn.Linear(512, 256),
# #     nn.ReLU(),
# #     nn.Linear(256, 5)
# # ))

# new_layers = OrderedDict(layers)
# new_layers['output_layer'] = 

# # Update the model with the new OrderedDict
# model.model = nn.Sequential(new_layers)

# print(list(model.model.children())[-1])

# new_fc_layer = torch.nn.Linear(in_features=model.model[9].out_channels, out_features=100)  # Example layer


model.model.add_module('output_regression_layer', nn.Sequential(
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 5)
))

model.train()

# print(model.model)