import torch
import torch.nn as nn
import torch.optim as optim
import torch.func


"""
Two types of models, 
dict models that take name info, a base class and a layer dict
and a custom model that is a dict that takes a name and a class
"""
class Model0_base(nn.Module):
  def __str__(self):
    return 'Model0'

  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(1, 10)
    self.lin2 = nn.Linear(10, 10)
    self.lin3 = nn.Linear(10, 1)

  def forward(self, x):
    x = self.lin1(x)
    x = 2 * nn.functional.sigmoid(x) - 1
    x = self.lin2(x)
    x = 2 * nn.functional.sigmoid(x) - 1
    return self.lin3(x)

class Model0_1(nn.Module):
  def __str__(self):
    return 'Model0'

  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(1, 20)
    self.lin2 = nn.Linear(20, 20)
    self.lin3 = nn.Linear(20, 1)

  def forward(self, x):
    x = self.lin1(x)
    x = 2 * nn.functional.sigmoid(x) - 1
    x = self.lin2(x)
    x = 2 * nn.functional.sigmoid(x) - 1
    return self.lin3(x)

class Model0_2(nn.Module):
  def __str__(self):
    return 'Model0'

  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(1, 50)
    self.lin2 = nn.Linear(50, 50)
    self.lin3 = nn.Linear(50, 1)

  def forward(self, x):
    x = self.lin1(x)
    x = nn.functional.sigmoid(x)
    x = self.lin2(x)
    x = nn.functional.sigmoid(x)
    return self.lin3(x)

class Model0_3(nn.Module):
  def __str__(self):
    return 'Model0'

  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(1, 10)
