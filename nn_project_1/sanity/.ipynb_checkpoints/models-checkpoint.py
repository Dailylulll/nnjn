import torch
import torch.nn as nn
import torch.func
import os


"""
Two types of models, 
dict models that take name info, a base class and a layer dict
and a custom model that is a dict that takes a name and a class

need model that returns this is the independent variable str
need flex model that can take sequence dicts for skinny experiments

need model loader!
"""


def get_model(name):
  match name:
    case 'mlp0':
      return mlp0
    case 'mlp1':
      
  if name == 'mlp0':
      return mlp0
  elif name == 'mlp1':
      return mlp1
  elif name == 'mlp1_4':
      return mlp1_4
  elif name == 'mlp2_4':
      return newclass
  else:
      raise ValueError('name')


class mlp0(nn.Module):
  name = 'mlp0'

  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(2, 2)

  def forward(self, x):
    x = self.lin1(x)
    return torch.nn.functional.softmax(x, dim=1)

class mlp1(nn.Module):
  name = 'mlp1'

  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(2, 4)
    self.lin2 = nn.Linear(4, 4)
    self.lin3 = nn.Linear(4, 2)

  def forward(self, x):
    x = nn.functional.sigmoid(self.lin1(x))
    x = nn.functional.sigmoid(self.lin2(x))
    return torch.nn.functional.softmax(self.lin3(x), dim=1)

class mlp1_4(nn.Module):
  name = 'mlp1_4'

  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(2, 4)
    self.lin2 = nn.Linear(4, 4)
    self.lin3 = nn.Linear(4, 4)

  def forward(self, x):
    x = nn.functional.sigmoid(self.lin1(x))
    x = nn.functional.sigmoid(self.lin2(x))
    return torch.nn.functional.softmax(self.lin3(x), dim=1)

class newclass(nn.Module):
  name = 'mlp2_4'

  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(2, 4)
    self.lin2 = nn.Linear(4, 4)
    self.lin3 = nn.Linear(4, 4)

  def forward(self, x, alpha):
    x = CustomSigmoid.apply(self.lin1(x), alpha)
    x = CustomSigmoid.apply(self.lin2(x), alpha)
    return torch.nn.functional.softmax(self.lin3(x), dim=1)

  def loop_alpha(epoch):
    return 0.75 + (torch.sin(torch.tensor(epoch)) + 1) *.25

class CustomSigmoid(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, alpha):
      ctx.alpha = alpha
      ctx.save_for_backward(input)
      return 1 / (1 + torch.exp(-ctx.alpha * input))

  @staticmethod
  def backward(ctx, grad_output):
      input, = ctx.saved_tensors
      alpha = ctx.alpha
      sigmoid_output = 1 / (1 + torch.exp(-alpha * input))
      grad_input = grad_output * sigmoid_output * (1 - sigmoid_output) * alpha
      return grad_input, None  # None because alpha is not a learned parameter


def save_trained_model(model, experiment, trial):
  cwd = os.getcwd()
  path = os.path.join(cwd, 'trained_models')
  path = os.path.join(path, f'{experiment["experiment_var"]}_{experiment["model"]}_{trial}.pth')
  print(path)
  torch.save(model.state_dict(), path)


def save_init_model(model):
  cwd = os.getcwd()
  path = os.path.join(cwd, 'start_models')
  for i in range(100):
    tmp_path = os.path.join(path, f'{model.name}_{i}.pth')
    if not os.path.exists(tmp_path):
      torch.save(model.state_dict(), tmp_path)
      break


def load_trained_model(file):
  cwd = os.getcwd()
  path = os.path.join(cwd, 'trained_models')
  path = os.path.join(path, file)
  return torch.load(path)


def load_init_model(model):
  cwd = os.getcwd()
  path = os.path.join(cwd, 'start_models')
  path = os.path.join(path, f'{model}.pth')
  return torch.load(path)
