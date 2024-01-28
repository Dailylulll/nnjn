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
    case('model0'):
      return Model0
    case('variable'):
      return Variable
    case _:
      raise ValueError('name')


class Variable:
  def __str__(self):
    return 'Model is the hyperparameter'


class Model0(nn.Module):
  def __str__(self):
    return 'Model0'

  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(1, 10)
    self.lin2 = nn.Linear(10, 10)
    self.lin3 = nn.Linear(10, 1)

  def forward(self, x):
    x = self.lin1(x)
    x = nn.functional.sigmoid(x)
    x = self.lin2(x)
    x = nn.functional.sigmoid(x)
    return self.lin3(x)


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
    tmp_path = os.path.join(path, f'{model.name()}_init{i}.pth')
    if not os.path.exists(tmp_path):
      torch.save(model.state_dict(), tmp_path)
      break


def load_trained_model(file):
  cwd = os.getcwd()
  path = os.path.join(cwd, 'trained_models')
  path = os.path.join(path, file)
  return torch.load(path)


def load_init_model(file):
  cwd = os.getcwd()
  path = os.path.join(cwd, 'start_models')
  path = os.path.join(path, file)
  return torch.load(path)
