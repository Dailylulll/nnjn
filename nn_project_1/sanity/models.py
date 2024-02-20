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
      return mlp1
    case 'mlp1_4':
      return mlp1_4
    case 'cnn0':
      return cnn0
    case 'cnn0_4':
      return cnn0_4
    case 'rbf0':
      return rbf0
    case 'rbf0_4':
      return rbf0_4
    case _:
      raise ValueError('name')


class rbf_node(nn.Module):
    def __init__(self, center, sigma=1.0):
        super().__init__()
        self.center =  nn.Parameter(torch.tensor(center), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=False)
    
    def forward(self, x):
      dist = torch.sum((x - self.center) ** 2, axis=1)
      return torch.exp(-dist / (2 * self.sigma ** 2))
  
    
class rbf0(nn.Module):
  name = 'rbf0'

  def __init__(self):
    super().__init__()
    self.rbfs = None
    self.lin1 = nn.Linear(5, 2)

  def post_init(self, centers, sigmas):
    self.rbfs = nn.ModuleList([rbf_node(*z) for z in zip(centers, sigmas)])

  def forward(self, x):
    x = [rbf(x) for rbf in self.rbfs]
    x = torch.stack(x, dim=1)
    x = self.lin1(x)
    return nn.functional.softmax(x, dim=1)
  
class rbf0_4(nn.Module):
  name = 'rbf0_4'

  def __init__(self):
    super().__init__()
    self.rbfs = None
    self.lin1 = nn.Linear(5, 4)

  def post_init(self, centers, sigmas):
    self.rbfs = nn.ModuleList([rbf_node(*z) for z in zip(centers, sigmas)])

  def forward(self, x):
    x = [rbf(x) for rbf in self.rbfs]
    x = torch.stack(x, dim=1)
    x = self.lin1(x)
    return nn.functional.softmax(x, dim=1)
    

class cnn0(nn.Module):
  name = 'cnn0'

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv1d(in_channels=1, 
                           out_channels=1, 
                           kernel_size=2, 
                           stride=1, 
                           padding=1)
    self.lin1 = nn.Linear(3, 2)

  def forward(self, x):
    x = x.unsqueeze(1)
    x = nn.functional.sigmoid(self.conv1(x))
    x = x.view(x.size(0), -1)
    x = self.lin1(x)
    return nn.functional.softmax(x, dim=1)

class cnn0_4(nn.Module):
  name = 'cnn0_4'

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv1d(in_channels=1, 
                           out_channels=1, 
                           kernel_size=2, 
                           stride=1, 
                           padding=1)
    self.lin1 = nn.Linear(3, 4)

  def forward(self, x):
    x = x.unsqueeze(1)
    x = nn.functional.sigmoid(self.conv1(x))
    x = x.view(x.size(0), -1)
    x = self.lin1(x)
    return nn.functional.softmax(x, dim=1)
  
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
    x = torch.flatten(self.lin3(x))
    return torch.nn.functional.softmax(x, dim=1)
  

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
