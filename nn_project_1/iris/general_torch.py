import random
import numpy as np
import torch


def init_seeds(seed, device):
  random.seed(seed),
  np.random.seed(seed),
  torch.manual_seed(seed),
  if device == 'cuda':
    torch.cuda.manual_seed_all(seed)


def get_optim(name):
  match name:
    case 'sgd':
      return torch.optim.SGD
    case 'adam':
      return torch.optim.Adam
    case _:
      raise ValueError('name')


def get_error(name):
  match name:
    case 'mse':
      return torch.nn.MSELoss
    case 'mae':
      return torch.nn.L1Loss
    case 'bce':
      return torch.nn.BCELoss
    case _:
      raise ValueError('name')




