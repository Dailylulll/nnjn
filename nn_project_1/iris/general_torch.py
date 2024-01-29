import random
import numpy as np
import torch
import sklearn.metrics as mt


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


def write_train_info(exp, writer, model, t, p, error, file):
  if exp['model_type'] == 'classification':
    t = torch.stack(t, dim=0)
    p = torch.stack(p, dim=0)
    target = torch.argmax(t, dim=1).tolist()
    pred = torch.argmax(p, dim=1).tolist()
    writer.add_scalar(f'{file}/accuracy', mt.accuracy_score(target, pred, average='weighted'))
    writer.add_scalar(f'{file}/precision', mt.precision_score(target, pred, average='weighted'))
    writer.add_scalar(f'{file}/f1', mt.f1_score(target, pred, average='weighted'))
    writer.add_scalar(f'{file}/recall', mt.recall_score(target, pred, average='weighted'))
  writer.add_scalar(f'{file}/error', error.item())
  if file == 'train':
    weights = []
    grads = []
    for param in model.parameters():
      if param.grad is not None:
        grads.append(param.grad.item())
      weights.append(param.data.item())
    writer.add_histogram(f'{file}/gradients', grads)
    writer.add_histogram(f'{file}/weights', weights)


def write_test_info(exp, writer, t, p, error, file):
  if exp['model_type'] == 'classification':
    t = torch.stack(t, dim=0)
    p = torch.stack(p, dim=0)
    target = torch.argmax(t, dim=1).tolist()
    pred = torch.argmax(p, dim=1).tolist()
    writer.add_scalar(f'{file}/accuracy', mt.accuracy_score(target, pred, average='weighted'))
    writer.add_scalar(f'{file}/precision', mt.precision_score(target, pred, average='weighted'))
    writer.add_scalar(f'{file}/f1', mt.f1_score(target, pred, average='weighted'))
    writer.add_scalar(f'{file}/recall', mt.recall_score(target, pred, average='weighted'))
  writer.add_scalar(f'{file}/error', error.item())
