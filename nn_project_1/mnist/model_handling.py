'''
saving and loading functions
when saving or loading a model, the file_name will be the variation, the model num, and the trial, the data will be the
state dict
'''

import torch
import os


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