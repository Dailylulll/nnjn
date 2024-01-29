'''
Save every experiment state into a single dict, into a list, with its index being the key
all the experimental data can go into a pd, a variation and a trial number
what does torchinfo do? third party lib
    #  'preds': [], This is problematic with pandas, just use statistics, and save the pred lists seperatesly
'''

import torch
import pandas as pd
import os
import pickle
import sklearn.metrics as mt


def pred_dict(experiment, trial, preds):
  return {
    'model_file_str': f'{experiment["experiment_var"]}_{experiment["model"]}_{trial}.pth',
    'preds': preds.copy()
  }


def experiment_dict():
  return {
    'model': None,
    'model_init_file': None,
    'optim': None,
    'optim_dict': None,
    'error_f': None,
    'data_type': None,
    'device': None,
    'data_file': None,
    'seed': None,
    'dl_dict': None,  # data loader
    'independent_var': None,
  }


def col_dict():
  return {
    'trial_num': -1,
    'variation': 'none',
    'epoch': -1,
    'error_rate': 0.0,
    'accuracy': 0.0,
    'precision': 0.0,
    'f1': 0.0,
    'recall': 0.0,
    'total_train_time': 0.0,  # in device time,
    'inference_time': 0.0,
    'memory_usage': 0
  }


def initialize_dataframe():
  df = pd.DataFrame(columns=col_dict().keys())
  return df


def add_entry(df: pd.DataFrame, entry_dict: dict):
  df.loc[len(df)] = entry_dict


def save_experiment(experiment):
  cwd = os.getcwd()
  path = os.path.join(cwd, 'experiments')
  path = os.path.join(path, f'{experiment["experiment_var"]}')
  os.makedirs(path, exist_ok=True)
  path = os.path.join(path, f'{experiment["experiment_var"]}/dict.pkl')
  with open(path, 'wb') as file:
    pickle.dump(experiment, file)


def load_experiment(dir):
  cwd = os.getcwd()
  path = os.path.join(cwd, f'experiments/{dir}/dict.pkl')
  with open(path, 'rb') as file:
    exp = pickle.load(file)
  return exp


def save_preds(experiment, preds):
  cwd = os.getcwd()
  path = os.path.join(cwd, 'experiments')
  path = os.path.join(path, f'{experiment["experiment_var"]}/{preds["model_file_str"]}.pkl')
  with open(path, 'wb') as file:
    pickle.dump(preds, file)
  pass


def load_preds(dir, file):
  cwd = os.getcwd()
  path = os.path.join(cwd, f'experiments/{dir}/{file}')
  with open(path, 'rb') as open_file:
    preds = pickle.load(open_file)
  return preds


def load_tensor_dict(file):
  cwd = os.getcwd()
  path = os.path.join(cwd, f'data/{file}')
  return torch.load(path)


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