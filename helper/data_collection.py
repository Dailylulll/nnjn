'''
Save every experiment state into a single dict, into a list, with its index being the key
all the experimental data can go into a pd, a variation and a trial number
what does torchinfo do? third party lib
    #  'preds': [], This is problematic with pandas, just use statistics, and save the pred lists seperatesly
'''

import pandas as pd

def pred_dict(experiment, trial, preds):
  return {
    'model_file_str': f'{experiment["experiment_var"]}_{experiment["model"]}_{trial}.pth',
    'preds': preds.copy()
  }

def experiment_dict():
  return {
    'data_frame': None,
    'model': None,
    'optimizer_dict': None,
    'error_function': None,
    'data_type': None,
    'device': None,
    'data_set': None,
    'data_loader_seed': None,
    'data_batch_size': None,
    'random_batch': False,
    'experiment_var': None,
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


def save_experiment():
  pass


def load_experiment():
  pass

def save_preds():
  pass

def load_preds():
  pass
