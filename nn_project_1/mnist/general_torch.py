import random
import numpy as np
import torch

# need get optim and error functions that take string and return functions....
def init_seeds(seed, device):
  random.seed(seed),
  np.random.seed(seed),
  torch.manual_seed(seed),
  if device == 'cuda':
    torch.cuda.manual_seed_all(seed)