import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import sklearn.metrics as mt


'''load data into pandas, and use seaborne to display data'''

def list_unbatch(list_of_batched_tensors):
  return torch.cat(list_of_batched_tensors, dim=0)

def get_one_hot_encoding(float_encoded_tensor):
  return torch.argmax(float_encoded_tensor, dim=1)

def unbatch_and_one_hot(list_of_batched_tensors):
  return get_one_hot_encoding(list_unbatch(list_of_batched_tensors))

def write_classification_train_info(writer, model, t, p, error, file, epoch):
  target = unbatch_and_one_hot(t)
  pred = unbatch_and_one_hot(p)
  writer.add_scalar(f'{file}/accuracy', mt.accuracy_score(target, pred, normalize=True), epoch)
  writer.add_scalar(f'{file}/precision', mt.precision_score(target, pred, average='weighted', zero_division=0), epoch)
  writer.add_scalar(f'{file}/f1', mt.f1_score(target, pred, average='weighted'), epoch)
  writer.add_scalar(f'{file}/recall', mt.recall_score(target, pred, average='weighted'), epoch)
  writer.add_scalar(f'{file}/error', error.item(), epoch)
  if file == 'train':
    for name, param in model.named_parameters():
      if param.requires_grad: 
        writer.add_histogram(f'{file}/grads', param.grad, epoch)
      writer.add_histogram(f'{file}/weights', param, epoch)


def write_classification_test_info(writer, hparams, t, p, error):
  target = unbatch_and_one_hot(t)
  pred = unbatch_and_one_hot(p)
  e = error.item()
  acc = mt.accuracy_score(target, pred, normalize=True)
  precision = mt.precision_score(target, pred, average='weighted', zero_division=0)
  f1 = mt.f1_score(target, pred, average='weighted')
  recall = mt.recall_score(target, pred, average='weighted')
  writer.add_hparams(hparams, 
                           metric_dict={
                             "accuracy":acc,
                             "precision":precision,
                             "f1":f1,
                             "recall":recall,
                             "error":e
                           })


def line_graph(x, y_list, variation_list, title, ylabel, shape=(10, 6)):
  plt.figure(shape)
  for i in range(y_list):
    plt.plot(x, y_list[i], marker='o', linestyle='-', label=variation_list[i])
  plt.title(title)
  plt.xlabel('Epochs')
  plt.ylabel(ylabel)
  plt.legend()
  plt.grid(True)
  plt.show()


'''bar chart for free params, device time, memory size, and accuracy'''


def bar_plot(y_list, x_list, variation_list, title, ylabel, xlabel, shape=(10, 6)):
  plt.figure(figsize=shape)
  for i in range(x_list):
    plt.bar(variation_list[i], y_list[i])
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()


def img_show(image, label, pred=None):
  plt.imshow(image, cmap='gray')
  if pred is None:
    plt.title(f"Label: {label}")
  else:
    plt.title(f"Label: {label} Pred: {pred}")
  plt.show()


def multi_img_show(images, labels, preds=None, shape=(30, 20)):
  cols = 5
  rows = int(len(images) / cols) + 1
  plt.figure(figsize=shape)
  index = 1
  if preds is None:
    for x in zip(images, labels):
      image = x[0]
      plt.subplot(rows, cols, index)
      plt.imshow(image, cmap='gray')
      plt.title(f'label: {x[1]}', fontsize=15)
      index += 1
  else:
    for x in zip(images, labels, preds):
      image = x[0]
      plt.subplot(rows, cols, index)
      plt.imshow(image, cmap='gray')
      plt.title(f'l: {x[1]} p: {x[2]}', fontsize=15)
      index += 1


def plot_multiclass_confusion_matrix(t, p, class_names):  # todo
    t = unbatch_and_one_hot(t).tolist()
    p = unbatch_and_one_hot(p).tolist()
    cm = confusion_matrix(t, p, labels=class_names)
    '''cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)'''

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plotting using the created axes
    sns.heatmap(cm, annot=False, fmt='g', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    # Return the figure object
    return fig


def plot_decision_boundary(exp, clf, X, y, step_size=0.01):
    """
    Plots the decision boundary for a classifier.

    Parameters:
    clf: A trained classifier with a .predict() method.
    X: 2D numpy array of data points.
    y: 1D numpy array of labels.
    step_size: The resolution of the decision boundary (default: 0.01).
    """

    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))

    # Predict the function value for the whole grid
    mesh = np.c_[xx.ravel(), yy.ravel()]
    t_mesh = torch.tensor(mesh, requires_grad=False).to(exp['data_type']).to(exp['device'])
    Z = clf(t_mesh)
    Z = torch.argmax(Z, dim=1)
    Z = Z.reshape(xx.shape).detach().numpy()
    y = torch.argmax(y, dim=1)

    # Plot the contour and training examples
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(xx, yy, Z, alpha=0.5, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary')

    # Show plot
    return fig

# Example usage:
# Assume clf is your trained classifier, and X_train and y_train are your data and labels
# plot_decision_boundary(clf, X_train, y_train)




'''
seaborn?
give chatgpt data format and ask it to generate cool seaborn graphs
'''