import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


'''load data into pandas, and use seaborne to display data'''


'''
accuracy, precision, f1 and recall,
error rate change over time'''


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


def plot_multiclass_confusion_matrix(actual, predicted, class_names):
    cm = confusion_matrix(actual, predicted, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Plotting
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


'''
seaborn?
'''