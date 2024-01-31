import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import sklearn.metrics as mt


'''load data into pandas, and use seaborne to display data'''


'''
accuracy, precision, f1 and recall,
error rate change over time'''


def initialize_class_dataframe():
  """ Initialize a DataFrame with specified columns. """
  columns = ['Run', 'Accuracy', 'Precision', 'F1_Score', 'Recall', 'Error']
  return pd.DataFrame(columns=columns)


def add_class_data(df, key, accuracy, precision, f1_score, recall, error):
  """ Add a new row of data to the DataFrame using row indexing. """
  df.loc[len(df)] = [key, accuracy, precision, f1_score, recall, error]
  return df

def write_train_info(exp, writer, model, t, p, error, file, epoch):
  if exp['model_type'] == 'classification':
    t = torch.stack(t, dim=0).squeeze()
    p = torch.stack(p, dim=0).squeeze()
    target = torch.argmax(t, dim=1).tolist()
    pred = torch.argmax(p, dim=1).tolist()
    writer.add_scalar(f'{file}/accuracy', mt.accuracy_score(target, pred, normalize=True), epoch)
    writer.add_scalar(f'{file}/precision', mt.precision_score(target, pred, average='weighted', zero_division=0), epoch)
    writer.add_scalar(f'{file}/f1', mt.f1_score(target, pred, average='weighted'), epoch)
    writer.add_scalar(f'{file}/recall', mt.recall_score(target, pred, average='weighted'), epoch)
  writer.add_scalar(f'{file}/error', error.item(), epoch)
  if file == 'train':
    for name, param in model.named_parameters():
      writer.add_histogram(f'{file}/grads', param.grad, epoch)
      writer.add_histogram(f'{file}/weights', param, epoch)


def write_test_info(exp, df, t, p, error):
  e = error.item()
  if exp['model_type'] == 'classification':
    t = torch.stack(t, dim=0).squeeze()
    p = torch.stack(p, dim=0).squeeze()
    target = torch.argmax(t, dim=1).tolist()
    pred = torch.argmax(p, dim=1).tolist()
    acc = mt.accuracy_score(target, pred, normalize=True)
    precision = mt.precision_score(target, pred, average='weighted', zero_division=0)
    f1 = mt.f1_score(target, pred, average='weighted')
    recall = mt.recall_score(target, pred, average='weighted')
    add_class_data(df, exp['run'], acc, precision, f1, recall, e)


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


def plot_multiclass_confusion_matrix(actual, predicted, class_names):  # todo
    cm = confusion_matrix(actual, predicted, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Plotting
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def create_parallel_coordinate_graph(data, error_values):  # todo
    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Add error values as a column in the DataFrame
    df['Inference_Error'] = error_values

    # Create the parallel coordinate plot
    fig = px.parallel_coordinates(df, color="Inference_Error",
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  color_continuous_midpoint=2)
    return fig

# Example Usage
'''data = [
    {'feature1': 10, 'feature2': 0.5, 'feature3': 3},
    {'feature1': 15, 'feature2': 0.6, 'feature3': 2},
    # Add more dictionaries as needed
]
error_values = [0.1, 0.2]  # Corresponding error values'''

fig = create_parallel_coordinate_graph(data, error_values)
fig.show()


'''
seaborn?
'''