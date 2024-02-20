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


def write_classification_train_info(writer, model, t, p, error, file, epoch):
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


def write_classification_test_info(writer, hparams, target, pred, error):
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


def plot_multiclass_confusion_matrix(actual, predicted, class_names):  # todo
    cm = confusion_matrix(actual, predicted, labels=class_names)
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


def create_parallel_coordinate_graph(run_dicts):  # todo
    df_list = []
    for item in run_dicts:
        # Flatten hparams and add test_error
        flattened = {**item['hparams']['optim_dict'], **item['hparams']['dl_dict'], 'error_f': item['hparams']['error_f'],
                     'test_error': item['test_error']}
        df_list.append(flattened)
    df = pd.DataFrame(df_list)

    print('cleared loop')
    # Calculate correlation to test_error and sort features
    correlation = df.corr()['test_error'].abs().sort_values(ascending=False)
    sorted_features = correlation.index.tolist()
    sorted_features.remove('test_error')  # Remove 'test_error' itself from the list

    print('cleared correlation')

    # Create parallel coordinates plot
    fig = px.parallel_coordinates(df, color="test_error", dimensions=sorted_features,
                                  color_continuous_scale=px.colors.diverging.Tealrose)

    fig.show()


def seaborn_swarm(df):  # todo, sort this mess
    sns.pairplot(df, corner=True)
    plt.show()

    # heatmap for correlation matrix, maybe good for run dict too?
    corr = df[['Accuracy', 'Precision', 'F1_Score', 'Recall', 'Error']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

    # lineplot for trends over tuns
    df['Run'] = pd.Categorical(df['Run'])
    # Plotting each performance metric over runs
    metrics = ['Accuracy', 'Precision', 'F1_Score', 'Recall', 'Error']
    for metric in metrics:
      sns.lineplot(data=df, x='Run', y=metric, marker='o').set_title(metric + ' over Runs')
      plt.show()

    # boxplots for distribution of each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, metric in enumerate(['Accuracy', 'Precision', 'F1_Score', 'Recall', 'Error']):
      sns.boxplot(data=df, y=metric, ax=axes[i])
    axes[-1].set_visible(False)  # Hide the last subplot if it's unused
    plt.tight_layout()
    plt.show()

    # point plots for detialed metric comparison across runs
    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(metrics):
      plt.subplot(2, 3, i + 1)
      sns.pointplot(x='Run', y=metric, data=df, join=False)
      plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # bar plots for mean comparison of metrics across runs
    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(metrics):
      plt.subplot(2, 3, i + 1)
      sns.barplot(x='Run', y=metric, data=df, ci=None)
      plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # swarm plots for individual observations across runs
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
      plt.subplot(2, 3, i + 1)
      sns.swarmplot(x='Run', y=metric, data=df)
      plt.title(metric)
      plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # violin plots for detailed distribution comparison
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
      plt.subplot(2, 3, i + 1)
      sns.violinplot(x='Run', y=metric, data=df)
      plt.title(metric)
      plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # bar plots of metrics
    # Melt the DataFrame to long format for easier plotting with Seaborn
    df_long = pd.melt(df, id_vars=['Run'], value_vars=['Accuracy', 'Precision', 'F1_Score', 'Recall', 'Error'],
                      var_name='Metric', value_name='Value')

    # Create bar plots
    plt.figure(figsize=(18, 10))
    sns.barplot(data=df_long, x='Metric', y='Value', hue='Run')
    plt.title('Comparison of Metrics Across Runs')
    plt.ylabel('Value')
    plt.xlabel('Metric')
    plt.legend(title='Run', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


'''
seaborn?
give chatgpt data format and ask it to generate cool seaborn graphs
'''