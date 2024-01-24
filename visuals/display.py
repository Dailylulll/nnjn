import matplotlib.pyplot as plt

'''Add title for what it is, caption for training material'''
def train_v_test(trainX, trainY, testX, testY):
  plt.figure(figsize=(10, 8))
  plt.scatter(trainX, trainY, c="b", s=4, label="Training data")
  plt.scatter(testX, testY, c="r", s=4, label="Testing data")
  plt.legend()
  plt.show()


'''Need a viewer like on mnist site, to show multiple images and labels at the same time'''