import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import torch.nn.functional as F

# My setup of matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('text', usetex=False)
plt.rc('font', family='times')
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')


num_classes = 4
classes = ["Mild", "Moderate", "No Dementia", "Very Mild"]

test_accuracies = np.load("results/test_accuracies_gmlp.npy")
losses = np.load("results/losses_gmlp.npy")
plt.plot(losses, linewidth=3)
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.grid()
plt.show()

plt.plot(test_accuracies, linewidth=3)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

conf_matrix_data = np.load("results/conf_matrix_gmlp.npy", allow_pickle=True).item()
conf_matrix = np.array([conf_matrix_data[f"{i}"] for i in range(num_classes)])
conf_matrix = np.eye(num_classes)
sns.heatmap(conf_matrix, annot=True, cbar=False, xticklabels=classes, yticklabels=classes)
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.show()

