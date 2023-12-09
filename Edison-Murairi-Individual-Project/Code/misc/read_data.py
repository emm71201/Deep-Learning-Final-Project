import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib.gridspec import SubplotSpec
import cv2
import sys
import os

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

PATH = "../data/Data/"
nrows, ncols = 3,5
fig, axes = plt.subplots(nrows,ncols, figsize=(10,7))
# clear subplots
for i in range(nrows):
    for j in range(ncols):
        axes[i,j].axis('off')

dementia = []
row = 0
folders = ['Non_Demented', 'Very_mild_Dementia', 'Mild_Dementia']
dementia = ['Non_Demented', 'Very_mild_Dementia', 'Mild_Dementia']
for folder in folders:
    col = 0
    for file in os.listdir(PATH + folder)[:ncols]:

        complete_path = os.path.join(PATH, folder, file)
        image = mpimg.imread(complete_path)
        axes[row][col].imshow(image)
        col += 1
    row += 1
grid = plt.GridSpec(nrows, ncols)
for i in range(len(dementia)):
    row_title = dementia[i]
    create_subtitle(fig, grid[i, ::], row_title)
#plt.savefig("../Figures/examples_images.jpg", bbox_inches='tight')
plt.show()

# Exploring the dataset
# df = {}
# for folder in os.listdir(PATH):
#     tmp = []
#     for file in os.listdir(PATH + folder)[:]:
#         tmp.append(file)
#     df[folder] = tmp
# counts = []
# for key, item in df.items():
#     print(f"{key} : {len(item)}")
#     counts.append(len(item))
#
# fig, ax = plt.subplots()

# fruits = ['apple', 'blueberry', 'cherry', 'orange']
# counts = [40, 100, 30, 55]
# bar_labels = ['red', 'blue', '_red', 'orange']
# bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
#
# ax.bar(dementia, counts, color=bar_colors)
#
# ax.set_ylabel('Counts')
#ax.set_title('Fruit supply by kind and color')
#ax.legend(title='Fruit color')

# plt.show()








