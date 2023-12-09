import numpy as np
import pandas as pd
import matplotlib as mpl
import random
import os

kmax = 100
def get_demo(demo_folder="gmlp_demo"):
    ii = random.choice(range(kmax))
    image = os.path.join(demo_folder, f"img{ii}.png")
    labels = np.load(os.path.join(demo_folder, f"labels{ii}.npy"))

    return image, labels
