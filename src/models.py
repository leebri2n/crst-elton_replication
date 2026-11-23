import sklearn
from sklearn import preprocessing, metrics
from sklearn import linear_model, kernel_ridge, svm, ensemble, neighbors
import pandas as pd
import numpy as np
import os

import featurizations

def dataset(mols_list, feat_scheme={}):
    df = np.array()

    return df




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def __main__():
    return

if __name__ == "__main__":
    data_path = os.path.abspath("/scratch/leebrian/elton/Machine-Learning-Energetic-Molecules-Notebooks/datasets/")
    mols, df = featurizations.load_data_as_mols(data_path)

    featurized_data = dataset(mols)