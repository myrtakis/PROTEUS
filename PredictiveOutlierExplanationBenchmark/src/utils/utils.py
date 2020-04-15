from scipy.io import loadmat
import numpy as np
import pandas as pd


def convert_mat_to_csv(path):
    mat = loadmat(path)
    vars = np.arange(mat['X'].shape[1]+1)
    vars = ['Var' + str(v) for v in vars]
    vars[-1] = 'is_anomaly'
    csv = np.hstack((mat['X'], mat['y']))
    csv = np.vstack((vars, csv))
    path = path.replace('mat', 'csv')
    pd.DataFrame(csv).to_csv(path, header=False, index=False)
