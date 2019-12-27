from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd

sesLib = importr('MXM')

if __name__ == '__main__':
    df = pd.read_csv('datasets/breast_lof_031_010.csv')
    pandas2ri.activate()
    rdf = ro.conversion.py2rpy(df.iloc[0:100, :])
    sesObject = sesLib.SES('is_anomaly', rdf, max_k=5, threshold=0.5)
    selectedVars = sesObject.slots['selectedVars']
    print(selectedVars)
