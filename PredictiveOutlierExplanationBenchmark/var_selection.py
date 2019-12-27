from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd

sesLib = importr('MXM')


def run_ses(var_sel_conf, train_data, max_k, alpha):
    pandas2ri.activate()
    rdf = ro.conversion.py2rpy(train_data)
    sesObject = sesLib.SES('is_anomaly', rdf, max_k=max_k, threshold=alpha)
    selectedVars = sesObject.slots['selectedVars']
    print(selectedVars)


VAR_SELECTION_MAP = {
    "ses": run_ses
}