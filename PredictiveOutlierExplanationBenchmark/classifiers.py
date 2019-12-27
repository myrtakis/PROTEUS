
def run_iforest(classifier_conf):
    return None


def run_random_forest(classifier_conf):
    return None


CLASSIFIERS_MAP = {
    'iforest':  run_iforest,
    'rf':       run_random_forest
}
