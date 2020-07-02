import itertools
from configpkg.FeatureSelectionConfig import *
from configpkg.ClassifiersConfig import *
from holders.ModelConf import ModelConf
from models.FeatureSelection import FeatureSelection
from models.Classifier import Classifier


id_key = None
params_key = None


def generate_param_combs():
    fsel_conf_combs = []
    classifiers_conf_combs = []
    global id_key, params_key
    for fsel_conf in FeatureSelectionConfig.list_all_feature_selection_algs():
        id_key = FeatureSelectionConfig.id_key()
        params_key = FeatureSelectionConfig.params_key()
        fsel_id = fsel_conf[id_key]
        fsel_params = fsel_conf[params_key]
        fsel_param_keys = list(fsel_params.keys())
        fsel_param_vals = fsel_params.values()
        params_combs = list(itertools.product(*fsel_param_vals))
        fsel_conf_combs.extend(build_key_value_params(fsel_param_keys, params_combs, fsel_id))

    for classifier_conf in ClassifiersConfig.list_all_classifiers():
        classifier_id = classifier_conf[ClassifiersConfig.id_key()]
        classifier_params = classifier_conf[ClassifiersConfig.params_key()]
        classifier_param_keys = list(classifier_params.keys())
        classifier_param_vals = classifier_params.values()
        params_combs = list(itertools.product(*classifier_param_vals))
        omit_combs = None
        if ClassifiersConfig.omit_combinations_key() in classifier_conf:
            omit_combs = classifier_conf[ClassifiersConfig.omit_combinations_key()]
        classifiers_conf_combs.extend(build_key_value_params(classifier_param_keys, params_combs, classifier_id, omit_combs))
    return fsel_conf_combs, classifiers_conf_combs


def omit_configuration(params, omit_combs):
    omit = True
    na = 'na'
    if omit_combs is None:
        return not omit

    omitted_combs = {}

    for item in omit_combs:
        for pparam, v1 in item[ClassifiersConfig.prime_param_key()].items():
            if str(params[pparam]).lower() != str(v1).lower():
                continue
            omitted_combs = item[ClassifiersConfig.combs_key()]
            for oparam in item[ClassifiersConfig.combs_key()]:
                if str(params[oparam]).lower() != na:
                    return omit

    for param, val in params.items():
        if str(val).lower() == na and param not in omitted_combs:
            return omit

    return not omit


def build_key_value_params(params_keys, params_combs, alg_id, omit_combs=None):
    params_kv = []
    for t in params_combs:
        kv = {}
        counter = 0
        for v in t:
            kv[params_keys[counter]] = v
            counter += 1
        if not omit_configuration(kv, omit_combs):
            params_kv.append({id_key: alg_id, params_key: kv})
    return params_kv


def create_model_confs(fsel_conf_combs, classifiers_conf_combs):
    model_confs = {}
    for fsel_conf in fsel_conf_combs:
        for clf_conf in classifiers_conf_combs:
            fsel_clf_id = fsel_conf[FeatureSelectionConfig.id_key()] + '_' + clf_conf[ClassifiersConfig.id_key()]
            model_confs.setdefault(fsel_clf_id, [])
            model_confs[fsel_clf_id].append(ModelConf(fsel_conf, clf_conf))
    return model_confs
