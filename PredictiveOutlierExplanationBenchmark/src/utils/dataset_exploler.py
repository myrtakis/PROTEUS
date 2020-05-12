import openml

MIN_FEATURES = 40
MAJORITY_CLASS = 0.8
MAX_MISSING_VALUES = 0

FEATURES_KEY = 'NumberOfFeatures'
MAJORITY_CLASS_KEY = 'MajorityClassSize'
MISSING_VALUES_KEY = 'NumberOfInstancesWithMissingValues'
SAMPLES_KEY = 'NumberOfInstances'
MINORITY_CLASS_KEY = 'MinorityClassSize'

COL_NAME = 'COL_NAME'
ACTIVATE = 'ACTIVATE'
OPERATION = 'OPERATION'

filters = [
    {COL_NAME: FEATURES_KEY, OPERATION: lambda col, samples=None: col > MIN_FEATURES, ACTIVATE: True},
    {COL_NAME: MAJORITY_CLASS_KEY, OPERATION: lambda col, samples: col / samples >= MAJORITY_CLASS, ACTIVATE: True},
    {COL_NAME: MISSING_VALUES_KEY, OPERATION: lambda col, samples=None: col <= MAX_MISSING_VALUES, ACTIVATE: True}
]


def explore_datasets():
    datasets_df = openml.datasets.list_datasets(output_format='dataframe')
    datasets_df = apply_filters(datasets_df)
    print()


def apply_filters(datasets_df):
    datasets_df = datasets_df.dropna()
    for filter_data in filters:
        if not filter_data[ACTIVATE]:
            continue
        col_name = filter_data[COL_NAME]
        num_of_samples = datasets_df[SAMPLES_KEY]
        operation = filter_data[OPERATION]
        datasets_df = datasets_df[operation(datasets_df[col_name], num_of_samples)]
        print(filter_data[COL_NAME], datasets_df.shape)
    return datasets_df


if __name__ == '__main__':
    explore_datasets()
    pass