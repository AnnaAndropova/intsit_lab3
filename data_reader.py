import pandas as pd


def make_dicts(df, names):
    for name in names:
        count = 0
        dict = {}
        for i in df[name].unique():
            dict[i] = count
            count += 1
        globals().update({'dict_{}'.format(name): dict})


def make_numeric_data(df, names):
    for name in names:
        df[name] = df[name].map(globals()['dict_' + name])


def read_data():
    train_set = pd.read_csv('data/adult.data')
    test_set = pd.read_csv('data/adult.test')
    column_labels = ('age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                     'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                     'wage_class')
    train_set.columns = column_labels
    test_set.columns = column_labels

    X_train = train_set.copy()
    X_test = test_set.copy()

    non_numeric_columns = ('workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                           'native_country', 'wage_class')
    make_dicts(X_train, non_numeric_columns)
    make_numeric_data(X_train, non_numeric_columns)
    make_numeric_data(X_test, non_numeric_columns)
    return X_train, X_test
