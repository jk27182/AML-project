import pickle

import pandas as pd


with open('data/all_data.pickle', 'rb') as file:
    all_data = pickle.load(file)

orig_characteristics = all_data['OrigCharacteristics.dta']
orig_characteristics_columns = [
    #'Deal',
    'type',
    'CutoffLTV',
    'CutoffDSCR',
    'CutoffCpn',
    'log_bal',
    'fixed',
    'buildingage',
    'CutoffOcc',
    'year_priced',
    'quarter_type',
    'AmortType',
    # 'MSA',
    'qy',
    'Size',

    'OVER_w',
    'past_over',
    'high_overstatement2', # is 100% dependent on Over_w, if we predict this we get 100% accuracy
    'Distress',
    #'non_perf'
]
orig_data = orig_characteristics[orig_characteristics_columns]
target_col = 'Distress'
orig_data_with_dummies = pd.get_dummies(
    orig_data,
    columns=[
        'AmortType',
        # 'MSA',
        'type'
    ]
)
clean_data = orig_data_with_dummies[
    orig_data_with_dummies.notna().all(axis=1)
]

dummy_cols = [col for col, dtype in clean_data.dtypes.items() if dtype == bool]
for dummy_col in dummy_cols:
    clean_data[dummy_col] = clean_data[dummy_col].map({True: 1, False:0})

# Percentage of clean data from whole dataset
print('percentage of clean data and all data ', len(clean_data) / len(orig_data_with_dummies))

y = clean_data[target_col].astype('U32')
X = clean_data.drop(columns=target_col)

with open('data/clean_data.pickle', 'wb') as file:
    pickle.dump({'X': X, 'y': y},file)