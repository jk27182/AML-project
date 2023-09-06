from pathlib import Path
from IPython.display import display
from functools import partial
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import joblib

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, recall_score, accuracy_score, precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

import optuna

import interpret
from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression
interpret.set_visualize_provider(interpret.provider.InlineProvider())


with open('all_data.pickle', 'rb') as file:
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

sample_size = len(y)
print('sample size ', sample_size)

global_res_models = {}
# objective for optuna hyperparameter search
def objective(trial, clf_type, X, y):
    res = {
        'scores': [],
        'models': [],
    }
    if clf_type == 'EBM':
        learning_rate = trial.suggest_float('learning_rate', 0.0009, 0.0015, step=0.0004)
        smoothing_rounds = trial.suggest_int('smoothing_rounds',0, 4, step=2) # default is 0

        model_name = "EBM"
        clf = ExplainableBoostingClassifier(
            random_state=42,
            learning_rate=learning_rate,
            smoothing_rounds=smoothing_rounds,
        )
    
    if clf_type == 'LogisticRegression':
        penalty = trial.suggest_categorical("penalty", ["elasticnet", "l2", 'l1'])
        C = trial.suggest_float("C", 0.001, 2, step=0.5)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        max_iter = trial.suggest_int('max_iter', 500, 1000, step=100)
        if penalty == 'elasticnet' or penalty == 'l1':
            solver = 'saga'
        else:
            solver = trial.suggest_categorical('solver', ['newton-cholesky', 'lbfgs', 'newton-cg'])

        model_name = "LogisticRegression"
        clf = LogisticRegression(
            random_state=42,
            penalty=penalty,
            C=C,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter
        )

    # Use stratified K Fold instead of normal k fold to preserve class imbalance
    strat_kfold = StratifiedKFold(n_splits=2)
    # for train_idx, test_idx in tqdm(strat_kfold.split(X, y), total=strat_kfold.get_n_splits(), desc='K fold'):
    for fold, (train_idx, test_idx) in enumerate(strat_kfold.split(X, y)):
        print(f'Currently in fold: {fold}')
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]

        X_test = X.iloc[test_idx, :]
        y_test = y.iloc[test_idx]

        X_train_oversampled, y_train_oversampled = SMOTE().fit_resample(X_train, y_train)
        print('Oversampled data')

        print('Starting to fit data')
        start_time = time.perf_counter()
        clf.fit(X_train_oversampled, y_train_oversampled)
        total_time = time.perf_counter() - start_time
        print(f'Finished fitting data, took {total_time} sec.')

        auc_roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        res['scores'].append(auc_roc)
        res['models'].append(clf)

    trial.set_user_attr(key='scores_and_models', value=res)
    global_res_models.update({model_name: res})
    return np.mean(auc_roc)

# current best 0.54
# study_ebm = optuna.create_study(storage="mysql://root@localhost/ebm",direction='maximize', study_name='EBM_study')
# optuna.delete_study(storage="mysql://root@localhost/ebm", study_name='EBM_study')
study_ebm = optuna.create_study(
    direction='maximize',
    # storage="mysql://root@localhost/ebm",
    study_name='EBM_study'
)
print('test succesful')
objective_edm = partial(objective, clf_type='EBM', X=X.head(50), y=y.head(50))
study_ebm.optimize(objective_edm, n_trials=2, show_progress_bar=True)

best_params_ebm = study_ebm.best_params
best_ebm_value = study_ebm.best_value
with open('model/best_params_ebm.joblib', 'wb') as file:
    joblib.dump({'params':best_params_ebm, 'value':best_ebm_value}, file)

# no current best
# optuna.delete_study(storage="mysql://root@localhost/logreg", study_name='LogisticRegression_study')
study_LR = optuna.create_study(
    direction='maximize',
    # storage="mysql://root@localhost/logreg",
    study_name='LogisticRegression_study'
)
objective_LR = partial(objective, clf_type='LogisticRegression', X=X.head(50), y=y.head(50))
study_LR.optimize(objective_LR, n_trials=50, show_progress_bar=True)

best_params_LR = study_LR.best_params
best_LR_value = study_LR.best_value
with open('model/best_params_LR.joblib', 'wb') as file:
    joblib.dump({'params':best_params_LR, 'value':best_LR_value}, file)