import sys
from pathlib import Path
from IPython.display import display
from functools import partial
import pickle
import time
import datetime
                                                                           
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import joblib
import ray
from ray import tune

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, recall_score, accuracy_score, precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

import optuna

from nam.data import NAMDataset
from nam.config import defaults
from nam.data import FoldedDataset
from nam.models import NAM
from nam.models import get_num_units
from nam.trainer import LitNAM
from nam.types import Config
from nam.utils import parse_args
from nam.utils import plot_mean_feature_importance
from nam.utils import plot_nams
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


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
    # 'qy',
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
        'type',
        'year_priced'
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

y = clean_data[target_col].head(100)
X = clean_data.drop(columns=target_col).head(100)

sample_size = len(y)
print('sample size ', sample_size)

config = defaults()
feature_cols = X.columns
X['Distress'] = y
config.wandb = False
config.num_workers = 4
config.cross_val = False

def objective(cfg):
     # hyperparams = {
     #      "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
     #      "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 1.0, log=True),
     #      "output_regularization": trial.suggest_float("output_regularization", 0.01, 1.0, log=True),
     #      "dropout": trial.suggest_float("dropout", 0.01, 1.0, log=True),
     #      "feature_dropout": trial.suggest_float("feature_dropout", 0.01, 1.0, log=True),
     #      "batch_size": trial.suggest_categorical("batch_size", [128, 512, 1024]),
     #      "hidden_sizes": trial.suggest_categorical("hidden_sizes", [[],[32], [64, 32]])
     # }
     config.update(**cfg)
     print(config)
     nam_dataset = NAMDataset(
          config,
          data_path=X,
          features_columns=feature_cols,
          targets_column='Distress',
     )

     current_datetime = datetime.datetime.now()
     formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
     logdir = f"NAM_run_{formatted_datetime}"
     config.logdir = logdir

     nam_model = NAM(
          config=config,
          name='Testing_NAM',
          num_inputs=len(nam_dataset[0][0]),
          num_units=get_num_units(config, nam_dataset.features)
     )
     # NAM Training
     data_loaders = nam_dataset.train_dataloaders()
     scores = []
     for fold, (train_loader, val_loader) in enumerate(data_loaders):
          tb_logger = TensorBoardLogger(
               save_dir=config.logdir,
               name=f'{nam_model.name}',
               version=f'fold_{fold + 1}')

          checkpoint_callback = ModelCheckpoint(
               filename=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.4f}",
               monitor='val_loss',
               save_top_k=config.save_top_k,
               mode='min'
          )
          litmodel = LitNAM(config, nam_model)
          pl.Trainer()
          trainer = pl.Trainer(
               logger=tb_logger,
               max_epochs=config.num_epochs,
               callbacks=checkpoint_callback,
          )
          trainer.fit(
               litmodel,
               train_dataloaders=train_loader,
               val_dataloaders=val_loader)
          print('=-----------------=============')
          print('Callback metrics')
          tmp_auroc = trainer.callback_metrics['AUROC_metric'].item()
          scores.append(tmp_auroc)

     return {'loss' :np.mean(scores)}

n_trials = 1
# if sys.argv[1] == 'max':
#      direction = 'maximize'
# if sys.argv[1] == 'min':
#      direction = 'minimize'

# study_nam = optuna.create_study(
#      direction=direction,
#      # storage="mysql://root@localhost/nam",
#      study_name='nam_study'
# )
# study_nam.optimize(objective, n_trials=n_trials, show_progress_bar=True)
# best_params_nam = study_nam.best_params
# best_nam_value = study_nam.best_value
# with open(f'models/best_params_nam_{direction}.joblib', 'wb') as file:
#      joblib.dump({'params':best_params_nam, 'value':best_nam_value}, file)
# If you
param_space = {
     'params': {
          # "lr": tune.loguniform(1e-4, 1e-1),
          # "l2_regularization": tune.loguniform(0.01, 1.0),
          # "output_regularization": tune.loguniform(0.01, 1.0),
          # "dropout": tune.loguniform(0.01, 1.0),
          # "feature_dropout": tune.loguniform(0.01, 1.0),
          # "batch_size": tune.choice([128, 512, 1024]),
          # "hidden_sizes": tune.choice([[], [32], [64, 32]]),
          "num_epochs": tune.choice([1])
     }
     } 
trainable_with_resources = tune.with_resources(objective, {"cpu": 4})
tuner = tune.Tuner(
    trainable_with_resources,
    param_space=param_space,
    tune_config=tune.TuneConfig(num_samples=10),
    run_config=ray.air.RunConfig(name="my_tune_run")
)
results = tuner.fit()
# result = tune.run(
#     objective,
#     num_samples=n_trials,
#     metric="loss",
#     mode="min",
#     ,
#     resources_per_trial={"cpu": 4}, 
#     storage_path="/Users/janik/Documents/Master/KIT/Semester 4/Advanced Machine Learning Projekt/models/ray_results",  
#     name="NAM",  
#     verbose=1  
# )
# tuner.fit()
best_trial = results.get_best_trial("loss", "min", "last")
best_loss = best_trial.last_result['loss']
best_config = best_trial.config
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
