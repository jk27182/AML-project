import pickle
import datetime
                                                                           
import numpy as np
import pandas as pd


import joblib
import ray
from ray import tune

from nam.data import NAMDataset, FoldedDataset
from nam.config import defaults
from nam.models import NAM
from nam.models import get_num_units
from nam.trainer import LitNAM

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


with open('data/all_data.pickle', 'rb') as file:
    all_data = pickle.load(file)

orig_characteristics = all_data['OrigCharacteristics.dta']
orig_characteristics_columns = [
    'type',
    'CutoffLTV',
    'CutoffDSCR',
    'CutoffCpn',
    'log_bal',
    'fixed',
    'buildingage',
    'CutoffOcc',
    'quarter_type',
    'AmortType',
    'Size',
    'OVER_w',
    'past_over',
    'high_overstatement2',
    'Distress',
]
orig_data = orig_characteristics[orig_characteristics_columns]
target_col = 'Distress'
orig_data_with_dummies = pd.get_dummies(
    orig_data,
    columns=[
        'AmortType',
        'type',
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

y = clean_data[target_col]
X = clean_data.drop(columns=target_col)

sample_size = len(y)
print('sample size ', sample_size)

config = defaults()
feature_cols = X.columns
X['Distress'] = y
config.wandb = False
config.num_workers = 4
config.cross_val = True
config.num_folds = 2
config.batch_size = 512

def objective(cfg):
     params = cfg['params']
     print(params)
     config.update(**params)
     print(config)
     nam_dataset = FoldedDataset(
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
               log_every_n_steps=25
          )
          trainer.fit(
               litmodel,
               train_dataloaders=train_loader,
               val_dataloaders=val_loader)
          tmp_auroc = trainer.callback_metrics['AUROC_metric'].item()
          scores.append(tmp_auroc)

     return {'loss' :np.mean(scores)}

n_trials = 20
param_space = {
     'params': {
          "lr": tune.loguniform(1e-4, 1e-1),
          "l2_regularization": tune.loguniform(0.01, 1.0),
          "output_regularization": tune.loguniform(0.01, 1.0),
          "dropout": tune.loguniform(0.01, 1.0),
          "feature_dropout": tune.loguniform(0.01, 1.0),
          "batch_size": tune.choice([128, 512, 1024]),
          "hidden_sizes": tune.choice([[], [32], [64, 32]]),
          "num_epochs": tune.choice([10, 25])
     }
     } 
trainable_with_resources = tune.with_resources(objective, {"cpu": 8})
tuner = tune.Tuner(
    trainable_with_resources,
    param_space=param_space,
    tune_config=tune.TuneConfig(num_samples=n_trials),
    run_config=ray.air.RunConfig(
         name="NAM_hyperparameter_run",
         storage_path='/Users/janik/Documents/Master/KIT/Semester4/Advanced_Machine_Learning_Projekt/models/ray_results'
     )
)
results = tuner.fit()

best_min_trial = results.get_best_result(metric="loss", mode="min", scope="all")
best_max_trial = results.get_best_result(metric="loss", mode="max", scope="all")

best_min_config = best_min_trial.config
best_max_config = best_max_trial.config
print(f"Best min trial config: {best_min_trial.config}")
print(f"Best max trial config: {best_max_trial.config}")

with open('models/best_params_nam_minimize.joblib', 'wb') as file:
     joblib.dump(best_min_config, file)

with open('models/best_params_nam_maximize.joblib', 'wb') as file:
     joblib.dump(best_max_config, file)
