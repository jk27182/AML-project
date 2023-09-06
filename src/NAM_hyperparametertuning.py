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


# with open('all_data.pickle', 'rb') as file:
#     all_data = pickle.load(file)

# orig_characteristics = all_data['OrigCharacteristics.dta']
# orig_characteristics_columns = [
#     #'Deal',
#     'type',
#     'CutoffLTV',
#     'CutoffDSCR',
#     'CutoffCpn',
#     'log_bal',
#     'fixed',
#     'buildingage',
#     'CutoffOcc',
#     'year_priced',
#     'quarter_type',
#     'AmortType',
#     # 'MSA',
#     'qy',
#     'Size',

#     'OVER_w',
#     'past_over',
#     'high_overstatement2', # is 100% dependent on Over_w, if we predict this we get 100% accuracy
#     'Distress',
#     #'non_perf'
# ]
# orig_data = orig_characteristics[orig_characteristics_columns]

# target_col = 'Distress'
# orig_data_with_dummies = pd.get_dummies(
#     orig_data,
#     columns=[
#         'AmortType',
#         # 'MSA',
#         'type'
#     ]
# )
# clean_data = orig_data_with_dummies[
#     orig_data_with_dummies.notna().all(axis=1)
# ]

# dummy_cols = [col for col, dtype in clean_data.dtypes.items() if dtype == bool]
# for dummy_col in dummy_cols:
#     clean_data[dummy_col] = clean_data[dummy_col].map({True: 1, False:0})

# # Percentage of clean data from whole dataset
# print('percentage of clean data and all data ', len(clean_data) / len(orig_data_with_dummies))
# clean_data.drop(columns=target_col)
# y = clean_data[target_col].astype('U32')
# X = clean_data.drop(columns=target_col)

with open('data/clean_data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

y = data_dict['y']
X = data_dict['X']
# x = np.random.normal(0,2, (500,4))
# clean_data = pd.DataFrame(x, columns= ['a', 'b', 'c', 'd'])
# target_col = 'a'
# Neural Additive Model
config = defaults()
feature_cols = X.columns
X['Distress'] = y
nam_dataset = NAMDataset(
    config,
    data_path=X,
    features_columns=feature_cols,
    targets_column='Distress',
)
nam_model = NAM(
    config=config,
    name='Testing_NAM',
    num_inputs=len(nam_dataset[0][0]),
    num_units=get_num_units(config, nam_dataset.features)
)
# NAM Training
data_loaders = nam_dataset.train_dataloaders()
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
trainer.logger.experiment.summary


trainer.test(litmodel, dataloaders=nam_dataset.test_dataloaders())
fig = plot_mean_feature_importance(litmodel.model, nam_dataset)
fig = plot_nams(litmodel.model, nam_dataset, num_cols= 3)