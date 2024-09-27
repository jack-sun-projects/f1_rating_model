"""
This module will split the transformed dataset into X and y then train the
model.
"""

import pickle as pk

import pandas as pd
from pygam import LogisticGAM, s, l, te
from loguru import logger

from f1_rating_model.src.model.pipeline.data_preparation import prepare_data


def build_model() -> None:

    """
    Splits the prepared dataset into X and y, the trains and saves a GAM on
    the dataset.
    """

    logger.info('Starting model building pipeline')

    results_full, results_predictions = prepare_data()

    X_col_drop = [
        'year',
        'round',
        'date',
        'race_name',
        'num_finishing_drivers',
        'dob',
        'status',
        'positionOrder',
        'position_percentile',
    ]
    y_col = 'position_percentile'

    X, y = _split_X_y(results_full, X_col_drop, y_col)

    model = _train_model(X, y)

    _save_model(model)

    return X, results_predictions


def _split_X_y(results: pd.DataFrame,
               X_col_drop: list[str],
               y_col: str,
               ) -> tuple[pd.DataFrame, pd.Series]:

    """
    Splits the prepared dataset into X (features) and y (target variable).
    """

    logger.info('Defining X and y variables')

    # splitting dataset into independent and dependent variables
    X = results.drop(X_col_drop, axis=1)
    y = results[y_col]

    return X, y


def _train_model(X: pd.DataFrame,
                 y: pd.Series
                 ) -> LogisticGAM:

    """
    Trains a logistic GAM on the dataset using a combination of linear,
    spline, and tensor terms.
    """

    logger.info('Training logistic GAM model')

    # creating GAM terms
    logistic_gam_terms = l(0) + l(1) + s(3) + te(0, 1) + te(0, 3) + l(2) \
        + l(4) + s(5, n_splines=5, spline_order=2)

    for n in range(6, len(X.columns)):
        logistic_gam_terms = logistic_gam_terms + l(n)

    # fitting model
    logistic_gam = LogisticGAM(logistic_gam_terms).fit(X, y)

    return logistic_gam


def _save_model(model: LogisticGAM) -> None:

    """
    Uses the pickle library to save a version of the trained GAM.
    """

    model_save_path = 'f1_rating_model/src/model/models/f1_v1'
    logger.info('Saving model to: {model_save_path}')
    with open(model_save_path, 'wb') as model_file:
        pk.dump(model, model_file)
