"""
This module loads the data from locally saved .csv files that contains all
Formula 1 information from 1950 to now.
"""

import pandas as pd
from loguru import logger


def load_data() -> tuple[pd.DataFrame, ...]:

    """
    Reads the 5 csv files that will be used to form the dataset for the
    machine learning model.
    """

    logger.info('Loading raw data')

    results = pd.read_csv(
        'f1_rating_model/src/model/raw/results.csv')
    races = pd.read_csv(
        'f1_rating_model/src/model/raw/races.csv')
    drivers = pd.read_csv(
        'f1_rating_model/src/model/raw/drivers.csv')
    constructors = pd.read_csv(
        'f1_rating_model/src/model/raw/constructors.csv')
    status = pd.read_csv(
        'f1_rating_model/src/model/raw/status.csv')
    pre_f1_yoe = pd.read_csv(
        'f1_rating_model/src/model/raw/pre_f1_yoe.csv')

    return results, races, drivers, constructors, status, pre_f1_yoe
