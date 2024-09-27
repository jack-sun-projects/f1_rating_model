"""
This module prepares the csv files for training for the machine learning model.
"""

import datetime

import pandas as pd
import numpy as np
from loguru import logger

from f1_rating_model.src.model.pipeline.data_collection import load_data


def prepare_data() -> pd.DataFrame:

    """
    Prepare the dataset for model training by appending the separate csv files
    and transforming them.
    """

    logger.info('Starting data preprocessing')

    results, races, drivers, constructors, status, pre_f1_yoe = load_data()
    results_concat, pre_f1_yoe = _concat_data(results,
                                              races,
                                              drivers,
                                              constructors,
                                              status,
                                              pre_f1_yoe,)
    results_full, results_predictions = _transform_data(results_concat,
                                                        pre_f1_yoe,)

    return results_full, results_predictions


def _concat_data(
        results: pd.DataFrame,
        races: pd.DataFrame,
        drivers: pd.DataFrame,
        constructors: pd.DataFrame,
        status: pd.DataFrame,
        pre_f1_yoe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Concatenates the csv files and removes unused columns.
    """

    logger.info('Appending csv files')

    results_full = results.merge(races, how='left', on='raceId')
    results_full = results_full.merge(drivers, how='left', on='driverId')
    results_full = results_full.merge(
        constructors,
        how='left',
        on='constructorId',
        )
    results_full = results_full.merge(status, how='left', on='statusId')

    results_full['driver_name'] = (
        results_full['forename']
        + " "
        + results_full['surname']
        )
    results_full = results_full.rename(
        {'name_x': 'race_name',
         'name_y': 'constructor_name'},
        axis=1,
        )
    results_full['constructor_year'] = (
        results_full['year'].astype('str')
        + " "
        + results_full['constructor_name']
        )

    results_full = results_full.loc[
        results_full['race_name'] != 'Indianapolis 500', :]

    results_full = results_full[[
        'year',
        'round',
        'date',
        'race_name',
        'driver_name',
        'dob',
        'constructor_year',
        'positionOrder',
        'status',
        'laps'
        ]]

    return results_full, pre_f1_yoe


def _transform_data(
        results_full: pd.DataFrame,
        pre_f1_yoe: pd.DataFrame,
) -> pd.DataFrame:

    """
    Cleans the data to get it ready for model training.
    """

    logger.info('Transforming data')

    # classifying finish status
    results_full.loc[
        results_full['status'].str.contains('Finished|Lap'), 'dnf'
        ] = 0
    results_full['dnf'] = results_full['dnf'].fillna(1)
    results_full.loc[
        results_full['status']
        .str.contains('Accident|Collision|Spun'), 'status'
        ] = 'retired_human_error'
    results_full.loc[
        (~(results_full['status'].str.contains('retired_human_error')))
        & (~(results_full['status'].str.contains('Finished|Lap'))), 'status'
        ] = 'retired_technical_error'
    results_full.loc[
        results_full['status'].str.contains('Finished|Lap'), 'status'
        ] = 'finished'

    # determining % race not completed
    results_full = results_full.merge(
        results_full.groupby(['year', 'round'])
        .max(['laps'])['laps'].reset_index(),
        how='left', on=['year', 'round'])
    results_full['race%_not_completed'] = \
        1-results_full['laps_x'] / results_full['laps_y']
    results_full.loc[
        results_full['status'] == 'finished', 'race%_not_completed'
        ] = 0
    results_full = results_full.drop(['laps_x', 'laps_y'], axis=1)

    # determining number of drivers in each race
    results_full = results_full.merge(
        results_full.loc[
            results_full['status'] == 'finished', :]
        .groupby(['year', 'race_name'])
        .agg('count')['round']
        .reset_index(),
        on=['year', 'race_name'], how='left')
    results_full = results_full.merge(
        results_full.groupby(['year', 'race_name'])
        .agg('count')['round_x']
        .reset_index(),
        on=['year', 'race_name'], how='left', suffixes=(None, '_x'))
    results_full = results_full.rename(
        {'round_x': 'round',
         'round_y': 'num_finishing_drivers',
         'round_x_x': 'num_drivers'}, axis=1
         )
    results_full['finish_ratio'] = \
        results_full['num_finishing_drivers'] / results_full['num_drivers']

    # filtering out drivers with less than 5 total finishes
    min_races_filter = results_full.loc[
        ~(results_full['status'] == 'retired_technical_error'), :
        ]
    min_races_filter = min_races_filter \
        .groupby(['driver_name']) \
        .size() \
        .reset_index(name='count') \
        .sort_values('count', ascending=False)
    min_races_filter = min_races_filter.loc[min_races_filter['count'] >= 5, :]
    results_full = results_full.merge(
        min_races_filter['driver_name'], how='inner', on='driver_name')

    # determining years of experience, capped at 4
    pre_f1_yoe = pre_f1_yoe.loc[
        pre_f1_yoe['raced_flag'] > 0,
        ['driver_name', 'year']]

    num_races = results_full.groupby(['driver_name', 'year']) \
        .agg(num_races=('driver_name', 'count')) \
        .reset_index()
    num_races = num_races.loc[num_races['num_races'] >= 3, :]
    num_races = pd.concat(
        [num_races, pre_f1_yoe], axis=0) \
        .sort_values(['driver_name', 'year'])
    num_races['num_races'] = num_races['num_races'].fillna('pre_f1_counted')

    yoe = pd.concat([results_full, pre_f1_yoe])
    yoe = yoe.groupby('driver_name') \
        .agg(first_year=('year', 'min'), last_year=('year', 'max')) \
        .reset_index()
    yoe = yoe.merge(
        pd.DataFrame(
            np.arange(1946, datetime.date.today().year+1), columns=['year']),
        how='cross')
    yoe = yoe.loc[
        (yoe['year'] >= yoe['first_year']) & (yoe['year'] <= yoe['last_year']),
        ['driver_name', 'year']]
    yoe = yoe.merge(num_races, on=['driver_name', 'year'], how='left')
    yoe['yoe'] = yoe.groupby('driver_name')['num_races'] \
        .transform(lambda x: x.shift().rolling(4, 0).count())
    results_full = results_full.merge(
        yoe[['driver_name', 'year', 'yoe']],
        on=['driver_name', 'year'],
        how='left',)

    # determining age and years from prime
    results_full['dob'] = pd.to_datetime(results_full['dob'])
    results_full['date'] = pd.to_datetime(results_full['date'])
    results_full['age'] = \
        (results_full['date'] - results_full['dob']).dt.days/365.25

    # adding position percentile for log gam model
    results_full['position_percentile'] = \
        (results_full['positionOrder']-1) / (results_full['num_drivers']-1)

    # sorting values for alignment and clarity
    results_full = results_full.sort_values(
        ['year', 'round', 'positionOrder'],
        ascending=[False, True, True]) \
        .reset_index(drop=True)

    # creating results table before dummying drivers and constructor-years for
    # rankings generation
    results_predictions = results_full.reset_index(drop=True).copy()

    # creating dummies for drivers and constructor-year pairs
    results_full = pd.concat(
        [results_full, pd.get_dummies(results_full['driver_name'])], axis=1) \
        .drop(['driver_name'], axis=1)
    results_full = pd.concat(
        [results_full, pd.get_dummies(results_full['constructor_year'])],
        axis=1) \
        .drop(['constructor_year'], axis=1)

    return results_full, results_predictions
