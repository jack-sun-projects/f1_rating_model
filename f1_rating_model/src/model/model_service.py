from pathlib import Path
import pickle as pk
import datetime

import numpy as np
import pandas as pd
from loguru import logger

from model.pipeline.train_model import build_model


class ModelService:

    def __init__(self):
        self.model = None
        self.X = None
        self.results = None

    def load_model(self, model_name='f1_v1'):

        model_path = Path(f'f1_rating_model/src/model/models/{model_name}')

        if not model_path.exists():
            logger.warning("Model was not found")
            logger.info('Building model')
            _, _ = build_model()

        self.model = pk.load(open(
            f'f1_rating_model/src/model/models/{model_name}', 'rb'))

    def generate_rankings(self,
                          X,
                          results,
                          ranking_type,
                          year,
                          top=25,
                          ):

        logger.info('Generating rankings')

        # getting error term
        errors = pd.concat([results, pd.DataFrame(self.model.predict_proba(X),
                                                  columns=['prediction'])],
                           axis=1)
        errors['error'] = errors['position_percentile'] - errors['prediction']
        errors = errors['error']

        # removing impact of constructor
        first_constructor_index = pd.DataFrame(X.columns).loc[
            X.columns.str.contains('1950'), :].index[0]  # this seems very slow
        X.iloc[:, first_constructor_index:] = 0
        X = pd.concat([X, results['status']], axis=1)

        # normalizing impact of other non-driver factors
        X = X.drop('status', axis=1)
        X['num_drivers'] = 20
        X['finish_ratio'] = 0.875

        # getting normalized predictions
        predictions = self.model.predict_proba(X)

        # calculating 'true' driver score
        full_predictions = pd.concat(
            [results,
             pd.DataFrame(predictions, columns=['prediction']),
             errors],
            axis=1)
        full_predictions = full_predictions.loc[
            full_predictions['status'] != 'retired_technical_error', :]
        full_predictions['score'] = \
            full_predictions['prediction'] + full_predictions['error']

        # getting the first and last year of each driver's career
        driver_min_max = full_predictions.groupby('driver_name').agg(
            first_year=('year', 'min'),
            last_year=('year', 'max')
            ).reset_index()
        driver_min_max = driver_min_max.merge(
            pd.DataFrame(
                np.arange(1950, datetime.date.today().year+1),
                columns=['year']),
            how='cross')
        driver_min_max = driver_min_max.loc[
            (driver_min_max['year'] >= driver_min_max['first_year']) &
            (driver_min_max['year'] <= driver_min_max['last_year']),
            ['driver_name', 'year']]

        # filtering out seasons in which drivers finished less than 3 races
        # (default value=3)
        rankings = full_predictions.groupby(by=['driver_name', 'year']).agg(
            num_counting_races=('status',
                                lambda x: ((~x.str.contains(
                                    'retired_technical_error')))
                                .count()),
            score=('score', 'median')
            ).reset_index()
        rankings = rankings.loc[
            rankings['num_counting_races'] >= 3, :] \
            .drop(['num_counting_races'], axis=1)
        rankings = driver_min_max.merge(
            rankings,
            how='left',
            on=['driver_name', 'year'])

        # determining ranking based on selected ranking_type
        if ranking_type == 'annual':
            rankings = rankings.loc[rankings['score'].notnull(), :] \
                .sort_values('score').reset_index(drop=True)
            rankings = rankings.loc[rankings['year'] == year, :]
            rankings['score'] = rankings['score'].round(3)
        if ranking_type == 'moving_average':
            rankings['moving_average'] = rankings \
                .groupby('driver_name')['score'] \
                .transform(lambda x: x.rolling(3, 3).mean())
            rankings = rankings.loc[rankings['moving_average'].notnull(), :]
            rankings = rankings.sort_values('moving_average') \
                .groupby('driver_name').head(1) \
                .reset_index(drop=True).drop(['score'], axis=1)
            rankings['year'] = (rankings['year']-2).astype('str') \
                + " - " \
                + rankings['year'].astype('str')
            rankings['moving_average'] = rankings['moving_average'].round(3)

        return rankings.head(top)
