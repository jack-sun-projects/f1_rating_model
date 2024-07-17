#%%
import numpy as np
import pandas as pd
import datetime

#%%
full_predictions=pd.read_csv('full_ratings.csv')


    # getting the first and last year of each driver's career
driver_min_max=full_predictions.groupby('driver_name').agg(
        first_year=('year', 'min')
        ,last_year=('year', 'max')
    ).reset_index()
driver_min_max=driver_min_max.merge(pd.DataFrame(np.arange(1950, datetime.date.today().year+1), columns=['year']), how='cross')
driver_min_max=driver_min_max.loc[(driver_min_max['year']>=driver_min_max['first_year'])&(driver_min_max['year']<=driver_min_max['last_year']), ['driver_name', 'year']]

    # filtering out seasons in which drivers finished less than 3 races
rankings=full_predictions.groupby(by=['driver_name', 'year']).agg(
        num_counting_races=('status', lambda x: ((~x.str.contains('retired_technical_error'))).count()),
        score=('score', 'median')
    ).reset_index()
rankings=rankings.loc[rankings['num_counting_races']>=3, :].drop(['num_counting_races'], axis=1)
rankings=driver_min_max.merge(rankings, how='left', on=['driver_name', 'year'])

rankings=rankings.loc[:, ['driver_name', 'score']].groupby('driver_name').apply(lambda x: x.nlargest(5, 'score').mean())['score']
rankings