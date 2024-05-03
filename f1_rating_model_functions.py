from tqdm import tqdm
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pygam import LinearGAM, s, f, l, te

def fit_model(model, X, y):

    print("Fitting model...")

    # fitting the model
    fitted_model=model.fit(X, y)
    print("Complete", "\n")

    # making predictions
    y_pred=fitted_model.predict(X)

    # calculating error
    mae=mean_absolute_error(y, y_pred)
    r2=r2_score(y, y_pred)

    # explain why we don't use CV score ( we need all historical data!  )
    # cv_mae=-np.mean(cross_val_score(model, X, y, scoring="neg_mean_absolute_error"))

    print("Evaluation results: ", "\n", "MAE: ", round(mae, 2), "\n", "R-squared: ", round(r2, 3), sep="")

    return fitted_model

def get_rankings(results, model, X, model_type, save_results):
    
    # getting predictions

    results_full_pred=pd.concat([results.sort_values(['year', 'round', 'positionOrder'], ascending=[False, True, True]).reset_index(drop=True), pd.DataFrame(model.predict(X), columns=['prediction']).reset_index(drop=True)], axis=1)
    results_full_pred['error']=results_full_pred['positionOrder']-results_full_pred['prediction']

    # joining in coefficients relevant to ranking

    if model_type=='ridge':

        features=pd.concat([pd.DataFrame(X.columns), pd.DataFrame(model.coef_)], axis=1)
        features.columns=['feature', 'coefficient']

        results_full_pred=pd.merge(results_full_pred, features, how='left', left_on='driver_name', right_on='feature')
        results_full_pred=results_full_pred.rename({'coefficient':'driver_coefficient'}, axis=1)
        results_full_pred['yoe_coefficient']=features.loc[features['feature']=='yoe', 'coefficient'].iloc[0]*results_full_pred['yoe']
        results_full_pred['years_from_prime_coefficient']=features.loc[features['feature']=='years_from_prime', 'coefficient'].iloc[0]*results_full_pred['years_from_prime']
        results_full_pred['human_error_coefficient']=features.loc[features['feature']=='human_error', 'coefficient'].iloc[0]*results_full_pred['human_error']
        results_full_pred['score']=results_full_pred.loc[:, ['error', 'driver_coefficient', 'yoe_coefficient', 'human_error_coefficient']].sum(axis=1)

    elif model_type=='gam':
        
        features=pd.concat([pd.DataFrame(X.columns[0:4]), pd.DataFrame(model.coef_[0:4])], axis=1)
        features.columns=['feature', 'coefficient']
        con_year_features=pd.DataFrame(X.columns, columns=['feature']).tail(results['constructor_year'].nunique()).reset_index(drop=True)
        con_year_coef=pd.DataFrame(model.coef_[:-1], columns=['coefficient']).tail(results['constructor_year'].nunique()).reset_index(drop=True)
        con_year_features_coef=pd.concat([con_year_features, con_year_coef], axis=1)
        features=pd.concat([features, con_year_features_coef], axis=0)

        results_full_pred=pd.merge(results_full_pred, features, how='left', left_on='constructor_year', right_on='feature')
        results_full_pred=results_full_pred.rename({'coefficient':'constructor_year_coefficient'}, axis=1)
        results_full_pred['tech_problems_coef']=features.loc[features['feature']=='technical_problems', 'coefficient'].iloc[0]*results_full_pred['technical_problems']
        results_full_pred['num_drivers_coef']=features.loc[features['feature']=='num_drivers', 'coefficient'].iloc[0]*results_full_pred['num_drivers']
        results_full_pred['dnf_ratio_coef']=features.loc[features['feature']=='dnf_ratio', 'coefficient'].iloc[0]*results_full_pred['dnf_ratio']
        results_full_pred['score']=results_full_pred['prediction']-results_full_pred['constructor_year_coefficient']-results_full_pred['tech_problems_coef']-results_full_pred['num_drivers_coef']-results_full_pred['dnf_ratio_coef']
    
    if save_results==True:
        results_full_pred.to_csv("results_predictions_table.csv")
    else:
        pass

    # grouping and calculating score by year

    driver_min_max=results_full_pred.groupby('driver_name').agg(
        first_year=('year', 'min')
        ,last_year=('year', 'max')
    ).reset_index()
    driver_min_max=driver_min_max.merge(pd.DataFrame(np.arange(1950, datetime.date.today().year+1), columns=['year']), how='cross')
    driver_min_max=driver_min_max.loc[(driver_min_max['year']>=driver_min_max['first_year'])&(driver_min_max['year']<=driver_min_max['last_year']), ['driver_name', 'year']]

    rankings=results_full_pred.groupby(by=['driver_name', 'year']).agg(
        num_counting_races=('technical_problems', lambda x: (x==0).sum()),
        score=('score', 'median')
    ).reset_index()
    rankings=rankings.loc[rankings['num_counting_races']>=3, :].drop(['num_counting_races'], axis=1)

    rankings=driver_min_max.merge(rankings, how='left', on=['driver_name', 'year'])
    rankings['3yma_score']=rankings.groupby('driver_name')['score'].transform(lambda x: x.rolling(3, 3).mean())
    overall_rankings=rankings.groupby('driver_name').agg({'3yma_score':'min'}).reset_index().sort_values('3yma_score')
    overall_rankings=overall_rankings.loc[overall_rankings['3yma_score'].notnull(), :]

    return overall_rankings

def get_race_predictions(results, model, X, year, round):

    results_full_pred=pd.concat([results.sort_values(['year', 'round', 'positionOrder'], ascending=[False, True, True]).reset_index(drop=True), pd.DataFrame(model.predict(X), columns=['prediction']).reset_index(drop=True)], axis=1)
    race_results_predictions=results_full_pred.loc[(results_full_pred['year']==year) & (results_full_pred['round']==round), :]
    race_results_predictions['error']=race_results_predictions['positionOrder']-race_results_predictions['prediction']

    return race_results_predictions

def get_coefficient(results, model, X, feature_name):

    if model==ridge:
        features=pd.concat([pd.DataFrame(X.columns), pd.DataFrame(model.coef_)], axis=1)

    elif model==gam:
        features=pd.concat([pd.DataFrame(X.columns), pd.DataFrame(model.coef_[:-1])], axis=1)

    features.columns=['feature', 'coefficient']

    return features.loc[features['feature']==feature_name, :]