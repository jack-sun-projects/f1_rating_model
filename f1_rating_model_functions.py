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
from statsmodels.stats.outliers_influence import variance_inflation_factor

def fit_model(model, X, y):

    print("Fitting model...")

    # fitting the model
    fitted_model=model.fit(X, y)
    print("Complete", "\n")

    return fitted_model

def evaluate_model(model, model_type, X, y, results):

    # calculating error
    if "logistic" in model_type:
        y_pred=pd.DataFrame(model.predict_proba(X), columns=['proba_prediction'])
        results_full=pd.concat([results.reset_index(drop=True), y_pred.reset_index(drop=True)], axis=1)
        results_full['prediction']=results_full['proba_prediction']*(results_full['num_drivers']-1)+1
        results_full['error']=results['positionOrder']-results_full['prediction']
        results_full['abs_error']=abs(results_full['error'])
        mae=mean_absolute_error(results_full['positionOrder'], results_full['prediction'])
        mse=mean_squared_error(results_full['positionOrder'], results_full['prediction'])
        r2=r2_score(results_full['positionOrder'], results_full['prediction'])
    else:
        y_pred=model.predict(X)
        mae=mean_absolute_error(y, y_pred)
        mse=mean_squared_error(y, y_pred)
        r2=r2_score(y, y_pred)

        # explain why we don't use CV score ( we need all historical data!  )
        # cv_mae=-np.mean(cross_val_score(model, X, y, scoring="neg_mean_absolute_error"))

    print("Evaluation results: ", "\n", "MAE: ", round(mae, 2), "\n", "MSE: ", round(mse, 2), "\n", "R-squared: ", round(r2, 3), sep="")

def get_rankings(model, model_type, X, results, ranking="3yma", min_races_season=3):
    
    # getting error term
    if "logistic" in model_type:
        errors=pd.concat([results, pd.DataFrame(model.predict_proba(X), columns=['prediction'])], axis=1)
        errors['error']=errors['position_percentile']-errors['prediction']
    else:
        errors=pd.concat([results, pd.DataFrame(model.predict(X), columns=['prediction'])], axis=1)
        errors['error']=errors['positionOrder']-errors['prediction']
    errors=errors['error']

    # getting predictions without non-driver factors and calculating 'true' score
    X_scores=X.copy()
    first_constructor_index=pd.DataFrame(X_scores.columns).loc[X_scores.columns.str.contains('1950'), :].index[0] # this seems very slow
    X_scores.iloc[:, first_constructor_index:]=0
    X_scores=pd.concat([X_scores, results['status']], axis=1)
    #X_scores=X_scores.loc[X_scores['status']!='retired_technical_error', :]

    if "ridge" in model_type: #should we predict on specific car/other factors as long as they're even for better interpretability?
        X_scores.loc[X_scores['status']!='retired_human_error', 'dnf']=0
        X_scores=X_scores.drop('status', axis=1)
        X_scores['num_drivers']=20
    if "gam" in model_type:
        #X_scores.loc[X_scores['status']=='retired_technical_error', ['dnf', 'race%_not_completed']]=0
        X_scores=X_scores.drop('status', axis=1)
        X_scores['num_drivers']=20
        X_scores['finish_ratio']=0.875

    if "logistic" in model_type:
        predictions=model.predict_proba(X_scores)
    else:
        predictions=model.predict(X_scores)

    full_predictions=pd.concat([results, pd.DataFrame(predictions, columns=['prediction']), errors], axis=1)
    full_predictions=full_predictions.loc[full_predictions['status']!='retired_technical_error', :]
    full_predictions['score']=full_predictions['prediction']+full_predictions['error']
    
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
        score=('score', 'median') #median or mean? 
    ).reset_index()
    rankings=rankings.loc[rankings['num_counting_races']>=min_races_season, :].drop(['num_counting_races'], axis=1)
    rankings=driver_min_max.merge(rankings, how='left', on=['driver_name', 'year'])

    # grouping and calculating score by year
    if ranking=='annual':
        rankings=rankings.loc[rankings['score'].notnull(), :].sort_values('score').reset_index(drop=True)
        rankings['score']=rankings['score'].round(3)
        return rankings
    if ranking=='3yma':
        rankings['3yma_score']=rankings.groupby('driver_name')['score'].transform(lambda x: x.rolling(3, 3).mean())
        rankings=rankings.loc[rankings['3yma_score'].notnull(), :]
        rankings=rankings.sort_values('3yma_score').groupby('driver_name').head(1).reset_index(drop=True).drop(['score'], axis=1)
        rankings['year']=(rankings['year']-2).astype('str') + " - " + rankings['year'].astype('str')
        rankings['3yma_score']=rankings['3yma_score'].round(3)
        return rankings
    
def get_constructor_rankings(model, model_type, X):
    
    first_constructor_index=pd.DataFrame(X.columns).loc[X.columns.str.contains('1950'), :].index[0] # this seems very slow
    constructor_list=pd.DataFrame(X.columns[first_constructor_index:], columns=['constructor_year'])
    if "gam" in model_type:
        constructor_coefficients=pd.DataFrame(model.coef_[-(len(X.columns)-first_constructor_index)-1:-1], columns=['coefficient'])
    else:
        constructor_coefficients=pd.DataFrame(model.coef_[-(len(X.columns)-first_constructor_index):], columns=['coefficient'])
    constructor_rankings=pd.concat([constructor_list, constructor_coefficients], axis=1).sort_values('coefficient').reset_index(drop=True)
    constructor_rankings['coefficient']=constructor_rankings['coefficient'].round(2)

    return constructor_rankings

def get_race_predictions(model, model_type, X, results, year_range=[], round_range=[], driver=[]): #maybe add scores and stuff?

    if "logistic" in model_type:
        results_full_pred=pd.concat([results.reset_index(drop=True), pd.DataFrame(model.predict_proba(X), columns=['prediction']).reset_index(drop=True)], axis=1)
    else:
        results_full_pred=pd.concat([results.reset_index(drop=True), pd.DataFrame(model.predict(X), columns=['prediction']).reset_index(drop=True)], axis=1)
        
    if year_range:
        results_full_pred=results_full_pred.loc[(results_full_pred['year']>=year_range[0]) & (results_full_pred['year']<=year_range[1]), :]
    else: pass

    if round_range:
        results_full_pred=results_full_pred.loc[(results_full_pred['round']>=round_range[0]) & (results_full_pred['round']<=round_range[1]), :]
    else: pass

    if driver:
        results_full_pred=results_full_pred.loc[results_full_pred['driver_name'].isin(driver), :]
    else: pass
    
    if "logistic" in model_type:
        results_full_pred['prediction']=results_full_pred['prediction']*(results_full_pred['num_drivers']-1)+1
        results_full_pred['error']=results_full_pred['positionOrder']-results_full_pred['prediction']
    else:
        results_full_pred['error']=results_full_pred['positionOrder']-results_full_pred['prediction']

    results_full_pred=results_full_pred.drop([
        'dob'
        ,'dnf'
        ,'race%_not_completed'
        ,'num_finishing_drivers'
        ,'num_drivers'
        ,'finish_ratio'
        ,'yoe'
        ,'years_from_prime'
	,'dnf_interaction'
	,'dnf_race%_interaction'
	,'dnf_finish_interaction'
    ], axis=1)
    results_full_pred[['age', 'prediction', 'error']]=results_full_pred[['age', 'prediction', 'error']].round(1)

    return results_full_pred

def get_coefficient(model, model_type, X, feature_name):

    if model_type=='ridge':
        features=pd.concat([pd.DataFrame(X.columns), pd.DataFrame(model.coef_)], axis=1)

    features.columns=['feature', 'coefficient']
    return features.loc[features['feature']==feature_name, :]

def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif