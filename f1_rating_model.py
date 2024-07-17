import numpy as np
import pandas as pd
import datetime
from pygam import LinearGAM, GAM, LogisticGAM, s, f, l, te
import pickle

# reading csv files
results=pd.read_csv("H:/Random files/Data analyst/f1/f1 rating model/raw/results.csv")
races=pd.read_csv("H:/Random files/Data analyst/f1/f1 rating model/raw/races.csv")
drivers=pd.read_csv("H:/Random files/Data analyst/f1/f1 rating model/raw/drivers.csv", encoding='utf-8-sig')
constructors=pd.read_csv("H:/Random files/Data analyst/f1/f1 rating model/raw/constructors.csv")
status=pd.read_csv("H:/Random files/Data analyst/f1/f1 rating model/raw/status.csv")

# merging tables
results_concat=results.merge(races, how='left', on='raceId')
results_concat=results_concat.merge(drivers, how='left', on='driverId')
results_concat=results_concat.merge(constructors, how='left', on='constructorId')
results_concat=results_concat.merge(status, how='left', on='statusId')

results_concat['driver_name']=results_concat['forename']+" "+results_concat['surname']
results_concat=results_concat.rename({'name_x':'race_name', 'name_y':'constructor_name'}, axis=1)
results_concat['constructor_year']=results_concat['year'].astype('str')+" "+results_concat['constructor_name']
results_concat=results_concat.loc[results_concat['race_name']!='Indianapolis 500',:]

results_concat=results_concat[[
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

# classifying finish status
results_full=results_concat.copy()
results_full.loc[results_full['status'].str.contains('Finished|Lap'), 'dnf']=0
results_full['dnf']=results_full['dnf'].fillna(1)
results_full.loc[results_full['status'].str.contains('Accident|Collision|Spun'), 'status']='retired_human_error'
results_full.loc[(~(results_full['status'].str.contains('retired_human_error')))&(~(results_full['status'].str.contains('Finished|Lap'))),'status']='retired_technical_error'
results_full.loc[results_full['status'].str.contains('Finished|Lap'), 'status']='finished'

# creating technical problems dummy variable
#results_full.loc[results_full['status']=='retired_technical_error', 'technical_problems']=1
#results_full['technical_problems']=results_full['technical_problems'].fillna(0)

# creating human error dummy variable
#results_full.loc[results_full['status']=='retired_human_error', 'human_error']=1
#results_full['human_error']=results_full['human_error'].fillna(0)

# determining % race not completed
results_full=results_full.merge(results_full.groupby(['year', 'round']).max(['laps'])['laps'].reset_index(), how='left', on=['year', 'round'])
results_full['race%_not_completed']=1-results_full['laps_x']/results_full['laps_y']
results_full.loc[results_full['status']=='finished', 'race%_not_completed']=0
results_full=results_full.drop(['laps_x', 'laps_y'], axis=1)

# determining number of drivers in each race
results_full=results_full.merge(results_full.loc[results_full['status']=='finished', :].groupby(['year', 'race_name']).agg('count')['round'].reset_index(), on=['year', 'race_name'], how='left')
results_full=results_full.merge(results_full.groupby(['year', 'race_name']).agg('count')['round_x'].reset_index(), on=['year', 'race_name'], how='left', suffixes=(None, '_x'))
results_full=results_full.rename({'round_x':'round', 'round_y':'num_finishing_drivers', 'round_x_x':'num_drivers'}, axis=1)
results_full['finish_ratio']=results_full['num_finishing_drivers']/results_full['num_drivers']
#results_full['dnf_ratio']=(results_full['num_drivers']-results_full['num_finishing_drivers'])/results_full['num_drivers']

# filtering out drivers with less than 5 total finishes
min_races_filter=results_full.loc[~(results_full['status']=='retired_technical_error'), :]
min_races_filter=min_races_filter.groupby(['driver_name']).size().reset_index(name='count').sort_values('count', ascending=False)
min_races_filter=min_races_filter.loc[min_races_filter['count']>=5, :]

results_full=results_full.merge(min_races_filter['driver_name'], how='inner', on='driver_name')

# determining years of experience, capped at 4
pre_1950_drivers=pd.read_csv("H:/Random files/Data analyst/f1/f1 rating model/raw/earliest_drivers.csv")
pre_1950_drivers=pre_1950_drivers.loc[pre_1950_drivers['raced_flag']>0, ['driver_name', 'year']]

num_races=results_full.groupby(['driver_name', 'year']).agg(num_races=('driver_name', 'count')).reset_index()
num_races=num_races.loc[num_races['num_races']>=3, :]
num_races=pd.concat([num_races, pre_1950_drivers], axis=0).sort_values(['driver_name', 'year'])
num_races['num_races']=num_races['num_races'].fillna('pre_f1_counted')

yoe=pd.concat([results_full, pre_1950_drivers])
yoe=yoe.groupby('driver_name').agg(
        first_year=('year', 'min')
        ,last_year=('year', 'max')
    ).reset_index()
yoe=yoe.merge(pd.DataFrame(np.arange(1946, datetime.date.today().year+1), columns=['year']), how='cross')
yoe=yoe.loc[(yoe['year']>=yoe['first_year'])&(yoe['year']<=yoe['last_year']), ['driver_name', 'year']]
yoe=yoe.merge(num_races, on=['driver_name', 'year'], how='left')
yoe['yoe']=yoe.groupby('driver_name')['num_races'].transform(lambda x: x.shift().rolling(4, 0).count())
results_full=results_full.merge(yoe[['driver_name', 'year', 'yoe']], on=['driver_name', 'year'], how='left')

# determining age and years from prime
results_full['dob']=pd.to_datetime(results_full['dob'])
results_full['date']=pd.to_datetime(results_full['date'])
results_full['age']=(results_full['date']-results_full['dob']).dt.days/365.25
results_full.loc[results_full['age']<25, 'years_from_prime']=25-results_full['age']
results_full.loc[results_full['age']>30, 'years_from_prime']=results_full['age']-30
results_full.loc[results_full['years_from_prime'].isna(), 'years_from_prime']=0

# creating possible dnf interaction term for GAM models
#results_full['dnf_interaction']=results_full[['dnf', 'race%_not_completed', 'finish_ratio']].product(axis=1)
#results_full['dnf_race%_interaction']=results_full[['dnf', 'race%_not_completed']].product(axis=1)
#results_full['dnf_finish_interaction']=results_full[['dnf', 'finish_ratio']].product(axis=1)

# adding yoe for drivers that started earlier than 1953
#earliest_drivers=results_full.groupby('driver_name').agg({'year':'min'}).reset_index()
#earliest_drivers=earliest_drivers.loc[earliest_drivers['year']<=1953, 'driver_name'].unique()

# adding position percentile for log gam model
results_full['position_percentile']=(results_full['positionOrder']-1)/(results_full['num_drivers']-1)

# removing drivers who dnf'd due to technical reasons
#results_full=results_full.loc[results_full['status']!='retired_technical_error', :]

# sorting values for alignment and clarity
results_full=results_full.sort_values(['year', 'round', 'positionOrder'], ascending=[False, True, True]).reset_index(drop=True)

# creating results table before dummying drivers and constructor-years for rankings generation
results_predictions=results_full.reset_index(drop=True).copy()

# creating dummies for drivers and constructor-year pairs
results_full=pd.concat([results_full, pd.get_dummies(results_full['driver_name'])], axis=1).drop(['driver_name'], axis=1)
results_full=pd.concat([results_full, pd.get_dummies(results_full['constructor_year'])], axis=1).drop(['constructor_year'], axis=1)

# splitting dataset into independent and dependent variables
X_logistic_gam=results_full.drop([
    'year',
    'round',    
    'date',
    'race_name',
    'num_finishing_drivers',
    'dob',
    'status',
    'positionOrder',
    'position_percentile',
    'years_from_prime'
], axis=1)
y_logistic_gam=results_full['position_percentile']

# creating GAM terms
logistic_gam_terms=l(0)+l(1)+s(3)+te(0, 1)+te(0, 3)+l(2)+l(4)+s(5, n_splines=5, spline_order=2)
for n in range(6, len(X_logistic_gam.columns)):
    logistic_gam_terms=logistic_gam_terms+l(n)

# fitting model
logistic_gam=LogisticGAM(logistic_gam_terms).fit(X_logistic_gam, y_logistic_gam)

# pickling model
with open('f1_rating_model.pkl', 'wb') as file:
    pickle.dump(logistic_gam, file)

def get_rankings(model, X, results, ranking="3yma", min_races_season=3, download=False):
    
    # getting error term
    errors=pd.concat([results, pd.DataFrame(model.predict_proba(X), columns=['prediction'])], axis=1)
    errors['error']=errors['position_percentile']-errors['prediction']
    errors=errors['error']

    # getting predictions without non-driver factors and calculating 'true' score
    X_scores=X_logistic_gam.copy()
    first_constructor_index=pd.DataFrame(X_scores.columns).loc[X_scores.columns.str.contains('1950'), :].index[0] # this seems very slow
    X_scores.iloc[:, first_constructor_index:]=0
    X_scores=pd.concat([X_scores, results['status']], axis=1)

    X_scores=X_scores.drop('status', axis=1)
    X_scores['num_drivers']=20
    X_scores['finish_ratio']=0.875

    predictions=model.predict_proba(X_scores)

    full_predictions=pd.concat([results, pd.DataFrame(predictions, columns=['prediction']), errors], axis=1)
    full_predictions=full_predictions.loc[full_predictions['status']!='retired_technical_error', :]
    full_predictions['score']=full_predictions['prediction']+full_predictions['error']

    if download is True:
        full_predictions.drop([
            'dnf'
            ,'num_finishing_drivers'
            ,'date'
            ,'years_from_prime'
        ], axis=1).to_csv('full_ratings.csv', index=False, encoding='utf-8-sig')
    else:
        None
        
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
    
logistic_gam_annual_ranking=get_rankings(logistic_gam, X_logistic_gam, results_predictions, ranking='annual', download=True)