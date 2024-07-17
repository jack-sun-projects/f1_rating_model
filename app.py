from flask import Flask, render_template, request
import datetime
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import base64

with open('f1_rating_model.pkl', 'rb') as f:
    model = pickle.load(f)

full_predictions=pd.read_csv('H:/Random files/Data analyst/f1/f1 rating model/full_ratings.csv')

app=Flask(__name__)

def get_rankings(ranking='ma', ma_years=3, min_races_season=3):
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
    rankings=rankings.loc[rankings['num_counting_races']>=min_races_season, :].drop(['num_counting_races'], axis=1)
    rankings=driver_min_max.merge(rankings, how='left', on=['driver_name', 'year'])

    # grouping and calculating score by year
    if ranking=='annual':
        rankings=rankings.loc[rankings['score'].notnull(), :].sort_values('score').reset_index(drop=True)
        rankings['score']=rankings['score'].round(3)
        rankings['rank']=np.arange(len(rankings))+1
        rankings=rankings.iloc[:, [3, 0, 1, 2]]
        return rankings
    if ranking=='ma':
        rankings['ma_score']=rankings.groupby('driver_name')['score'].transform(lambda x: x.rolling(ma_years, ma_years).mean())
        rankings=rankings.loc[rankings['ma_score'].notnull(), :]
        rankings=rankings.sort_values('ma_score').groupby('driver_name').head(1).reset_index(drop=True).drop(['score'], axis=1)
        rankings['year']=(rankings['year']-(ma_years-1)).astype('str') + " - " + rankings['year'].astype('str')
        rankings['ma_score']=rankings['ma_score'].round(3)
        rankings['rank']=np.arange(len(rankings))+1
        rankings=rankings.iloc[:, [3, 0, 1, 2]]
        return rankings

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/driver_career_ranking_3yma.html')
def driver_career_ranking_3yma():
    df=get_rankings().head(25)
    html_table=df.to_html(classes='table table-striped', index=False)
    return render_template('3yma_peak_ranking.html', table=html_table)

@app.route('/driver_career_ranking_5yma.html')
def driver_career_ranking_5yma():
    df=get_rankings(ma_years=5).head(25)
    html_table=df.to_html(classes='table table-striped', index=False)
    return render_template('5yma_peak_ranking.html', table=html_table)

@app.route('/year_list.html')
def year_list():
    years=list(range(1950, datetime.date.today().year+1))
    return render_template('year_list.html', numbers=years)

@app.route('/submit_year', methods=['POST'])
def submit_year():
    selected_year=request.form.get("numbers")
    ranking=get_rankings(ranking='annual')
    ranking=ranking.loc[ranking['year']==int(selected_year), :]
    ranking['rank']=np.arange(len(ranking))+1
    html_table=ranking.to_html(classes='table table-striped', index=False)
    return render_template('annual_driver_ranking.html', table=html_table)

@app.route('/driver_list.html')
def driver_list():
    drivers=list(sorted(get_rankings(ranking='annual')['driver_name'].unique()))
    return render_template('driver_list.html', numbers=drivers)

@app.route('/submit_driver', methods=['POST'])
def submit_driver():
    # create df
    selected_driver=request.form.get("numbers")
    ranking=get_rankings(ranking='annual')
    ranking=ranking.loc[ranking['driver_name']==selected_driver, :].sort_values('year')
    ranking=ranking.drop(['rank', 'driver_name'], axis=1)
    display_table=ranking.transpose()
    display_table.columns=display_table.iloc[0].astype('int')
    display_table=display_table[1:]
    html_table=display_table.to_html(classes='table table-striped', index=False)

    # create line graph
    plt.figure()
    plt.title(selected_driver+' career annual scores')
    plt.plot(ranking['year'], ranking['score'])
    plt.xticks(ranking['year'], rotation=90)
    plt.ylabel('Score')
    plt.ylim(0, 1)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('driver_career_annual_scores.html', table=html_table, plot_url=plot_url)

if __name__=='__main__':
    app.run(port=3000, debug=True)

# plan:
    # create array for general prediction with error
    # create array for equal playing field prediction