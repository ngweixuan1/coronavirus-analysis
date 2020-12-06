import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

def es(covid, days_forecasted):
    countries = covid['location'].unique()
    forecasted = pd.DataFrame({'location': [], 'period': [], 'new_cases_per_million':[]})
    for i in range(len(countries)):
        temp = covid[covid['location'] == countries[i]].sort_values('date')
        if temp['new_cases_per_million'].dropna().count()>1:
            temp['new_cases_per_million']= temp['new_cases_per_million'].apply(lambda x: 0 if x<0 else x)
            model = ExponentialSmoothing(np.array(temp['new_cases_per_million']), trend = 'additive', damped = True)
            fit = model.fit()    
            temp = pd.DataFrame((fit.forecast(days_forecasted)),  columns=["new_cases_per_million"])
            temp['location'] = countries[i]
            temp['period'] = pd.DataFrame(np.linspace(1,days_forecasted,days_forecasted))
            forecasted = pd.concat([forecasted, temp])
    forecasted['new_cases_per_million'] = forecasted['new_cases_per_million'].apply(lambda x: max(0,x))
    return forecasted