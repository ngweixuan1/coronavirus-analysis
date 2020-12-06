import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
import os,sys,inspect
from load_data import *
from rf import *
from es import *

#load data
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dirname = os.path.dirname(currentdir)
parent = os.path.join(dirname, 'dataset\\')

covid = load_covid()
policy = load_policy()
mobility = load_mobility()
temperature = load_temperature(parent)
sentiments = load_sentiments(parent)

#merge data, extract train test sets
df = covid.merge(policy, left_on=["location", "date"], right_on= ["CountryName", "Date"], how="left").dropna(subset = ["Date", "location", "CountryName"] , how='any').dropna(subset = ["Date", "location", "CountryName"] , how='any').fillna(0)
X = df[['C1_School closing', 'C2_Workplace closing', 
       'C3_Cancel public events', 'C4_Restrictions on gatherings',
       'C5_Close public transport', 
       'C6_Stay at home requirements',
       'C7_Restrictions on internal movement', 
       'C8_International travel controls', 'E1_Income support', 
       'E2_Debt/contract relief', 'E3_Fiscal measures',
       'E4_International support', 'H1_Public information campaigns',
       'H2_Testing policy', 'H3_Contact tracing',
       'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',
       'H6_Facial Coverings' ]].fillna(0)
y = df['new_cases_per_million'].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#RandomforestRegressor to get best_features
best_features = rf(X_train, X_test, y_train, y_test, [20, 25, 30, 35]) #max_depth chosen based on initial exploration

#Repeat for mobility due to different data granularities
df2 = mobility.merge(covid, left_on= ["country_region", "date"], right_on=  ["location", "date"], how="left").dropna(subset = ["date", "location"] , how='any').fillna(0)
X = df2[['retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline']].fillna(0)
y = df2['new_cases_per_million'].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_features2 = rf(X_train, X_test, y_train, y_test, [5, 10, 15])

sd = mobility.groupby(["country_region", "date"]).agg({'retail_and_recreation_percent_change_from_baseline': 'mean',
       'grocery_and_pharmacy_percent_change_from_baseline': 'mean',
       'parks_percent_change_from_baseline': 'mean',
       'transit_stations_percent_change_from_baseline': 'mean',
       'workplaces_percent_change_from_baseline': 'mean',
       'residential_percent_change_from_baseline': 'mean'}).reset_index()
policy = policy.groupby(["CountryName", "Date"]).mean().reset_index()

# Merge all features and rolling up different granularities
df_temp = covid.merge(sd, left_on = ['location', 'date'], right_on = ['country_region', 'date'], how ="left")
df_temp = df_temp.dropna(subset=["country_region", 'date'])
df_temp2 = df_temp.merge(temperature, left_on = ['location', 'date'], how ="left", right_on = ["location", "DummyDate"])
df = df_temp2.merge(policy, left_on=["location", "date"], right_on= ["CountryName", "Date"], how="left")
df = df.merge(sentiments, left_on=["location", "date"], right_on=["Name", "Date"], how="left")
df = df.dropna(subset=["CountryName", 'Date_x']).fillna(0)
features = [*best_features, *best_features2, "AverageTemperature", "Sentiment Score", "Positive", "Negative", "ratio"]

#Lasso
X = df[list(features)]
y= df['new_cases_per_million']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lamda = np.linspace(0,1.0, 11)
r2 = []
for i in range(len(lamda)):
    clf = Lasso(alpha=lamda[i], positive=True).fit(X_train, y_train)
    pred = clf.predict(X_test)
    r2.append(r2_score(y_test, pred))
best_lambda = lamda[np.argmax(r2)]
clf = Lasso(alpha=lamda[i], positive=True).fit(X_train, y_train)
coefficient_pd = pd.DataFrame({'columns': list(X.columns), 'coef' : list(clf.coef_)})
coefficients = np.array(coefficient_pd[coefficient_pd['coef']>0])

# Exponential Smoothing
days_forecasted = 30
forecasted = es(covid, days_forecasted)
last_date = max(df['date'])
forecasted['date'] = forecasted['period'].apply(lambda x: last_date + timedelta(days=x))
df_last = df[df['date']==last_date].reset_index()

output = forecasted.merge(df_last, left_on = ['location'], right_on = ['country_region'], how ='left').dropna(subset=['country_region']).rename({'new_cases_per_million_x': 'new_cases_per_million', 'location_x':'location', 'date_x':'date'}, axis=1)

output['addition_high'] = 0
output['addition_low'] = 0
final_features = coefficient_pd[coefficient_pd['coef']>0]['columns'].to_list()
final_features = ["date", "location", *final_features]
ouput_coefficients = df_last[final_features]


for i in coefficients:
    column = i[0]
    scaling = i[1]
    if column[2] != "_":
        difference = output[column].max() - output[column] 
        output['addition_high'] += scaling*difference
        difference2 = (output[column] - output[column].min()).abs()
        output['addition_low'] -= scaling*difference2
        ouput_coefficients[column+'_high'] = output[column].max()
        ouput_coefficients[column+'_low'] = output[column].min()
    else:
        difference = output[column] - output[column].min() 
        output['addition_high'] += scaling*difference
        difference2 = (output[column].max() - output[column]).abs()
        output['addition_low'] -= scaling*difference2
        ouput_coefficients[column+'_high'] = output[column].min()
        ouput_coefficients[column+'_low'] = output[column].max()
    
output['new_cases_per_million_low'] = output['addition_low'] + output['new_cases_per_million'] 
output['new_cases_per_million_low'] =  output['new_cases_per_million_low'].apply(lambda x: max(0,x))
output['new_cases_per_million_high'] = output['addition_high'] + output['new_cases_per_million'] 

condition = (covid['date'] - last_date).dt.days <=0
final_output = covid[condition]
df_countries = df['location'].unique().tolist()
final_output = final_output[final_output['location'].isin(df_countries)]
final_output['new_cases_per_million_high'] = covid['new_cases_per_million']
final_output['new_cases_per_million_low'] = covid['new_cases_per_million']
final_output2 = pd.concat([final_output, output[['location', 'date','new_cases_per_million','new_cases_per_million_low', 'new_cases_per_million_high' ]]])
final_output2.to_csv(parent + "prediction.csv")
ouput_coefficients.to_csv(parent + "ouput_coefficients.csv")