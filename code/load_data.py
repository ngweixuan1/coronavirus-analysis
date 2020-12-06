import pandas as pd
import numpy as np
from datetime import timedelta

def load_covid():
       covid = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
       covid['date'] = covid['date'].apply(lambda x: pd.to_datetime(x)).dt.date
       covid = covid[["location", "date", "new_cases_per_million"]].fillna(0)
       return covid

def load_policy():
       policy = pd.read_csv("https://pandemicdatalake.blob.core.windows.net/public/raw/covid-19/covid_policy_tracker/latest/CovidPolicyTracker.csv")
       policy['Date'] = policy['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d')).dt.date
       policy = policy[['CountryName', 'Date', 'C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 
              'C4_Restrictions on gatherings', 'C5_Close public transport', 'C6_Stay at home requirements', 'C7_Restrictions on internal movement', 
              'C8_International travel controls', 'E1_Income support','E2_Debt/contract relief', 'E3_Fiscal measures','E4_International support', 
              'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing','H4_Emergency investment in healthcare', 'H5_Investment in vaccines','H6_Facial Coverings' ]]
       policy = policy.applymap(lambda x: -x if isinstance(x, float) else x)
       return policy

def load_mobility():
       mobility = pd.read_csv("https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv")
       mobility = mobility[mobility['country_region_code'].notna()].drop(["sub_region_2", "metro_area", "iso_3166_2_code", "census_fips_code"], axis=1)
       mobility['date'] = mobility['date'].apply(lambda x: pd.to_datetime(x)).dt.date
       return mobility

def load_temperature(parent):
       data = pd.read_csv(parent + 'GlobalLandTemperaturesByCountry.csv')
       startdate = "2008-01-01"
       enddate = "2012-12-31"
       temperature1 = data[(data["dt"] > startdate) & (data["dt"] < enddate)]
       temperature1 = temperature1.groupby(["Country","dt"])["AverageTemperature"].mean().reset_index().sort_values(by=['Country','dt'])
       temperature1['Year']=[d.split('-')[0] for d in temperature1.dt]
       temperature1['Month']=[d.split('-')[1] for d in temperature1.dt]
       temperature1['Day']=[d.split('-')[2] for d in temperature1.dt]
       temperature1 = temperature1.groupby(["Country","Month","Day"])["AverageTemperature"].mean().reset_index().sort_values(by=['Country','Day',"Month"])
       temperature1["DummyDate"]  = "2020" + "-" + temperature1["Month"] + "-" + temperature1["Day"]
       temperature1 = temperature1[["Country","DummyDate","AverageTemperature"]]
       temperature = temperature1.rename(columns = {"Country":"location"})
       return temperature

def load_sentiments(parent):
       country_converter = pd.read_csv("https://datahub.io/core/country-list/r/data.csv")
       tweets = pd.read_csv(parent + "final_op_sentiments_daily.csv")
       sentiments = tweets.merge(country_converter, how="left", left_on="Country" , right_on="Code")
       return sentiments