# Covid-Recovery-Dashboard-DVA

## Overview
We have built a "COVID recovery dashboard" to help businesses take decisions to recover from the pandemic and return to the new normalcy. We have used an innovative approach by combining various factors to predict and guide the public to make meaningful decisions.

## Dashboard

Dashboard link: https://covid-recovery-dashboard.herokuapp.com/

The dashboard has two sections:

1. Key Performance Indicators

Show various COVID statistics like new cases, deaths and sentiment polarity across various countries



2. Prediction Engine

To allow the users visualize the impact of strict and lenient policies for decision makers


### Data
All the datasets required for visualization are present in the `datasets` folder
### Covid19 Cases:
Download https://covid.ourworldindata.org/data/owid-covid-data.csv as owid-covid-data.csv
### Temperature: 
Register an account at https://data.world/data-society/global-climate-change-data. Download "GlobalLandTemperaturesByCountry.csv".
### Tweets:
Register an account and download all csvs at https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset




The above dataset are downloaded and placed into the dataset folder.

## Installation and Execution 
#### Bringing up the dashboard is a simple two-step process:
1. Install all the required packages:
`pip install -r requirements.txt`
2. Execute the dash app
`python app.py`

#### Data Extraction and Transformation

The following data collection/analysis scripts need not be executed to view the final results. Added them here just for reference.

Please navigate to `code` folder to execute the following scripts

##### Infection/Policy Data

| File Name         | Description                                                                                                                                                                                                                           |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **main.py:**      | To run the random forest feature selection and exponential smoothing.   Inputs: dataset/GlobalLandTemperaturesByCountry.csv and dataset/final_op_sentiments_daily.csv Outputs: dataset/prediction.csv and dataset/output_coefficients |
| **load_data.py:** | Called in main.py to load necessary dataset                                                                                                                                                                                           |
| **rf.py:**        | Called in main.py to run the random forest algorithm                                                                                                                                                                                  |
| **es.py:**        | Called in main.py to run exponential smoothing                                                                                                                                                                                        |

##### Flight Data
| File Name                              | Description                                                                                                 |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **flights_data_cleaning.R:**           | for data cleaning and munging                                                                               |
| **flights_aggregation_for_viz.ipynb:** | For aggregation of results   Input: dataset/flights_with_capacity.ipynb Output: dataset/merged_airlines.csv |
| **Risk Factor.ipynb:**                 | For risk factor computation  Input: dataset/merged_airlines.csv Output: dataset/risk_factor.csv             |

##### Tweets
| File Name                         | Description                                                                                                                                                                                                                                                                                                |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **twitter_covid_ds.py:**          | The purpose of this is to annotate individual tweets with the country of origin.  Input: Download all the csvs from https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset and place it in a folder. Please provide the folder path in line 13 and run the program  Output: tweets1.csv |
| **DVA Sentiment Analysis.ipynb:** | The purpose of this is aggregate the data and compute the statistics of tweets  Input: tweets1.csv obtained from previous script  Output: final_op_sentiments_daily.csv/ final_op_sentiments_weekly.csv                                                                                                    |
