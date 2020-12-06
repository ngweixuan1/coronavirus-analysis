import requests
import json
import pandas as pd
import os
from datetime import datetime
import math
import numpy as np
import time

# TODO: Enter your Bearer Token Here
bearer_token = "AAAAAAAAAAAAAAAAAAAAAEfOIwEAAAAAzBbjeOoP4XpWEqydXtsOodfiCfk%3DXnDQNejU60wVcxAcJANbk6FosEQEm2yQPyXwuqhFa4pnHxrjkJ"
# TODO: Change your path please
path = 'D:\\Datasets\\COVID Tweets New'

head = {'Authorization': f'Bearer {bearer_token}',
        'Content-type': 'application/json'}
url = "https://api.twitter.com/labs/2/tweets?expansions=geo.place_id"

# Get the list of all Dataframes
dfs = []
final_df = None
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        df = pd.read_csv(f"{path}\\{filename}", header=None)
        separator = '_' if '_' in filename else '-'
        df['Date'] = datetime.strptime(filename.split(separator)[0]+"/2020", '%B%d/%Y').strftime('%m/%d/%y')
        dfs.append(df)
    final_df = pd.concat(dfs)

print(final_df.head())
print(final_df.shape)
final_df['Country'] = np.nan
pd.options.mode.chained_assignment = None
chunks = math.ceil(final_df.shape[0] / 90000)
print('Chunks', chunks)
for idx in range(0, final_df.shape[0]+89999, 90000):
    if idx > final_df.shape[0]:
        break
    max_idx = min(idx+90000, final_df.shape[0])
    values = final_df[0][idx:max_idx].values
    for i in range(0, 90000, 100):
        ids = ",".join([str(id) for id in values[i:i+100]])
        data = {
            "ids": ids,
            "place.fields": "country_code"
        }
        response = requests.get(url = url, params = data, headers = head)
        if str(response.status_code) != "200":
            print(response.status_code)
            print("Limit Exceeded... Please wait for 20 mins :(")
            time.sleep(20*60)
            print("Done Waiting. Continuing with the request..")
            response = requests.get(url = url, params = data, headers = head)
        resp = json.loads(response.text)
        places = resp['includes']['places']
        country_map = {}
        countries = {}
        for place in places:
            country_map[place['id']] = place['country_code']
        for datum in resp['data']:
            final_df.loc[final_df[0] == int(datum['id']),'Country'] = \
                country_map.get(datum.get('geo', {'place_id': None})['place_id'], None)

        final_df.to_csv('tweets1.csv', ',', mode='w+')
        print(idx+i+100, " Done")