import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pycountry_convert import country_alpha2_to_country_name, country_name_to_country_alpha2, \
    country_name_to_country_alpha3, map_country_alpha2_to_country_alpha3, map_country_alpha3_to_country_alpha2
from pycountry_convert.convert_country_alpha2_to_continent_code import COUNTRY_ALPHA2_TO_CONTINENT_CODE
from dash.dependencies import Output, Input
from pycountry_convert.country_wikipedia import WIKIPEDIA_COUNTRY_NAME_TO_COUNTRY_ALPHA2
import time
import datetime

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

tabs_styles = {
    'height': '20px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    "font-size": "17px"
    # 'padding': '6px',
    # 'fontWeight': 'bold'
}

tab_selected_style = {
    # 'borderBottom': '1px solid #d6d6d6',
    # 'borderTop': '1px solid #d6d6d6',
    'backgroundColor': '#006296',
    'color': 'white',
    'fontWeight': 'bold',
    "font-size": "17px"
    # 'padding': '6px'
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'COVID-19 Recovery Dashboard'
app.config['suppress_callback_exceptions'] = True
server = app.server
text = "For **{0}**, map shows the countries with which it should open its " \
       "air corridor basis the **{1}** Strictness Level of Policy. The map also shows the COVID Infection Rate for the countries from which the " \
       "flights will operate."


def try_convert(country_name):
    try:
        return country_name_to_country_alpha3(country_name)
    except:
        return None


def get_infection_policy():
    inf_policy_df = pd.read_csv("dataset/owid-covid-data.csv")

    def country_convert(x):
        return country_name_to_country_alpha3(x)

    def to_timestamp(value):
        return time.mktime(datetime.datetime.strptime(value, "%Y-%m-%d").timetuple())

    inf_policy_df['Country Name'] = inf_policy_df.iso_code
    inf_policy_df['New Cases'] = inf_policy_df.new_cases_smoothed
    inf_policy_df['New Deaths'] = inf_policy_df.new_deaths_smoothed
    inf_policy_df['timestamp'] = inf_policy_df.date.apply(lambda x: to_timestamp(x))
    return inf_policy_df


def get_tweets_data():
    tweets_df = pd.read_csv("dataset/final_op_sentiments_weekly.csv")
    return tweets_df


flights_data = pd.read_csv("dataset/merged-airlines.csv")
flights_data['CC'] = flights_data['source_airport_country'].apply(lambda x: try_convert(x))
latlon = pd.read_csv("dataset/latlon.csv", encoding='latin-1')
inf_policy = pd.read_csv("dataset/owid-covid-data.csv").sort_values('date')
lim_date = '2020-11-16'
inf_choropleth_recent_data = inf_policy[inf_policy.date == ('%s' % lim_date)]
a2toa3 = map_country_alpha2_to_country_alpha3()
a3toa2 = map_country_alpha3_to_country_alpha2()
arima = pd.read_csv("dataset/prediction.csv")
risk_factors = pd.read_csv('dataset/risk_factor.csv')
arima['CC'] = arima['location'].apply(lambda x: try_convert(x))

inf_policy_df = get_infection_policy()

continent_filter = {}
for country_code, continent in COUNTRY_ALPHA2_TO_CONTINENT_CODE.items():
    countries = continent_filter.get(continent, [])
    try:
        countries.append(country_alpha2_to_country_name(country_code))
    except:
        continue
    continent_filter[continent] = countries

country_options = []
for country, country_code in WIKIPEDIA_COUNTRY_NAME_TO_COUNTRY_ALPHA2.items():
    country_options.append({'label': country, 'value': country_code})


def get_filtered_map():
    return html.Div([
        html.Div([
            html.H5("Select Country and Policy", className="bold"),
            html.P("Filter by Continent", className="control_label"),
            get_filter_by_continent(),
            html.P("Select the Country", className="control_label"),
            get_filter_by_country(),
            html.P("Select Strictness Level of Policy", className="control_label"),
            dcc.RadioItems(
                id='strictness',
                options=[
                    {'label': 'Lowest Level', 'value': 'low'},
                    {'label': 'Highest Level', 'value': 'high'}
                ],
                value='low',
                className="dcc_control"
            ),
            html.Span("Policy Selected:", className="control_label"),
            html.Span("Moderate", id="policy_selected", className="control_label"),
            html.P(),
            html.Div(id="policy-indicator", style={'padding': '0px 10px 10px 10px'})
        ], className="pretty_container three columns"),
        html.Div([
            html.P(dcc.Markdown(text.format("United States", "Lowest")), id="text", className="label"),
            html.P(),
            html.Button('View Positive Rate', id='submit-val', n_clicks=0),
            dcc.Loading(id="loading-1",
                        children=[dcc.Graph(
                            id='world_map', config={
                                "displaylogo": False,
                            }
                        )], type="default"),
            html.Hr(),
            html.Div([
                html.Div([
                    dcc.Graph(id="arima", config={
                        "displaylogo": False,
                    })
                ], className="pretty_container ten columns"),
                html.Div(
                    id='metrics',
                    className="pretty_container two columns")
            ], className="row")
        ], className="pretty_container nine columns"),
        html.Hr(),
        html.Div("Last Updated At: 20/11/2020", className="last-banner")
    ], className="row")


def get_kpi_plots():
    df = get_infection_policy()

    return html.Div([
        html.Div([
            html.Div([
                html.P("Select the Continent", className="control_label", style={"color": "white"}),
                get_filter_by_continent(id="kpi-continent")], className="three columns"),
            html.Div([
                html.P("Select the Country", className="control_label", style={"color": "white"}),
                get_filter_by_country(id="kpi-country")], className="three columns")
        ], className="row"),
        html.Div([
            html.Div([
                html.H5("New COVID-19 cases spread across the world", className="main-header"),
                dcc.Loading(id="loading-1",
                            children=[html.Div([dcc.Graph(
                                id='new_cases', config={
                                    "displaylogo": False,
                                }
                            )])], type="default"),
                html.Hr(),
                html.H5("New COVID-19 deaths spread across the world", className="main-header"),
                dcc.Loading(id="loading-icon",
                            children=[html.Div([dcc.Graph(
                                id='new_deaths_per_million', config={
                                    "displaylogo": False,
                                }
                            )])], type="default"),
                html.Hr(),
                html.Div("Last Updated At: 20/11/2020", className="last-banner"),
            ], className="pretty_container seven columns"),
            html.Div([
                html.H5("Statistics trend per country", className="main-header"),
                html.P("New Cases", className="sub-section-header"),
                dcc.Graph(id='x-time-series-new-cases', config={
                    "displaylogo": False,
                }),
                html.P("New Deaths", className="sub-section-header"),
                dcc.Graph(id='x-time-series-new-deaths', config={
                    "displaylogo": False,
                }),
                html.P("Sentiment polarity of tweets", className="sub-section-header"),
                dcc.Graph(id='sentiment-tweets', config={
                    "displaylogo": False,
                }),
                html.Hr(),
                html.Div("Last Updated At: 20/11/2020", className="last-banner"),
            ], className="pretty_container five columns", id='rightCol')
        ], className="row"),
    ], id="mainContainer"
    )


def get_filter_by_country(id=None):
    id = "country-selector" if id is None else id
    return dcc.Dropdown(
        id=id,
        options=country_options,
        value="US",
        className="dcc_control"
    )


def get_filter_by_continent(id=None):
    id = "continent-selector" if id is None else id
    return dcc.Dropdown(
        id=id,
        options=[
            {'label': 'All', 'value': 'all'},
            {'label': 'Africa', 'value': 'AF'},
            {'label': 'Asia', 'value': 'AS'},
            {'label': 'Europe', 'value': 'EU'},
            {'label': 'Oceania', 'value': 'OC'},
            {'label': 'North America', 'value': 'NA'},
            {'label': 'South America', 'value': 'SA'}
        ],
        value='all',
        className="dcc_control"
    )


app.layout = html.Div([
    html.Div([
        html.H1('😷   COVID-19 Recovery Dashboard'),
        html.H6('Team 162 - DVA Nightwalkers ✰ Final Project ✰ CSE 6242 ✰ Fall 2020'),
        html.A(html.Button('Reset zoom and applied filters ↺', className='refresh row'), href='/'),
        html.P(),
    ], style={'textAlign': 'center'}),
    dcc.Tabs(id="tabs-styled-with-props", value='tab-1', children=[
        dcc.Tab(label='Key Performance Indicators', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Prediction Engine', value='tab-2', style=tab_style, selected_style=tab_selected_style),
    ], colors={
        "border": "#006296",
        "primary": "#006296",
        "background": "white"
    }),
    html.Div(id='tabs-conninet-props'),
    html.Hr()
])


@app.callback(Output('arima', 'figure'),
              [Input('country-selector', 'value'), Input('strictness', 'value')])
def render_arima(country_code, strictness):
    country_code = a2toa3[country_code]
    filtered_arima = arima[arima["CC"] == country_code]
    lim_df = filtered_arima[filtered_arima['date'] < lim_date]
    x = filtered_arima['date']
    y = filtered_arima['new_cases_per_million']
    xd = lim_df['date']
    yd = lim_df['new_cases_per_million']
    val = "high" if strictness == 'low' else 'low'
    yl = filtered_arima[f'new_cases_per_million_{val}']
    ydl = lim_df[f'new_cases_per_million_{val}']
    fig = go.Figure([
        go.Scatter(
            name='New Cases Per Million',
            x=xd,
            y=ydl,
            line=dict(color='red'),
            mode='lines'
        ),
        go.Scatter(
            name='Prediction with the Policy',
            x=x,
            y=yl,
            line=dict(color='blue', dash='dash'),
            mode='lines',
        ),
        go.Scatter(
            name='Prediction without the Policy',
            x=x,
            y=y,
            mode='lines',
            line=dict(color='red', dash='dash'),
        )
    ])
    fig.update_xaxes(rangeslider_visible=True)
    return fig


@app.callback(Output('tabs-conninet-props', 'children'),
              [Input('tabs-styled-with-props', 'value')])
def render_conninet(tab):
    if tab == 'tab-1':
        return get_kpi_plots()
    elif tab == 'tab-2':
        return get_filtered_map()


@app.callback(Output('country-selector', 'options'),
              [Input('continent-selector', 'value')])
def continent_filer_options(continent):
    if continent == 'all':
        return country_options
    else:
        options = []
        countries = continent_filter[continent]
        for country in countries:
            options.append({
                'label': country,
                'value': country_name_to_country_alpha2(country)
            })
        return options


@app.callback(Output('metrics', 'children'),
              [Input('country-selector', 'value'), Input('strictness', 'value')])
def div_options(country, strictness):
    cont = None
    country_ = a2toa3[country]
    if strictness == 'high':
        fl_df = risk_factors[risk_factors['iso_code'] == country_]
        countries = fl_df.sources_y.iloc[0]
        countries = html.Ul(
            [html.Li(country_alpha2_to_country_name(a3toa2[x]), className="label") for x in countries.split(';')])
        cont = html.Div([
            dcc.Markdown("Air corridor can be opened to the following countries", className="label"),
            countries
        ])
    else:
        cont = dcc.Markdown("Air corridor opened to all the countries", className="label")
    cont_df = arima[arima.location == country_alpha2_to_country_name(country)]
    warn_div = None

    if cont_df.size > 0:
        date_ = cont_df[cont_df.date == max(cont_df.date)]
        act = date_['new_cases_per_million']
        if strictness == 'high':
            act = int(act.iloc[0])
            pred = int(date_['new_cases_per_million_low'].iloc[0])
        else:
            act = int(act.iloc[0])
            pred = int(date_['new_cases_per_million_high'].iloc[0])

    else:
        act = pred = "NA"
    if cont_df.size > 0 and pred > act:
        pct_ch = round((pred - act) / max(act, 1), 2)
        if pct_ch > 0.1:
            warn_div = dcc.Markdown("More than 10% of surge in cases detected! This policy is not recommended.",
                                    className="red label")

    content = html.Div([
        dcc.Markdown("Expected New Cases Per Million", className="label"),
        dcc.Markdown(f"### {act}", className="red"),
        dcc.Markdown("Adjusted New Cases Per Million", className="label"),
        dcc.Markdown(f"### {pred}", className="blue"),
        warn_div,
        cont
    ])

    return content


@app.callback(Output('kpi-country', 'options'),
              [Input('kpi-continent', 'value')])
def kpi_continent_filer_options(continent):
    if continent == 'all':
        return country_options
    else:
        options = []
        countries = continent_filter[continent]
        for country in countries:
            options.append({
                'label': country,
                'value': country_name_to_country_alpha2(country)
            })
        return options


@app.callback(Output("loading-1", "children"))
def input_triggers_spinner():
    time.sleep(1)
    return None


@app.callback(
    Output('new_cases', 'figure'),
    [Input('kpi-continent', 'value'), Input('kpi-country', 'value')]
)
def kpi_plots(continent_code, country_code):
    continent = {
        "AF": "africa",
        "AS": "asia",
        "NA": "north america",
        "SA": "south america",
        "EU": "europe",
        "AU": "australia",
        "OC": None,
        "all": None
    }[continent_code]
    inf_choropleth_recent_data = inf_policy
    fig = px.choropleth(inf_choropleth_recent_data, locationmode="ISO-3",
                        locations='iso_code', color='positive_rate',
                        color_continuous_scale="ylgnbu", template='seaborn',
                        range_color=[0, 0.2], scope=continent, projection='natural earth',
                        animation_frame='date')
    if continent_code == "OC":
        fig.update_geos(
            lataxis_range=[-50, 0], lonaxis_range=[50, 250]
        )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 5
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    return fig


@app.callback(
    Output('new_deaths_per_million', 'figure'),
    [Input('kpi-continent', 'value'), Input('kpi-country', 'value')]
)
def kpi_plots_deaths(continent_code, country_code):
    continent = {
        "AF": "africa",
        "AS": "asia",
        "NA": "north america",
        "SA": "south america",
        "EU": "europe",
        "AU": "australia",
        "OC": None,
        "all": None
    }[continent_code]
    inf_choropleth_recent_data = inf_policy
    inf_choropleth_recent_data['new deaths/M'] = inf_choropleth_recent_data['new_deaths_per_million']
    fig = px.choropleth(inf_choropleth_recent_data, locationmode="ISO-3", locations='iso_code',
                        color='new deaths/M',
                        color_continuous_scale='matter',
                        animation_frame='date',
                        template='seaborn', range_color=[0, 0.5], scope=continent,
                        projection='natural earth')
    if continent_code == "OC":
        fig.update_geos(
            lataxis_range=[-50, 0], lonaxis_range=[50, 250]
        )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 5
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


@app.callback(
    [Output('policy-indicator', 'children'), Output('policy_selected', 'children'), Output('policy_selected', 'style')],
    [Input('strictness', 'value')])
def policy_indicator(strictness):
    policy = {
        "low": [
            {'label': 'International Travel', 'value': 'yes'},
            {'label': 'Revert Stay At Home Order', 'value': 'yes'},
            {'label': 'Grocery Stores - No timings', 'value': 'yes'},
            {'label': 'Pharmacies - Open All', 'value': 'yes'}
        ],
        "high": [
            {'label': 'International Travel', 'value': 'rest'},
            {'label': 'Revert Stay At Home Order', 'value': 'no'},
            {'label': 'Grocery Stores - No timings', 'value': 'no'},
            {'label': 'Pharmacies - Open All', 'value': 'no'}
        ]
    }
    restrictions = policy[strictness]
    elements = []
    for restriction in restrictions:
        elements.append(html.P())
        value = restriction['value']
        label = restriction['label']
        resp = "✔️" if value == "yes" else "❌" if value == "no" else "⚠️"
        elements.append(html.Span(resp, className=f"{value} label"))
        elements.append(html.Span(f" {label}", className="label"))

    strictness_lbl = {
        'high': 'Strict',
        'med': 'Moderate',
        'low': 'Lenient'
    }
    color = {
        'low': 'green',
        'med': 'orange',
        'high': 'red'
    }

    return elements, strictness_lbl[strictness], {'color': color[strictness]}


@app.callback(
    [Output('world_map', 'figure'), Output('text', 'children'), Output('submit-val', 'children')],
    [Input('country-selector', 'value'), Input('strictness', 'value'), Input('submit-val', 'n_clicks')])
def update_graph(country_code, strictness, clicks):
    country = country_alpha2_to_country_name(country_code)
    dest_lat = latlon.loc[latlon['name'] == country]['latitude'].iloc[0]
    dest_lon = latlon.loc[latlon['name'] == country]['longitude'].iloc[0]
    dest_flights = flights_data[flights_data['dest_airport_country'] == country]
    if dest_flights.size == 0:
        fig = px.scatter_geo(lat=[dest_lat], lon=[dest_lon], projection='natural earth')
        markdown = dcc.Markdown("#### NO DATA AVAILABLE FOR THE SELECTED COUNTRY")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    else:
        if clicks % 2 == 0:
            fig = px.choropleth(dest_flights, locationmode="ISO-3", locations='CC', color='flight_capacity',
                                color_continuous_scale="spectral", template='seaborn', projection='natural earth')
            label = "View Positive Rate"
        else:
            fig = px.choropleth(inf_choropleth_recent_data, locationmode="ISO-3", locations='iso_code',
                                color='positive_rate', color_continuous_scale="reds", template='seaborn',
                                projection='natural earth')
            label = "View Flight Capacity"

        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

        country_3 = a2toa3[country_code]
        country_cr = risk_factors[risk_factors['iso_code'] == country_3]
        for val in dest_flights.itertuples():
            source = val[1]
            if strictness == 'high' and not country_name_to_country_alpha3(source) in country_cr['sources_y'].iloc[0]:
                continue
            try:
                lat = latlon.loc[latlon['name'] == source]['latitude'].iloc[0]
                lon = latlon.loc[latlon['name'] == source]['longitude'].iloc[0]
                fig = fig.add_scattergeo(lat=[lat, dest_lat], lon=[lon, dest_lon], line=dict(width=1, color='#1F1F1F'),
                                         mode='lines+text', text="✈️", showlegend=False)
            except:
                continue
        strictness_level = {
            'low': "Lowest",
            'med': "Moderate",
            'high': "Highest"
        }[strictness]

        markdown = dcc.Markdown(text.format(country, strictness_level), className="label")
    return fig, markdown, label


def create_time_series(dff, text):
    fig = px.scatter(dff, x='date', y='new_cases')
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(type='linear')
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=text)
    fig.update_layout(height=300, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    fig.update_xaxes(rangeslider_visible=True)

    return fig


@app.callback(
    Output('x-time-series-new-cases', 'figure'),
    [Input('kpi-country', 'value')])
def update_y_time_series(country_code):
    country_code = a2toa3[country_code]
    dff = inf_policy_df[inf_policy_df['iso_code'] == country_code]
    return create_time_series(dff, f"New cases spike at {country_code}")


def create_time_series_deaths(dff, text):
    fig = px.scatter(dff, x='date', y='new_deaths')
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(type='linear')
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=text)
    fig.update_layout(height=300, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    fig.update_xaxes(rangeslider_visible=True)

    return fig


@app.callback(
    Output('x-time-series-new-deaths', 'figure'),
    [Input('kpi-country', 'value')])
def update_y_time_series(country_code):
    country_code = a2toa3[country_code]
    dff = inf_policy_df[inf_policy_df['iso_code'] == country_code]
    return create_time_series_deaths(dff, f"New deaths spike at {country_code}")


def create_tweets(dff, text):
    fig = px.scatter(dff, x='Date', y='Sentiment Score')
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(type='linear')
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=text)
    fig.update_layout(height=300, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    fig.update_xaxes(rangeslider_visible=True)

    return fig


@app.callback(
    Output('sentiment-tweets', 'figure'),
    [Input('kpi-country', 'value')])
def update_tweets(country_code):
    df = get_tweets_data()
    dff = df[df['Country'] == country_code]
    return create_tweets(dff, f"Tweets on {a2toa3[country_code]}")


if __name__ == '__main__':
    app.run_server()
