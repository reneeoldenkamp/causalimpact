import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import scipy

'''
Covid data from RIVM with IC admissions:
'https://data.rivm.nl/data/covid-19/COVID-19_ic_opnames_tm_03102021.csv'
And weather data from KNMI https://daggegevens.knmi.nl/.
Combined into one panda dataframe, for time index 27-02-2020 untill 3-10-2021.
'''
def data_loader():
    '''
    Version:
    Date_of_report: Date and time at which the datafile is made bij RIVM
    Date_of_statistics: Date of admission to the IC or date that the admission is
        registered by NICE (YYYY-MM-DD)
    IC_admission_notification: Amount of new notifications of admission of covid-19
        patients at the IC, registered by NICE (register_date)
    IC_admission: Amount of new admissions of covid-19 patients at the IC, registered
        by NICE (admission date)
    Een patiënt kan meerdere keren op de IC worden opgenomen. In dit open
    databestand is één IC opname per patiënt opgenomen. RIVM en de NICE registratie
    hebben de methode om in dergelijke gevallen de meest relevante IC opnamedatum te
    bepalen zoveel mogelijk gelijk getrokken, maar de aantallen kunnen iets afwijken
    van de gegevens zoals gepresenteerd door de NICE registratie
    '''
    fields = ['Date_of_statistics','IC_admission_notification','IC_admission']
    df_IC = pd.read_csv('https://data.rivm.nl/data/covid-19/COVID-19_ic_opnames_tm_03102021.csv',
        sep = ";",
        usecols=fields)
    df_IC.rename(columns={"Date_of_statistics":"Date"}, inplace = True)
    df_IC["Date"] = pd.to_datetime(df_IC['Date'])
    df_IC = df_IC.set_index('Date')

    '''
    Weather in de Bilt in a period of 27-02-2020 untill 3-10-2021
    FG: Etmaalgemiddelde windsnelheid (in 0.1 m/s)
    TG: Etmaalgemiddelde temperatuur (in 0.1 graden Celsius)
    Q: Globale straling (in J/cm2)
    RH: Etmaalsom neerslag (in 0.1 mm)
    DR: Duur van de neerslag (in 0.1 uur)
    UG: Etmaalgemiddelde relatieve vochtigheid (in procenten)
    '''
    df_weather = pd.read_csv('weather.txt',
        skiprows=13,
        skipinitialspace=True,
        usecols = ['YYYYMMDD','TG', 'FG', 'Q', 'DR', 'RH', 'UG'])
    df_weather['YYYYMMDD'] = df_weather['YYYYMMDD'].apply(lambda x: pd.to_datetime(str(x), format="%Y%m%d"))
    df_weather['TG'] = df_weather['TG'].div(10)
    df_weather.rename(columns={"YYYYMMDD":"Date"}, inplace=True)
    df_weather = df_weather.set_index('Date')

    df = pd.concat([df_IC, df_weather], axis=1)
    df = df.reset_index()

    mask = (df['Date']>'2020-06-01')&(df['Date']<='2020-11-4')
    df_second_wave = df.loc[mask]

    return df_IC, df_weather, df, df_second_wave

'''
Plots the covid data and the weather data to the time points
'''
def data_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'],
        y = df['FG'],
        mode = 'lines',
        name = "Etmaal gemiddelde windsnelheid" ,
        line = dict(color='orange')))
    fig.add_trace(go.Scatter(x = df['Date'],
        y = df['TG'],
        mode = 'lines',
        name = "Etmaalgemiddelde temperatuur",
        line = dict(color='red')))
    fig.add_trace(go.Scatter(x = df['Date'],
        y = df['UG'],
        mode = 'lines',
        name = "Etmaalgemiddelde relatieve vochtigheid",
        line = dict(color='gray')))
    fig.add_trace(go.Scatter(x = df['Date'],
        y = df['IC_admission'],
        mode = 'lines',
        name = 'IC admission',
        line = dict(color='black')))
    fig.add_trace(go.Scatter(x = df['Date'],
        y = df['RH'],
        mode = 'lines',
        name = 'Etmaalsom neerslag',
        line = dict(color='blue')))
    fig.add_trace(go.Scatter(x = df['Date'],
        y = df['DR'],
        mode = 'lines',
        name = 'Duur van de neerslag',
        line = dict(color='green')))
    fig.update_layout(
        title="Dataset",
        xaxis_title="Time points",
        yaxis_title="Number",
        font_family = "Courier New",
        font_color  = "black",
        title_font_family = "Courier New",
        title_font_size = 28)
    # fig.show()

'''
Scatter plots of the different weather datasets to the IC admissions of the
RIVM dataset.
'''
def scatter_plot(df, method):
    corr = df.corr(method = method)
    corr_FG = method + " correlation = " + "{:.2f}".format(corr['IC_admission'][2])
    corr_TG = method + " correlation = " + "{:.2f}".format(corr['IC_admission'][3])
    corr_Q = method + " correlation = " + "{:.2f}".format(corr['IC_admission'][4])
    corr_DR = method + " correlation = " + "{:.2f}".format(corr['IC_admission'][5])
    corr_RH = method + " correlation = " + "{:.2f}".format(corr['IC_admission'][6])
    corr_UG = method + " correlation = " + "{:.2f}".format(corr['IC_admission'][7])

    fig_2 = make_subplots(rows = 3, cols = 2,
        subplot_titles=(corr_FG, corr_TG, corr_Q, corr_DR, corr_RH, corr_UG))
    fig_2.add_trace(go.Scatter(x = df['IC_admission'],
        y = df['FG'],
        mode = 'markers',
        name = '24-hour average wind speed'),
        row=1, col=1)
    fig_2.add_trace(go.Scatter(x = df['IC_admission'],
        y = df['TG'],
        mode = 'markers',
        name = '24-hour average temperature'),
        row=1, col=2)
    fig_2.add_trace(go.Scatter(x = df['IC_admission'],
        y = df['Q'],
        mode = 'markers',
        name = 'Global radiation'),
        row=2, col=1)
    fig_2.add_trace(go.Scatter(x = df['IC_admission'],
        y = df['DR'],
        mode = 'markers',
        name = 'Duration of precipitation'),
        row=2, col=2)
    fig_2.add_trace(go.Scatter(x = df['IC_admission'],
        y = df['RH'],
        mode = 'markers',
        name = '24-hour sum of precipitation'),
        row=3, col=1)
    fig_2.add_trace(go.Scatter(x = df['IC_admission'],
        y = df['UG'],
        mode = 'markers',
        name = '24-hour average relative humidity'),
        row=3, col=2)
    fig_2.update_layout(
        title="IC admissions vs Weather parameters",
        font_family = "Courier New",
        font_color  = "black",
        title_font_family = "Courier New",
        title_font_size = 28,
        showlegend = False)

    fig_2.update_xaxes(title_text="IC admissions", row=1, col=1)
    fig_2.update_xaxes(title_text="IC admissions", row=1, col=2)
    fig_2.update_xaxes(title_text="IC admissions", row=2, col=1)
    fig_2.update_xaxes(title_text="IC admissions", row=2, col=2)
    fig_2.update_xaxes(title_text="IC admissions", row=3, col=1)
    fig_2.update_xaxes(title_text="IC admissions", row=3, col=2)

    fig_2.update_yaxes(title_text="24-hour average wind speed", row=1, col=1)
    fig_2.update_yaxes(title_text="24-hour average temperature", row=1, col=2)
    fig_2.update_yaxes(title_text="Global raadiation", row=2, col=1)
    fig_2.update_yaxes(title_text="Duration of precipitation", row=2, col=2)
    fig_2.update_yaxes(title_text="24-hour sum of precipitation", row=3, col=1)
    fig_2.update_yaxes(title_text="24-hour average relative humidity", row=3, col=2)

    # fig_2.show()

def correlation_matrix(df, method):
    corr = df.corr(method = method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr = corr.mask(mask)
    fig_3 = px.imshow(corr, text_auto=True,
        x = ['IC notifications', ' IC admission', 'FG: Wind speed',
        'TG: temperature', 'Q: radiance', 'DR: time precipitation',
        'RH: sum precipitation', 'UG: humidity'],
        y = ['IC notifications', ' IC admission', 'FG: Wind speed',
        'TG: temperature', 'Q: radiance', 'DR: time precipitation',
        'RH: sum precipitation', 'UG: humidity'],
        color_continuous_scale='RdBu')
    fig_3.update_layout(
        title_text = 'Correlation Matrix, method: '+ method,
        title=dict(
            font_size = 24,
            x = 0.5
        ),
        xaxis_showgrid = False,
        yaxis_showgrid = False)
    # fig_3.show()

df_IC, df_weather, df, df_second_wave = data_loader()

data_plot(df_second_wave)
scatter_plot(df, "pearson")
correlation_matrix(df, "pearson")
