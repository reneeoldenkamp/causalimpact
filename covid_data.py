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
import plotly.io as pio

from causalimpact import CausalImpact
import pmdarima as pm
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

import time

from analyse_model_plotly import *

def data_loader():
    """ Load data from RIVM with IC admissions:
    'https://data.rivm.nl/data/covid-19/COVID-19_ic_opnames_tm_03102021.csv'
    And weather data from KNMI https://daggegevens.knmi.nl/.
    Combined into one panda dataframe, for time index 27-02-2020 untill 3-10-2021.

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
    'IC_admission_notification',

    Return
    ------
    df_IC: Pandas Dataframe
        Dataframe with IC admisscions
    df_weather:
        Data with al the weather coefficients
    df: Pandas Dataframe
        Combined dataset of IC admissions and weather data
    df_second_wave: Pandas Dataframe
        Combined dataset of IC admissions and weather data for specific time period of the second wave
    """

    fields = ['Date_of_statistics','IC_admission']
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

    mask = (df['Date']>'2020-05-01')&(df['Date']<='2020-12-13')
    df_second_wave = df.loc[mask]

    return df_IC, df_weather, df, df_second_wave


def data_plot(df):
    """ Plots the covid data and the weather data to the time points

    Parameters
    ----------
    df: Pandas Dataframe
        dataset
    """
    lockdown = '2020-10-14'
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
    fig.add_trace(go.Scatter(x = df.index,
        y = df['IC_admission'],
        mode = 'lines',
        name = 'IC admission',
        line = dict(color='black')))
    fig.add_vline(x=lockdown, line_width=1, line_dash="dash", line_color="blue")
    fig.add_annotation(x=lockdown, y=100, text='2020-10-14: Partial lockdown', showarrow=False, xanchor="center", font=dict(color='blue'))
    fig.add_vline(x='2020-11-04', line_width=1, line_dash="dash", line_color="green")
    fig.add_annotation(x='2020-11-04', y=110, text='2020-11-04: Extra measures', showarrow=False, xanchor='center', font=dict(color='green'))
    fig.add_vline(x='2020-12-14', line_width=1, line_dash="dash", line_color="red")
    fig.add_annotation(x='2020-12-14', y=120, text='2020-11-14: Strict lockdown', showarrow=False, xanchor='center',
                       font=dict(color='red'))
    fig.add_vline(x='2020-09-01', line_width=1, line_dash="dash", line_color="orange")
    fig.add_annotation(x='2020-09-01', y=60, text='2020-09-01', showarrow=False, xanchor='center',
                       font=dict(color='orange'))
    fig.add_vline(x='2020-09-23', line_width=1, line_dash="dash", line_color="orange")
    fig.add_annotation(x='2020-09-23', y=80, text='2020-09-23', showarrow=False, xanchor='center',
                       font=dict(color='orange'))
    fig.add_vline(x='2020-09-14', line_width=1, line_dash="dash", line_color="orange")
    fig.add_annotation(x='2020-09-14', y=70, text='2020-09-14', showarrow=False, xanchor='center',
                       font=dict(color='orange'))
    fig.add_vline(x='2020-09-30', line_width=1, line_dash="dash", line_color="orange")
    fig.add_annotation(x='2020-09-30', y=90, text='2020-09-30', showarrow=False, xanchor='center',
                       font=dict(color='orange'))
    fig.add_vline(x='2020-02-27', line_width=1, line_dash="dash", line_color="purple")
    fig.add_annotation(x='2020-02-27', y=126, text='2020-02-27', showarrow=False, xanchor='center',
                       font=dict(color='purple'))
    fig.add_vline(x='2020-03-30', line_width=1, line_dash="dash", line_color="purple")
    fig.add_annotation(x='2020-03-30', y=126, text='2020-03-30', showarrow=False, xanchor='center',
                       font=dict(color='purple'))
    fig.add_vline(x='2020-05-01', line_width=1, line_dash="dash", line_color="purple")
    fig.add_annotation(x='2020-05-01', y=126, text='2020-05-01', showarrow=False, xanchor='center',
                       font=dict(color='purple'))

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
        title="Covid-19 Data",
        xaxis_title="Date",
        yaxis_title="Number of new IC admissions",
        font_family = "Courier New",
        font_size=20,
        font_color  = "black",
        title_font_family = "Courier New",
        title_font_size = 38)
    fig.show()


def scatter_plot(df, method):
    """
    Scatter plots of the different weather datasets to the IC admissions of the
    RIVM dataset.

    Parameters
    ----------
    df: Pandas Dataframe
        dataset
    method: Pearson or Spearson correlation method
    """
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
    """ Plots a correlation matrix

    Parameters
    ----------
    df: Pandas Dataframe
        dataset
    method:
        Pearson or Spearman correlation metric

    """
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

def run_covid_data_causalimpact(data, data_real, lockdown, int, int_1, end_date, name):
    """
    Parameters
    ----------
    data: Pandas Dataframe
        Covid dataset
    data_real: Pandas Dataframe
        Covid dataset
    lockdown: datetime
        Date of the partial lockdown, the real intervention
    int: datetime
        Date of fake intervention to split data in validation sets
    int_1: datetime
        1 day after the fake intervention
    name: text
        Name that is used to save the plots and table

    Returns
    -------
    predictions: Pandas Dataframe
        forecast made by the model
    data_real:
        real data to compare the predictions with
    run_time: float
        duration of the model to get predictions
    aic: float
        aic score for the chosen model
    coef: Pandas Dataframe
        all coefficients used by the model
    coef_values.loc['beta.x1']['coef']:
        coefficient of exogenous dataset
    coef_values.loc['beta.x1']['std_err']:
        standard error of exgenous dataset
    coef_values.loc['beta.x1']['pvalues']
        P>|z| of exogenous dataset
    """

    # Run Causalimpact package to get the model
    start_time = time.time()
    impact = CausalImpact(data[:end_date], data_real[:end_date], ['2020-05-01', int], [int_1, end_date],
                          model_args={'level': 'lltrend', 'trend': 'lltrend', 'week_season': True,
                                      'freq_seasonal': [{'period': 365, 'harmonics': 1}], 'exponential': False,
                                      'standardize_data': False, 'data_name': "IC_admission"})

    # Run the model to get the prediction and confidence interval results
    aic, llf, params, coef, sterr, pvalues = impact.run()
    run_time = time.time() - start_time

    coef_values = pd.DataFrame(
        {'Name': params, 'coef': coef, 'std_err': sterr, "pvalues": pvalues}
    )
    coef_values.set_index("Name", inplace=True)
    impact.plot(data_real, fname="covid/"+name)

    predictions = impact.inferences['point_pred']
    ci_low = impact.inferences['point_pred_lower']
    ci_up = impact.inferences['point_pred_upper']
    x_axis = data_real.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis[4:],
        y=ci_low[4:],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        name='uncertainty (low)',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_axis[4:],
        y=ci_up[4:],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        name='uncertainty (high)',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=data_real,
        mode='lines',
        line=dict(color='blue'),
        name='historic'
    ))
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=predictions,
        mode='lines',
        name='forecast',
        line=dict(color='red', dash='dash')
    ))
    fig.add_vline(x=lockdown, line_width=1, line_dash="dash", line_color="black")
    fig.add_vline(x=int, line_width=1, line_dash="dash", line_color="green")
    fig.add_annotation(x=lockdown, y=150, text='Lockdown', showarrow=False)
    fig.add_annotation(x=int, y=170, text='Intervention', showarrow=False)
    fig.update_layout(
        yaxis_title='number of new IC-admissions',
        xaxis_title='date',
        title='Forecast (CausalImpact) of new IC-admissions ',
        hovermode="x"
    )
    fig.show()
    image_name = "covid/" + name + "CI.png"
    pio.write_image(fig, file=image_name, width=1500, height=1000)
    return predictions, data_real, run_time, aic, coef_values, coef_values.loc['beta.x1']['coef'], coef_values.loc['beta.x1'][
        'std_err'], coef_values.loc['beta.x1']['pvalues']

def run_covid_data_ARIMAX(data, data_real, lockdown, int, int_1, name, end_date):
    """

    Parameters
    ----------
    data: Pandas Dataframe
        Covid dataset
    data_real: Pandas Dataframe
        Covid dataset
    lockdown: datetime
        Date of the partial lockdown, the real intervention
    int: datetime
        Date of fake intervention to split data in validation sets
    name: text
        Name that is used to save the plots and table

    Returns
    -------
    predictions: Pandas Dataframe
        forecast made by the model
    data_real:
        real data to compare the predictions with
    run_time: float
        duration of the model to get predictions
    aic: float
        aic score for the chosen model
    coef: Pandas Dataframe
        all coefficients used by the model
    st_err: Pandas Dataframe
        error of all coefficients used by the model
    pvalues: Pandas Dataframe
        P>|z| values for all coefficients used by the model
    -------

    """
    endo_data = data['IC_admission'][:end_date]
    exo_data = data.drop('IC_admission', axis=1)[:end_date]
    train = endo_data[:int]
    len_pred = len(endo_data) - len(train)
    start_time = time.time()
    decomposition = seasonal_decompose(train, model='additive', period=7)
    seasonal = decomposition.seasonal
    train -= seasonal
    # from pmdarima.datasets import load_lynx
    # from pmdarima.arima.utils import nsdiffs
    # # load lynx
    # lynx = load_lynx()
    #
    # # estimate number of seasonal differences using a Canova-Hansen test
    # D = nsdiffs(lynx,
    #             m=365,  # commonly requires knowledge of dataset
    #             max_D=5,
    #             test='ch')  # -> 0
    #
    # # or use the OCSB test (by default)
    # nsdiffs(lynx,
    #         m=365,
    #         max_D=5,
    #         test='ocsb')  # -> 0
    # print(D)
    # breakpoint()
    model = pm.auto_arima(train,
                          start_p=1, start_q=1, start_P=3, start_Q=3,
                          max_p=3, max_q=3, max_P=3, max_Q=3, seasonal=True,
                          stepwise=True, suppress_warnings=True, m=365, D=0, max_D=1,
                          error_action='ignore', trace=True)

    model = model.fit(train, exo_data[:int])
    aic = model.aic()
    coef = model.params()
    st_err = model.bse()
    pvalues = model.pvalues()
    predictions, ci = model.predict(len_pred,
                                    exo_data[int_1:end_date],
                                    return_conf_int=True)
    index_pred = len(data_real[:int])
    predictions += seasonal[:len_pred].values

    run_time = time.time() - start_time

    x_axis = data_real.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis[index_pred:],
        y=ci[:,0],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        name='uncertainty (low)',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_axis[index_pred:],
        y=ci[:,1],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        name='uncertainty (high)',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=data_real,
        mode='lines',
        line=dict(color='blue'),
        name='historic'
    ))
    fig.add_trace(go.Scatter(
        x=x_axis[index_pred:],
        y=predictions,
        mode='lines',
        name='forecast',
        line=dict(color='red', dash='dash')
    ))
    fig.add_vline(x=lockdown, line_width=1, line_dash="dash", line_color="black")
    fig.add_vline(x=int, line_width=1, line_dash="dash", line_color="green")
    fig.add_annotation(x=lockdown, y=150, text='Lockdown', showarrow=False)
    fig.add_annotation(x=int, y=170, text='Intervention', showarrow=False)
    fig.update_layout(
        yaxis_title='number of new IC-admissions',
        xaxis_title='date',
        title='Forecast (ARIMAX) of new IC-admissions ',
        hovermode="x"
    )
    fig.show()
    image_name = "covid/" + name + "CI.png"
    pio.write_image(fig, file=image_name, width=1500, height=1000)
    return predictions, data_real, run_time, aic, coef, st_err, pvalues

def run_covid_data_xgb(data, data_real, lockdown, intervention_1, name, end_date):
    """

    Parameters
    ----------
    data: Pandas Dataframe
        Covid dataset
    data_real: Pandas Dataframe
        Covid dataset
    lockdown: datetime
        Date of the partial lockdown, the real intervention
    int: datetime
        Date of fake intervention to split data in validation sets
    name: text
        Name that is used to save the plots and table:

    Returns
    -------
    predictions: Pandas Dataframe
        Forecast made by the model
    data_real:
        Real data to compare the predictions with
    feature_importance:
        Coefficient of exogenous data
    run_time: float
        Duration of the model to get predictions
    """
    endo_data = data['IC_admission'][:end_date]
    exo_data = data.drop('IC_admission', axis=1)[:end_date]
    len_pred = len(endo_data) - len(endo_data[:intervention_1])
    intervention = intervention_1
    int = len(endo_data[:intervention_1])

    start_time = time.time()

    data_diff = np.diff(endo_data)
    decomposition = seasonal_decompose(data_diff, model='additive', period=7)
    seasonal = decomposition.seasonal
    data_diff = data_diff - seasonal
    #
    # decomposition_2 = seasonal_decompose(data_diff, model='additive', period=365)
    # seasonal_2 = decomposition_2.seasonal
    # data_diff -= seasonal_2

    # Split data
    exo_train = exo_data[:intervention]
    exo_test = exo_data[int:]
    endo_train = data_diff[:int]

    # Make predictions on stationary data
    model = XGBRegressor(booster='gblinear')
    model.fit(exo_train, endo_train)
    feature_importance = model.coef_
    predictions = model.predict(exo_test)

    predictions += seasonal[:len_pred]

    predictions = np.concatenate(([endo_data[int]], predictions)).cumsum()

    run_time = time.time() - start_time

    x_axis = data_real.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=data_real,
        mode='lines',
        line=dict(color='blue'),
        name='historic'
    ))
    fig.add_trace(go.Scatter(
        x=x_axis[int:],
        y=predictions,
        mode='lines',
        name='forecast',
        line=dict(color='red', dash='dash')
    ))
    fig.add_vline(x=lockdown, line_width=1, line_dash="dash", line_color="black")
    fig.add_vline(x=intervention, line_width=1, line_dash="dash", line_color="green")
    fig.add_annotation(x=lockdown, y=150, text='Lockdown', showarrow=False)
    fig.add_annotation(x=intervention, y=170, text='Intervention', showarrow=False)
    fig.update_layout(
        yaxis_title='number of new IC-admissions',
        xaxis_title='date',
        title='Forecast (XGBoost) of new IC-admissions ',
        hovermode="x"
    )
    fig.show()
    image_name = "covid/" + name + "CI.png"
    pio.write_image(fig, file=image_name, width=1500, height=1000)
    return predictions, data_real, feature_importance, run_time


df_IC, df_weather, df, df_second_wave = data_loader()
# shift weather data
df_weather = df_weather.shift(periods=-20, fill_value=0)
df = pd.concat([df_IC, df_weather], axis=1)
# data_plot(df[:'2020-11-30'])
# print(df)
# df.set_index("Date", inplace=True)

# data_plot(df[:'2020-12-14'])
# df_second_wave.set_index("Date", inplace=True)
# data_real = df_second_wave['IC_admission']
# df =df['2020-03-30':'2020-10-14']
# df = df_second_wave
df = df['2020-05-01':'2020-11-30']
data_real = df['IC_admission']

lockdown = '2020-10-14'
intervention = '2020-10-14'
end_date = '2020-11-24'
int_1 = '2020-10-15'
runs = 1
name = "shift_real_20days"
# name = 'shift_seas1_val2'
lags = 20

predictions_ci, data_real, run_time_ci, aic_ci, coef_values, coef_ci, sterr_ci, pvalues_ci = run_covid_data_causalimpact(
    df, data_real, lockdown, intervention, int_1, end_date, name+"_ci.png")
predictions_ARIMAX, data_real, run_time_ARIMAX, aic_ARIMAX, coef_ARIMAX, sterr_ARIMAX, pvalues_ARIMAX = run_covid_data_ARIMAX(
    df, data_real, lockdown, intervention, int_1, name+"_ARIMAX.png", end_date)
predictions_xgb, data_real, feature_importance, run_time_xgb = run_covid_data_xgb(df, data_real, lockdown, intervention, name+"_xgb.png", end_date)

predictions_ci = predictions_ci[int_1:]
ME_ci, MSE_ci, MAPE_ci, RMSE_ci, MAE_ci = analyse_model(predictions_ci, data_real[:end_date], int_1)
# mean_ci, std_ci = plot_normal_distributed(predictions_ci, data_real, 'pre-intervention', intervention)
plot_autocorrelation(predictions_ci, data_real[intervention:], name+"CI", lags)
plot_Partial_ACF(predictions_ci, data_real[intervention:], name+"CI", lags)


ME_ARIMAX, MSE_ARIMAX, MAPE_ARIMAX, RMSE_ARIMAX, MAE_ARIMAX = analyse_model(predictions_ARIMAX, data_real[:end_date], int_1)
# mean_ARIMAX, std_ARIMAX = plot_normal_distributed(predictions_ARIMAX, data_real, 'pre-intervention', intervention)
plot_autocorrelation(predictions_ARIMAX, data_real[intervention:], name+"ARIMAX", lags)
plot_Partial_ACF(predictions_ARIMAX, data_real[intervention:], name+"ARIMAX", lags)

predictions_xgb = predictions_xgb[1:]
ME_xgb, MSE_xgb, MAPE_xgb, RMSE_xgb, MAE_xgb = analyse_model(predictions_xgb, data_real[:end_date], int_1)
# mean_xgb, std_xgb = plot_normal_distributed(predictions_xgb, data_real, 'pre-intervention', intervention)
plot_autocorrelation(predictions_xgb, data_real[int_1:end_date], name+"xgb", 2)
plot_Partial_ACF(predictions_xgb, data_real[int_1:end_date], name+"xgb", 2)
#
# MAPE_ARIMAX, MAPE_xgb = 0, 0
# RMSE_ARIMAX, RMSE_xgb = 0, 0
# MAE_ARIMAX, MAE_xgb = 0, 0
# aic_ARIMAX = 0
# run_time_ARIMAX, run_time_xgb = 0, 0
analysis = pd.DataFrame()
analysis['Model'] = ['CausalImpact', 'ARIMAX', 'XGBoost']
# analysis['Runs'] = [runs, runs, runs]
# analysis['mean_residuals'] = [mean_ci, mean_ARIMAX, mean_xgb]
# analysis['std_residuals'] = [std_ci, std_ARIMAX, std_xgb]
# analysis['ME'] = [ME_ci_tot, ME_ARIMAX_tot, ME_xgb_tot]
# analysis['MSE'] = [MSE_ci_tot, MSE_ARIMAX_tot, MSE_xgb_tot]
analysis['MAPE'] = [MAPE_ci, MAPE_ARIMAX, MAPE_xgb]
analysis['RMSE'] = [RMSE_ci, RMSE_ARIMAX, RMSE_xgb]
analysis['MAE'] = [MAE_ci, MAE_ARIMAX, MAE_xgb]
analysis['AIC'] = [aic_ci, aic_ARIMAX, 0]
# analysis['Loglikelihood'] = [0,0,0]
# analysis['Beta coef'] = [coef_ci,coef_ARIMAX, feature_importance]
# analysis['std err'] = [std_ci,std_xgb,0]
# analysis['Beta z score'] = [0,0,0]
# analysis['Beta P>|z|'] = [pvalues_ci, pvalues_ARIMAX,0]
analysis['Run time'] = [run_time_ci, run_time_ARIMAX, run_time_xgb]

table = analysis.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format,
)
print(table)
coef_ARIMAX = coef_ARIMAX.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format,)
coef_ci = coef_values.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format,)
outF = open("covid/"+name+".txt", "w")
outF.write(table)
outF.write(coef_ci)
outF.write(coef_ARIMAX)
outF.close()
