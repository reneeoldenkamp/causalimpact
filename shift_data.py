from covid_data import data_loader, correlation_matrix

import pandas as pd
from numpy import array
import numpy as np
import plotly.graph_objects as go

'''
Shift the weather data with x days, to see if the correlation of the weather
data with the IC admissions is higher on different delays. And plot the
correlation of the IC_admissions with the weather component over the different
shifts.
shift: amount of days to shift
method: correlation method, "pearson" or "spearman"
'''
def shift_data(shifts, method):
    df_IC, df_weather, df, df_second_wave = data_loader()
    corr_list = []
    for i in range(shifts+1):
        # Shift IC admissions
        df_weather = df_weather.shift(periods=1, fill_value=0)
        df = pd.concat([df_IC, df_weather], axis=1)
        df = df.reset_index()
        corr_matrix = df.corr(numeric_only = True, method = method)

        # Select only correlation with IC-admission
        corr_list.append(corr_matrix['IC_admission'])

    # Make lists for correlation for each weather index
    FG = []
    TG = []
    Q = []
    DR = []
    RH = []
    UG = []
    # Add correlation of each lag to each weather index list
    for i in range(len(corr_list)):
        FG.append(corr_list[i][1])
        TG.append(corr_list[i][2])
        Q.append(corr_list[i][3])
        DR.append(corr_list[i][4])
        RH.append(corr_list[i][5])
        UG.append(corr_list[i][6])

    # Plot correlation over different shifts
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = np.arange(len(FG)),
        y = FG,
        mode = 'lines',
        name = "FG: Wind speed",
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = np.arange(len(FG)),
        y = TG,
        mode = 'lines',
        name = "TG: Temperature",
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = np.arange(len(FG)),
        y = Q,
        mode = 'lines',
        name = "Q: Radiance",
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
    x = np.arange(len(FG)),
    y = DR,
    mode = 'lines',
    name = "DR: Duration precipitation",
    line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = np.arange(len(FG)),
        y = RH,
        mode = 'lines',
        name = "RH: Sum precipitation",
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = np.arange(len(FG)),
        y = UG,
        mode = 'lines',
        name = "UG: Humidity",
        line = dict(width = 3)))
    fig.update_layout(
        xaxis_title="Lag",
        yaxis_title="Pearson correlation",
        font_family = "Arial",
        font_color  = "black",
        font_size = 30,
        title=dict(
            text = 'Pearson correlation ICU admissions and weather factors for different lags',
            font_size = 35,
            x = 0.5)
        )
    fig.show()

shift_data(100, "pearson")
