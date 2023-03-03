import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.api as sm

import plotly.graph_objects as go
import plotly.express as px

from load_data import correlation_matrix

'''
datapoints: amount of datapoints that need to be computed
trend: "stationary", "linear" or "exponential"
seasonality: "season_0" (no season), "season_1" (1 sin season),
"season_2" (1 sin season and a weekly season)
int: timestep of intervention
post_int: trend after intervention "linear", "exponential" or "none"
'''
def make_dataset(datapoints, trend, seasonality, int, post_int):
    ar = np.array([1, 0.5, 0.4])
    ma = np.array([1, 0.5, 0.4])
    data = sm.tsa.arma_generate_sample(ar, ma, datapoints, scale = 0.5)
    # To get only positive numbers, add a value to sample
    data += 5

    # change the constant for linear and exponential based on the amount of datapoints
    if trend == 'linear':
        x = np.arange(datapoints)
        data = 0.05*x + data

    elif trend == 'exponential':
        x = np.arange(datapoints)
        data = 0.002*(x**2) + data

    # seasonality is 1 sin wave in the data
    if seasonality == 'season_1':
        x = np.linspace(-np.pi,np.pi,datapoints)
        y = np.sin(x)*2
        data += y
        season_name = "year season"

    # If there is a weekly seasonality
    elif seasonality =='season_2':
        x = np.linspace(-np.pi,np.pi,datapoints)
        y = np.sin(x)*2
        data += y
        for i in range(len(data)):
            if i%6 == 0 or i%7 ==0:
                data[i] -= 2
        season_name = "year and weekly season"
    else:
        season_name = "no season"

    data_int = data.copy()
    if post_int == "linear":
        x = np.arange(datapoints-int)
        data_int[int:] = data_int[int:] - 0.1*x
    elif post_int == "exponential":
        x = np.arange(datapoints-intervention)
        data_int[int:] = data_int[int:] - 0.02*(x**2)

    # values can not be smaller than 0
    zero_values = data_int < 0
    data_int[zero_values] = 0

    subtitle = "Trend pre-intervention: " + trend + ", Seasonality: "+ season_name + ", Trend post-intervention: "+ post_int + ", Intervention: " + str(int)
    return data, data_int, subtitle

'''
Generate exogenous datasets with different noise and log shifts
'''
def generate_exo_data(data):
    exo_data = data.copy()
    exo_data += np.random.randn(len(data))*1.5
    exo_data = pd.DataFrame({'exo_data': exo_data}, columns=['exo_data'])
    exo_data = exo_data.shift(periods=5, fill_value=2)

    return exo_data.squeeze()

def format_title(title, subtitle, subtitle_font_size=25):
    title = f'{title}'
    if not subtitle:
        return title
    subtitle = f'<span style="font-size: {subtitle_font_size}px;">{subtitle}</span>'
    return f'{title}<br>{subtitle}'

'''
Generate synthetic dataset adn 4 exogenous datasets. Plot the datasets and plots
a correlation matrix.
datapoints: number of datapoints
int: time point of the Intervention
method: correlation method "pearson" or "spearman"
'''
def gen_plot_data(datapoints, int, method):
    # compute synthetic dataset
    data, data_int, subtitle = make_dataset(datapoints, "linear", "season_0", int, "linear")

    # exogenous dataset 1-4
    data_1, data_int_1, subtitle_1 = make_dataset(datapoints, "stationary", "season_0", int, "none")
    data_2, data_int_2, subtitle_2 = make_dataset(datapoints, "stationary", "season_1", int, "none")
    data_3, data_int_3, subtitle_3 = make_dataset(datapoints, "linear", "season_0", int, "none")
    data_4, data_int_4, subtitle_4 = make_dataset(datapoints, "linear", "season_1", int, "none")

    # generate exo_data
    exo_data_1 = generate_exo_data(data_1)
    exo_data_2 = generate_exo_data(data_2)
    exo_data_3 = generate_exo_data(data_3)
    exo_data_4 = generate_exo_data(data_4)

    # exo_data_2 add double sin season
    exo_data_2 += np.sin(np.linspace(-2*np.pi,2*np.pi,datapoints))

    #exo_data_3 add extra noise
    exo_data_3 += np.random.randn(len(data))*2

    # exo_data_4 add value (20), season (1/2 pi), noise and linear descent
    exo_data_4 = data_4+20
    data_4 += np.sin(np.linspace(-0.5*np.pi,0.5*np.pi,datapoints))
    exo_data_4 += np.random.randn(len(data))*7
    x = np.arange(datapoints)
    exo_data_4 = exo_data_4 - 0.05*x

    df = pd.DataFrame({'control_data':data, 'exo_data_1':exo_data_1,
        'exo_data_2':exo_data_2, 'exo_data_3': exo_data_3, 'exo_data_4': exo_data_4},
        columns = ['control_data', 'exo_data_1', 'exo_data_2', 'exo_data_3', 'exo_data_4'])

    # calculate and plot the correlation_matri
    corr = df.corr(method = method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr = corr.mask(mask)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
    # fig_corr.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = np.arange(datapoints),
        y = data_int,
        mode = 'lines',
        name = 'Data with effect of the intervention',
        line = dict(width = 4)))
    fig.add_trace(go.Scatter(
        x = np.arange(datapoints),
        y = data,
        mode = 'lines',
        name = 'Data without effect of the intervention',
        line = dict(width = 4)))
    fig.add_trace(go.Scatter(
        x = np.arange(datapoints),
        y = exo_data_1,
        mode = 'lines',
        name = "Exo-data_1"))
    fig.add_trace(go.Scatter(
        x = np.arange(datapoints),
        y = exo_data_2,
        mode = 'lines',
        name = "Exo-data_2"))
    fig.add_trace(go.Scatter(
        x = np.arange(datapoints),
        y = exo_data_3,
        mode = 'lines',
        name = "Exo-data_3"))
    fig.add_trace(go.Scatter(
        x = np.arange(datapoints),
        y = exo_data_4,
        mode = 'lines',
        name = "Exo-data_4"))
    fig.add_vline(x=intervention, line_width=1, line_dash="dash", line_color="black")
    fig.update_layout(
        xaxis_title="Time points",
        yaxis_title="IC admissions",
        font_family = "Arial",
        font_color  = "black",
        font_size = 30,
        title=dict(
            text = format_title("Synthetic dataset with and without effect of the intervention",subtitle),
            font_size = 38,
            x = 0.5),
        legend=dict(
            yanchor="top",
            y=0.90,
            xanchor="left",
            x=0.01)
        )
    # fig.show()
    return data_int, exo_data_1, exo_data_2, exo_data_3, exo_data_4

# Make the synthetic dataset
datapoints = 200
intervention = 150

data_int, exo_data_1, exo_data_2, exo_data_3, exo_data_4 = gen_plot_data(datapoints, intervention, "pearson")

data = pd.DataFrame({
    'data_int':data_int, 'exo_data_1':exo_data_1, 'exo_data_2':exo_data_2,
    'exo_data_3':exo_data_3, 'exo_data_4':exo_data_4},
    columns = ['data_int', 'exo_data_1', 'exo_data_2', 'exo_data_3', 'exo_data_4'])

from causalimpact import CausalImpact

impact = CausalImpact(data, [0, intervention], [intervention+1, datapoints-1])
impact_2 = CausalImpact(data, [0, intervention], [intervention+1, datapoints-1],
        model_args={'level':'lltrend'})
impact.run()
impact.summary()
impact.plot()

impact_2.run()
impact_2.summary()
impact_2.plot()
