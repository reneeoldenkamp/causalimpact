import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.api as sm

import plotly.graph_objects as go
import plotly.express as px

from scipy import stats
from scipy.special import boxcox, inv_boxcox

from load_data import correlation_matrix

'''
Generate dataset.
datapoints: amount of datapoints that need to be computed
trend: "stationary", "linear" or "exponential"
seasonality: "season_0" (no season), "season_1" (1 sin season),
"season_2" (1 sin season and a weekly season)
int: timestep of intervention
post_int: trend after intervention "linear", "exponential" or "none"
'''
def gen_dataset(datapoints, trend, seasonality, int, post_int):
    ar = np.array([1, 0.5, 0.4])
    ma = np.array([1, 0.5, 0.4])
    data = sm.tsa.arma_generate_sample(ar, ma, datapoints, scale = 0.5)

    # To get only positive numbers, add a value to sample
    data += 5

    # Change the data to have linear or exponential growth
    if trend == 'linear':
        x = np.arange(datapoints)
        data = 0.05*x + data

    elif trend == 'exponential':
        x = np.arange(datapoints)
        data = 0.002*(x**2) + data

    # Seasonality is 1 sin wave in the data
    if seasonality == 'season_1':
        x = np.linspace(-np.pi,np.pi,datapoints)
        y = np.sin(x)*2
        data += y
        season_name = "year season"

    # If there is a weekly seasonality
    elif seasonality == 'season_2':
        x = np.linspace(-np.pi,np.pi,datapoints)
        y = np.sin(x)*2
        data += y
        for i in range(len(data)):
            if i%6 == 0 or i%7 ==0:
                data[i] -= 2
        season_name = "year and weekly season"
    else:
        season_name = "no season"

    # Set linear or exponential effect of the intervention to the data
    data_int = data.copy()
    if post_int == "linear":
        x = np.arange(datapoints-int)
        data_int[int:] = data_int[int:] - 0.1*x
    elif post_int == "exponential":
        x = np.arange(datapoints-intervention)
        data_int[int:] = data_int[int:] - 0.02*(x**2)

    # Values can not be smaller than 0
    zero_values = data_int < 0
    data_int[zero_values] = 0

    subtitle = "Trend pre-intervention: " + trend + ", Seasonality: "+ season_name + ", Trend post-intervention: "+ post_int + ", Intervention: " + str(int)
    return data, data_int, subtitle

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
def gen_exo_data(datapoints, int, method, trend, season, post_trend):
    # Generate synthetic endogene dataset
    data, data_int, subtitle = gen_dataset(datapoints, trend, season, int, post_trend)

    # Generat exogenous dataset 1-4
    exo_data_1, data_int_1, subtitle_1 = gen_dataset(datapoints, "stationary", "season_2", int, "none")
    exo_data_2, data_int_2, subtitle_2 = gen_dataset(datapoints, "stationary", "season_1", int, "none")
    exo_data_3, data_int_3, subtitle_3 = gen_dataset(datapoints, "linear", "season_0", int, "none")
    exo_data_4, data_int_4, subtitle_4 = gen_dataset(datapoints, "linear", "season_1", int, "none")

    # Adjust exo datasets to get a lower correlation with endo data
    exo_data_1 += np.random.randn(len(data))*1.5
    exo_data_1 = pd.DataFrame({'exo_data_1' : exo_data_1}, columns=['exo_data_1'])
    exo_data_1 = exo_data_1.shift(periods=5, fill_value=2)
    exo_data_1 = exo_data_1.squeeze()

    exo_data_2 += np.random.randn(len(data))*1.5
    exo_data_2 = pd.DataFrame({'exo_data_2' : exo_data_3}, columns=['exo_data_2'])
    exo_data_2 = exo_data_2.shift(periods=5, fill_value=2)
    exo_data_2 = exo_data_2.squeeze()
    exo_data_2 += np.sin(np.linspace(-2*np.pi,2*np.pi,datapoints))

    exo_data_3 += np.random.randn(len(data)) * 1.5
    exo_data_3 = pd.DataFrame({'exo_data_3' : exo_data_3}, columns=['exo_data_3'])
    exo_data_3 = exo_data_3.shift(periods=5, fill_value=2)
    exo_data_3 = exo_data_3.squeeze()
    exo_data_3 += np.random.randn(len(data))*2

    exo_data_4 += np.random.randn(len(data)) * 1.5
    exo_data_4 = pd.DataFrame({'exo_data_4' : exo_data_4}, columns=['exo_data_4'])
    exo_data_4 = exo_data_4.shift(periods=5, fill_value=2)
    exo_data_4 = exo_data_4.squeeze()
    exo_data_4 = exo_data_4+20
    exo_data_4 += np.sin(np.linspace(-0.5*np.pi,0.5*np.pi,datapoints))
    exo_data_4 += np.random.randn(len(data))*7
    x = np.arange(datapoints)
    exo_data_4 = exo_data_4 - 0.05*x

    # Combine all datasets into pandas dataframe
    df = pd.DataFrame({'control_data':data, 'exo_data_1':exo_data_1,
        'exo_data_2':exo_data_2, 'exo_data_3': exo_data_3, 'exo_data_4': exo_data_4},
        columns = ['control_data', 'exo_data_1', 'exo_data_2', 'exo_data_3', 'exo_data_4'])

    # Calculate and plot the correlation_matrix
    corr = df.corr(method = method)
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # corr = corr.mask(mask)
    # fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
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
    fig.show()
    data_real = data
    return data_real, data_int, exo_data_1, exo_data_2, exo_data_3, exo_data_4, corr['control_data']

# Make the synthetic dataset
''' 
If stationary: 'level' = llevel, 'trend' = llevel 
If linear: 'level' = lltrend, 'trend' = lltrend
If exponential: "exponential" = True, 'level' = ?, 'trend' = ? 

If season_0, no freq_seasonal 
If season_1, 'freq_seasonal': [{'period':200, 'harmonics':1}] 
if season_2, 'freq_seasonal': [{'period':200, 'harmonics':1}, {'period':7, 'harmonics':??]  
'''
datapoints = 200
intervention = 150
trend = "stationary"
season = "season_1"
post_trend = "linear"

data_real, data_int, exo_data_1, exo_data_2, exo_data_3, exo_data_4, correlation = gen_exo_data(datapoints, intervention, "pearson", trend, season, post_trend)

data = pd.DataFrame({
    'data_int':data_int, 'exo_data_1':exo_data_1, 'exo_data_2':exo_data_2,
    'exo_data_3':exo_data_3, 'exo_data_4':exo_data_4},
    columns = ['data_int', 'exo_data_1', 'exo_data_2', 'exo_data_3', 'exo_data_4'])

data_high = pd.DataFrame({'data_int':data_int, 'exo_data_2':exo_data_2}, columns=['data_int', 'exo_data_2'])
data_medium = pd.DataFrame({'data_int':data_int, 'exo_data_3':exo_data_3}, columns=['data_int', 'exo_data_3'])
data_low = pd.DataFrame({'data_int':data_int, 'exo_data_4':exo_data_4}, columns=['data_int', 'exo_data_4'])

from causalimpact import CausalImpact

impact_four = CausalImpact(data, data_real, [0, intervention], [intervention+1, datapoints-1],
        model_args={'level':'llevel', 'trend':'llevel', 'freq_seasonal': [{'period':200, 'harmonics':1}], 'standardize_data':True})

impact_high = CausalImpact(data_high, data_real, [0, intervention], [intervention+1, datapoints-1],
        model_args={'level':'llevel', 'trend':'llevel', 'freq_seasonal': [{'period':200, 'harmonics':1}], 'standardize_data':True})

impact_medium = CausalImpact(data_medium, data_real, [0, intervention], [intervention+1, datapoints-1],
        model_args={'level':'llevel', 'trend':'llevel', 'freq_seasonal': [{'period':200, 'harmonics':1}], 'standardize_data':True})

impact_low = CausalImpact(data_low, data_real, [0, intervention], [intervention+1, datapoints-1],
        model_args={'level':'llevel', 'trend':'llevel', 'freq_seasonal': [{'period':200, 'harmonics':1}], 'standardize_data':True})


impact_low.run()
impact_low.summary()
impact_low.plot(data_real)

impact_medium.run()
impact_medium.summary()
impact_medium.plot(data_real)

impact_high.run()
impact_high.summary()
impact_high.plot(data_real)

impact_four.run()
impact_four.summary()
impact_four.plot(data_real)

from analyse_model_plotly import *

plot_data(impact_low, data_real, intervention)
mean_low, std_low = plot_normal_distributed(impact_low, data_real, 'pre-intervention', intervention)
# plot_residuals(impact_low, data_real, 'pre-intervention', intervention)
plot_autocorrelation(impact_low, data_real, intervention, 40)
plot_Partial_ACF(impact_low, data_real, intervention, 40)
plot_difference(impact_low, data_real, intervention)
analyse_model(impact_low, data_real, intervention)
ME_low, MSE_low, RMSE_low, MAE_low = analyse_model(impact_low, data_real, intervention)

plot_data(impact_medium, data_real, intervention)
mean_medium, std_medium = plot_normal_distributed(impact_medium, data_real, 'pre-intervention', intervention)
# plot_residuals(impact_medium, 'pre-intervention', intervention)
plot_autocorrelation(impact_medium, data_real, intervention, 40)
plot_Partial_ACF(impact_medium, data_real, intervention, 40)
plot_difference(impact_medium, data_real, intervention)
analyse_model(impact_medium, data_real, intervention)
ME_medium, MSE_medium, RMSE_medium, MAE_medium = analyse_model(impact_medium, data_real, intervention)

plot_data(impact_high, data_real, intervention)
mean_high, std_high = plot_normal_distributed(impact_high, data_real, 'pre-intervention', intervention)
# plot_residuals(impact_high, 'pre-intervention', intervention)
plot_autocorrelation(impact_high, data_real, intervention, 40)
plot_Partial_ACF(impact_high, data_real, intervention, 40)
plot_difference(impact_high, data_real, intervention)
analyse_model(impact_high, data_real, intervention)
ME_high, MSE_high, RMSE_high, MAE_high = analyse_model(impact_high, data_real, intervention)

plot_data(impact_four, data_real, intervention)
mean_four, std_four = plot_normal_distributed(impact_four, data_real, 'pre-intervention', intervention)
# plot_residuals(impact_four, data_real, 'pre-intervention', intervention)
plot_autocorrelation(impact_four, data_real, intervention, 40)
plot_Partial_ACF(impact_four, data_real, intervention, 40)
plot_difference(impact_four, data_real, intervention)
ME_four, MSE_four, RMSE_four, MAE_four = analyse_model(impact_four, data_real, intervention)


analysis = pd.DataFrame()
analysis['trend'] = [trend, trend, trend, trend ]
analysis['season'] = [season, season, season, season]
analysis['name'] = ['Four', 'High', 'Medium', 'Low']
analysis['correlation'] = [correlation[1], correlation[2], correlation[3], correlation[4]]
analysis['mean_residuals'] = [mean_four, mean_high, mean_medium, mean_low]
analysis['std_residuals'] = [std_four, std_high, std_medium, std_low]
analysis['ME'] = [ME_four, ME_high, ME_medium, ME_low]
analysis['MSE'] = [MSE_four, MSE_high, MSE_medium, MSE_low]
analysis['RMSE'] = [RMSE_four, RMSE_high, RMSE_medium, RMSE_low]
analysis['MAE'] = [MAE_four, MAE_high, MAE_medium, MAE_low]
print(analysis)


