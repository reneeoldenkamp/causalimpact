import numpy as np
import pandas as pd
import math
import pylab as py
import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_percentage_error

from data_seasonality import make_dataset_year

def plot_data(model, control_data, int):
    """ Plot endogeneous and exogenous data and the predicted data by the model

    Parameters
    -----------

    """
    # Put all data in panda dataframe
    data = pd.DataFrame({'Observed_data': model.inferences['response'],
        'Predicted_data':model.inferences['point_pred'],
        'Control_data': control_data.squeeze(),
        'Time_points': range(0, len(control_data))},
        columns = ['Observed_data', 'Predicted_data', 'Control_data', 'Time_points'])
    title = "Observed data, predicted data and control data with an intervention at time step " + str(int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = data['Time_points'],
        y = data['Observed_data'],
        mode = 'lines',
        name = "Observed data",
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = data['Time_points'],
        y = data['Predicted_data'],
        mode = 'lines',
        name = "Predicted data",
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = data['Time_points'],
        y = data['Control_data'],
        mode = 'lines',
        name = "Control data",
        line = dict(width = 3)))
    fig.add_vline(x=int, line_width=1, line_dash="dash", line_color="black")
    fig.update_layout(
        xaxis_title="Time points",
        yaxis_title="New IC admissions",
        font_family = "Arial",
        font_color  = "black",
        font_size = 30,
        title=dict(
            text = title,
            font_size = 38,
            x = 0.5),
        legend=dict(
            yanchor="top",
            y=0.90,
            xanchor="left",
            x=0.01)
        )
    # fig.show()

'''
Check if the noise of the model fit is normally distributed
Only before intervention, where the model fits the data
type: choose between 'pre-intervention', 'post-intervention' and 'whole dataset'
int: time point of intervention
'''

def plot_normal_distributed(predictions, real_data, type, int):
    if type == 'pre-intervention':
        residuals = predictions - real_data[int:]

    if type == 'post-intervention':
        residuals = predictions - real_data[int:]

    if type == 'whole_dataset':
        residuals = predictions - real_data[int:]

    fig = ff.create_distplot([residuals],
        ['Distribution of the residuals'],
        curve_type = 'kde',
        bin_size=0.2)

    # Add vertical lines for mean and std
    title = "Distribution of residuals: " + type
    mean = np.mean(residuals)
    name_mean = 'Mean:' + ("{:.3f}".format(mean))
    std = np.std(residuals)
    name_std_pos = "Std:"+("{:.3f}".format(std))

    # Add the normal distribution
    fig2 = ff.create_distplot([residuals],
        ['Distribution of the residuals'],
        curve_type = 'normal',
        bin_size=0.2)
    normal_x = fig2.data[1]['x']
    normal_y = fig2.data[1]['y']
    fig.add_traces(go.Scatter(
        x=normal_x,
        y=normal_y,
        mode = 'lines',
        line = dict(color='red', width = 1),
        name = 'Normal distribution'))
    fig.add_trace(go.Scatter(
        x = [mean,mean],
        y = [-0.1, 0.6],
        mode = 'lines',
        line = dict(color = 'black', width = 2, dash = 'dash'),
        name = name_mean))
    fig.add_trace(go.Scatter(
        x = [std,std],
        y = [-0.1, 0.6],
        mode = 'lines',
        line = dict(color = 'green', width = 2, dash = 'dash'),
        name = name_std_pos))
    fig.update_layout(
        font_family = "Arial",
        font_color  = "black",
        font_size = 30,
        title=dict(
            text = title,
            font_size = 38,
            x = 0.5),
        legend=dict(
            yanchor="top",
            y=0.90,
            xanchor="left",
            x=0.01)
        )
    # fig.show()

    # qq-plot
    # sm.qqplot(predictions - real_data[int:], line='45')
    # py.show()
    return mean, std

'''
Plot the control data and the model data. Check if the control data falls within
the precision interval of the model data.
type: choose between 'pre-intervention', 'post-intervention' and 'whole dataset'
int: time point of intervention
!!! first precicion interval is really big, so taken out of the set!!!
'''
def plot_control_data(model, control_data, type, int):
    if type == 'pre-intervention':
        data = pd.DataFrame(
            {'Predicted_data':model.inferences['point_pred'][1:int],
            'Control_data': control_data[1:int].squeeze(),
            'Time_points': range(0, len(control_data[1:int])),
            'Pred_upper': model.inferences['point_pred_upper'][1:int],
            'Pred_lower': model.inferences['point_pred_lower'][1:int],
            'Observed_data': model.inferences['response'][1:int]},
            columns = ['Predicted_data', 'Control_data', 'Time_points',
            'Pred_upper', 'Pred_lower', 'Observed_data'])

    if type == 'post-intervention':
        data = pd.DataFrame(
            {'Predicted_data':model.inferences['point_pred'][int:],
            'Control_data': control_data[int:].squeeze(),
            'Time_points': range(int-1, len(control_data[1:])),
            'Pred_upper': model.inferences['point_pred_upper'][int:],
            'Pred_lower': model.inferences['point_pred_lower'][int:],
            'Observed_data': model.inferences['response'][int:]},
            columns = ['Predicted_data', 'Control_data', 'Time_points',
            'Pred_upper', 'Pred_lower', 'Observed_data'])

    if type == 'whole_dataset':
        data = pd.DataFrame(
            {'Predicted_data':model.inferences['point_pred'][1:],
            'Control_data': control_data[1:].squeeze(),
            'Time_points': range(0, len(control_data[1:])),
            'Pred_upper': model.inferences['point_pred_upper'][1:],
            'Pred_lower': model.inferences['point_pred_lower'][1:],
            'Observed_data': model.inferences['response'][1:]},
            columns = ['Predicted_data', 'Control_data', 'Time_points',
            'Pred_upper', 'Pred_lower', 'Observed_data'])

    fig = go.Figure()
    fig.add_trace(go.Scatter
        (x = data['Time_points'],
        y = data['Predicted_data'],
        mode = 'lines',
        name = "Predicted data",
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = data['Time_points'],
        y = data['Control_data'],
        mode = 'lines',
        name = "Control data",
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = data['Time_points'],
        y = data['Pred_upper'],
        mode = 'lines',
        marker = dict(color='#444'),
        line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(
        x = data['Time_points'],
        y = data['Pred_lower'],
        mode = 'lines',
        marker = dict(color='#444'),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False))
    fig.add_trace(go.Scatter(
        x = data['Time_points'],
        y = data['Observed_data'],
        mode = 'lines',
        name = 'Observed data',
        line = dict(width = 3)))
    if type == 'whole_dataset':
        fig.add_vline(x = int, line_dash = 'dash', line_color = 'green')

    title = "Control data and predicted data with precision interval of 95%: "+type
    fig.update_layout(
        xaxis_title="Time points",
        yaxis_title="New IC admissions",
        font_family = "Arial",
        font_color  = "black",
        font_size = 30,
        title=dict(
            text = title,
            font_size = 38,
            x = 0.5),
        legend=dict(
            yanchor="top",
            y=0.90,
            xanchor="left",
            x=0.01)
        )
    fig.show()

'''
Plot the residuals over time untill the intervention, plot the mean.
type: choose between 'pre-intervention', 'post-intervention' and 'whole dataset'
int: time point of intervention
'''
def plot_residuals(model, control_data, type, int):
    residuals = model.inferences['point_pred'] - control_data.squeeze()
    if type == 'pre-intervention':
        data = pd.DataFrame(
            {'Residuals': residuals[:int],
            'Time_points': range(0,len(residuals[:int]))},
            columns =['Residuals', 'Time_points'])

    if type == 'post-intervention':
        data = pd.DataFrame(
            {'Residuals': residuals[int:],
            'Time_points': range(int,len(residuals))},
            columns =['Residuals', 'Time_points'])

    if type == 'whole_dataset':
        data = pd.DataFrame(
            {'Residuals': residuals,
            'Time_points': range(0,len(residuals))},
            columns =['Residuals', 'Time_points'])

    mean = np.mean(data['Residuals'])
    name_mean = 'Mean:' + ("{:.3f}".format(mean))
    std = np.std(data['Residuals'])
    name_std_pos = "Std:"+("{:.3f}".format(std))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = data['Time_points'],
        y = data['Residuals'],
        mode = 'lines',
        name = "Residuals",
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = data['Time_points'],
        y = [data['Residuals'].mean()]*len(data),
        mode = 'lines',
        name=name_mean,
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = data['Time_points'],
        y = [std]*len(data),
        mode = 'lines',
        name = name_std_pos,
        visible='legendonly',
        line = dict(width = 3)))

    title = "Residuals and mean of the residuals: "+type
    fig.update_layout(
        xaxis_title="Time points",
        yaxis_title="New IC admissions",
        font_family = "Arial",
        font_color  = "black",
        font_size = 30,
        title=dict(
            text = title,
            font_size = 38,
            x = 0.5),
        legend=dict(
            yanchor="top",
            y=0.90,
            xanchor="left",
            x=0.01)
        )
    fig.show()

'''
Plot the autocorrelation
int: time point of intervention
lags: number of lags
'''
def plot_autocorrelation(predictions, data_real, name, lags):
    # print(predictions, data_real)
    residuals = predictions - data_real
    print(len(residuals))
    # print(residuals)
    acf_data, ci = acf(residuals, nlags = lags, alpha=0.05)
    print("HALLPO", ci)
    # print(acf_data)

    data = pd.DataFrame({'ACF': acf_data, 'Lags': range(0,len(acf_data))},
        columns =['ACF', 'Lags'])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = data['Lags'],
        y = data['ACF'],
        name = "ACF",
        width = 0.1,
        showlegend=False))
    fig.add_trace(go.Scatter(
        x = data['Lags'],
        y = data['ACF'],
        mode = 'markers',
        line=dict(color = 'black'),
        name = "ACF",
        showlegend=False))

    ci_list = []
    for i in range(lags+1):
        # ci_list.append(1.96/(np.sqrt(len(residuals)-i)))
        ci_list.append(2/np.sqrt(len(residuals)))
    ci_list_lower = list(map(lambda x: -x, ci_list))


    fig.add_trace(go.Scatter(
        x = data['Lags'][1:],
        y = ci_list[1:],
        mode = 'lines',
        marker = dict(color='#444'),
        line=dict(width=0),
        showlegend=False))
    fig.add_trace(go.Scatter(
        x = data['Lags'][1:],
        y = ci_list_lower[1:],
        mode = 'lines',
        marker = dict(color='#444'),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False))
    fig.update_layout(
        xaxis_title="Lag",
        yaxis_title="ACF",
        font_family = "Arial",
        font_color  = "black",
        font_size = 40,
        title=dict(
            text = "Autocorrelation Function (ACF)",
            font_size = 48,
            x = 0.5),
        legend=dict(
            yanchor="top",
            y=0.90,
            xanchor="left",
            x=0.01)
        )
    # fig.show()
    # import os
    # if not os.path.exists("images"):
    #     os.mkdir("images")
    # fig.write("images/fig1.png")
    image_name = "images/"+name+"_ACF.png"
    pio.write_image(fig, file=image_name, width=1500 , height=1000)

'''
Plot partial autocorrelation
int: time point of intervention
lags: number of lags
'''
def plot_Partial_ACF(predictions, data_real, name, lags):
    residuals = predictions - data_real
    pacf_data, ci = pacf(residuals, nlags = lags, alpha=0.05)

    data = pd.DataFrame(
        {'PACF': pacf_data, 'Lags': range(0,len(pacf_data))},
        columns =['PACF', 'Lags'])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = data['Lags'],
        y = data['PACF'],
        name = "PACF",
        width = 0.1,
        showlegend=False))
    fig.add_trace(go.Scatter(
        x = data['Lags'],
        y = data['PACF'],
        mode = 'markers',
        line=dict(color = 'black'),
        name = "PACF",
        showlegend=False))

    ci_list = []
    for i in range(lags+1):
        ci_list.append(1.96/(np.sqrt(len(residuals))))
    ci_list_lower = list(map(lambda x: -x, ci_list))

    fig.add_trace(go.Scatter(
        x = data['Lags'][1:],
        y = ci_list[1:],
        mode = 'lines',
        marker = dict(color='#444'),
        line=dict(width=0),
        showlegend=False))
    fig.add_trace(go.Scatter(
        x = data['Lags'][1:],
        y = ci_list_lower[1:],
        mode = 'lines',
        marker = dict(color='#444'),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False))
    fig.update_layout(
        xaxis_title="Time points",
        yaxis_title="IC admissions",
        font_family = "Arial",
        font_color  = "black",
        font_size = 30,
        title=dict(
            text="Partial Autocorrelation",
            font_size = 38,
            x = 0.5),
        legend=dict(
            yanchor="top",
            y=0.90,
            xanchor="left",
            x=0.01)
        )
    # fig.show()
    image_name = "images/" + name + "_PACF.png"
    pio.write_image(fig, file=image_name, width=1500 , height=1000)

from operator import sub
'''
Plots the precision of the model.
It plots three lines:
the difference in IC admissions predicted by the model and the real (observed) data.
the difference in IC admissions of the control data and the real (observed) data.
the difference of the above two, which shows the precision of the model.
'''
def plot_difference(model, control_data, int):
    pred_diff = model.inferences['point_pred'] - model.inferences['response']
    real_diff = control_data.squeeze() - model.inferences['response']
    precision_model = pred_diff - real_diff
    df = pd.DataFrame(
        {'pred_diff':pred_diff,
        'real_diff':real_diff,
        'precision_model':precision_model,
        'Time_points':range(0,len(control_data))})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = df['Time_points'][int:],
        y = df['pred_diff'][int:],
        mode = 'lines',
        name = 'Effect of intervention predicted by the model',
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = df['Time_points'][int:],
        y = df['real_diff'][int:],
        mode = 'lines',
        name = 'Real effect of intervention',
        line = dict(width = 3)))
    fig.add_trace(go.Scatter(
        x = df['Time_points'][int:],
        y = df['precision_model'][int:],
        mode = 'lines',
        name = 'Precision of the model',
        line = dict(width = 3)))
    fig.update_layout(
        xaxis_title="Time points",
        yaxis_title="Difference in IC admissions",
        font_family = "Arial",
        font_color  = "black",
        font_size = 30,
        title=dict(
            text="Real effect of the intervention, and effect predicted by the model",
            font_size = 38,
            x = 0.5),
        legend=dict(
            yanchor="top",
            y=0.90,
            xanchor="left",
            x=0.01)
        )
    fig.show()

'''
Quantitative measures to analyse the model
int: time point of intervention
'''

def analyse_model(predictions, data_real, int):
    """ Quantitative measures to analyse the model

    Parameters
    ----------
    predictions:
        predicted forecast
    data_real:
        real values of forecast
    int:
        point in dataset of the intervention

    Return
    ------
    ME:

    MSE:

    MAPE:

    RMSE:

    MAE:
    """

    ME = (np.sum((data_real[int:] - predictions)))/len(predictions)
    MSE = (np.sum((data_real[int:] - predictions)**2))/len(predictions)
    RMSE = math.sqrt(MSE)
    MAE = (np.sum(abs(data_real[int:] - predictions)))/len(predictions)
    MAPE = mean_absolute_percentage_error(data_real[int:], predictions)
    residuals = predictions - data_real[int:]
    # residuals = pd.DataFrame({'residuals':residuals}, columns=['residuals'])
    # ljung = []
    # for i in range(20):
    #     ljung.append(acorr_ljungbox(residuals, lags=[i+1], return_df=True))

    return ME, MSE, MAPE, RMSE, MAE