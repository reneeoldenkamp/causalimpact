import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.api as sm

import plotly.graph_objects as go

from scipy import stats
from scipy.special import boxcox, inv_boxcox

def gen_dataset(datapoints, trend, seasonality, int, post_int):
    """ Generate dataset.

    Parameters
    ----------
    datapoints: amount of datapoints that need to be computed
    trend: "stationary", "linear" or "exponential"
    seasonality: "season_0" (no season), "season_1" (1 sin season),
        "season_2" (1 sin season and a weekly season)
    int: timestep of intervention
    post_int: trend after intervention "linear", "exponential" or "none"

    Returns
    -------
    data without effect of intervention, data with effect of intervention, subtitle
    """

    ar = np.array([1])
    ma = np.array([1])

    # , 0.5, 0.4 , 0.5, 0.4
    data = sm.tsa.arma_generate_sample(ar, ma, datapoints, scale = 0.5)

    # To get only positive numbers, add a value to sample
    data += 5

    # Make dataframe with different components so can be plotted
    components = pd.DataFrame({'arma':data}, columns=['arma'])

    # Seasonality is 1 sin wave in the data
    season_year = np.linspace(0,0,datapoints)
    season_week = np.linspace(0,0,datapoints)
    if seasonality == 'season_1':
        x = np.arange(0, datapoints / 20, 0.05)
        season_year = np.sin(x)*2
        data += season_year
        season_name = "year season"

    # If there is a weekly seasonality
    elif seasonality == 'season_2':
        # x = np.linspace(-np.pi,np.pi,(1/3)*datapoints)
        x = np.arange(0,datapoints/10,0.1)
        y = np.sin(x)
        data += y
        j = 0
        data_1 = data.copy()
        for i in range(len(data)):
            j += 1
            if j == 6:
                data[i] -= 2
            if j == 7:
                data[i] -= 2
                j = 0
        season_name = "year and weekly season"
        season_week = data - data_1
        # print(data, data_1, season_week)

    else:
        season_name = "no season"

    # Change the data to have linear or exponential growth
    lin_trend = np.linspace(0,0,datapoints)
    exp_trend = np.linspace(0,0,datapoints)
    if trend == 'linear':
        x = np.arange(datapoints)
        data = (0.05 * x) + data
        lin_trend = 0.05*x

    elif trend == 'exponential':
        x = np.arange(datapoints)

        # data = np.exp(x/150) * data
        # exp_trend = np.exp(x/150)

        # data = (0.002 * (x ** 2)) * data
        # exp_trend = 0.002 * (x ** 2)

        # with inv_boxcox (period is 62)
        data = (0.1 * x) + data
        data = inv_boxcox(data, 0.5)

    data += 1

    # Values can not be smaller than 0
    zero_values = data < 0
    data[zero_values] = 0

    # Set linear or exponential effect of the intervention to the data
    data_int = data.copy()
    if post_int == "linear":
        x = np.arange(datapoints-int)
        data_int[int:] = data_int[int:] - 0.1*x
    elif post_int == "exponential":
        x = np.arange(datapoints-int)
        data_int[int:] = data_int[int:] - 0.02*(x**2)

    zero_values_int = data_int < 0
    data_int[zero_values_int] = 0

    subtitle = "Trend pre-intervention: " + trend + ", Seasonality: "+ season_name + ", Trend post-intervention: "+ post_int + ", Intervention: " + str(int)

    # # Plot components
    # components['season_1'] = season_year
    # components['linear'] = lin_trend
    # components['exponential'] = exp_trend
    # components['season_2'] = season_week
    #
    # fig, axes = plt.subplots(nrows = 5, ncols = 1)
    # components['arma'].plot(ax=axes[0])
    # components['season_1'].plot(ax=axes[1])
    # components['season_2'].plot(ax=axes[2])
    # components['linear'].plot(ax=axes[3])
    # components['exponential'].plot(ax=axes[4])

    return data, data_int, subtitle

def format_title(title, subtitle, subtitle_font_size=25):
    title = f'{title}'
    if not subtitle:
        return title
    subtitle = f'<span style="font-size: {subtitle_font_size}px;">{subtitle}</span>'
    return f'{title}<br>{subtitle}'


def gen_exo_data(datapoints, int, method, trend, season, post_trend):
    """ Generate synthetic dataset and 4 exogenous datasets. Plot the datasets and plots
    a correlation matrix.

    Parameters
    ----------
    datapoints: number of datapoints
    int: time point of the intervention
    method: correlation method "pearson" or "spearman"

    Returns
    -------
    data_real: endogeneous dataset without intervention
    data_int: endogeneous dataset with effect of intervention
    exo_data_1, exo_data_2, exo_data_3, exo_data_4: exogeneous datasets
    corr['control_data']: correlation matrix
    """

    # Generate synthetic endogenous dataset
    data, data_int, subtitle = gen_dataset(datapoints, trend, season, int, post_trend)

    # Generate exogenous dataset 1-4
    if season == 'season_0':
        exo_data_1, data_int_1, subtitle_1 = gen_dataset(datapoints, "linear", "season_0", int, "none")
        exo_data_1 += np.random.randn(len(data)) * 8
        exo_data_2, data_int_2, subtitle_2 = gen_dataset(datapoints, "stationary", "season_1", int, "none")
        exo_data_3, data_int_3, subtitle_3 = gen_dataset(datapoints, "linear", "season_0", int, "none")
        exo_data_4, data_int_4, subtitle_4 = gen_dataset(datapoints, "linear", "season_2", int, "none")
        exo_data_4 += np.random.randn(len(data)) * 5

    if season == 'season_1':
        exo_data_1, data_int_1, subtitle_1 = gen_dataset(datapoints, "stationary", "season_1", int, "none")
        exo_data_1 += np.random.randn(len(data)) * 5
        exo_data_2, data_int_2, subtitle_2 = gen_dataset(datapoints, "stationary", "season_1", int, "none")
        exo_data_2 += np.sin(np.linspace(-2 * np.pi, 2 * np.pi, datapoints))
        exo_data_3, data_int_3, subtitle_3 = gen_dataset(datapoints, "linear", "season_1", int, "none")
        exo_data_3 += np.random.randn(len(data)) * 1.5
        exo_data_4, data_int_4, subtitle_4 = gen_dataset(datapoints, "stationary", "season_1", int, "none")
        exo_data_4 = pd.DataFrame({'exo_data_4': exo_data_4}, columns=['exo_data_4'])
        exo_data_4 = exo_data_4.shift(periods=5, fill_value=2)
        exo_data_4 = exo_data_4.squeeze()
        x = np.arange(datapoints)
        exo_data_4 = exo_data_4 - 0.05 * x

    if season == 'season_2':
        exo_data_1, data_int_1, subtitle_1 = gen_dataset(datapoints, "stationary", "season_1", int, "none")
        exo_data_1 += np.random.randn(len(data)) * 5
        exo_data_2, data_int_2, subtitle_2 = gen_dataset(datapoints, "stationary", "season_2", int, "none")
        exo_data_2 += np.sin(np.linspace(-2 * np.pi, 2 * np.pi, datapoints))
        exo_data_3, data_int_3, subtitle_3 = gen_dataset(datapoints, "exponential", "season_1", int, "none")
        exo_data_3 += np.random.randn(len(data)) * 1.5
        exo_data_4, data_int_4, subtitle_4 = gen_dataset(datapoints, "linear", "season_1", int, "none")
        exo_data_4 = pd.DataFrame({'exo_data_4': exo_data_4}, columns=['exo_data_4'])
        exo_data_4 = exo_data_4.shift(periods=5, fill_value=2)
        exo_data_4 = exo_data_4.squeeze()
        x = np.arange(datapoints)
        exo_data_4 = exo_data_4 - 0.065 * x

    # Combine all datasets into pandas dataframe
    df = pd.DataFrame({'control_data':data, 'exo_data_1':exo_data_1,
        'exo_data_2':exo_data_2, 'exo_data_3': exo_data_3, 'exo_data_4': exo_data_4},
        columns = ['control_data', 'exo_data_1', 'exo_data_2', 'exo_data_3', 'exo_data_4'])

    # Calculate the correlation_matrix
    corr = df[:int].corr(method = method)

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
    fig.add_vline(x=int, line_width=1, line_dash="dash", line_color="black")
    fig.update_layout(
        xaxis_title="Time points",
        yaxis_title="IC admissions",
        font_family = "Arial",
        font_color  = "black",
        font_size = 30,
        title=dict(
            text = format_title("Synthetic dataset with and without effect of the intervention", subtitle),
            font_size = 38,
            x = 0.5),
        legend=dict(
            yanchor="top",
            y=0.90,
            xanchor="left",
            x=0.01)
        )
    # fig.show()
    data_real = data
    return data_real, data_int, exo_data_1, exo_data_2, exo_data_3, exo_data_4, corr['control_data']



# data_real, data_int, exo_data_1, exo_data_2, exo_data_3, exo_data_4, correlation = gen_exo_data(datapoints,
#                                                                                                 intervention,
#                                                                                                 "pearson", trend,
#                                                                                                 season, post_trend)
#
# # data = pd.DataFrame({
# #     'data_int':data_int, 'exo_data_1':exo_data_1, 'exo_data_2':exo_data_2,
# #     'exo_data_3':exo_data_3, 'exo_data_4':exo_data_4},
# #     columns = ['data_int', 'exo_data_1', 'exo_data_2', 'exo_data_3', 'exo_data_4'])
#
# data_high = pd.DataFrame({'data_int': data_int, 'exo_data_3': exo_data_3}, columns=['data_int', 'exo_data_3'])
# # data_medium = pd.DataFrame({'data_int':data_int, 'exo_data_2':exo_data_2}, columns=['data_int', 'exo_data_2'])
# # data_low = pd.DataFrame({'data_int':data_int, 'exo_data_1':exo_data_1}, columns=['data_int', 'exo_data_1'])
#
# # impact_four = CausalImpact(data, data_real, [0, intervention], [intervention+1, datapoints-1],
# #         model_args={'level':'lltrend', 'trend':'lltrend', 'nseasons':0, 'exponential':True, 'standardize_data':False})
#
# impact = CausalImpact(data, data_real, [0, int], [int+1, datapoints-1],
#         model_args={'level':'lltrend', 'trend':'lltrend', 'week_season':True,
#         'freq_seasonal':[{'period':62, 'harmonics':1}], 'exponential':True, 'standardize_data':False})
#
# # impact_medium = CausalImpact(data_medium, data_real, [0, intervention], [intervention+1, datapoints-1],
# #         model_args={'level':'lltrend', 'trend':'lltrend',
# #         'freq_seasonal':[{'period':200, 'harmonics':1}],'nseasons':1, 'exponential':True,'standardize_data':False})
# #
# # impact_low = CausalImpact(data_low, data_real, [0, intervention], [intervention+1, datapoints-1],
# #         model_args={'level':'lltrend', 'trend':'lltrend',
# #         'freq_seasonal':[{'period':200, 'harmonics':1}],'nseasons':1, 'exponential':True,'standardize_data':False})
#
# #
# # impact_low.run()
# # impact_low.summary()
# # impact_low.plot(data_real)
# #
# # impact_medium.run()
# # impact_medium.summary()
# # impact_medium.plot(data_real)
# #
# impact_high.run()
# impact_high.summary()
# impact_high.plot(data_real)
#
# predictions = impact_high.inferences['point_pred'][int:]
# #
# # impact_four.run()
# # impact_four.summary()
# # impact_four.plot(data_real)





    # # plot_data(impact_low, data_real, intervention)
    # mean_low, std_low = plot_normal_distributed(impact_low, data_real, 'pre-intervention', intervention)
    # # plot_residuals(impact_low, data_real, 'pre-intervention', intervention)
    # # plot_autocorrelation(impact_low, data_real, intervention, 40)
    # # plot_Partial_ACF(impact_low, data_real, intervention, 40)
    # # plot_difference(impact_low, data_real, intervention)
    # # analyse_model(impact_low, data_real, intervention)
    # ME_low, MSE_low, MAPE_low, RMSE_low, MAE_low = analyse_model(impact_low, data_real, intervention)
    #
    # # plot_data(impact_medium, data_real, intervention)
    # mean_medium, std_medium = plot_normal_distributed(impact_medium, data_real, 'pre-intervention', intervention)
    # # plot_residuals(impact_medium, 'pre-intervention', intervention)
    # # plot_autocorrelation(impact_medium, data_real, intervention, 40)
    # # plot_Partial_ACF(impact_medium, data_real, intervention, 40)
    # # plot_difference(impact_medium, data_real, intervention)
    # # analyse_model(impact_medium, data_real, intervention)
    # ME_medium, MSE_medium, MAPE_medium, RMSE_medium, MAE_medium = analyse_model(impact_medium, data_real, intervention)
    #
    # # plot_data(impact_high, data_real, intervention)
    # mean_high, std_high = plot_normal_distributed(impact_high, data_real, 'pre-intervention', intervention)
    # # plot_residuals(impact_high, 'pre-intervention', intervention)
    # # plot_autocorrelation(impact_high, data_real, intervention, 40)
    # # plot_Partial_ACF(impact_high, data_real, intervention, 40)
    # # plot_difference(impact_high, data_real, intervention)
    # # analyse_model(impact_high, data_real, intervention)
    # ME_high, MSE_high, MAPE_high, RMSE_high, MAE_high = analyse_model(impact_high, data_real, intervention)
    #
    # # plot_data(impact_four, data_real, intervention)
    # mean_four, std_four = plot_normal_distributed(impact_four, data_real, 'pre-intervention', intervention)
    # # plot_residuals(impact_four, data_real, 'pre-intervention', intervention)
    # # plot_autocorrelation(impact_four, data_real, intervention, 40)
    # # plot_Partial_ACF(impact_four, data_real, intervention, 40)
    # # plot_difference(impact_four, data_real, intervention)
    # ME_four, MSE_four, MAPE_four, RMSE_four, MAE_four = analyse_model(impact_four, data_real, intervention)
    #
    # analysis = pd.DataFrame()
    # analysis['trend'] = [trend, trend, trend, trend ]
    # analysis['season'] = [season, season, season, season]
    # analysis['name'] = ['Four', 'High', 'Medium', 'Low']
    # analysis['correlation'] = [correlation[4], correlation[3], correlation[2], correlation[1]]
    # analysis['mean_residuals'] = [mean_four, mean_high, mean_medium, mean_low]
    # analysis['std_residuals'] = [std_four, std_high, std_medium, std_low]
    # analysis['ME'] = [ME_four, ME_high, ME_medium, ME_low]
    # analysis['MSE'] = [MSE_four, MSE_high, MSE_medium, MSE_low]
    # analysis['MAPE'] = [MAPE_four, MAPE_high, MAPE_medium, MAPE_low]
    # analysis['RMSE'] = [RMSE_four, RMSE_high, RMSE_medium, RMSE_low]
    # analysis['MAE'] = [MAE_four, MAE_high, MAE_medium, MAE_low]
    # print(analysis.to_latex(index=False,
    #                   formatters={"name": str.upper},
    #                   float_format="{:.3f}".format,
    # ))
    # print(analysis)