from synthetic_data import *
from analyse_model_plotly import *

from causalimpact import CausalImpact
import pmdarima as pm
from xgboost import XGBRegressor

from statsmodels.tsa.seasonal import seasonal_decompose
import time

def make_synthetic_data(datapoints, intervention, trend, season, post_trend):
    """ Make synthetic data

    Parameters
    ----------
    datapoints: int
        amount of total datapoint
    intervention: int
        datapoint where intervention takes place
    trend:
        stationary, linear, exponential
    season:
        season_0, season_1, season_2
    post_trend:
        linear, exponential

    Return
    ------
    data: Pandas Dataframe
        dataset with all endogeneous and exogeneous datasets
    data_high: Pandas Dataframe
        dataset with endogeneous data and 1 exogeneous data with high correlation
    data_medium: Pandas Dataframe
        dataset with endogeneous data and 1 exogeneous data with medium correlation
    data_low: Pandas Dataframe
        dataset with endogeneous data and 1 exogeneous data with low correlation
    data_real:
        real data to compare predictions with

    """
    data_real, data_int, exo_data_1, exo_data_2, exo_data_3, exo_data_4, correlation = gen_exo_data(datapoints,
                                                                                                    intervention,
                                                                                                    "pearson", trend,
                                                                                                    season, post_trend)

    data = pd.DataFrame({
        'data_int':data_int, 'exo_data_1':exo_data_1, 'exo_data_2':exo_data_2,
        'exo_data_3':exo_data_3, 'exo_data_4':exo_data_4},
        columns = ['data_int', 'exo_data_1', 'exo_data_2', 'exo_data_3', 'exo_data_4'])

    data_high = pd.DataFrame({'data_int': data_int, 'exo_data': exo_data_3}, columns=['data_int', 'exo_data'])
    data_medium = pd.DataFrame({'data_int':data_int, 'exo_data':exo_data_2}, columns=['data_int', 'exo_data'])
    data_low = pd.DataFrame({'data_int':data_int, 'exo_data':exo_data_1}, columns=['data_int', 'exo_data'])

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=np.arange(len(data['data_int'])),
    #     y=data['data_int'],
    #     mode='lines',
    #     line=dict(color='blue'),
    #     name='data'
    # ))
    # fig.add_vline(x=100, line_width=1, line_dash="dash", line_color="black")
    # fig.add_vline(x=126, line_width=1, line_dash="dash", line_color="black")
    # fig.add_vline(x=500, line_width=1, line_dash="dash", line_color="black")
    # fig.add_annotation(x=100, y=9, text='No full season', showarrow=False, xanchor="left")
    # fig.add_annotation(x=126, y=10, text='1 full season', showarrow=False, xanchor="left")
    # fig.add_annotation(x=500, y=9.5, text='More than 1 full season', showarrow=False, xanchor="center")
    # fig.update_layout(
    #     xaxis_title="Date",
    #     yaxis_title="Number of new ICU admissions",
    #     font_family="Arial",
    #     font_size=35,
    #     font_color="black",
    #     title=dict(
    #         text="Different end dates of training set",
    #         font_size=45,
    #         x=0.5)
    # )
    # fig.show()

    return data, data_high, data_medium, data_low, data_real

def run_synthetic_data_causalimpact(data, data_real, seasonality, name, int):
    """ Run CausalImpact on synthetic data

    Parameters
    ----------
    data: Pandas Dataframe
        endo and exo data
    data_real:
        real data to compare the predictions with
    int: int
        time of intervention

    Return
    ------
    predictions: Pandas Dataframe
        forecast made by the model
    data_real: Pandas Dataframe
        real data to compare the predictions with
    run_time: float
        duration of the model to get predictions
    aic: float
        aic score for the chosen model
        aic score for the chosen model
    coef_values: Pandas Dataframe
        all coefficients used by the model
    coef_values.loc['beta.x1']['coef']:
        coefficient of exogenous dataset
    coef_values.loc['beta.x1']['std_err']:
        standard error of exgenous dataset
    coef_values.loc['beta.x1']['pvalues']
        P>|z| of exogenous dataset
    """
    # Run Causalimpact package to get the mdoel
    start_time = time.time()
    impact = CausalImpact(data, data_real, [0, int], [int+1, len(data)-1],
            model_args={'level':'lltrend', 'trend':'lltrend', 'week_season':True,
           'freq_seasonal':[{'period':seasonality[0], 'harmonics':1}],
            'exponential':False, 'standardize_data':False})

    # Run the model to get the prediction and confidence interval results
    aic, llf, params, coef, sterr, pvalues = impact.run()
    run_time = time.time() - start_time

    coef_values = pd.DataFrame(
        {'Name': params, 'coef':coef, 'std_err':sterr, "pvalues":pvalues}
    )
    coef_values.set_index("Name", inplace=True)
    impact.plot(data_real, fname="images/"+name)

    predictions = impact.inferences['point_pred'][int:]
    ci_low = impact.inferences['point_pred_lower']
    ci_up = impact.inferences['point_pred_upper']

    x_axis = np.arange(len(predictions))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis + 500,
        y=ci_low[int:],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        name='uncertainty (low)',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_axis + 500,
        y=ci_up[int:],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        name='uncertainty (high)',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(data_real[:int])),
        y=data_real[:int],
        mode='lines',
        line=dict(color='blue'),
        name='Observed data'
    ))
    fig.add_trace(go.Scatter(
        x=x_axis + 500,
        y=predictions,
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(int, len(data_real) + int),
        y=data_real[int:],
        mode='lines',
        name='Real data',
        line=dict(color='black')
    ))
    fig.update_layout(
        yaxis_title='New ICU admissions',
        xaxis_title='Date',
        title_text='Forecast (BSTS model) without an exponential argument',
        hovermode="x",
        font_size=35,
        title=dict(
            font_size=45,
            x=0.5),
    )
    # fig.show()
    image_name = "images/" + name + "CI.png"
    pio.write_image(fig, file=image_name, width=1500, height=1000)

    return predictions, data_real, run_time, aic, coef_values, coef_values.loc['beta.x1']['coef'], coef_values.loc['beta.x1']['std_err'], coef_values.loc['beta.x1']['pvalues']

def run_synthetich_data_ARIMA(data, data_real, seasonality, name, intervention):
    """ Run ARIMAX model on synthetic data.

    Parameters
    ----------
   data: Pandas Dataframe
        endo and exo time series data
    data_real: ndarray
        real data to compare the predictions with
    int: int
        timestep where intervention starts

    Return
    ------
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
    """
    endo_data = data['data_int']
    if data.shape[1] > 2:
        exo_data_1 = data['exo_data_1']
        exo_data_2 = data['exo_data_2']
        exo_data_3 = data['exo_data_3']
        exo_data_4 = data['exo_data_4']

        exo_data = pd.DataFrame({
            'exo_data_1': exo_data_1, 'exo_data_2': exo_data_2,
            'exo_data_3': exo_data_3, 'exo_data_4': exo_data_4},
            columns=['exo_data_1', 'exo_data_2', 'exo_data_3', 'exo_data_4'])
    else:
        exo_data = pd.DataFrame({'exo_data': data['exo_data']}, columns=['exo_data'])

        # exo_data = pd.DataFrame({'exo_data':endo_data}, columns=['exo_data'])
    train = endo_data[:intervention]

    start_time = time.time()
    if len(seasonality) > 1:
        decomposition = seasonal_decompose(endo_data, model='additive', period=seasonality[1])
        seasonal = decomposition.seasonal
        train -= seasonal
    model = pm.auto_arima(train,
                        start_p=1, start_q=1, start_P=3, start_Q=3,
                        max_p=3, max_q=3, max_P=3, max_Q=3, seasonal=True,
                        stepwise=True, suppress_warnings=True,
                          m=int(seasonality[0]),
                          D=1, max_D=1,
                        error_action='ignore', trace=True)
    model = model.fit(train, exo_data[:intervention])
    aic = model.aic()
    coef = model.params()
    st_err = model.bse()
    pvalues = model.pvalues()
    predictions, ci = model.predict(100,
                                    exo_data[intervention:],
                                    return_conf_int=True)
    if len(seasonality) > 1:
        predictions += seasonal[intervention-1:]
    run_time = time.time()-start_time
    # print(model.summary())
    # model.plot_diagnostics(figsize=(14,10))
    # plt.show()
    x_axis = np.arange(train.shape[0] + predictions.shape[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis+126,
        y=ci[:, 0],
        # y=ci_lower,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        name='uncertainty (low)',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_axis+126,
        y=ci[:, 1],
        # y=ci_upper,
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
        y=train,
        mode='lines',
        line=dict(color='blue'),
        name='Historic data'
    ))
    fig.add_trace(go.Scatter(
        x=x_axis+126,
        y=predictions,
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(intervention, len(data_real) + intervention),
        y=data_real[intervention:],
        mode='lines',
        name='Real data',
        line=dict(color='black')
    ))
    fig.update_layout(
        yaxis_title='Number of new ICU admissions',
        xaxis_title='Time point',
        title='Forecast (ARIMA) of new ICU admissions ',
        hovermode="x",
        font_size = 20,
        title_font_size = 30
    )
    # fig.show()
    image_name = "images/" + name + "ARIMAX.png"
    pio.write_image(fig, file=image_name, width=1500, height=1000)
    return predictions, data_real, run_time, aic, coef, st_err, pvalues

def run_synthetic_data_xgboost(data, data_real, seasonality, name, intervention):
    """ Run xgboost model on synthetic data.

    Parameters
    ----------
    data: Pandas Dataframe
        Endogenous and exogenous data
    data_real: ndarray
        Real data to compare the predictions with
    seasonality: list
        List with duration of all seasonalities in the data
    name: text
        Name that is used to save the plots and table
    intervention: int
        Timestep where intervention starts

    Return
    ------
    predictions: Pandas Dataframe
        forecast made by the model
    data_real:
        real data to compare the predictions with
    feature_importance:
        coefficient of exogenous data
    run_time: float
        duration of the model to get predictions
    """

    if data.shape[1] > 2:
        exo_data_1 = data['exo_data_1']
        exo_data_2 = data['exo_data_2']
        exo_data_3 = data['exo_data_3']
        exo_data_4 = data['exo_data_4']

        exo_data = pd.DataFrame({
            'exo_data_1': exo_data_1, 'exo_data_2': exo_data_2,
            'exo_data_3': exo_data_3, 'exo_data_4': exo_data_4},
            columns=['exo_data_1', 'exo_data_2', 'exo_data_3', 'exo_data_4'])
    else:
        exo_data = data['exo_data']
    print(exo_data.head())
    start_time = time.time()

    # Differentiate data to make the data stationary
    data_diff = np.diff(data['data_int'])
    if len(seasonality) > 0:
        period = int(seasonality[0])
        print(period)
        decomposition = seasonal_decompose(data_diff, model='additive', period = int(seasonality[0]))
        seasonal = decomposition.seasonal
        data_diff -= seasonal
        if len(seasonality) > 1:
            decomposition = seasonal_decompose(data_diff, model='additive', period = seasonality[1])
            seasonal_2 = decomposition.seasonal
            data_diff -= seasonal_2

    # Split data
    exo_train = exo_data[:intervention]
    exo_test = exo_data[intervention:]
    endo_train = data_diff[:intervention]

    # Make predictions on stationary data
    model = XGBRegressor(booster='gblinear')
    model.fit(exo_train, endo_train)
    feature_importance = model.coef_
    predictions = model.predict(exo_test)

    # Make data non stationary again
    if len(seasonality) > 0:
        predictions += seasonal[intervention-1:]
        if len(seasonality) > 1:
            predictions += seasonal_2[intervention-1:]
    predictions = np.concatenate(([data['data_int'][intervention]], predictions)).cumsum()

    run_time = time.time()-start_time

    # Plot predictions
    fig = go.Figure()
    x_axis = np.arange(0, len(exo_train))
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=data['data_int'],
        mode='lines',
        line=dict(color='blue'),
        name='historic'
    ))
    fig.add_trace(go.Scatter(
        x=x_axis+500,
        y=predictions,
        mode='lines',
        name='forecast',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(intervention, len(data_real) + intervention),
        y=data_real[intervention:],
        mode='lines',
        name='real data',
        line=dict(color='black')
    ))
    fig.update_layout(
        yaxis_title='number of new IC-admissions',
        xaxis_title='date',
        title='Forecast (GBDT) of new IC-admissions ',
        hovermode="x"
    )
    # fig.show()
    image_name = "images/" + name + "xgb.png"
    pio.write_image(fig, file=image_name, width=1500, height=1000)

    return predictions[1:], data_real, feature_importance, run_time



datapoints = 600
intervention = 500
trend = "exponential"
season = "season_2"
post_trend = "exponential"
seasonality = [125.75, 7]#[125.75, 7]
lags = 40
runs = 1
name = "slecht_exp_trend"
# name = "exponetial_season_2"
# name = "acf_goed"+trend+'_'+season+'_'
# data_four, data_high, data_medium, data_low, data_real = make_synthetic_data(datapoints, intervention, trend, season, post_trend)
# data = data_high
# # data = data[:200]
#
# predictions_ci, data_real, run_time_ci, aic_ci, llf_ci, coef_ci, sterr_ci, pvalues_ci = run_synthetic_data_causalimpact(data, data_real, seasonality, name, intervention)
# plot_autocorrelation(predictions_ci, data_real[intervention:], name+"CI", lags)
# plot_Partial_ACF(predictions_ci, data_real[intervention:], name+"CI", lags)
# #
# # predictions_ARIMAX, data_real, run_time_ARIMAX, aic_ARIMAX, coef_ARIMAX, sterr_ARIMAX, pvalues_ARIMAX = run_synthetich_data_ARIMA(data, data_real, seasonality, name, intervention)
# # predictions_xgb, data_real, feature_importance, run_time_xgb = run_synthetic_data_xgboost(data, data_real, seasonality, name, intervention)
#
ME_ci_tot, MSE_ci_tot, MAPE_ci_tot, RMSE_ci_tot, MAE_ci_tot = 0, 0, 0, 0, 0
mean_ci_tot, std_ci_tot = 0, 0
ME_ARIMAX_tot, MSE_ARIMAX_tot, MAPE_ARIMAX_tot, RMSE_ARIMAX_tot, MAE_ARIMAX_tot = 0, 0, 0, 0, 0
mean_ARIMAX_tot, std_ARIMAX_tot = 0, 0
ME_xgb_tot, MSE_xgb_tot, MAPE_xgb_tot, RMSE_xgb_tot, MAE_xgb_tot = 0, 0, 0, 0, 0
mean_xgb_tot, std_xgb_tot = 0, 0
run_time_ci_tot, run_time_ARIMAX_tot, run_time_xgb_tot = 0, 0, 0
aic_ci_tot, aic_ARIMAX_tot = 0, 0
coef_ci_tot, coef_ARIMAX_tot, coef_xgb_tot = 0, 0, 0
sterr_ci_tot, sterr_ARIMAX_tot = 0, 0
pvalues_ci_tot, pvalues_ARIMAX_tot = 0, 0



for i in range(runs):
    data_four, data_high, data_medium, data_low, data_real = make_synthetic_data(datapoints, intervention, trend, season, post_trend)
    # breakpoint()
    data = data_high
    mean_real = np.mean(data_real[intervention:])
    # data = data[:226]
    # data_real = data_real[:226]
#
    predictions_ci, data_real, run_time_ci, aic_ci, coef_values, coef_ci, sterr_ci, pvalues_ci = run_synthetic_data_causalimpact(data, data_real, seasonality, name, intervention)
    # predictions_ARIMAX, data_real, run_time_ARIMAX, aic_ARIMAX, coef_ARIMAX, sterr_ARIMAX, pvalues_ARIMAX = run_synthetich_data_ARIMA(data, data_real, seasonality, name, intervention)
    # predictions_xgb, data_real, feature_importance, run_time_xgb = run_synthetic_data_xgboost(data, data_real, seasonality, name, intervention)
    # mean_pred_ci = np.mean(predictions_ci)

    # ME_ci, MSE_ci, MAPE_ci, RMSE_ci, MAE_ci = analyse_model(predictions_ci, data_real, intervention)
    # mean_ci, std_ci = plot_normal_distributed(predictions_ci, data_real, 'pre-intervention', intervention)
    plot_autocorrelation(predictions_ci, data_real[intervention:], name+"CI", lags)
    plot_Partial_ACF(predictions_ci, data_real[intervention:], name+"CI", lags)
    breakpoint()
    # mean_pred_ARIMAX = np.mean(predictions_ARIMAX)
    # ME_ARIMAX, MSE_ARIMAX, MAPE_ARIMAX, RMSE_ARIMAX, MAE_ARIMAX = analyse_model(predictions_ARIMAX, data_real, intervention)
    # mean_ARIMAX, std_ARIMAX = plot_normal_distributed(predictions_ARIMAX, data_real, 'pre-intervention', intervention)
    # plot_autocorrelation(predictions_ARIMAX, data_real[intervention:], name+"ARIMAX", lags)
    # plot_Partial_ACF(predictions_ARIMAX, data_real[intervention:], name+"ARIMAX", lags)

    ME_xgb, MSE_xgb, MAPE_xgb, RMSE_xgb, MAE_xgb = analyse_model(predictions_xgb, data_real, intervention)
    mean_xgb, std_xgb = plot_normal_distributed(predictions_xgb, data_real, 'pre-intervention', intervention)
    plot_autocorrelation(predictions_xgb, data_real[intervention:], name+"xgb", lags)
    plot_Partial_ACF(predictions_xgb, data_real[intervention:], name+"xgb", lags)

    ME_ci_tot += ME_ci
    MSE_ci_tot += MSE_ci
    MAPE_ci_tot += MAPE_ci
    RMSE_ci_tot += RMSE_ci
    MAE_ci_tot += MAE_ci
    mean_ci_tot += mean_ci
    std_ci_tot += std_ci
    # ME_ARIMAX_tot += ME_ARIMAX
    # MSE_ARIMAX_tot += MSE_ARIMAX
    # MAPE_ARIMAX_tot += MAPE_ARIMAX
    # RMSE_ARIMAX_tot += RMSE_ARIMAX
    # MAE_ARIMAX_tot += MAE_ARIMAX
    # mean_ARIMAX_tot += mean_ARIMAX
    # std_ARIMAX_tot += std_ARIMAX
    ME_xgb_tot += ME_xgb
    MSE_xgb_tot += MSE_xgb
    MAPE_xgb_tot += MAPE_xgb
    RMSE_xgb_tot += RMSE_xgb
    MAE_xgb_tot += MAE_xgb
    mean_xgb_tot += mean_xgb
    std_xgb_tot = std_xgb
    run_time_ci_tot += run_time_ci
    # run_time_ARIMAX_tot += run_time_ARIMAX
    run_time_xgb_tot += run_time_xgb
    aic_ci_tot += aic_ci
    # aic_ARIMAX_tot += aic_ARIMAX
    coef_ci_tot += coef_ci
    # coef_ARIMAX_tot += coef_ARIMAX['exo_data']
    coef_xgb_tot += feature_importance
    sterr_ci_tot += sterr_ci
    # sterr_ARIMAX_tot += sterr_ARIMAX['exo_data']
    pvalues_ci_tot += pvalues_ci
    # pvalues_ARIMAX_tot += pvalues_ARIMAX['exo_data']

analysis = pd.DataFrame()
analysis['Trend'] = [trend, trend, trend]
analysis['Season'] = [season, season, season]
analysis['Model'] = ['CausalImpact', 'ARIMAX', 'XGBoost']
# analysis['Runs'] = [runs, runs, runs]
analysis['mean_residuals'] = [mean_ci_tot/runs, mean_ARIMAX_tot/runs, mean_xgb_tot/runs]
analysis['std_residuals'] = [std_ci_tot/runs, std_ARIMAX_tot/runs, std_xgb_tot/runs]
# analysis['ME'] = [ME_ci_tot/runs, ME_ARIMAX_tot/runs, ME_xgb_tot/runs]
# analysis['MSE'] = [MSE_ci_tot/runs, MSE_ARIMAX_tot/runs, MSE_xgb_tot/runs]
analysis['MAPE'] = [MAPE_ci_tot/runs, MAPE_ARIMAX_tot/runs, MAPE_xgb_tot/runs]
analysis['RMSE'] = [RMSE_ci_tot/runs, RMSE_ARIMAX_tot/runs, RMSE_xgb_tot/runs]
analysis['MAE'] = [MAE_ci_tot/runs, MAE_ARIMAX_tot/runs, MAE_xgb_tot/runs]
analysis['AIC'] = [aic_ci_tot/runs,aic_ARIMAX_tot/runs, 0]
# analysis['Loglikelihood'] = [0,0,0]
analysis['Beta coef'] = [coef_ci_tot/runs,coef_ARIMAX_tot/runs, coef_xgb_tot/runs]
analysis['std err'] = [std_ci_tot/runs,std_xgb_tot/runs,0]
# analysis['Beta z score'] = [0,0,0]
analysis['Beta P>|z|'] = [pvalues_ci_tot/runs, pvalues_ARIMAX_tot/runs,0]
analysis['Run time'] = [run_time_ci_tot/runs, run_time_ARIMAX_tot/runs, run_time_xgb_tot/runs]

analysis['mean'] = [mean_pred_ci, 0, mean_real]

table = analysis.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format,
)
print(table)
# coef_ARIMAX = coef_ARIMAX.to_latex(index=True,
#                   formatters={"name": str.upper},
#                   float_format="{:.3f}".format,)
# std_err_ARIMAX = sterr_ARIMAX.to_latex(index=True,
#                   formatters={"name": str.upper},
#                   float_format="{:.3f}".format,)
coef_ci = coef_values.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format,)
outF = open("images/"+name+".txt", "w")
outF.write(table)
outF.write(coef_ci)
# outF.write(coef_ARIMAX)
# outF.write(std_err_ARIMAX)
outF.close()
