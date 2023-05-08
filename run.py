import numpy as np
from synthetic_data import *
from analyse_model_plotly import *

from causalimpact import CausalImpact

import pmdarima as pm
from pmdarima import model_selection

import xgboost as xgb

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

    return data, data_high, data_medium, data_low, data_real

def run_synthetic_data_causalimpact(data, data_real, int):
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
    predictions: Pandas Datafram
        forecast made by the model
    data_real:
        real data to compare the predictions with
    """
    # Run Causalimpact package to get the mdoel
    impact = CausalImpact(data, data_real, [0, int], [int+1, datapoints-1],
            model_args={'level':'lltrend', 'trend':'lltrend', 'week_season':True,
            'freq_seasonal':[{'period':62, 'harmonics':1}], 'exponential':True, 'standardize_data':False})

    # Run the model to get the prediction and confidence interval results
    impact.run()
    impact.summary()
    impact.plot(data_real)

    predictions = impact.inferences['point_pred'][int:]

    return predictions, data_real

def run_synthetich_data_ARIMA(data, data_real, int):
    """ Run ARIMAX model on synthetic data.

    Parameters
    ----------
   data: Pandas Dataframe
        endo and exo data
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
    """

    endo_data = data['data_int']
    exo_data = data['exo_data']

    train = endo_data[:int]
    model = pm.auto_arima(train, exogneous = exo_data,start_p=1, start_q=1, start_P=1, start_Q=1,
                        max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                        stepwise=True, suppress_warnings=True, D=10, max_D=10,
                        error_action='ignore')
    predictions, ci = model.predict(100, return_conf_int=True)
    x_axis = np.arange(train.shape[0] + predictions.shape[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis+500,  # x, then x reversed
        y=ci[:, 0],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        name='uncertainty (low)',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_axis+500,
        y=ci[:, 1],
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
        x=np.arange(int, len(data_real) + int),
        y=data_real[int:],
        mode='lines',
        name='real data',
        line=dict(color='black')
    ))
    fig.update_layout(
        yaxis_title='number of new IC-admissions',
        xaxis_title='date',
        title='Forecast (ARIMA) of new IC-admissions ',
        hovermode="x"
    )
    fig.show()

    print(data_real.type)
    return predictions, data_real

def run_synthetic_data_xgboost(data, data_real, int):
    """ Run xgboost model on synthetic data.

    Parameters
    ----------
    data: Pandas Dataframe
        endo and exo data
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
    """

    y = data['data_int']
    X = data_high['exo_data']
    X_train = X[:int]
    X_test = X[int:]
    y_train = y[:int]

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    fig = go.Figure()
    x_axis = np.arange(0, len(X_train))
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=y_train,
        mode='lines',
        line=dict(color='blue'),
        name='historic'
    ))
    fig.add_trace(go.Scatter(
        x=x_axis + 500,
        y=predictions,
        mode='lines',
        name='forecast',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(int, len(data_real) + int),
        y=data_real[int:],
        mode='lines',
        name='real data',
        line=dict(color='black')
    ))
    fig.update_layout(
        yaxis_title='number of new IC-admissions',
        xaxis_title='date',
        title='Forecast (XGBoost) of new IC-admissions ',
        hovermode="x"
    )
    fig.show()
    print(data_real.type)

    return predictions, data_real

datapoints = 600
int = 500
trend = "linear"
season = "season_0"
post_trend = "stationair"

data_four, data_high, data_medium, data_low, data_real = make_synthetic_data(datapoints, int, trend, season, post_trend)

data = data_high

# predictions, data_real = run_synthetic_data_causalimpact(data, data_real, int)
predictions, data_real = run_synthetich_data_ARIMA(data, data_real, int)
# predictions, data_real = run_synthetic_data_xgboost(data, data_real, int)

print(analyse_model_1(predictions, data_real, int))