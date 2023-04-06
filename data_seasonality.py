import numpy as np
import pandas as pd
import math
import pylab as py
import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from causalimpact import CausalImpact
from scipy import stats

'''
input data
Datapoint: number of datapoints that must be generated
Seasonality: number of seasons
intervention: timepoint of the intervention
exponential: set to True or False
'''
def make_dataset_year(datapoints, seasonality, intervention, exponential, weekly):
    # datapoint must be a mutliple of Seasonality
    if datapoints%7 != 0:
        raise ValueError("Datapoint must be a multiple of 7 weekdays")

    # set value of weekdays between 5 and 10, and foor weekends 3-5
    time = np.arange(7)
    if weekly == True:
        values = np.where(time < 5, np.random.randint(3,5, time.shape), np.random.randint(1,2, time.shape))

    else:
        values = np.random.randint(3,5,7)

    # Repeat the pattern 20 times
    x = []
    weeks = int(datapoints/7)
    for i in range(weeks):
        for j in range(7):
            x.append(values[j])

    # import 4 seasons seasonality
    season_2 = int(datapoints/seasonality)
    count = 0
    while count < seasonality:
        if count%2 == 0:
            for i in range(season_2):
                x[(season_2*count)+i] = x[(season_2*count)+i]+i/2
            count += 1
        else:
            for i in range(season_2):
                x[(season_2*count)+i] = x[(season_2*count)+i]+(season_2-i)/2
            count += 1

    # Make exponential growth in data
    if exponential == True:
        x += np.logspace(1,6, num=datapoints, base=2)

    # Add noise
    x += np.random.randn(datapoints)*2

    # Make y
    y = 1.2 * x + np.random.normal(size=datapoints)

    # no_interventiom is dataset if no intervention took place
    control_data = y
    control_data = pd.DataFrame({'control_data':control_data}, columns=['control_data'])

    # Adjust data after intervention
    if exponential == True:
        y[intervention:] = y[intervention:] - np.logspace(1,6,num=datapoints-intervention, base=2)
    else:
        y[intervention:] -= 5

    # Make pandas dataframe
    data = pd.DataFrame({'y':y, 'x':x}, columns=['y','x'])
    for col in data.columns:
        print(col)

    pre_period = [0, intervention]
    post_period = [intervention+1, datapoints-1]
    return data, pre_period, post_period, control_data
