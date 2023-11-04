# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 16:08:31 2023

@author: brech
"""

# %% Initialization

# Required libraries
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import feather
from scipy import stats
import scikit_posthocs as sp

# Plot config
sns.set_style('whitegrid')


# %% Data preparation

# Load LSE data
data = feather.read_dataframe('generated_data\\lst_dataset.feather')
data = data.reset_index()
data['lst'] -= 273.15
data['lst_mea'] -= 273.15
data['lst_med'] -= 273.15

# %% Check for outliers, variance and normality

# OUTLIERS

# Cleveland dotplot
# LSE outliers
sns.scatterplot(data=data, x='index', y='lse')

# LST outliers
sns.scatterplot(data=data, x='index', y='lst')

# Three outliers in LSE, none in LST.

# Get outliers indexes and drop observations
outliers = data[(data.lse < 0.95) | (data.lst < 12)].index.tolist()
data = data.drop(outliers).reset_index(drop=True)


# NORMALITY

# Histograms
sns.histplot(data=data, x='lst')

# Groups are clearly non-parametric.


# %% Summarise data

# Calculate absolute errors
data['abs_error_mea'] = abs(data.lst - data.lst_mea)
data['abs_error_med'] = abs(data.lst - data.lst_med)

# Summarise
summary = data.iloc[:, list(range(3, 8))].describe().transpose()
summary = summary.drop('count', axis=1)
summary['IQR'] = summary['75%'] - summary['25%']


# %% Scatter plots

# Function for retrieving RMSE, MAE and Bias
def rmse_mae_bias(data, var1, var2, groupby):

    # Calculate error, absolute error and square error
    data['bias'] = (data[var1] - data[var2])
    data['mae'] = data['bias'].abs()
    data['sq_error'] = data['bias']**2

    results = data.groupby(groupby)[
        [var1, var2, 'bias', 'mae', 'sq_error']].mean()

    results['rmse'] = np.sqrt(results['sq_error'])

    return results


# Calculate error metrics
data_long = data.melt(id_vars=['index', 'datetime', 'lse', 'lst'],
                      value_vars=['lst_mea', 'lst_med'])
errors_metrics = rmse_mae_bias(data_long, 'lst', 'value', 'variable')


def scatter_w_metrics(data, x, y, col, col_order, error_metrics):

    # Scatter plots
    grid2 = sns.FacetGrid(data=data, col=col, col_wrap=2, col_order=col_order,
                          height=4)
    grid2.map_dataframe(sns.scatterplot, x, y, size=0.02)
    grid2.map_dataframe(lambda data, **kws:
                        plt.axline(xy1=(20, 20), slope=1, ls='--',
                                   color='black', zorder=0))

    # Add error metrics to plot
    for ax, estm, lab in zip(grid2.axes.flat, col_order, ['a)', 'b)']):

        ax.set(title=None, xlabel='LST (°C)', ylabel='LST (°C)')

        bias = error_metrics.loc[estm].bias
        mae = error_metrics.loc[estm].mae
        rmse = error_metrics.loc[estm].rmse

        label = (f'Bias = {bias:.3f} \nMAE = {mae:.3f} \nRMSE = {rmse:.3f}')

        ax.text(15, 38.5, label,
                bbox=dict(edgecolor='black', facecolor='white', lw=0.2))

        ax.text(12.5, 48, lab, size=12)


scatter_w_metrics(data=data_long, x='lst', y='value', col='variable',
                  col_order=['lst_mea', 'lst_med'],
                  error_metrics=errors_metrics)


plt.savefig('plots\\scatter_plot.tif', dpi=300)
