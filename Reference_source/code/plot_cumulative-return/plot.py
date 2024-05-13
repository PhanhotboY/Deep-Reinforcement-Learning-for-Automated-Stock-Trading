#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division

import datetime
from collections import OrderedDict
from functools import wraps

import empyrical as ep
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy as sp
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter

import timeseries
import utils


def plot_rolling_returns(returns, color, label,
                         factor_returns=None,
                         live_start_date=None,
                         logy=False,
                         cone_std=None,
                         legend_loc='best',
                         volatility_match=False,
                         cone_function=timeseries.forecast_cone_bootstrap,
                         ax=None,
                         **kwargs):
    """
    Plots cumulative rolling returns versus some benchmarks'.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Additionally, a non-parametric cone plot may be added to the
    out-of-sample returns region.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    logy : bool, optional
        Whether to log-scale the y-axis.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - See timeseries.forecast_cone_bounds for more details.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    volatility_match : bool, optional
        Whether to normalize the volatility of the returns to those of the
        benchmark returns. This helps compare strategies with different
        volatilities. Requires passing of benchmark_rets.
    cone_function : function, optional
        Function to use when generating forecast probability cone.
        The function signiture must follow the form:
        def cone(in_sample_returns (pd.Series),
                 days_to_project_forward (int),
                 cone_std= (float, or tuple),
                 starting_value= (int, or float))
        See timeseries.forecast_cone_bootstrap for an example.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative returns')
    ax.set_title('Cumulative returns')
    ax.set_yscale('log' if logy else 'linear')
    ax.grid(True)

    if volatility_match and factor_returns is None:
        raise ValueError('volatility_match requires passing of '
                         'factor_returns.')
    elif volatility_match and factor_returns is not None:
        bmark_vol = factor_returns.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    cum_rets = ep.cum_returns(returns, 1.0)

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if factor_returns is not None:
        cum_factor_returns = ep.cum_returns(
            factor_returns[cum_rets.index], 1.0)
        cum_factor_returns.plot(lw=2, color='gray',
                                label=factor_returns.name, alpha=0.60,
                                ax=ax, **kwargs)

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_cum_returns = cum_rets.loc[cum_rets.index < live_start_date]
        oos_cum_returns = cum_rets.loc[cum_rets.index >= live_start_date]
    else:
        is_cum_returns = cum_rets
        oos_cum_returns = pd.Series([])

    is_cum_returns.plot(lw=2, color=color, alpha=2.0,
                        ax=ax, **kwargs)

    if len(oos_cum_returns) > 0:
        oos_cum_returns.plot(lw=4, color='red', alpha=0.6,
                             label='Live', ax=ax, **kwargs)

        if cone_std is not None:
            if isinstance(cone_std, (float, int)):
                cone_std = [cone_std]

            is_returns = returns.loc[returns.index < live_start_date]
            cone_bounds = cone_function(
                is_returns,
                len(oos_cum_returns),
                cone_std=cone_std,
                starting_value=is_cum_returns[-1])

            cone_bounds = cone_bounds.set_index(oos_cum_returns.index)
            for std in cone_std:
                ax.fill_between(cone_bounds.index,
                                cone_bounds[float(std)],
                                cone_bounds[float(-std)],
                                color='steelblue', alpha=0.5)

    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=None, framealpha=0.5)
    ax.axhline(1.0, linestyle='--', color='black', lw=1.5)

    # return ax


DRL_ddpg_strat = pd.read_csv('DRL_ddpg_strat.csv')
DRL_ppo_strat = pd.read_csv('DRL_ppo_strat.csv')
DRL_a2c_strat = pd.read_csv('DRL_a2c_strat.csv')
dow_strat = pd.read_csv('dow_strat.csv')
ensemble_strat = pd.read_csv('ensemble_strat.csv')

DRL_ddpg_strat = DRL_ddpg_strat.fillna(0).set_index('Date')
DRL_ppo_strat = DRL_ppo_strat.fillna(0).set_index('Date')
DRL_a2c_strat = DRL_a2c_strat.fillna(0).set_index('Date')
dow_strat = dow_strat.fillna(0).set_index('Date')
ensemble_strat = ensemble_strat.fillna(0).set_index('Date')


# Cumulative Returns
fig = plt.figure(figsize=(15, 8))
#plt.subplot(2, 1, 1)
plot_rolling_returns(DRL_ddpg_strat, color='green', label='DDPG')
plot_rolling_returns(DRL_ppo_strat, color='blue', label='PPO')
plot_rolling_returns(DRL_a2c_strat, color='red', label='A2C')
plot_rolling_returns(dow_strat, color='orange', label='DJIA')
plot_rolling_returns(ensemble_strat, color='magenta', label='Ensemble')

plt.grid(True)
plt.legend()
plt.show()
