
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

from collections import OrderedDict
from functools import partial

import empyrical as ep
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
from sklearn import linear_model

DEPRECATION_WARNING = ("Risk functions in pyfolio.timeseries are deprecated "
                       "and will be removed in a future release. Please "
                       "install the empyrical package instead.")


# @deprecated(msg=DEPRECATION_WARNING)
def cum_returns(returns, starting_value=0):
    """
    Compute cumulative returns from simple returns.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    starting_value : float, optional
       The starting returns (default 1).
    Returns
    -------
    pandas.Series
        Series of cumulative returns.
    Notes
    -----
    For increased numerical accuracy, convert input to log returns
    where it is possible to sum instead of multiplying.
    """

    return ep.cum_returns(returns, starting_value=starting_value)


def simulate_paths(is_returns, num_days,
                   starting_value=1, num_samples=1000, random_seed=None):
    """
    Gnerate alternate paths using available values from in-sample returns.
    Parameters
    ----------
    is_returns : pandas.core.frame.DataFrame
        Non-cumulative in-sample returns.
    num_days : int
        Number of days to project the probability cone forward.
    starting_value : int or float
        Starting value of the out of sample period.
    num_samples : int
        Number of samples to draw from the in-sample daily returns.
        Each sample will be an array with length num_days.
        A higher number of samples will generate a more accurate
        bootstrap cone.
    random_seed : int
        Seed for the pseudorandom number generator used by the pandas
        sample method.
    Returns
    -------
    samples : numpy.ndarray
    """

    samples = np.empty((num_samples, num_days))
    seed = np.random.RandomState(seed=random_seed)
    for i in range(num_samples):
        samples[i, :] = is_returns.sample(num_days, replace=True,
                                          random_state=seed)

    return samples


def summarize_paths(samples, cone_std=(1., 1.5, 2.), starting_value=1.):
    """
    Gnerate the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns.
    Parameters
    ----------
    samples : numpy.ndarray
        Alternative paths, or series of possible outcomes.
    cone_std : list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.
    Returns
    -------
    samples : pandas.core.frame.DataFrame
    """

    cum_samples = ep.cum_returns(samples.T,
                                 starting_value=starting_value).T

    cum_mean = cum_samples.mean(axis=0)
    cum_std = cum_samples.std(axis=0)

    if isinstance(cone_std, (float, int)):
        cone_std = [cone_std]

    cone_bounds = pd.DataFrame(columns=pd.Float64Index([]))
    for num_std in cone_std:
        cone_bounds.loc[:, float(num_std)] = cum_mean + cum_std * num_std
        cone_bounds.loc[:, float(-num_std)] = cum_mean - cum_std * num_std

    return cone_bounds


def forecast_cone_bootstrap(is_returns, num_days, cone_std=(1., 1.5, 2.),
                            starting_value=1, num_samples=1000,
                            random_seed=None):
    """
    Determines the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns. Future cumulative mean and
    standard devation are computed by repeatedly sampling from the
    in-sample daily returns (i.e. bootstrap). This cone is non-parametric,
    meaning it does not assume that returns are normally distributed.
    Parameters
    ----------
    is_returns : pd.Series
        In-sample daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    num_days : int
        Number of days to project the probability cone forward.
    cone_std : int, float, or list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.
    starting_value : int or float
        Starting value of the out of sample period.
    num_samples : int
        Number of samples to draw from the in-sample daily returns.
        Each sample will be an array with length num_days.
        A higher number of samples will generate a more accurate
        bootstrap cone.
    random_seed : int
        Seed for the pseudorandom number generator used by the pandas
        sample method.
    Returns
    -------
    pd.DataFrame
        Contains upper and lower cone boundaries. Column names are
        strings corresponding to the number of standard devations
        above (positive) or below (negative) the projected mean
        cumulative returns.
    """

    samples = simulate_paths(
        is_returns=is_returns,
        num_days=num_days,
        starting_value=starting_value,
        num_samples=num_samples,
        random_seed=random_seed
    )

    cone_bounds = summarize_paths(
        samples=samples,
        cone_std=cone_std,
        starting_value=starting_value
    )

    return cone_bounds
