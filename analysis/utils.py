import pandas as pd
import numpy as np
# jax imports
import jax.numpy as jnp
from jax.scipy.special import erf
from jax.numpy import sqrt, pi
from jax import random

# numpyro imports
import numpyro as npy
import numpyro.distributions as dist
from numpyro import infer

def fit_model(
    obs,
    model,
    parameters,
    outlier_check=False,
    l_x_new=1000,
    key=random.PRNGKey(0),
):
    """
    Fitting model function using bayesian inference.

    Input:
        - obs (tuple): tuple of x_obs and y_obs
        - model (function): model function
        - parameters (list): list of parameters
        - l_x_new (int): number of points for the mutual information calculation
        - key (jax.random.PRNGKey): random key

    Output:
        - post_pred (dict): dictionary of posterior predictive samples, including the parameters and observables.
        - x_new (np.array): new x values for the plot


    """
    key, key_ = random.split(key)
    kernel = infer.NUTS(model, step_size=0.2)
    mcmc = infer.MCMC(kernel, num_warmup=500, num_samples=1000)

    key, key_ = random.split(key)

    x_obs, y_obs = obs
    x_obs = jnp.array(x_obs)
    y_obs = jnp.array(y_obs)
    mcmc.run(key_, x_obs=x_obs, y_obs=y_obs)
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()

    observables = ["obs"]

    key, key_ = random.split(key)

    x_new = jnp.logspace(
        jnp.log10(jnp.min(x_obs) / 4),
        jnp.log10(jnp.max(x_obs)),
        l_x_new,
    )

    post_predictive = infer.Predictive(
        model,
        posterior_samples=posterior_samples,
        return_sites=parameters + observables,
    )
    post_pred = post_predictive(key_, x_new)

    if outlier_check:

        for i in range(x_obs.shape[0]):
            st_mu = post_pred["obs"].mean(axis=0)
            st_std = post_pred["obs"].std(axis=0)
            differences = jnp.array([])
            for i, x in enumerate(x_obs):
                idx = jnp.argmin(jnp.abs(x_new - x))
                difference = jnp.abs(st_mu[idx] - y_obs[i])
                differences = jnp.append(differences, difference)

            idx_max_difference = jnp.argmax(differences)
            if differences[idx_max_difference] > 4 * st_std[idx_max_difference]:
                print(
                    f"outlier detected at {x_obs[idx_max_difference]}, datapoint {idx_max_difference}"
                )
                x_obs = jnp.delete(x_obs, idx_max_difference)
                y_obs = jnp.delete(y_obs, idx_max_difference)

                x_new = jnp.logspace(
                    jnp.log10(jnp.min(x_obs) / 4),
                    jnp.log10(jnp.max(x_obs)),
                    l_x_new,
                )

                print("refitting model...")
                mcmc.run(key_, x_obs=x_obs, y_obs=y_obs)
                mcmc.print_summary()
                posterior_samples = mcmc.get_samples()
                post_predictive = infer.Predictive(
                    model,
                    posterior_samples=posterior_samples,
                    return_sites=parameters + observables,
                )
                post_pred = post_predictive(key_, x_new)
    return post_pred, x_new


def calculate_st_at_cmc(x_new, post_pred):
    x_new = x_new / 1000  # convert to M
    cmc = post_pred["cmc"].mean(axis=0)
    st_at_cmc = post_pred["obs"][:, np.argmin(np.abs(x_new - cmc))].mean() * 1000
    st_at_cmc_std = post_pred["obs"][:, np.argmin(np.abs(x_new - cmc))].std() * 1000
    st_at_cmc_relative_err = st_at_cmc_std / st_at_cmc * 100
    return st_at_cmc 


def calculate_C20(x_new, post_pred: dict):
    """ """

    st_50 = 50 / 1000  # convert to N/m
    st_fit_mu = post_pred["obs"].mean(axis=0)
    st_fit_std = post_pred["obs"].std(axis=0)

    # find the index where the surface tension is closest to 50 mN/m
    idx = np.argmin(np.abs(st_fit_mu - st_50))

    st_fit_lower = st_fit_mu - st_fit_std
    st_fit_upper = st_fit_mu + st_fit_std

    idx_lower = np.argmin(np.abs(st_fit_lower - st_50))
    idx_upper = np.argmin(np.abs(st_fit_upper - st_50))

    c_20_mu = x_new[idx] * 1000  # concentration at 50 mN/m

    c_20_lower = x_new[idx_lower] * 1000  # concentration at 50 mN/m - std
    c_20_upper = x_new[idx_upper] * 1000  # concentration at 50 mN/m + std

    c_20_std = (c_20_upper - c_20_lower) / 2  # std of the concentration at 50 mN/m

    c_20_std_rela = c_20_std / c_20_mu * 100  # relative std of the concentration at 50 mN/m
    
    return c_20_mu

def calculate_gamma_max(x_new, post_pred: dict, n, interval = 10):
    R = 8.31446261815324
    T = 293.15

    log_c = np.log(x_new) #natural log
    st = post_pred["obs"].mean(axis=0)
    slopes = []
    gammas = []
    avg_xs = []
    for i in range(len(log_c) - interval + 1):
        x_window = log_c[i:i+interval]
        y_window = st[i:i+interval]
        # Linear fit: slope is coefficient at index 0
        slope = np.polyfit(x_window, y_window, 1)[0]
        slopes.append(slope)
        gamma =  (-1/(n*R*T))*slope
        gammas.append(gamma)
        avg_x = np.mean(x_window)
        avg_xs.append(avg_x)
    
    return np.max(gammas)




