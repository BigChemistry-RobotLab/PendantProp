import numpy as np

# jax imports
import jax.numpy as jnp
from jax import random

# numpyro imports
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


def calc_st_at_cmc(x_new, post_pred):
    x_new = x_new / 1000  # convert to M
    cmc = post_pred["cmc"].mean(axis=0)
    st_at_cmc = post_pred["obs"][:, np.argmin(np.abs(x_new - cmc))].mean()
    st_at_cmc_std = post_pred["obs"][:, np.argmin(np.abs(x_new - cmc))].std()
    st_at_cmc_relative_err = st_at_cmc_std / st_at_cmc * 100
    return st_at_cmc, st_at_cmc_std, st_at_cmc_relative_err
