import jax.numpy as jnp
from jax.scipy.special import erf
from jax.numpy import sqrt, pi
import numpyro as npy
import numpyro.distributions as dist
from jax import random

def APNModel_S1(cS0, cmc, r):
    """
    Model function for calculating the monomeric concentration of a surfactant given the CMC and total surfactant cocentration.
    Model function taken from https://doi.org/10.1016/j.jcis.2011.12.037

    Parameters:

        Independent variables:
            - cS0 (np.array): list of total surfactant concentrations

        Dependent variables:
            - cS1 (np.array): list of monomeric surfactant concentrations

        Fitting parameters:
            - cmc (float): Critical micelle concentration
            - r (float): relative transition width

    """
    s0 = cS0 / cmc
    # A = 2 / (1 + sqrt(2 / pi) * r * jnp.exp(-1 / (2 * r * r)) + erf(1 / (sqrt(2) * r)))
    A = 1
    cS1 = cmc * (
        1
        - (A / 2)
        * (
            sqrt(2 / pi) * r * jnp.exp(-((s0 - 1) ** 2) / (2 * r * r))
            + (s0 - 1) * (erf((s0 - 1) / (sqrt(2) * r)) - 1)
        )
    )
    return cS1


def szyszkowski(cS0, theta):
    """
    Szyszkowski model for surface tension

    Parameters:

        Independent variables:
            - cS0 (np.array): list of total surfactant concentrations

        Dependent variables:
            - g (np.array): list of surface tension values

        Fitting parameters (theta):
            - cmc (float): Critical micelle concentration
            - a (float): constant
            - Kad (float): constant

    """
    cmc, gamma_max, Kad = theta
    cS1 = APNModel_S1(cS0, cmc, r=0.001)
    R = 8.314  # J/(mol*K)
    T = 294.15  # K (21 degrees Celsius)
    g = 72.8 / 1000 - R * T * gamma_max * jnp.log(1 + Kad * cS1)  # N/m
    return g

def szyszkowski_model(x_obs, y_obs=None):
    """
    Bayesian model for the Szyszkowski model.

    Parameters:
        - x_obs (np.array): list of total surfactant concentrations
        - y_obs (np.array): list of surface tension values
    """
    cmc = npy.sample("cmc", dist.Uniform(0, jnp.max(x_obs)))
    gamma_max = npy.sample("gamma_max", dist.Uniform(0, jnp.max(x_obs) / 10))
    Kad = npy.sample("Kad", dist.Uniform(0, 100000))

    sigma = npy.sample("sigma", dist.Exponential(100000))

    mu = szyszkowski(x_obs, theta=(cmc, gamma_max, Kad))

    npy.sample("obs", dist.Normal(mu, sigma), obs=y_obs)


def szyszkowski_g0(cS0, theta):
    """
    Szyszkowski model for surface tension

    Parameters:

        Independent variables:
            - cS0 (np.array): list of total surfactant concentrations

        Dependent variables:
            - g (np.array): list of surface tension values

        Fitting parameters (theta):
            - cmc (float): Critical micelle concentration
            - a (float): constant
            - Kad (float): constant

    """
    cmc, gamma_max, Kad, g0 = theta
    cS1 = APNModel_S1(cS0, cmc, r=0.001)
    R = 8.314  # J/(mol*K)
    T = 294.15  # K (21 degrees Celsius)
    g = g0 - R * T * gamma_max * jnp.log(1 + Kad * cS1)  # N/m
    return g

def szyszkowski_g0_model(x_obs, y_obs=None):
    """
    Bayesian model for the Szyszkowski model with g0.

    Parameters:
        - x_obs (np.array): list of total surfactant concentrations
        - y_obs (np.array): list of surface tension values
    """
    cmc = npy.sample("cmc", dist.Uniform(0, jnp.max(x_obs)))
    gamma_max = npy.sample("gamma_max", dist.Uniform(0, jnp.max(x_obs) / 10))
    Kad = npy.sample("Kad", dist.Uniform(0, 100000))
    g0 = npy.sample("g0", dist.Normal(72.8 / 1000, 1e-3))

    sigma = npy.sample("sigma", dist.Exponential(100000))

    mu = szyszkowski_g0(x_obs, theta=(cmc, gamma_max, Kad, g0))

    npy.sample("obs", dist.Normal(mu, sigma), obs=y_obs)


def dynamic_st(t, theta):
    """
    reference: Hua and Rosen, 1987: https://doi.org/10.1016/0021-9797(88)90203-2
    """
    st_messo, t_star, n = theta
    st_t = (72.8 - st_messo) / (1 + (t / t_star) ** n) + st_messo
    return st_t


def dynamic_st_model(x_obs, y_obs=None):
    # define priors
    st_messo = npy.sample("st_messo", dist.Uniform(20, 60))
    # st_0 = npy.sample("st_0", dist.Uniform(60, 80))
    t_star = npy.sample("t_star", dist.Uniform(0, 1000))  # TODO change prior
    n = npy.sample("n", dist.Uniform(0, 10))

    sigma = npy.sample("sigma", dist.Exponential(20.0))

    # define st time curve
    mu = dynamic_st(x_obs, theta=(st_messo, t_star, n))

    # define likelihood
    npy.sample("obs", dist.TruncatedNormal(mu, sigma, low=0), obs=y_obs)
