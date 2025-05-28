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
    if x_obs.size == 0:
        # Log a warning and return a default value
        print("Warning: x_obs is empty. Skipping computation.")
        return None  # or return a default value like 0 or an empty array
    
    cmc = npy.sample("cmc", dist.Uniform(0, jnp.max(x_obs)))
    gamma_max = npy.sample("gamma_max", dist.Uniform(0, jnp.max(x_obs) / 10))
    Kad = npy.sample("Kad", dist.Uniform(0, 100000))

    sigma = npy.sample("sigma", dist.Exponential(100000))

    mu = szyszkowski(x_obs, theta=(cmc, gamma_max, Kad))

    npy.sample("obs", dist.Normal(mu, sigma), obs=y_obs)
