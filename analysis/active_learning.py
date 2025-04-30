import jax.numpy as jnp
from jax import random
from numpyro import infer
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

from utils.logger import Logger
from utils.data_processing import smooth
from utils.load_save_functions import load_settings


class ActiveLearner:
    def __init__(
        self,
        model,
        parameters: list,
        resolution: int = 1000,
        experiments_dir="experiments",
    ):
        self.key = random.PRNGKey(42)
        self.model = model
        self.parameters = parameters
        self.resolution = resolution
        self.lower_bound_factor = 0.25
        self.higher_bound_factor = 1
        self.tolerance = 4
        self.obs = None
        self.post_pred = None
        self.x_new = None
        settings = load_settings()
        self.logger = Logger(
            name="protocol",
            file_path=f"{experiments_dir}/{settings['EXPERIMENT_NAME']}/meta_data",
        )

        # plot settings
        # plt.rc("text", usetex=True)

    def suggest(self, results: pd.DataFrame, solution_name: str, outlier_check=False):
        results_solution = results.loc[results["solution"] == solution_name]
        x_obs = results_solution["concentration"].to_numpy() / 1000
        y_obs = results_solution["surface tension eq. (mN/m)"].to_numpy() / 1000
        obs = (x_obs, y_obs)
        self.fit(obs=obs, outlier_check=outlier_check)
        suggested_concentration, st_at_suggested_concentration = (
            self.bayesian_suggestion()
        )
        # self.plot_suggestion(x_suggestion=suggested_concentration)
        suggested_concentration = float(suggested_concentration[0] * 1000)  # back to mM
        st_at_suggested_concentration = float(
            st_at_suggested_concentration[0] * 1000
        )  # back to mN/m

        return suggested_concentration, st_at_suggested_concentration

    def fit(self, obs: tuple, outlier_check=False):
        key, key_ = random.split(self.key)
        kernel = infer.NUTS(self.model, step_size=0.2)
        mcmc = infer.MCMC(kernel, num_warmup=500, num_samples=1000)

        self.obs = obs
        x_obs, y_obs = self.obs
        mcmc.run(key_, x_obs=x_obs, y_obs=y_obs)
        self.logger.info("analysis: fitting model to data")
        posterior_samples = mcmc.get_samples()

        observables = ["obs"]

        key, key_ = random.split(key)
        self.x_new = jnp.logspace(
            jnp.log10(jnp.min(x_obs) * self.lower_bound_factor),
            jnp.log10(jnp.max(x_obs) * self.higher_bound_factor),
            self.resolution,
        )

        post_predictive = infer.Predictive(
            self.model,
            posterior_samples=posterior_samples,
            return_sites=self.parameters + observables,
        )
        self.post_pred = post_predictive(key_, self.x_new)

        if outlier_check:
            no_outlier = True
            for i in range(x_obs.shape[0]):
                st_mu = self.post_pred["obs"].mean(axis=0)
                st_std = self.post_pred["obs"].std(axis=0)
                differences = jnp.array([])
                for i, x in enumerate(x_obs):
                    idx = jnp.argmin(jnp.abs(self.x_new - x))
                    difference = jnp.abs(st_mu[idx] - y_obs[i])
                    differences = jnp.append(differences, difference)

                idx_max_difference = jnp.argmax(differences)
                if (
                    differences[idx_max_difference]
                    > self.tolerance * st_std[idx_max_difference]
                ):
                    no_outlier = False
                    self.logger.warning(
                        f"analysis: outlier detected at {x_obs[idx_max_difference]}, datapoint {idx_max_difference}"
                    )
                    x_obs = jnp.delete(x_obs, idx_max_difference)
                    y_obs = jnp.delete(y_obs, idx_max_difference)

                    x_new = jnp.logspace(
                        jnp.log10(jnp.min(x_obs) * self.lower_bound_factor),
                        jnp.log10(jnp.max(x_obs) * self.higher_bound_factor),
                        self.resolution,
                    )

                    self.logger.info("analysis: refitting model")
                    mcmc.run(key_, x_obs=x_obs, y_obs=y_obs)
                    mcmc.print_summary()
                    posterior_samples = mcmc.get_samples()
                    post_predictive = infer.Predictive(
                        self.model,
                        posterior_samples=posterior_samples,
                        return_sites=self.parameters + observables,
                    )
                    self.post_pred = post_predictive(key_, x_new)
            if no_outlier:
                self.logger.info("analysis: no outlier detected")

    def bayesian_suggestion(
        self, parameter_of_interest: str = "all", n_suggestions: int = 1
    ):
        self.logger.info("analysis: calculating bayesian suggestion")
        if parameter_of_interest == "all":
            U_of_interest = jnp.zeros(self.x_new.shape)
            for parameter in self.parameters:
                parameter_std = self.post_pred[parameter].std(axis=0)
                parameter_mean = self.post_pred[parameter].mean(axis=0)
                parameter_relative_std = parameter_std / parameter_mean
                U = mutual_info_regression(
                    self.post_pred["obs"], self.post_pred[parameter]
                )
                U_of_interest += parameter_relative_std * U
        else:
            U_of_interest = mutual_info_regression(
                self.post_pred["obs"], self.post_pred[parameter_of_interest]
            )
            U_of_interest = smooth(U_of_interest, 30)  # smooth

        # find peaks in U_of_interest
        peaks, _ = find_peaks(U_of_interest, distance=25)
        idx = peaks[jnp.argsort(U_of_interest[peaks])][-n_suggestions:]
        x_suggestion = self.x_new[idx]
        st_at_suggestion = self.post_pred["obs"][:, idx].mean(axis=0)

        return x_suggestion, st_at_suggestion

    def plot_fit(self, filename: str = "test"):
        x_obs, y_obs = self.obs
        x_obs = x_obs * 1000
        y_obs = y_obs * 1000
        x_new = self.x_new * 1000
        st_fit = self.post_pred["obs"].mean(axis=0) * 1000
        st_fit_std = self.post_pred["obs"].std(axis=0) * 1000

        fig, ax = plt.subplots()
        ax.plot(x_new, st_fit, c="C0", alpha=0.5, zorder=20)
        ax.fill_between(
            x_new,
            st_fit - 2 * st_fit_std,
            st_fit + 2 * st_fit_std,
            color="C0",
            alpha=0.2,
        )
        ax.scatter(x_obs, y_obs, c="C1", label="observed", zorder=10)
        ax.set_xscale("log")
        ax.set_xlabel("concentration (mM)", fontsize=15)
        ax.set_ylabel("surface tension (mN/m)", fontsize=15)
        filename = f"{filename}.png"
        fig.savefig(filename, dpi=400)

        self.logger.info("analysis: plotted fit")

    def plot_suggestion(self, x_suggestion, filename: str = "test"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # plot fit
        st_fit = self.post_pred["obs"].mean(axis=0)
        ax1.plot(
            self.x_new * 1000, st_fit * 1000, c="C0", alpha=0.5, zorder=20, label="fit"
        )
        x_obs, y_obs = self.obs
        ax1.scatter(x_obs * 1000, y_obs * 1000, c="C1", label="observed")
        for x in x_suggestion:
            ax1.axvline(x=x * 1000, color="C2", linestyle="--", label="suggestion")

        # mutual information
        U_total = jnp.zeros(self.post_pred["obs"].shape[1])
        for parameter in self.parameters:
            parameter_std = self.post_pred[parameter].std(axis=0)
            parameter_mean = self.post_pred[parameter].mean(axis=0)
            parameter_relative_std = parameter_std / parameter_mean
            U = smooth(
                parameter_relative_std
                * mutual_info_regression(
                    self.post_pred["obs"], self.post_pred[parameter]
                ),
                window_size=30,
            )
            U_total += U
            ax2.plot(self.x_new * 1000, U, label=parameter, alpha=0.5)
        ax2.plot(self.x_new * 1000, U_total, label="total", alpha=0.5)

        # settings
        ax1.set_ylim(20, 80)
        ax1.set_ylabel(r"ST (mN/m)", fontsize=15)
        ax1.set_xscale("log")
        ax2.set_xlabel(r"concentration (mM)", fontsize=15)
        ax2.set_ylabel(r"Mutual Information", fontsize=15)
        ax2.legend()
        ax1.legend()
        filename = f"{filename}.png"
        fig.savefig(filename, dpi=600)

    def get_properties(self):
        if self.post_pred is None:
            self.logger.info("analysis: model was not fitted on data!")
        properties = {}
        for parameter in self.parameters:
            prop_mean = self.post_pred[parameter].mean(axis=0)
            prop_std = self.post_pred[parameter].std(axis=0)
            prop_relative_std = prop_std / prop_mean * 100
            properties[parameter] = {
                "mean": prop_mean,
                "std": prop_std,
                "relative std": prop_relative_std,
            }
        self.logger.info("analysis: retrieved properties")
        return properties
