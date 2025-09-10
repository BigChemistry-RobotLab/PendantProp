import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.load_save_functions import load_settings
from hardware.opentrons.containers import Container
from utils.logger import Logger
from hardware.opentrons.configuration import Configuration
from hardware.opentrons.opentrons_api import OpentronsAPI

# from pyDOE import lhs
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class GridSearch:   
    def __init__(self, opentrons_api: OpentronsAPI):
        self.settings = load_settings()
        self.config = Configuration(opentrons_api=opentrons_api)
        # self.containers = self.config.load_containers()
        self.logger = Logger(
            name="protocol",
            file_path=f'experiments/{self.settings["EXPERIMENT_NAME"]}/meta_data',
        )
        self.min_pip_vol = self.settings["MINIMUM_PIPETTING_VOLUME"]
        self.well_vol = self.settings["WELL_VOLUME"]

    def generate_grid(
        self, solution1: str, solution2: str,
        x_max_conc: float = 1, y_max_conc: float = 1,
        x_dilution: int = 100, y_dilution: int = 100,
        n_samples: int = 96, plot=False
    ):
        x_min, x_max = (x_max_conc / x_dilution) / 2, x_max_conc / 2
        y_min, y_max = (y_max_conc / y_dilution) / 2, y_max_conc / 2

        # Log scale ranges
        log_x_min, log_x_max = np.log10(x_min), np.log10(x_max)
        log_y_min, log_y_max = np.log10(y_min), np.log10(y_max)

        # Sample n_samples points uniformly in log-space    Change this to better distribution later!
        log_x = np.random.uniform(log_x_min, log_x_max, n_samples)
        log_y = np.random.uniform(log_y_min, log_y_max, n_samples)

        x_random = 10 ** log_x
        y_random = 10 ** log_y

        output_df = pd.DataFrame({
            solution1: x_random,
            solution2: y_random
        })

        # Round values to 4 decimals
        output_df = output_df.round(4)

        # Plot the grid
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x_random, y_random, c='dodgerblue', alpha=0.7, edgecolors='k', label="Grid-Randomized Points")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"{solution1} concentration (mM)", fontsize=12)
            ax.set_ylabel(f"{solution2} concentration (mM)", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Rectangle of search space
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=2, edgecolor='red', facecolor='none', linestyle="--")
            ax.add_patch(rect)

            plt.tight_layout()
            plt.show()
        output_df.to_csv(f"experiments/{self.settings['EXPERIMENT_NAME']}/{solution1}_{solution2}_grid.csv")
        return output_df

    def process_df(self, df):
        columns = df.columns
        df = df.dropna()                                    
        for col in df.columns:                                  # removes rows with missing values
            df = df[df[col] >= 0]                               # and negatives should not be
        grid = list(df.itertuples(index=False, name=None))      # possible in the first place
        self.min_conc = df.min().to_list()  
        return grid, columns

    def generate_dilution_series(self, stock_conc, min_required, factor):
        dilutions = [stock_conc]
        while dilutions[-1] / factor >= min_required:
            dilutions.append(dilutions[-1] / factor)
        return dilutions 

    def is_point_feasible(self, target_conc_sol1, target_conc_sol2, dilution_series_sol1, dilution_series_sol2):
        for dilution_factor_sol1 in dilution_series_sol1:
            for dilution_factor_sol2 in dilution_series_sol2:
                vol_sol1 = (target_conc_sol1 / dilution_factor_sol1) * self.well_vol
                vol_sol2 = (target_conc_sol2 / dilution_factor_sol2) * self.well_vol
                vol_water = self.well_vol - vol_sol1 - vol_sol2

                if all([
                    vol_sol1 >= self.min_pip_vol or np.isclose(vol_sol1, 0),
                    vol_sol2 >= self.min_pip_vol or np.isclose(vol_sol2, 0),
                    vol_water >= self.min_pip_vol or np.isclose(vol_water, 0),
                    self.well_vol == round(vol_sol1 + vol_sol2 + vol_water,0),
                ]):
                    return True
        return False

    def count_total_dilutions(self, dilution_series1, dilution_series2):
        num_dilutions1 = len(dilution_series1) - 1          # -1 to account for original stock
        num_dilutions2 = len(dilution_series2) - 1          # which should always be at least 2x  
        total_dilutions = num_dilutions1 + num_dilutions2   # highest required concentration
        return total_dilutions  # should be less than or equal to amount of empty/available FT15 tubes

    def optimize_dilution_factor_upward(self, grid, starting_factor, max_red_points, stock1, stock2):
        factor = starting_factor
        best_dilutions1 = self.generate_dilution_series(stock1, self.min_conc[0], factor)
        best_dilutions2 = self.generate_dilution_series(stock2, self.min_conc[1], factor)
        best_feasibility = [
            self.is_point_feasible(c1, c2, best_dilutions1, best_dilutions2)
            for c1, c2 in grid
        ]

        while True:
            dilutions1 = self.generate_dilution_series(stock1, self.min_conc[0], factor)
            dilutions2 = self.generate_dilution_series(stock2, self.min_conc[1], factor)

            feasibility = [
                self.is_point_feasible(c1, c2, dilutions1, dilutions2)
                for c1, c2 in grid
            ]
            red_points = feasibility.count(False)

            print(f"[Up] Factor: {factor} -> Red Points: {red_points}")
            if red_points <= max_red_points and factor == 15:       # Don't know if this is necessary, but don't want to go too high(?)
                return factor, best_dilutions1, best_dilutions2, best_feasibility   # Maybe combine this with tube counting?
            if red_points <= max_red_points:
                best_dilutions1 = dilutions1
                best_dilutions2 = dilutions2
                best_feasibility = feasibility
                factor = round(factor + 0.1, 1)
            else: # increase factor until as high as possible to minimize amount of dilutions required
                factor = round(factor - 0.1, 1)
                print(f"Reducing by 0.1. Final factor is {factor}.")
                return factor, best_dilutions1, best_dilutions2, best_feasibility
                # limit reached, return the last correct factor

    def find_initial_feasible_factor(self, starting_factor, min_allowed_factor, max_red_points, stock1, stock2, grid):
        factor = starting_factor
        previous_length = 0
        while factor >= min_allowed_factor:
            dilutions1 = self.generate_dilution_series(stock1, self.min_conc[0], factor)
            dilutions2 = self.generate_dilution_series(stock2, self.min_conc[1], factor)
            feasibility = [
                self.is_point_feasible(c1, c2, dilutions1, dilutions2)
                for c1, c2 in grid
            ]
            red_points = feasibility.count(False)
            line = f"[Down] Factor: {factor} -> Red Points: {red_points}"
            padding = " " * max(0, previous_length - len(line))  # Clear any leftover characters

            print(f"\r{line}{padding}", end="", flush=True)
            previous_length = len(line)
            if red_points <= max_red_points:
                print()  
                return factor  
            factor = round(factor * 0.9, 1)
        raise ValueError("No possible dilution factor found within limits.")

    def find_optimal_dilution_setup(self, stock1, stock2, grid, max_red_points=0, min_allowed_factor=1.2):
        starting_factor = self.well_vol / self.min_pip_vol
        initial_factor = self.find_initial_feasible_factor(
            starting_factor, min_allowed_factor, max_red_points, stock1, stock2, grid
        )
        return self.optimize_dilution_factor_upward(
            grid, initial_factor, max_red_points, stock1, stock2
        )

    def generate_feasible_combinations_for_sample(self, form_scheme: pd.DataFrame, c1, c2, d1_list, d2_list):
        grouped_candidates = {}

        for d1 in d1_list:
            for d2 in d2_list:
                vol1 = round((c1 / d1) * self.well_vol, 2)
                vol2 = round((c2 / d2) * self.well_vol, 2)
                vol_water = round(self.well_vol - vol1 - vol2, 2)
                conc1 = round((vol1 * d1) / self.well_vol, 6)
                conc2 = round((vol2 * d2) / self.well_vol, 6)

                if all([
                    vol1 >= self.min_pip_vol or np.isclose(vol1, 0),
                    vol2 >= self.min_pip_vol or np.isclose(vol2, 0),
                    vol_water >= 0
                ]):
                    key = (c1, c2)
                    spread = max(vol1, vol2, vol_water) - min(vol1, vol2, vol_water)
                    candidate = {
                        "dil1": d1,
                        "vol1": vol1,
                        "conc1": conc1, # Make better names for columns
                        "dil2": d2,     # need to be recognizable TODO
                        "vol2": vol2,   # Need to round the volumes before recalculating
                        "conc2": conc2,
                        "vol_water": vol_water,
                        "spread": spread,
                    }

                    if key not in grouped_candidates:
                        grouped_candidates[key] = [candidate]
                    else:
                        grouped_candidates[key].append(candidate)

        rows_to_add = []
        for candidates in grouped_candidates.values():  # In case multiple possibilities
            best = min(candidates, key=lambda x: x["spread"])
            del best["spread"]
            rows_to_add.append(best)

        if rows_to_add:
            form_scheme = pd.concat([form_scheme, pd.DataFrame(rows_to_add)], ignore_index=True)

        return form_scheme

# class PointSelection:
#     """
#     This class manages the selection of new experimental points
#     for a Bayesian optimization process.
#     """

#     def __init__(self):
#         """
#         Initializes the parameters for the optimization.
#         """
#         self.max_bounds = [(1.0), (2.0)]
#         self.bound_dilution = 100
#         self.bounds = [
#             (self.max_bounds[0] / self.bound_dilution, self.max_bounds[0]),
#             (self.max_bounds[1] / self.bound_dilution, self.max_bounds[1]),
#         ]
#         self.grid_step_x = 0.1
#         self.grid_step_y = 0.2
#         self.initial_points = 8
#         self.batch_size = 8
#         self.n_candidates = 500
#         self.threshold = 0.05
#         self.min_separation = 0.15
#         self.round_decimals = 4
#         self.random_state = 42
#         self.rng = np.random.default_rng(self.random_state)
#         # Not sure how useful this is, mainly for less required computing and maybe useful for discrete values to formulate stocks?
#         self.accessible_space = self._build_accessible_space()

#     def _maximin_lhs(self, n_points):
#         """
#         Generates an initial set of diverse points using Latin Hypercube Sampling.
#         The best design is chosen by maximizing the minimum distance between points.
#         """
#         dim = len(self.bounds)
#         best_pts, best_min_dist = None, -np.inf
#         for _ in range(self.n_candidates):
#             s = lhs(dim, samples=n_points)
#             scaled = np.array(
#                 [
#                     s[:, i] * (self.bounds[i][1] - self.bounds[i][0])
#                     + self.bounds[i][0]
#                     for i in range(dim)
#                 ]
#             ).T
#             d = cdist(scaled, scaled)
#             np.fill_diagonal(d, np.inf)
#             m = d.min()
#             if m > best_min_dist:
#                 best_min_dist, best_pts = m, scaled
#         return best_pts

#     def _build_accessible_space(self):
#         """
#         Creates a discrete grid of all possible experimental points within the defined
#         bounds and resolution. This serves as the search space for the algorithm.
#         """
#         x_vals = np.round(
#             np.arange(self.bounds[0][0], self.bounds[0][1] + 1e-12, self.grid_step_x),
#             10,
#         )
#         y_vals = np.round(
#             np.arange(self.bounds[1][0], self.bounds[1][1] + 1e-12, self.grid_step_y),
#             10,
#         )
#         xx, yy = np.meshgrid(x_vals, y_vals)
#         return np.c_[xx.ravel(), yy.ravel()]

#     def _select_diverse_batch(self, points, sigmas):
#         """
#         Selects a new batch of points by prioritizing those with high uncertainty (sigma)
#         while enforcing a minimum separation distance to ensure diversity.
#         """
#         order = np.argsort(-sigmas)
#         selected = []
#         for idx in order:
#             p = points[idx]
#             if len(selected) == 0:
#                 selected.append(idx)
#                 if len(selected) == self.batch_size:
#                     break
#                 continue
#             dists = cdist(points[selected], p[None, :]).ravel()
#             if np.all(dists >= self.min_separation):
#                 selected.append(idx)
#                 if len(selected) == self.batch_size:
#                     break
#         if len(selected) < self.batch_size:
#             for idx in order:
#                 if idx not in selected:
#                     selected.append(idx)
#                     if len(selected) == self.batch_size:
#                         break
#         return np.array(selected, dtype=int)

#     def suggest_next_points(self, data=None):
#         """
#         The main function to suggest the next set of experimental points.

#         Args:
#             data (pd.DataFrame, optional): A DataFrame containing the 'X' and 'y'
#                                            data from previous experiments.
#                                            If None, initial points are suggested.

#         Returns:
#             np.ndarray: An array of shape (batch_size, 2) with the suggested points.
#             None: If the optimization has converged (highest uncertainty is below threshold).
#         """
#         if data is None:
#             # If no data is provided, suggest the initial set of points
#             return self._maximin_lhs(self.initial_points)

#         # If data is provided, perform Bayesian optimization to suggest the next batch
#         X = data[['x', 'y']].values
#         y = data['objective_value'].values

#         # Get a set of keys for all points that have already been tested
#         tested_grid_keys = set([tuple(np.round(row, self.round_decimals)) for row in X])

#         # Train the Gaussian Process model on the existing data
#         kernel = Matern(nu=2.5)
#         gp = GaussianProcessRegressor(
#             kernel=kernel,
#             alpha=1e-6,
#             normalize_y=True,
#             random_state=self.random_state,
#         )
#         gp.fit(X, y)

#         # Predict uncertainty for all points in the accessible space
#         _, sigma = gp.predict(self.accessible_space, return_std=True)

#         # Filter out points that have already been tested
#         all_grid_keys = [tuple(np.round(row, self.round_decimals)) for row in self.accessible_space]
#         mask_candidates = np.array(
#             [k not in tested_grid_keys for k in all_grid_keys]
#         )
#         candidates = self.accessible_space[mask_candidates]
#         sigma_candidates = sigma[mask_candidates]

#         # Check for stopping criteria
#         max_sigma = float(np.max(sigma_candidates)) if len(sigma_candidates) else 0.0
#         print(f"Max sigma among candidates: {max_sigma:.4f}")
#         if (len(candidates) == 0) or (max_sigma < self.threshold):
#             return None

#         # Select the next batch of diverse points
#         sel_idx_local = self._select_diverse_batch(candidates, sigma_candidates)
#         X_next = candidates[sel_idx_local]

#         return X_next
