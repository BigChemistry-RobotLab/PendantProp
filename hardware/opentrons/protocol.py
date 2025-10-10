import warnings
import pandas as pd
import numpy as np
import time
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.spatial.distance import cdist # Required for cdist
from scipy.spatial import cKDTree


# Suppress the specific FutureWarning of Pandas
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)


from analysis.plots import Plotter
from analysis.active_learning import ActiveLearner
from analysis.models import szyszkowski_model
from hardware.opentrons.opentrons_api import OpentronsAPI
from hardware.opentrons.droplet_manager import DropletManager
from hardware.opentrons.formulater import Formulater
from hardware.opentrons.configuration import Configuration
from hardware.opentrons.containers import Container
from hardware.cameras import PendantDropCamera
from hardware.sensor.sensor_api import SensorAPI
from hardware.opentrons.gridsearch import GridSearch
# from hardware.opentrons.gridsearch import PointSelection
from utils.search_containers import find_container, find_stock_tubes
from utils.load_save_functions import (
    load_settings,
    save_calibration_data,
    initialize_results,
    initialize_sigma,
    load_info,
    append_results,
    save_results,
    append_sigma
)
from utils.logger import Logger
from utils.utils import play_sound, calculate_average_in_column
import math


class Protocol:
    def __init__(
        self,
        opentrons_api: OpentronsAPI,
        sensor_api: SensorAPI,
        pendant_drop_camera: PendantDropCamera,
    ):
        self.settings = load_settings()
        self.logger = Logger(
            name="protocol",
            file_path=f'experiments/{self.settings["EXPERIMENT_NAME"]}/meta_data',
        )
        self.logger.info("Initialization starting...")
        self.opentrons_api = opentrons_api
        self.opentrons_api.initialise()
        self.sensor_api = sensor_api
        self.pendant_drop_camera = pendant_drop_camera
        self.config = Configuration(opentrons_api=opentrons_api)
        self.gridsearch = GridSearch(opentrons_api=opentrons_api)
        self.labware = self.config.load_labware()
        self.containers = self.config.load_containers()
        pipettes = self.config.load_pipettes()
        self.right_pipette = pipettes["right"]
        self.left_pipette = pipettes["left"]
        self.n_measurement_in_eq = 100  # number of data points which is averaged for equillibrium surface tension
        self.results = initialize_results()
        self.results_sigma = initialize_sigma()
        self.learner = ActiveLearner(
            model=szyszkowski_model,
            parameters=["cmc", "gamma_max", "Kad"],
        )
        self.plotter = Plotter()
        self.droplet_manager = DropletManager(
            left_pipette=self.left_pipette,
            containers=self.containers,
            pendant_drop_camera=self.pendant_drop_camera,
            opentrons_api=self.opentrons_api,
            plotter=self.plotter,
        )
        self.formulater = Formulater(
            left_pipette=self.left_pipette,
            right_pipette=self.right_pipette,
            containers=self.containers,
            labware=self.labware,
            opentrons_api=self.opentrons_api,
        )
        # self.ps = PointSelection(opentrons_api=self.opentrons_api)
        self.opentrons_api.home()
        self.batch_size = self.settings["BATCH_SIZE"]
        self.repeat = self.settings["WASH_REPEATS"]

        self.logger.info("Initialization finished.\n\n\n")
        # play_sound("Lets go.")

    def measure_wells(self) -> None:
        """
        Perform pendant drop measurements for all wells specified in the settings.
        """
        self.logger.info("Starting measure wells protocol...\n\n\n")
        self.settings = load_settings()  # Update settings
        well_info = load_info(file_name=self.settings["WELL_INFO_FILENAME"])
        wells_ids = well_info["location"].astype(str) + well_info["well"].astype(str)

        for well_id in wells_ids:
            self._measure_single_well(well_id)

        self.left_pipette.return_needle()
        self.logger.info("Finished measure wells protocol.\n\n\n")

    def _measure_single_well(self, well_id: str) -> None:
        """
        Measure pendant drop for a single well and process the results.
        """
        self.logger.info(f"Start pendant drop measurement of {well_id}.\n")
        dynamic_surface_tension, drop_volume, drop_count, init_drop_vol = (
            self.droplet_manager.measure_pendant_drop(
                source=self.containers[well_id],
                # max_measure_time=float(self.settings["EQUILIBRATION_TIME"]),
            )
        )

        drop_parameters = self._create_drop_parameters(
            drop_volume=drop_volume,
            measure_time=float(self.settings["EQUILIBRATION_TIME"]),
            drop_count=drop_count,
            init_drop_vol=init_drop_vol
        )

        self.formulater.wash(repeat=self.settings["WASH_REPEATS"])

        self._append_and_save_results(
            point_type="None",
            dynamic_surface_tension=dynamic_surface_tension,
            well_id=well_id,
            drop_parameters=drop_parameters,
            solution_name=self.containers[well_id].solution_name,
            plot_type="wells",
        )

        self.logger.info(f"End of pendant drop measurement of {well_id}.\n")

    # def prepare_grid_scan(
    #     self,
    #     solution1,
    #     solution2,
    #     sol1_max_conc,
    #     sol2_max_conc,
    #     x_dil_range,
    #     y_dil_range,
    #     samples,
    # ):
    #     self.logger.info(
    #         "Preparing grid scan...\n"
    #     )  # Introduce functionality to set bounds instead of range? so 10^-5 -> 10^1? give choice
    #     success = False
    #     while not success:
    #         try:
    #             df = self.gridsearch.generate_grid(
    #                 solution1=solution1,
    #                 solution2=solution2,
    #                 x_max_conc=sol1_max_conc,
    #                 y_max_conc=sol2_max_conc,
    #                 x_dilution=x_dil_range,
    #                 y_dilution=y_dil_range,
    #                 n_samples=samples,
    #                 plot=False,
    #             )
    #             grid, columns = self.gridsearch.process_df(df=df)

    #             if (solution1 or solution2) not in columns:
    #                 self.logger.error("Surfactants not found! Quitting...\n\n\n")
    #                 self.logger.info(
    #                     "Needs: ", solution1, " ", solution2, "\n Found: ", columns
    #                 )
    #                 return

    #             dilution_factor, bdil1, bdil2, _ = (
    #                 self.gridsearch.find_optimal_dilution_setup(
    #                     stock1=sol1_max_conc, stock2=sol2_max_conc, grid=grid
    #                 )
    #             )
    #             success = True
    #         except Exception as e:
    #             self.logger.info(f"Retrying, error: {e}.")
    #     tubes_req = self.gridsearch.count_total_dilutions(
    #         bdil1, bdil2
    #     )  # Add check for empty tubes after

    #     amount_empty = len(find_container(containers=self.containers, type="tube 15"))
    #     if tubes_req > amount_empty:
    #         self.logger.error("Not enough tubes to create dilutions! Quitting...\n\n\n")
    #         return

    #     if "form_scheme" not in locals():
    #         form_scheme = pd.DataFrame(
    #             columns=[
    #                 "well_id",
    #                 "dil1",
    #                 "vol1",
    #                 "conc1",
    #                 "dil2",
    #                 "vol2",
    #                 "conc2",
    #                 "vol_water",
    #             ]
    #         )
    #     for c1, c2 in grid:
    #         form_scheme = self.gridsearch.generate_feasible_combinations_for_sample(
    #             form_scheme, c1=c1, c2=c2, d1_list=bdil1, d2_list=bdil2
    #         )
    #     empty_wells = find_container(
    #         containers=self.containers, type="Plate well", amount=len(form_scheme)
    #     )
    #     form_scheme["well_id"] = empty_wells
    #     form_scheme.to_csv(
    #         f"experiments/{self.settings['EXPERIMENT_NAME']}/formulation_scheme.csv"
    #     )
    #     return form_scheme, tubes_req, dilution_factor

    # def formulate_gridscan(
    #     self,
    #     tubes_req,
    #     form_scheme: pd.DataFrame,
    #     solution1,
    #     solution2,
    #     dilution_factor,
    # ):
    #     useable_tubes = find_container(
    #         containers=self.containers, type="tube 15", amount=tubes_req
    #     )
    #     sum_vol1_per_conc1 = form_scheme.groupby("dil1")["vol1"].sum().reset_index()
    #     sum_vol2_per_conc2 = form_scheme.groupby("dil2")["vol2"].sum().reset_index()
    #     self.formulater.formulate_dilution_tube(
    #         dilution_df=sum_vol1_per_conc1,
    #         solution=solution1,
    #         dilution_factor=dilution_factor,
    #     )
    #     self.formulater.formulate_dilution_tube(
    #         dilution_df=sum_vol2_per_conc2,
    #         solution=solution2,
    #         dilution_factor=dilution_factor,
    #     )

    # def _perform_grid_measurement(
    #     self, solutions, form_scheme, concentrations, max_measure_time, well_volume
    #     ):
    #     batches = [
    #         form_scheme.iloc[i : i + self.batch_size]
    #         for i in range(0, len(form_scheme), self.batch_size)
    #     ]

    #     for idx, batch_df in enumerate(batches, start=1):
    #         self.logger.info(f"Processing batch {idx} with {len(batch_df)} wells")
    #         self.logger.info(batch_df)
    #         self.logger.info("-" * 40)

    #         self.formulater.formulate_batches(
    #             batch_df=batch_df,
    #             well_volume=well_volume
    #         )

    #         for _, row in batch_df.iterrows():
    #             well_id = row["well_id"]
    #             self.logger.info(
    #                 f"Start pendant drop measurement of {well_id}, "
    #                 f"containing {concentrations[0]} mM {solutions[0]} and "
    #                 f"{concentrations[1]} mM {solutions[1]}."
    #             )

    #             (   dynamic_surface_tension,
    #                 drop_volume,
    #                 drop_count,
    #                 measure_time,
    #                 wt_number,
    #             ) = self.droplet_manager.measure_pendant_drop(
    #                 source=well_id,
    #                 max_measure_time=max_measure_time
    #             )

    #             drop_parameters = self._create_drop_parameters(
    #                 drop_volume=drop_volume,
    #                 measure_time=measure_time,
    #                 drop_count=drop_count,
    #                 wt_number=wt_number,
    #             )

    #             self._append_and_save_results_binary(
    #                 dynamic_surface_tension=dynamic_surface_tension,
    #                 well_id=well_id,
    #                 drop_parameters=drop_parameters,
    #                 solutions=solutions,
    #                 concentrations=concentrations,
    #             )

    def active_learning_loop(
        self, solutions, dilution_range=100, upper_bounds=None, clearance=0.08
    ) -> None:
        """Runs the entire active learning experiment over multiple iterations.

        Args:
            solutions (list): A list of identifiers for the stock solutions being used.
            dilution_range (int, optional): Factor to determine the lower concentration
                bound from the upper bound. Defaults to 100.
            upper_bounds (list, optional): A list of floats specifying the maximum
                concentration for each solution. Defaults to None, which triggers
                a default of [1, 1].
            clearance (float, optional): The percentage of the boundary to avoid when
                generating the initial sample points. Defaults to 0.08.

        Returns:
            tuple: A tuple containing:
                - space (np.ndarray): The complete grid of all possible experimental points.
                - batch (list): The list of points from the final iteration of the loop.
        """

        # Initializing all variables
        max_measure_time = self.settings["MAX_MEASUREMENT_TIME"]
        self.well_vol = self.settings["WELL_VOLUME"]
        self.min_pip_vol = self.settings["MINIMUM_PIPETTING_VOLUME"]

        num_dimensions = len(solutions)
        if upper_bounds == None:
            upper_bounds_x = 1
            upper_bounds_y = 1
            upper_bounds = [upper_bounds_x, upper_bounds_y]
        lower_bounds = [ub / dilution_range for ub in upper_bounds]
        self.min_conc = lower_bounds
        al_iteration = 0

        # Building space, both fully functional
        space = self._build_accessible_space(
            upper_bounds=upper_bounds, lower_bounds=lower_bounds
        )
        batch = self.generate_constrained_qmc(
            num_points=self.batch_size,
            num_dimensions=num_dimensions,
            lower_bound=lower_bounds,
            upper_bound=upper_bounds,
            space=space,
            boundary_percent=clearance,
        )
        # Fully functional
        stock_x = find_container(containers=self.containers, content=solutions[0], type="tube 50")  
        stock_y = find_container(containers=self.containers, content=solutions[1], type="tube 50")

        stock_x.sort(reverse=True)
        stock_y.sort(reverse=True)

        new_stock_x = self.formulater.formulate_new_master_stock(upper_bounds=upper_bounds[0]*2, master_stock=stock_x[0], solution=solutions[0])
        new_stock_y = self.formulater.formulate_new_master_stock(upper_bounds=upper_bounds[1]*2, master_stock=stock_y[0], solution=solutions[1])

        dil_series_x, dil_series_y = self.find_optimal_dilution_series(
            stock1=self.containers[new_stock_x].get_concentration(solute_name=solutions[0]),
            stock2=self.containers[new_stock_y].get_concentration(solute_name=solutions[1]),
            grid=space,
        )

        dilution_count= len(dil_series_x+dil_series_y)
        empty_list = find_container(containers=self.containers, content="empty", type="tube", amount=dilution_count)
        assert dilution_count <= len(empty_list), f"There are less tubes then required, {dilution_count} vs. {len(empty_list)}."

        self.formulater.formulate_serial_dilution(goals=dil_series_x, vials=empty_list, solution=solutions[0], stock_location=new_stock_x)
        self.formulater.formulate_serial_dilution(goals=dil_series_x, vials=empty_list, solution=solutions[1], stock_location=new_stock_y)

        # Changing the type, since it throws an error otherwise
        batch = [row.tolist() if isinstance(row, np.ndarray) else row for row in batch]

        # FINDING REQUIRED VOLUMES AND DILUTION VIALS FOR FORMULATION
        initial_batch, initial_volume_data_list = (
            self.calculate_optimal_volumes_for_batch(
                batch=batch, d1_list=dil_series_x, d2_list=dil_series_y
            )
        )

        for (final_c1, final_c2), init_volumes in zip(
            initial_batch, initial_volume_data_list
        ):
            source_well = self.formulater.formulate_single_point(
                surfactant_1=solutions[0],
                concentration_1=final_c1,
                stock_conc_1=init_volumes["stock_1_vols"],
                volume_1=init_volumes["vol1_uL"],

                surfactant_2=solutions[1],
                concentration_2=final_c2,
                stock_conc_2=init_volumes["stock_2_vols"],
                volume_2=init_volumes["vol2_uL"],

                total_well_volume=self.well_vol,
            )
            # Start of measurement, should be fully functional
            (
                dynamic_surface_tension,
                drop_volume,
                drop_count,
                measure_time,
                wt_number,
                init_drop_vol,
            ) = self.droplet_manager.measure_pendant_drop(
                source=source_well, max_measure_time=max_measure_time
            )
            drop_parameters = self._create_drop_parameters(
                drop_volume=drop_volume,
                measure_time=measure_time,
                drop_count=drop_count,
                wt_number=wt_number,
                init_drop_vol=init_drop_vol,
            )
            self._append_and_save_results(
                dynamic_surface_tension=dynamic_surface_tension,
                well_id=source_well,
                drop_parameters=drop_parameters,
            )

        while al_iteration < 20:  # Placeholder

            X_next, sigma, pred = self.suggest_next_points(
                space=space, n_dimensions=num_dimensions, data=batch
            )

            new_batch, volume_data_list = self.calculate_optimal_volumes_for_batch(
                batch=batch, d1_list=dil_series_x, d2_list=dil_series_y
            )

            # Check extensively
            for (final_c1, final_c2), volumes in zip(new_batch, volume_data_list):
                source_well = self.formulater.formulate_single_point(
                    surfactant_1=solutions[0],
                    concentration1=final_c1,
                    volume_1=volumes["vol1_uL"],
                    surfactant_2=solutions[1],
                    concentration2=final_c2,
                    volume_2=volumes["vol2_uL"],
                    total_well_volume=self.well_vol,
                )

                # Start of measurement, should be fully functional
                (
                    dynamic_surface_tension,
                    drop_volume,
                    drop_count,
                    measure_time,
                    wt_number,
                    init_drop_vol,
                ) = self.droplet_manager.measure_pendant_drop(
                    source=source_well, max_measure_time=max_measure_time
                )
                drop_parameters = self._create_drop_parameters(
                    drop_volume=drop_volume,
                    measure_time=measure_time,
                    drop_count=drop_count,
                    wt_number=wt_number,
                    init_drop_vol=init_drop_vol,
                )
                self._append_and_save_results(
                    dynamic_surface_tension=dynamic_surface_tension,
                    well_id=source_well,
                    drop_parameters=drop_parameters,
                )

                append_sigma(
                results=self.results_sigma,
                well_id=source_well,
                containers=self.containers,
                pred=pred,
                sigma=sigma,
            )

            al_iteration += 1
            batch = X_next
            batch = [
                row.tolist() if isinstance(row, np.ndarray) else row for row in batch
            ]

        self.logger.info("Finished grid search protocol.\n\n\n")

    def _check_all_points_feasible(self, stock1: float, stock2: float, grid: list[list[float]], factor_x: float, factor_y: float) -> bool:
        """Helper to check if a specific factor pair is feasible for the entire grid."""
        # This reuses the logic from the original loops, but is now clearly separated.
        dil1 = self.generate_dilution_series(stock1, self.min_conc[0], factor_x)
        dil2 = self.generate_dilution_series(stock2, self.min_conc[1], factor_y)

        # Check all grid points. If any is infeasible, return False immediately.
        for c1, c2 in grid:
            if not self.is_point_feasible(c1, c2, dil1, dil2):
                return False
        return True

    def _find_max_factor_univariately_binary(
        self,
        stock_to_optimize: float,
        min_conc_to_optimize: float,
        fixed_stock: float,
        fixed_min_conc: float,
        fixed_factor: float, # Factor for the *other* component
        grid: list[list[float]],
        is_x_optimization: bool,
        min_allowed_factor: float,
        max_factor: float,
        step: float,
    ) -> float:
        """Uses binary search to find the highest feasible dilution factor for one component."""

        def check_fn(factor_opt):
            # Pass the factors in the correct order to the feasibility check
            if is_x_optimization:
                return self._check_all_points_feasible(stock_to_optimize, fixed_stock, grid, factor_opt, fixed_factor)
            else:
                return self._check_all_points_feasible(fixed_stock, stock_to_optimize, grid, fixed_factor, factor_opt)

        low = min_allowed_factor
        high = max_factor
        best_factor = min_allowed_factor

        # Use a fixed number of iterations for precision (e.g., 50 is typically enough)
        for _ in range(50):
            mid = (low + high) / 2
            if check_fn(mid):
                best_factor = mid
                low = mid # Try a higher factor
            else:
                high = mid # Too high, search lower

        # Round down the result to the nearest multiple of 'step' to align with the
        # original search granularity and ensure feasibility.
        best_factor = math.floor(best_factor / step) * step

        # Ensure result is within the explicit bounds
        return max(min_allowed_factor, min(max_factor, best_factor))

    def find_optimal_dilution_series(
        self,
        stock1: float,
        stock2: float,
        grid: list[list[float]],
        min_allowed_factor: float = 1.2,
        step: float = 0.1,
        max_factor: float = 20.0,
    ) -> tuple[list[float], list[float]]:
        """
        Finds the highest feasible dilution factors for both components that still
        allow formulation of all points in a discrete grid, returning the dilution series.
        Optimized using binary search for efficiency.
        """
        # 1. Optimize Factor X (fixed_factor_y starts at minimum allowed)
        fixed_factor_y = min_allowed_factor
        best_factor_x = self._find_max_factor_univariately_binary(
            stock1, self.min_conc[0], stock2, self.min_conc[1],
            fixed_factor_y, grid, is_x_optimization=True,
            min_allowed_factor=min_allowed_factor, max_factor=max_factor, step=step
        )
        

        
        
        # 2. Optimize Factor Y (fixed_factor_x is the newly found optimal factor)
        best_factor_y = self._find_max_factor_univariately_binary(
            stock2, self.min_conc[1], stock1, self.min_conc[0],
            best_factor_x, grid, is_x_optimization=False,
            min_allowed_factor=min_allowed_factor, max_factor=max_factor, step=step
        )

       

        
        # 3. Final Result Generation
        dil_series_x = self.generate_dilution_series(
            stock1, self.min_conc[0], best_factor_x
        )


        dil_series_y = self.generate_dilution_series(
            stock2, self.min_conc[1], best_factor_y
        )

        return dil_series_x, dil_series_y

    def generate_dilution_series(self, stock_conc, min_required, factor):
        dilutions = [stock_conc]
        while dilutions[-1] / factor >= min_required:
            dilutions.append(dilutions[-1] / factor)
        return dilutions

    def is_point_feasible(
        self,
        target_conc_sol1,
        target_conc_sol2,
        dilution_series_sol1,
        dilution_series_sol2,
    ):

        for dilution_factor_sol1 in dilution_series_sol1:

            for dilution_factor_sol2 in dilution_series_sol2:
                
                vol_sol1 = (target_conc_sol1 / dilution_factor_sol1) * self.well_vol
                vol_sol2 = (target_conc_sol2 / dilution_factor_sol2) * self.well_vol
                vol_water = self.well_vol - vol_sol1 - vol_sol2
                if all(
                    [
                        vol_sol1 >= self.min_pip_vol or np.isclose(vol_sol1, 0),
                        vol_sol2 >= self.min_pip_vol or np.isclose(vol_sol2, 0),
                        vol_water >= self.min_pip_vol or np.isclose(vol_water, 0),
                        self.well_vol == round(vol_sol1 + vol_sol2 + vol_water, 0),
                    ]
                ):
                    return True

        return False

    def calculate_optimal_volumes_for_batch(
        self, batch: list[list[float]], d1_list: list[float], d2_list: list[float]
    ) -> tuple[list[list[float]], list[dict[str]]]:
        """
        Calculates the required volumes and verified concentrations for a batch of
        target concentration pairs, choosing the stock dilutions (d1/d2)
        that minimize the volume spread.

        Args:
            batch (list[list[float]]): Target concentration pairs [[c1, c2], ...].
            d1_list (list[float]): The optimal stock dilution series for component 1.
            d2_list (list[float]): The optimal stock dilution series for component 2.

        Returns:
            tuple[list[list[float]], list[dict[str]]]:
                - Verified final concentrations [[final_c1, final_c2], ...].
                - list of volume dictionaries for pipetting.
        """
        new_batch = []
        volume_data_list = []

        sorted_d1_list = sorted(d1_list)
        sorted_d2_list = sorted(d2_list)
        for target_c1, target_c2 in batch:
            candidates = []
            for d1 in sorted_d1_list:
                for d2 in sorted_d2_list:
                    if (
                        target_c1 > 0
                        and d1 < target_c1
                        and not np.isclose(target_c1, 0)
                    ):
                        continue
                    if (
                        target_c2 > 0
                        and d2 < target_c2
                        and not np.isclose(target_c2, 0)
                    ):
                        continue

                    vol1 = (
                        round((target_c1 / d1) * self.well_vol)
                        if not np.isclose(target_c1, 0)
                        else 0
                    )
                    vol2 = (
                        round((target_c2 / d2) * self.well_vol)
                        if not np.isclose(target_c2, 0)
                        else 0
                    )
                    self.logger.info("vol1 and 2", vol1, vol2)
                    if (vol1 + vol2) > self.well_vol:
                        continue

                    vol_water = self.well_vol - vol1 - vol2
                    if vol_water < 0:
                        continue
                    vol_water = round(vol_water)

                    vol1_ok = np.isclose(vol1, 0) or vol1 >= self.min_pip_vol
                    vol2_ok = np.isclose(vol2, 0) or vol2 >= self.min_pip_vol
                    water_ok = np.isclose(vol_water, 0) or vol_water >= self.min_pip_vol

                    if not (vol1_ok and vol2_ok and water_ok):
                        continue

                    final_c1 = round((vol1 * d1) / self.well_vol, 6)
                    final_c2 = round((vol2 * d2) / self.well_vol, 6)
                    spread = max(vol1, vol2, vol_water) - min(vol1, vol2, vol_water)
                    candidates.append(
                        {
                            "stock1_conc": d1,
                            "stock2_conc": d2,
                            "vol1_uL": vol1,
                            "vol2_uL": vol2,
                            "water_uL": vol_water,
                            "final_conc1": final_c1,
                            "final_conc2": final_c2,
                            "spread": spread,
                        }
                    )

            if candidates:
                best = min(candidates, key=lambda x: x["spread"])
                new_batch.append([best["final_conc1"], best["final_conc2"]])
                volume_data_list.append(
                    {
                        "stock1_conc": best["stock1_conc"],
                        "vol1_uL": best["vol1_uL"],
                        "stock2_conc": best["stock2_conc"],
                        "vol2_uL": best["vol2_uL"],
                        "water_uL": best["water_uL"],
                    }
                )
            else:
                new_batch.append([np.nan, np.nan])
                volume_data_list.append(
                    {
                        "stock1_conc": np.nan,
                        "vol1_uL": np.nan,
                        "stock2_conc": np.nan,
                        "vol2_uL": np.nan,
                        "water_uL": np.nan,
                    }
                )

        return new_batch, volume_data_list

    def generate_constrained_qmc(
        self,
        num_points,  # Number of points to generate (e.g., 8)
        num_dimensions,  # Number of dimensions (number of concentrations)
        lower_bound,  # Array or list of lower bounds for each dimension 		Do I need bounds and space here? TODO
        upper_bound,  # Array or list of upper bounds for each dimension
        space,
        boundary_percent=0.08,  # Percentage to stay away from the outer boundary
    ):
        """Generates space-filling points using a constrained Latin Hypercube Sample.

        This method creates an initial set of experimental points that are well-distributed
        across the parameter space but avoids the extreme edges. It first generates
        points in a continuous space using LHS and then "snaps" each point to the
        nearest available point on the discrete experimental grid (`space`).

        Args:
            num_points (int): The number of sample points to generate.
            num_dimensions (int): The number of variables (e.g., concentrations).
            lower_bound (list or np.ndarray): The minimum value for each dimension.
            upper_bound (list or np.ndarray): The maximum value for each dimension.
            space (np.ndarray): The discrete grid of all possible experimental points.
            boundary_percent (float, optional): The percentage of the space to avoid
                at the boundaries. Defaults to 0.08.

        Returns:
            list: A list of lists, where each inner list represents a unique,
                selected point from the discrete `space`.
        """
        lower_bound = np.array(lower_bound, dtype=float)
        upper_bound = np.array(upper_bound, dtype=float)

        # 1. Define the inner bounds to exclude the outer 8%
        inner_lower_bound = lower_bound + (upper_bound - lower_bound) * boundary_percent
        inner_upper_bound = upper_bound - (upper_bound - lower_bound) * boundary_percent

        # 2. Generate points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=num_dimensions, seed=np.random.randint(1, 10000))
        sample = sampler.random(n=num_points)

        # 3. Scale the generated points to the inner bounds
        scaled_points = qmc.scale(
            sample, l_bounds=inner_lower_bound, u_bounds=inner_upper_bound
        )
        tree = cKDTree(space)
        used_indices = set()
        qmc_space_points = []

        for pt in scaled_points:
            # Get all grid points sorted by distance
            _, indices = tree.query(pt, k=len(space))
            for idx in indices:
                if idx not in used_indices:
                    qmc_space_points.append(space[idx])
                    used_indices.add(idx)
                    break
        return qmc_space_points

    def _build_accessible_space(self, upper_bounds, lower_bounds, points_per_axis=9):
        """Creates a discrete grid of all possible experimental points.

        This function defines the complete search space for the algorithm. It generates
        logarithmically spaced values for each dimension between the specified bounds
        and then creates a mesh grid of all possible combinations.

        Args:
            upper_bounds (list): A list of floats for the max value of each dimension.
            lower_bounds (list): A list of floats for the min value of each dimension.
            points_per_axis (int): The number of points on a single axis, total number will be ^2.

        Returns:
            np.ndarray: An array of shape (N, D) where N is the total number of
                        possible points and D is the number of dimensions.
        """
        x_vals = np.round(
            np.logspace(
                start=np.log10(upper_bounds[0]),
                stop=np.log10(lower_bounds[0]) + 1e-12,
                num=points_per_axis,
            ),
            2,
        )
        y_vals = np.round(
            np.logspace(
                start=np.log10(upper_bounds[1]),
                stop=np.log10(lower_bounds[1]) + 1e-12,
                num=points_per_axis,
            ),
            2,
        )
        xx, yy = np.meshgrid(x_vals, y_vals)
        return np.c_[xx.ravel(), yy.ravel()]

    def suggest_next_points(
        self, space, n_dimensions, data=None, round_decimals=3, threshold=1
    ):
        """Suggests the next batch of points to measure using a Gaussian Process model.

        This is the core of the active learning algorithm. It trains a Gaussian Process
        (GP) Regressor on all data collected so far. It then uses the trained model to
        predict the outcome and uncertainty (sigma) for every point in the entire
        `space`. Finally, it selects a diverse batch of new points from the regions of
        highest uncertainty to be measured next.

        Args:
            space (np.ndarray): The grid of all possible experimental points.
            n_dimensions (int): The number of experimental variables (dimensions).
            data (list, optional): A list of lists, where each inner list contains
                the coordinates and the measured result of a past experiment,
                e.g., `[[x1, y1, result1], [x2, y2, result2]]`. Defaults to None.
            round_decimals (int, optional): The number of decimal places to use when
                checking for previously tested points. Defaults to 3.
            threshold (float, optional): The uncertainty (sigma) value below which
                the algorithm may consider the space to be fully explored, stopping
                the process. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - X_next (np.ndarray): An array of shape (batch_size, D) with the
                suggested points for the next experiment.
                - max_sigma (float): The highest uncertainty value found among the
                candidate points.
                - pred (np.ndarray): The model's prediction for all points in `space`.
            Returns `None` if the stopping criteria (e.g., max_sigma < threshold) are met.
        """
        if data is None:
            self.logger.error("No data given, quitting...")
            return

        data_np = np.array(data)
        X = data_np[:, :n_dimensions]  # first 2 columns
        y = data_np[:, 2]  # last column

        # Get a set of keys for all points that have already been tested
        tested_grid_keys = set([tuple(np.round(row, round_decimals)) for row in X])

        # Train the Gaussian Process model on the existing data
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=42,
        )
        gp.fit(X, y)

        # Predict uncertainty for all points in the accessible space
        pred, sigma = gp.predict(space, return_std=True)

        # Filter out points that have already been tested
        all_grid_keys = [tuple(np.round(row, 3)) for row in space]
        mask_candidates = np.array([k not in tested_grid_keys for k in all_grid_keys])
        candidates = space[mask_candidates]
        sigma_candidates = sigma[mask_candidates]

        # Check for stopping criteria
        max_sigma = float(np.max(sigma_candidates)) if len(sigma_candidates) else 0.0
        self.logger.info(f"Max sigma among candidates: {max_sigma:.4f}")
        if (len(candidates) == 0) or (max_sigma < threshold):
            return None

        # Select the next batch of diverse points
        sel_idx_local = self._select_diverse_batch(candidates, sigma_candidates, 0.03)
        X_next = candidates[sel_idx_local]

        return X_next, max_sigma, pred

    def _select_diverse_batch(self, points, sigmas, min_separation):
        """Selects a batch of points that are both highly uncertain and spatially diverse.

        This function implements a greedy algorithm. It sorts candidate points by their
        uncertainty (sigma) and iteratively selects points. A point is only chosen if
        it is at least `min_separation` distance away from all other points already
        selected for the batch. This prevents the algorithm from clustering all new
        measurements in a single high-uncertainty region.

        Args:
            points (np.ndarray): An array of candidate points to choose from.
            sigmas (np.ndarray): The array of uncertainties corresponding to each point.
            min_separation (float): The minimum Euclidean distance required between any
                two points in the selected batch.

        Returns:
            np.ndarray: An array of integer indices corresponding to the selected
                        points from the input `points` array.
        """
        # 1. Sort the candidate points by descending uncertainty (highest sigma first)
        order = np.argsort(-sigmas)
        selected_indices = []
        excluded_indices = set()

        log_points = np.log10(points)
        all_dists = cdist(log_points, log_points)

        # 2. Iterate through the points in order of decreasing uncertainty
        for idx in order:
            if idx in excluded_indices:
                continue

            p = log_points[idx]

            # Always select the very first point (it has the highest uncertainty)
            if len(selected_indices) == 0:
                selected_indices.append(idx)
            else:
                dists = cdist(log_points[selected_indices], p[None, :]).ravel()
                if np.all(dists >= min_separation):
                    selected_indices.append(idx)
                else:
                    continue

            # Exclude 8 nearest neighbours
            neighbor_indices = np.argsort(all_dists[idx])[1:9]
            excluded_indices.update(neighbor_indices)

            # Stop once the desired batch size is reached
            if len(selected_indices) == self.batch_size:
                break

        selected_points = np.array(selected_indices, dtype=int)
        # Return the indices of the selected diverse points
        return selected_points

    def characterize_surfactant(self) -> None:
        """
        Characterize surfactants by performing explore and exploit phases.
        """
        self.logger.info("Starting characterization protocol...\n\n\n")
        self.settings = load_settings()  # Update settings
        characterization_info = load_info(
            file_name=self.settings["CHARACTERIZATION_INFO_FILENAME"]
        )
        explore_points = int(self.settings["EXPLORE_POINTS"])
        exploit_points = int(self.settings["EXPLOIT_POINTS"])

        # self._check_needle_position()
        # self.formulater.wash(repeat=self.repeat)
        for i, surfactant in enumerate(characterization_info["surfactant"]):
            self.logger.info(f"Start characterization of {surfactant}.\n\n")
            row_id = characterization_info["row id"][i]
            max_measure_time = float(characterization_info["measure time"][i])

            # Perform serial dilution
            self.formulater.serial_dilution(
                row_id=row_id,
                solution_name=surfactant,
                n_dilutions=explore_points,
                well_volume=float(self.settings["WELL_VOLUME"]),
                dilution_factor=float(self.settings["DILUTION_FACTOR"]),
            )

            # Explore phase
            self._perform_explore_phase(
                surfactant=surfactant,
                row_id=row_id,
                explore_points=explore_points,
                max_measure_time=max_measure_time,
            )

            # Exploit phase
            self._perform_exploit_phase(
                surfactant=surfactant,
                row_id=row_id,
                explore_points=explore_points,
                exploit_points=exploit_points,
                max_measure_time=max_measure_time,
            )
        if self.left_pipette.has_needle:
            self.left_pipette.return_needle()
        self.logger.info("Finished characterization protocol.\n\n\n")
        play_sound("DATA DATA.")

    def _perform_explore_phase(
        self, surfactant: str, row_id: str, explore_points: int, max_measure_time: float
    ) -> None:
        """
        Perform the explore phase for a given surfactant.
        """
        self.logger.info(f"Start explore phase for surfactant {surfactant}.\n")
        for i in reversed(
            range(explore_points)
        ):  # Reverse order for low to high concentration
            well_id_explore = f"{row_id}{i+1}"
            source_well = self.containers[well_id_explore]
            self.logger.info(
                f"Start pendant drop measurement of {source_well.WELL_ID}, containing {source_well.get_concentration()} mM {source_well.get_solution()}.\n"
            )
            (
                dynamic_surface_tension,
                drop_volume,
                drop_count,
                measure_time,
                wt_number,
                init_drop_vol
            ) = self.droplet_manager.measure_pendant_drop(
                source=source_well, max_measure_time=max_measure_time
            )
            drop_parameters = self._create_drop_parameters(
                drop_volume=drop_volume,
                measure_time=measure_time,
                drop_count=drop_count,
                wt_number=wt_number,
                init_drop_vol=init_drop_vol
            )
            self._append_and_save_results(
                dynamic_surface_tension=dynamic_surface_tension,
                well_id=well_id_explore,
                drop_parameters=drop_parameters,
            )

    def _perform_exploit_phase(
        self,
        surfactant: str,
        row_id: str,
        explore_points: int,
        exploit_points: int,
        max_measure_time: float,
    ) -> None:
        """
        Perform the exploit phase for a given surfactant.
        """
        self.logger.info(f"Start exploit phase for surfactant {surfactant}.\n")
        for i in range(exploit_points):
            well_id_exploit = f"{row_id}{explore_points+i+1}"

            suggest_concentration, st_at_suggestion = self.learner.suggest(
                results=self.results, solution_name=surfactant
            )
            self.formulater.wash(return_needle=False, repeat=self.repeat)
            self.formulater.formulate_exploit_point(
                suggest_concentration=suggest_concentration,
                solution_name=surfactant,
                well_volume=float(self.settings["WELL_VOLUME"]),
                well_id_exploit=well_id_exploit,
            )
            self.logger.info(
                f"Start pendant drop measurement of {well_id_exploit}, containing {self.containers[well_id_exploit].concentration} mM {surfactant}.\n"
            )
            self.left_pipette.mixing(
                container=self.containers[well_id_exploit], mix=("before", 20, 5)
            )
            dynamic_surface_tension, drop_volume, drop_count, init_drop_vol = (
                self.droplet_manager.measure_pendant_drop(
                    source=self.containers[well_id_exploit],
                    max_measure_time=max_measure_time,
                )
            )
            drop_parameters = self._create_drop_parameters(
                drop_volume=drop_volume,
                measure_time=max_measure_time,
                drop_count=drop_count,
            )
            self._append_and_save_results(
                point_type="exploit",
                dynamic_surface_tension=dynamic_surface_tension,
                well_id=well_id_exploit,
                drop_parameters=drop_parameters,
                solution_name=surfactant,
                plot_type="concentrations",
            )
        self.formulater.wash(return_needle=False, repeat=self.repeat)

    def _create_drop_parameters(
        self,
        drop_volume: float,
        measure_time: float,
        drop_count: int,
        wt_number: float,
        init_drop_vol: float,
    ) -> dict:
        """
        Create a dictionary of drop parameters.
        """
        return {
            "drop_volume": drop_volume,
            "measure_time": measure_time,
            "drop_count": drop_count,
            "wt_number": wt_number,
            "initial_drop_volume": init_drop_vol
        }

    def _append_and_save_results(
        self,
        dynamic_surface_tension: list,
        well_id: str,
        drop_parameters: dict,
    ) -> None:
        """
        Append results, save them, and plot the results. Plots based on the specified plot type.
        """
        all_solutes_sorted = sorted(self.containers[well_id].get_solution())
        if not all_solutes_sorted:
            all_solutes_sorted = ["water"] 
        self.results = append_results(
            results=self.results,
            dynamic_surface_tension=dynamic_surface_tension,
            well_id=well_id,
            drop_parameters=drop_parameters,
            n_eq_points=self.n_measurement_in_eq,
            containers=self.containers,
            sensor_api=self.sensor_api,
            all_solutes_sorted=all_solutes_sorted
        )
        save_results(self.results)
        container = self.containers[well_id]
        solutes_in_well = sorted(list(container.solutes.keys()))

        # If the list is empty, it's water; otherwise, join with '_'
        solution_name_for_plot = "water" if not solutes_in_well else "_".join(solutes_in_well)

        self.plotter.plot_results_concentration(
            df=self.results, solution_name=solution_name_for_plot
            )

    def _check_needle_position(self):
        self.left_pipette.pick_up_needle()
        self.left_pipette.move_to_well(container=self.containers["drop_stage"])

        time.sleep(30)
        # play_sound("Remove hands from the deck within 5 seconds.")
        play_sound("Beep")
        time.sleep(5)

        self.left_pipette.move_to_well(
            container=self.containers["drop_stage"],
            depth_offset=-10,
        )


### legacy ###

# def random_measurement(self) -> None:
# """
# Perform pendant drop measurements for all wells specified in the settings.
# """
# self.logger.info("Starting measure wells protocol...\n\n\n")
# self.settings = load_settings()
# well_info = load_info(file_name=self.settings["WELL_INFO_FILENAME"])
# wells_ids = well_info["location"].astype(str) + well_info["well"].astype(str)

# try:
#     df = pd.read_csv("experiments/csv_file.csv")
# except Exception as e:
#     self.logger.error(f"Error loading formulation CSV: {e}")
#     return

# if len(df) != len(wells_ids):
#     self.logger.error(f"Mismatch: {len(wells_ids)} wells but {len(df)} formulation rows.")
#     return

# for i, well_id in enumerate(wells_ids):
#     row = df.iloc[i]
#     conc_dict = row.dropna().to_dict()
#     self.logger.info(f"Row {i}: {conc_dict}")
# for i, well_id in enumerate(wells_ids):
#     row = df.iloc[i]
#     conc_dict = row.dropna().to_dict()
#     # self.formulater.formulate_random_single_well(well_id=well_id, concentrations=conc_dict)   single well

#     self._measure_single_well(well_id)

# def calibrate(self):
# self.logger.info("Starting calibration...")
# drop_parameters = {
# "drop_volume": 12,
# "max_measure_time": 60,
# "flow_rate": 1,
# }  # standard settings for calibration
# scale_t, drop_parameters = self.droplet_manager.measure_pendant_drop(
# source=self.containers["7A1"],
# drop_parameters=drop_parameters,
# calibrate=True,
# )
# save_calibration_data(scale_t)
# average_scale = calculate_average_in_column(x=scale_t, column_index=1)
# self.logger.info(f"Finished calibration, average scale is: {average_scale}")
# play_sound("Calibration done.")

# def characterize_surfactant_old(self):
#     self.logger.info("Starting characterization protocol...")

#     # general information
#     self.settings = load_settings()  # update settings
#     characterization_info = load_info(
#         file_name=self.settings["CHARACTERIZATION_INFO_FILENAME"]
#     )
#     explore_points = int(self.settings["EXPLORE_POINTS"])
#     exploit_points = int(self.settings["EXPLOIT_POINTS"])

#     for i, surfactant in enumerate(characterization_info["surfactant"]):
#         row_id = characterization_info["row id"][i]
#         measure_time = float(characterization_info["measure time"][i])
#         self.formulater.serial_dilution(
#             row_id=row_id,
#             solution_name=surfactant,
#             n_dilutions=explore_points,
#             well_volume=float(self.settings["WELL_VOLUME"]),
#         )
#         for i in reversed(
#             range(explore_points)
#         ):  # reverse order to go from low to high concentration
#             well_id_explore = f"{row_id}{i+1}"
#             drop_volume_suggestion = suggest_volume(
#                 results=self.results,
#                 next_concentration=float(
#                     self.containers[well_id_explore].concentration
#                 ),
#                 solution_name=surfactant,
#             )
#             drop_parameters = {
#                 "drop_volume": drop_volume_suggestion,
#                 "max_measure_time": measure_time,
#                 "flow_rate": float(self.settings["FLOW_RATE"]),
#             }
#             dynamic_surface_tension, drop_parameters = (
#                 self.droplet_manager.measure_pendant_drop(
#                     source=self.containers[well_id_explore],
#                     drop_parameters=drop_parameters,
#                 )
#             )
#             self.results = append_results(
#                 results=self.results,
#                 point_type="explore",
#                 dynamic_surface_tension=dynamic_surface_tension,
#                 well_id=well_id_explore,
#                 drop_parameters=drop_parameters,
#                 n_eq_points=self.n_measurement_in_eq,
#                 containers=self.containers,
#                 sensor_api=self.sensor_api,
#             )
#             save_results(self.results)
#             self.plotter.plot_results_concentration(
#                 df=self.results, solution_name=surfactant
#             )

#         self.formulater.wash(repeat=3, return_needle=True)

#         for i in range(exploit_points):
#             well_id_exploit = f"{row_id}{explore_points+i+1}"
#             suggest_concentration, st_at_suggestion = self.learner.suggest(
#                 results=self.results, solution_name=surfactant
#             )
#             drop_volume_suggestion = volume_for_st(st_at_suggestion)
#             self.formulater.formulate_exploit_point(
#                 suggest_concentration=suggest_concentration,
#                 solution_name=surfactant,
#                 well_volume=float(self.settings["WELL_VOLUME"]),
#                 well_id_exploit=well_id_exploit,
#             )
#             drop_parameters = {
#                 "drop_volume": drop_volume_suggestion,
#                 "max_measure_time": float(self.settings["EQUILIBRATION_TIME"]),
#                 "flow_rate": float(self.settings["FLOW_RATE"]),
#             }
#             dynamic_surface_tension, drop_parameters = (
#                 self.droplet_manager.measure_pendant_drop(
#                     source=self.containers[well_id_exploit],
#                     drop_parameters=drop_parameters,
#                 )
#             )
#             self.results = append_results(
#                 results=self.results,
#                 point_type="exploit",
#                 dynamic_surface_tension=dynamic_surface_tension,
#                 well_id=well_id_exploit,
#                 drop_parameters=drop_parameters,
#                 n_eq_points=self.n_measurement_in_eq,
#                 containers=self.containers,
#                 sensor_api=self.sensor_api,
#             )
#             save_results(self.results)
#             self.plotter.plot_results_concentration(
#                 df=self.results, solution_name=surfactant
#             )
#             self.formulater.wash(repeat=3, return_needle=True)

#     self.logger.info("Finished characterization protocol.")
#     play_sound("DATA DATA.")

# def measure_same_well(self, well_id: str, repeat: int = 3):
#     drop_parameters = {"drop_volume": 6, "max_measure_time": 60, "flow_rate": 1}
#     for i in range(repeat):
#         dynamic_surface_tension, drop_parameters = (
#             self.droplet_manager.measure_pendant_drop(
#                 source=self.containers[well_id], drop_parameters=drop_parameters
#             )
#         )
#         # self.left_pipette.wash()
#         df = pd.DataFrame(
#             dynamic_surface_tension, columns=["time (s)", "surface tension (mN/m)"]
#         )
#         df.to_csv(
#             f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{well_id}/dynamic_surface_tension_{i}.csv"
#         )
#     if self.left_pipette.has_needle:
#         self.left_pipette.return_needle()

# def measure_same_well_cali(self, well_id: str, repeat: int = 3):
#     drop_parameters = {"drop_volume": 11, "max_measure_time": 30, "flow_rate": 1}
#     for i in range(repeat):
#         scale = (
#             self.droplet_manager.measure_pendant_drop(
#                 source=self.containers[well_id], drop_parameters=drop_parameters, calibrate=True
#             )
#         )
#         # self.left_pipette.wash()
#         df = pd.DataFrame(
#             scale, columns=["time (s)", "scale"]
#         )
#         df.to_csv(
#             f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{well_id}/scale{i}.csv"
#         )

#     if self.left_pipette.has_needle:
#         self.left_pipette.return_needle()

# def measure_plate(self, well_volume: float, solution_name: str, plate_location: int):
# # TODO save results correctly!
# self.logger.info("Starting measure whole plate protocol...")
# self.droplet_manager.set_max_retries = 1
# self.logger.info(self.droplet_manager.MAX_RETRIES)
# self.settings = load_settings()  # update settings
# well_info = load_info(file_name=self.settings["WELL_INFO_FILENAME"])
# wells_ids = well_info["location"].astype(str) + well_info["well"].astype(str)
# # self.formulater.fill_plate(well_volume=well_volume, solution_name=solution_name, plate_location=plate_location)

# for i, well_id in enumerate(wells_ids):
#     drop_parameters = {
#         "drop_volume": float(well_info["drop volume (uL)"][i]),
#         "max_measure_time": float(self.settings["EQUILIBRATION_TIME"]),
#         "flow_rate": float(well_info["flow rate (uL/s)"][i]),
#     }
#     dynamic_surface_tension, drop_parameters = self.droplet_manager.measure_pendant_drop(
#         source=self.containers[well_id], drop_parameters=drop_parameters
#     )
#     self.results = append_results(
#         results=self.results,
#         point_type="None",
#         dynamic_surface_tension=dynamic_surface_tension,
#         well_id=well_id,
#         drop_parameters=drop_parameters,
#         n_eq_points=self.n_measurement_in_eq,
#         containers=self.containers,
#         sensor_api=self.sensor_api,
#     )
#     save_results(self.results)
#     self.plotter.plot_results_well_id(df=self.results)

# self.logger.info("Done measuring plate.")
# play_sound("DATA DATA.")
