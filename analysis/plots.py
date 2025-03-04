import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.load_save_functions import load_settings
from utils.logger import Logger
from utils.utils import smooth_list


class Plotter:

    def __init__(self):
        self.settings = load_settings()
        self.fontsize_labels = 15
        self.window_size = 20
        self.logger = Logger(
            name="protocol",
            file_path=f"experiments/{self.settings['EXPERIMENT_NAME']}/meta_data",
        )

    def plot_results_well_id(self, df: pd.DataFrame):
        try:
            if not df.empty:
                wells_ids = df["well id"]
                st_eq = df["surface tension eq. (mN/m)"]

                fig, ax = plt.subplots()
                ax.bar(wells_ids, st_eq, color="C0")
                ax.set_xlabel("Well ID", fontsize=self.fontsize_labels)
                ax.set_ylabel("Surface Tension Eq. (mN/m)", fontsize=self.fontsize_labels)
                ax.set_title(
                    f"Results experiment {self.settings['EXPERIMENT_NAME']}",
                    fontsize=self.fontsize_labels,
                )
                ax.tick_params(axis="x", rotation=90)
                plt.tight_layout()

                # save in experiment folder and plots cache for web interface
                plt.savefig(f"experiments/{self.settings['EXPERIMENT_NAME']}/results_plot.png")
                plt.savefig("server/static/plots_cache/results_plot.png")
                plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Plotter: could not create plot results with well IDs. Error: {e}")
            self._create_empty_plot("results_plot")

    def plot_dynamic_surface_tension(self, dynamic_surface_tension: list, well_id: str, drop_count: int):
        try:
            if dynamic_surface_tension:
                # Ensure consistent lengths for time and surface tension
                lengths = [len(item) for item in dynamic_surface_tension]
                min_length = min(lengths)
                dynamic_surface_tension = [
                    item[:min_length] for item in dynamic_surface_tension
                ]

                df = pd.DataFrame(
                    dynamic_surface_tension, columns=["time (s)", "surface tension (mN/m)"]
                )

                t = df["time (s)"]
                st = df["surface tension (mN/m)"]

                t_smooth = smooth_list(x=t, window_size=self.window_size)
                st_smooth = smooth_list(x=st, window_size=self.window_size)

                fig, ax = plt.subplots()
                ax.plot(t_smooth, st_smooth, lw=2, color="black")
                ax.set_xlim(0, t_smooth[-1] + 5)
                ax.set_ylim(20, 80)
                ax.set_xlabel("Time (s)", fontsize=self.fontsize_labels)
                ax.set_ylabel("Surface Tension (mN/m)", fontsize=self.fontsize_labels)
                ax.set_title(f"Well ID: {well_id}, drop count: {drop_count}", fontsize=self.fontsize_labels)
                ax.grid(axis="y")

                plt.savefig(f"experiments/{self.settings['EXPERIMENT_NAME']}/data/{well_id}/dynamic_surface_tension_plot_{drop_count}.png")
                plt.savefig("server/static/plots_cache/dynamic_surface_tension_plot.png")
                plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Plotter: could not create dynamic surface tension plot for well ID: {well_id}, drop count: {drop_count}. Error: {e}")
            self._create_empty_plot(f"dynamic_surface_tension_plot_{drop_count}")

    def plot_results_concentration(self, df: pd.DataFrame, solution_name: str):
        try:
            if not df.empty:
                df_solution = df.loc[df["solution"] == solution_name]
                concentrations = df_solution["concentration"]
                st_eq = df_solution["surface tension eq. (mN/m)"]

                fig, ax = plt.subplots()
                ax.scatter(concentrations, st_eq)
                ax.set_ylim(20, 80)
                ax.set_xscale("log")
                ax.set_xlabel("Concentration", fontsize=self.fontsize_labels)
                ax.set_ylabel("Surface Tension Eq. (mN/m)", fontsize=self.fontsize_labels)
                ax.set_title(
                    f"Results experiment {self.settings['EXPERIMENT_NAME']}, solution: {solution_name}",
                    fontsize=self.fontsize_labels,
                    )
                plt.tight_layout()

                # save in experiment folder and plots cache for web interface
                plt.savefig(f"experiments/{self.settings['EXPERIMENT_NAME']}/results_plot_{solution_name}.png")
                plt.savefig("server/static/plots_cache/results_plot.png")
                plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Plotter: could not create plot results with concentrations for solution: {solution_name}. Error: {e}")
            self._create_empty_plot(f"results_plot_{solution_name}")

    def _create_empty_plot(self, plot_name: str):
        fig, ax = plt.subplots()
        ax.set_title("Empty Plot")
        plt.savefig(f"experiments/{self.settings['EXPERIMENT_NAME']}/{plot_name}.png")
        plt.savefig(f"server/static/plots_cache/{plot_name}.png")
        plt.close(fig)