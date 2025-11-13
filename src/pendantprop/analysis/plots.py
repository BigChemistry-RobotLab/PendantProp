# Imports

## Packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

## Custom code
from opentrons_api.containers import Container
from pendantprop.utils.utils import smooth_list


class Plotter:
    def __init__(self, settings: dict):
        self.file_settings = settings["file_settings"]
        self.save_root = (
            f"{self.file_settings['output_folder']}/{self.file_settings['exp_tag']}"
        )
        self.server_plot_folder = self.file_settings["plot_folder_server"]
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.server_plot_folder, exist_ok=True)
        self.fontsize_labels = 15
        self.window_size = 20

    def plot_results_sample_id(self, df: pd.DataFrame):
        try:
            if not df.empty:
                sample_ids = df["sample id"]
                st_eq = df["surface tension eq. (mN/m)"]

                fig, ax = plt.subplots()
                ax.bar(sample_ids, st_eq, color="C0")
                ax.set_xlabel("Sample ID", fontsize=self.fontsize_labels)
                ax.set_ylabel(
                    "Surface Tension Eq. (mN/m)", fontsize=self.fontsize_labels
                )
                ax.set_title(
                    f"{self.file_settings['exp_tag']}",
                    fontsize=self.fontsize_labels,
                )
                ax.tick_params(axis="x", rotation=90)
                plt.tight_layout()

                # save in experiment folder and plots cache for web interface
                plt.savefig(
                        f"{self.save_root}/results_plot.png"
                )
                plt.savefig(f"{self.server_plot_folder}/results_plot.png")
                plt.close(fig)

        except Exception as e:
            print(
                f"Plotter: could not create plot results with well IDs. Error: {e}"
            )
            self._create_empty_plot("results_plot")

    def plot_dynamic_surface_tension(
        self, dynamic_surface_tension: list, container: Container, drop_count: int
    ):
        sample_id = container.sample_id
        try:
            if dynamic_surface_tension:
                # Ensure consistent lengths for time and surface tension
                lengths = [len(item) for item in dynamic_surface_tension]
                min_length = min(lengths)
                dynamic_surface_tension = [
                    item[:min_length] for item in dynamic_surface_tension
                ]

                df = pd.DataFrame(
                    dynamic_surface_tension,
                    columns=["time (s)", "surface tension (mN/m)"],
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
                ax.set_title(
                    f"Sample ID: {sample_id}, drop count: {drop_count}",
                    fontsize=self.fontsize_labels,
                )
                ax.grid(axis="y")
                dir = f"{self.save_root}/{self.file_settings['data_folder']}/{sample_id}"
                os.makedirs(dir, exist_ok=True)
                plt.savefig(
                    f"{dir}/dynamic_surface_tension_plot_{drop_count}.png"
                )
                plt.savefig(
                    f"{self.server_plot_folder}/dynamic_surface_tension_plot.png"
                )
                plt.close(fig)
                
        except Exception as e:
            print(
                f"Plotter: could not create dynamic surface tension plot for sample ID: {container.sample_id}, drop count: {drop_count}. Error: {e}"
            )
            self._create_empty_plot(f"dynamic_surface_tension_plot_{drop_count}")

    def plot_results_concentration(self, df: pd.DataFrame, solution_name: str):
        try:
            if not df.empty:
                df_solution = df.loc[df["solution"] == solution_name]
                # concentrations = df_solution["concentration"]
                # st_eq = df_solution["surface tension eq. (mN/m)"]
                point_types = df_solution["point type"]

                fig, ax = plt.subplots()

                # Plot explore points in blue (C0)
                explore_points = df_solution[point_types == "explore"]
                ax.scatter(
                    explore_points["concentration"],
                    explore_points["surface tension eq. (mN/m)"],
                    color="C0",
                    label="Explore",
                )

                # Plot exploit points in orange (C1)
                exploit_points = df_solution[point_types == "exploit"]
                ax.scatter(
                    exploit_points["concentration"],
                    exploit_points["surface tension eq. (mN/m)"],
                    color="C1",
                    label="Exploit",
                )

                ax.set_ylim(20, 80)
                ax.set_xscale("log")
                ax.set_xlabel("Concentration", fontsize=self.fontsize_labels)
                ax.set_ylabel(
                    "Surface Tension Eq. (mN/m)", fontsize=self.fontsize_labels
                )
                #TODO fix
                # ax.set_title(
                #     f"{self.file_settings['EXPERIMENT_NAME']}, solution: {solution_name}",
                #     fontsize=self.fontsize_labels,
                # )
                # ax.legend()
                plt.tight_layout()

                # save in experiment folder and plots cache for web interface
                # plt.savefig(
                #     f"experiments/{self.file_settings['EXPERIMENT_NAME']}/results_plot_{solution_name}.png"
                # )
                plt.savefig(f"{self.server_plot_folder}/results_plot.png")
                plt.close(fig)
        except Exception as e:
            print(
                f"Plotter: could not create plot results with concentrations for solution: {solution_name}. Error: {e}"
            )
            self._create_empty_plot(f"results_plot_{solution_name}")

    def _create_empty_plot(self, plot_name: str):
        fig, ax = plt.subplots()
        ax.set_title("Empty Plot")
        plt.savefig(
            f"{self.save_root}/{plot_name}.png"
        )
        plt.savefig(f"{self.server_plot_folder}/{plot_name}.png")
        plt.close(fig)
