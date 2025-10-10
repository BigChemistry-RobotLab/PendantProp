import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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
                    f"{self.settings['EXPERIMENT_NAME']}",
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


    # def plot_results_concentration(self, df: pd.DataFrame, solution_name: list):
    #     try:
    #         if not df.empty:
    #             df_solution = df[df["solution"] == solution_name].copy()

    #             seen = {}

    #             fig, ax = plt.subplots()

    #             for _, row in df_solution.iterrows():
    #                 conc = row["concentration"]
    #                 st = row["surface tension eq. (mN/m)"]
    #                 key = (solution_name, conc)

    #                 run_index = seen.get(key, 0)
    #                 color = f"C{run_index % 10}"

    #                 ax.scatter(conc, st, color=color, label=f"Run {run_index + 1}" if key not in seen else None)

    #                 seen[key] = run_index + 1

    #             ax.set_ylim(20, 80)
    #             ax.set_xscale("log")
    #             ax.set_xlabel("Concentration", fontsize=self.fontsize_labels)
    #             ax.set_ylabel("Surface Tension Eq. (mN/m)", fontsize=self.fontsize_labels)
    #             ax.set_title(
    #                 f"{self.settings['EXPERIMENT_NAME']}, solution: {solution_name}",
    #                 fontsize=self.fontsize_labels,
    #             )

    #             handles, labels = ax.get_legend_handles_labels()
    #             by_label = dict(zip(labels, handles))
    #             ax.legend(by_label.values(), by_label.keys())

    #             plt.tight_layout()

    #             plt.savefig(f"experiments/{self.settings['EXPERIMENT_NAME']}/results_plot_{solution_name}.png")
    #             plt.savefig("server/static/plots_cache/results_plot.png")
    #             plt.close(fig)

    #     except Exception as e:
    #         self.logger.warning(
    #             f"Plotter: could not create plot results with concentrations for solution: {solution_name}. Error: {e}"
    #         )
    #         self._create_empty_plot(f"results_plot_{solution_name}")

    def plot_results_concentration(self, df: pd.DataFrame, solution_name: list, mode: str = "total"):
        """
        Extended version of plot_results_concentration.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must include columns: surfactant_1..n, concentration_1..n, and surface tension eq. (mN/m)
        solution_name : list
            List of surfactants to plot (e.g. ["SDS", "NaCl"])
        mode : str
            "total" -> sum of concentrations (default)
            "ratio" -> plot per ratio group (approximate mixture composition)
            "heatmap" -> show 2D heatmap of conc_1 vs conc_2 vs surface tension
        """

        try:
            if df.empty:
                self.logger.warning("Plotter: provided DataFrame is empty, cannot plot.")
                return

            # --- 1️⃣ Identify surfactant/concentration columns dynamically
            surf_cols = [c for c in df.columns if c.startswith("surfactant_")]
            conc_cols = [c for c in df.columns if c.startswith("concentration_")]

            # --- 2️⃣ Filter rows matching the surfactant combination
            def row_matches(row):
                surf_in_row = [row[c] for c in surf_cols if pd.notna(row[c])]
                return sorted(surf_in_row) == sorted(solution_name)

            df_mix = df[df.apply(row_matches, axis=1)].copy()
            if df_mix.empty:
                self.logger.warning(f"No data found for {solution_name}")
                return

            # --- 3️⃣ MODE HANDLER
            if mode == "total":
                self._plot_total_concentration(df_mix, conc_cols, solution_name)

            elif mode == "ratio":
                self._plot_by_ratio(df_mix, conc_cols, solution_name)

            elif mode == "heatmap":
                self._plot_heatmap(df_mix, conc_cols, solution_name)

            else:
                raise ValueError(f"Unknown plot mode: {mode}")

        except Exception as e:
            label = "_".join(solution_name)
            self.logger.warning(f"Plotter failed for {label} in mode {mode}: {e}")
            self._create_empty_plot(f"results_plot_{label}_{mode}")

    def _plot_total_concentration(self, df_mix, conc_cols, solution_name):
        seen = {}
        fig, ax = plt.subplots()

        for _, row in df_mix.iterrows():
            total_conc = sum([row[c] for c in conc_cols if pd.notna(row[c])])
            st = row["surface tension eq. (mN/m)"]

            key = (tuple(sorted(solution_name)), total_conc)
            run_index = seen.get(key, 0)
            color = f"C{run_index % 10}"

            ax.scatter(total_conc, st, color=color, label=f"Run {run_index + 1}" if key not in seen else None)
            seen[key] = run_index + 1

        ax.set_xscale("log")
        ax.set_ylim(20, 80)
        ax.set_xlabel("Total Concentration (mM)", fontsize=self.fontsize_labels)
        ax.set_ylabel("Surface Tension Eq. (mN/m)", fontsize=self.fontsize_labels)
        ax.set_title(f"{self.settings['EXPERIMENT_NAME']}, mixture: {', '.join(solution_name)}", fontsize=self.fontsize_labels)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
        plt.tight_layout()
        plt.savefig(f"experiments/{self.settings['EXPERIMENT_NAME']}/results_plot_{'_'.join(solution_name)}_total.png")
        plt.close(fig)

    def _plot_by_ratio(self, df_mix, conc_cols, solution_name, tolerance=0.05):
        """
        Groups data points by approximate concentration ratio between surfactants
        and plots surface tension vs total concentration.

        Parameters
        ----------
        df_mix : pd.DataFrame
            Subset of data for a given mixture.
        conc_cols : list
            Columns like ["concentration_1", "concentration_2", ...].
        solution_name : list
            Names of surfactants (["SDS", "NaCl"], etc.)
        tolerance : float
            Allowed fractional difference between ratio groups (default = 0.05 = ±5%).
        """

        fig, ax = plt.subplots()

        # compute normalized ratios
        def get_ratio_vector(row):
            concs = np.array([row[c] for c in conc_cols], dtype=float)
            total = np.sum(concs)
            if total == 0:
                return None
            return concs / total  # normalized vector (sums to 1)

        df_mix["ratio_vector"] = df_mix.apply(get_ratio_vector, axis=1)

        # --- Group similar ratios manually based on tolerance ---
        ratio_groups = []
        group_labels = []

        for vec in df_mix["ratio_vector"]:
            if vec is None:
                group_labels.append(None)
                continue

            # check if close to an existing group
            found_group = False
            for gi, gvec in enumerate(ratio_groups):
                if np.allclose(vec, gvec, rtol=tolerance, atol=tolerance):
                    group_labels.append(gi)
                    found_group = True
                    break

            # if no match, create a new group
            if not found_group:
                ratio_groups.append(vec)
                group_labels.append(len(ratio_groups) - 1)

        df_mix["ratio_group"] = group_labels

        # --- Plot each ratio group separately ---
        for gi, group in df_mix.groupby("ratio_group"):
            if gi is None:
                continue
            total_conc = group[conc_cols].sum(axis=1)
            label = ", ".join([f"{s}:{v:.2f}" for s, v in zip(solution_name, ratio_groups[gi])])
            ax.scatter(total_conc, group["surface tension eq. (mN/m)"], label=label)

        ax.set_xscale("log")
        ax.set_ylim(20, 80)
        ax.set_xlabel("Total Concentration (mM)")
        ax.set_ylabel("Surface Tension Eq. (mN/m)")
        ax.set_title(f"Surface tension by ratio (±{int(tolerance*100)}%): {', '.join(solution_name)}")

        ax.legend()
        plt.tight_layout()
        plt.savefig(f"experiments/{self.settings['EXPERIMENT_NAME']}/results_plot_{'_'.join(solution_name)}_ratio.png")
        plt.close(fig)


    def _plot_heatmap(self, df_mix, conc_cols, solution_name):
        if len(conc_cols) < 2:
            self.logger.warning("Heatmap mode requires at least two surfactants.")
            return

        conc1, conc2 = conc_cols[:2]
        st_col = "surface tension eq. (mN/m)"

        # Pivot into 2D grid
        pivot_df = df_mix.pivot_table(
            index=conc2, columns=conc1, values=st_col, aggfunc="mean"
        )

        fig, ax = plt.subplots()
        c = ax.imshow(
            pivot_df.values,
            origin="lower",
            aspect="auto",
            extent=[
                pivot_df.columns.min(), pivot_df.columns.max(),
                pivot_df.index.min(), pivot_df.index.max(),
            ],
            cmap="coolwarm",  # Blue = high, red = low
            norm=Normalize(vmin=pivot_df.min().min(), vmax=pivot_df.max().max()),
        )

        ax.set_xlabel(f"{solution_name[0]} concentration (mM)")
        ax.set_ylabel(f"{solution_name[1]} concentration (mM)")
        ax.set_title(f"Surface tension heatmap: {', '.join(solution_name)}")

        fig.colorbar(ScalarMappable(norm=c.norm, cmap=c.cmap), ax=ax, label="Surface tension (mN/m)")
        plt.tight_layout()
        plt.savefig(f"experiments/{self.settings['EXPERIMENT_NAME']}/results_plot_{'_'.join(solution_name)}_heatmap.png")
        plt.close(fig)



    def _create_empty_plot(self, plot_name: str):
        fig, ax = plt.subplots()
        ax.set_title("Empty Plot")
        plt.savefig(f"experiments/{self.settings['EXPERIMENT_NAME']}/{plot_name}.png")
        plt.savefig(f"server/static/plots_cache/{plot_name}.png")
        plt.close(fig)
